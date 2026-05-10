"""Full multi-criteria assessment runner.

Orchestrates all assessment layers on a set of candidate spots:
    1. Skyglow PSF prediction (predicted SQM)
    2. ERA5 cloud cover (clear-night fraction)
    3. Sky View Factor (terrain openness)
    4. ESA WorldCover (land cover suitability)
    5. Terrain slope (accessibility)
    6. SSI computation (weighted combination)
    7. Final filtering (remove unsuitable spots)

This is the final step before export -- it takes raw OSM-enriched spots
and produces the scientifically-scored, multi-criteria-filtered output.
"""

import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def run_full_assessment(
    spots: gpd.GeoDataFrame,
    radiance: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    region: str = "mainland",
) -> gpd.GeoDataFrame:
    """Run all multi-criteria assessment layers on a set of spots.

    Args:
        spots: GeoDataFrame with Point geometry (output of enrichment step).
        radiance: Full 2D radiance grid (3-year median, all pixels).
        x_coords: VIIRS grid longitude coordinates.
        y_coords: VIIRS grid latitude coordinates.
        region: Region name for cloud cover cache lookup.

    Returns:
        GeoDataFrame with all assessment columns added and SSI computed.
    """
    t0 = time.time()
    spots = spots.copy()
    n_initial = len(spots)
    logger.info("=" * 60)
    logger.info("MULTI-CRITERIA ASSESSMENT: %d spots", n_initial)
    logger.info("=" * 60)

    # 1. Skyglow PSF -> predicted SQM
    logger.info("--- [1/6] Skyglow propagation (PSF convolution) ---")
    from stargazing_spots.skyglow import assess_spots as skyglow_assess
    spots = skyglow_assess(spots, radiance, x_coords, y_coords)
    logger.info("  Done: mean SQM = %.2f", spots["predicted_sqm"].mean())

    # 2. Cloud cover (CLAAS-3 5km satellite or ERA5 25km fallback)
    logger.info("--- [2/7] Cloud cover (clear-night fraction) ---")
    try:
        from stargazing_spots.cloud_cover import (
            ClearSkyConfig,
            compute_clear_night_fraction,
            resample_to_viirs_grid,
        )

        # Try CLAAS-3 (5km satellite-observed) first, fall back to ERA5 (25km model)
        claas3_file = Path(__file__).parent.parent / "input" / "portugal" / "cache" / "claas3_clear_night_mainland.nc"
        if claas3_file.exists() and region == "mainland":
            import xarray as xr
            clear_fraction_era5 = xr.open_dataarray(str(claas3_file))
            logger.info("  Using CLAAS-3 (5km satellite, 11200 grid cells)")
        else:
            config = ClearSkyConfig()
            clear_fraction_era5 = compute_clear_night_fraction(config=config, region=region)
            logger.info("  Using ERA5 (25km model reanalysis)")

        # Resample ERA5 grid to VIIRS coordinates, then sample at spot locations
        clear_resampled = resample_to_viirs_grid(clear_fraction_era5, x_coords, y_coords)

        # Sample at spot locations
        spot_cols = np.array([int(np.argmin(np.abs(x_coords - g.x))) for g in spots.geometry])
        spot_rows = np.array([int(np.argmin(np.abs(y_coords - g.y))) for g in spots.geometry])
        clear_at_spots = np.array([
            clear_resampled[r, c] if 0 <= r < clear_resampled.shape[0] and 0 <= c < clear_resampled.shape[1] else np.nan
            for r, c in zip(spot_rows, spot_cols)
        ])
        spots["clear_night_fraction"] = np.round(clear_at_spots, 3)
        logger.info("  Done: mean clear fraction = %.1f%%", np.nanmean(clear_at_spots) * 100)
    except Exception as e:
        logger.warning("  Cloud cover FAILED: %s (continuing without)", e)
        spots["clear_night_fraction"] = np.nan

    # 3-5. SVF + Land cover + Slope -- run in PARALLEL (all are independent HTTP I/O)
    logger.info("--- [3-5/7] SVF + Land cover + Slope (parallel) ---")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _compute_svf():
        from stargazing_spots.sky_view import compute_svf_at_points
        return compute_svf_at_points(spots)  # Returns (svf, dem_elevation) tuple

    def _compute_land_cover():
        from stargazing_spots.land_cover import get_land_cover_at_points
        return get_land_cover_at_points(spots)

    def _compute_slope():
        from stargazing_spots.accessibility import compute_slope_at_points
        return compute_slope_at_points(spots)

    svf_result = None
    lc_result = None
    slope_result = None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_compute_svf): "svf",
            executor.submit(_compute_land_cover): "land_cover",
            executor.submit(_compute_slope): "slope",
        }

        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
                if task_name == "svf":
                    svf_result = result
                    logger.info("  SVF done: mean=%.3f", np.nanmean(result))
                elif task_name == "land_cover":
                    lc_result = result
                    logger.info("  Land cover done")
                elif task_name == "slope":
                    slope_result = result
                    logger.info("  Slope done: mean=%.1f deg", np.nanmean(result))
            except Exception as e:
                logger.warning("  %s FAILED: %s", task_name, e)

    # Apply SVF results (returns tuple: svf_array, dem_elevation_array)
    if svf_result is not None:
        svf_values, dem_elev = svf_result
        spots["sky_view_factor"] = np.round(svf_values, 3)
        spots["dem_elevation_m"] = np.round(dem_elev, 0)
        # Use DEM elevation instead of OSM 'ele' tag (100% coverage, authoritative)
        spots["ele"] = dem_elev
    else:
        spots["sky_view_factor"] = np.nan
        spots["dem_elevation_m"] = np.nan

    # Apply land cover results
    if lc_result is not None:
        from stargazing_spots.land_cover import LAND_COVER_CLASSES
        class_codes, lc_scores = lc_result
        spots["land_cover_class"] = class_codes
        spots["land_cover_label"] = [LAND_COVER_CLASSES.get(c, "unknown") for c in class_codes]
        spots["land_cover_score"] = lc_scores
    else:
        spots["land_cover_score"] = np.nan

    # Apply slope results
    if slope_result is not None:
        from stargazing_spots.accessibility import classify_slope
        spots["terrain_slope_deg"] = np.round(slope_result, 1)
        scores_labels = [classify_slope(s) for s in slope_result]
        spots["slope_score"] = [s[0] for s in scores_labels]
        spots["slope_class"] = [s[1] for s in scores_labels]
    else:
        spots["slope_score"] = np.nan

    # 6. Nighttime accessibility check
    logger.info("--- [6/7] Nighttime accessibility ---")
    try:
        from stargazing_spots.access_check import assess_access
        spots = assess_access(spots)
        n_restricted = (~spots["night_accessible"]).sum()
        if n_restricted > 0:
            logger.info("  Warning: %d spots have restricted access", n_restricted)
    except Exception as e:
        logger.warning("  Access check FAILED: %s (continuing without)", e)

    # 7. Compute SSI (weighted combination of all layers)
    logger.info("--- [7/7] Stargazing Suitability Index (SSI) ---")
    from stargazing_spots.suitability import compute_ssi
    spots = compute_ssi(spots)

    # Summary
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("ASSESSMENT COMPLETE (%.1fs)", elapsed)
    logger.info("=" * 60)
    for cls in ["excellent", "good", "acceptable", "marginal", "poor"]:
        count = (spots["ssi_class"] == cls).sum()
        if count > 0:
            logger.info("  %s: %d spots", cls, count)
    logger.info("  Total: %d spots assessed", len(spots))

    return spots


def filter_unsuitable(spots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove spots that are physically unsuitable for stargazing.

    Uses vegetation-aware SVF as the primary sky-visibility filter instead
    of binary land cover classification.

    Args:
        spots: GeoDataFrame with assessment columns.

    Returns:
        Filtered GeoDataFrame with unsuitable spots removed.
    """
    n_initial = len(spots)

    # Primary filter: SVF-based sky visibility (strict: need 70%+ of sky visible)
    if "sky_view_factor" in spots.columns:
        obstructed = spots["sky_view_factor"] < 0.7
        n_obstructed = obstructed.sum()
        spots = spots[~obstructed]
        if n_obstructed > 0:
            logger.info("  Removed %d spots (SVF < 0.7: insufficient sky visibility)", n_obstructed)

    # Secondary filter: water and dense built-up only (NOT tree cover -- handled by SVF)
    if "land_cover_class" in spots.columns:
        unsuitable_lc = spots["land_cover_class"].isin([50, 80, 95])  # built-up, water, mangrove
        n_lc = unsuitable_lc.sum()
        spots = spots[~unsuitable_lc]
        if n_lc > 0:
            logger.info("  Removed %d spots (land cover: urban/water)", n_lc)

    # Tertiary filter: remove spots with explicitly restricted nighttime access
    if "night_accessible" in spots.columns:
        restricted = ~spots["night_accessible"]
        n_restricted = restricted.sum()
        spots = spots[~restricted]
        if n_restricted > 0:
            logger.info("  Removed %d spots (access=private/no, barrier=gate)", n_restricted)

    # Filter: SSI floor — keep only good and excellent stargazing locations
    # SSI < 65 = marginal/acceptable/poor → not worth recommending
    if "ssi_score" in spots.columns:
        too_low = spots["ssi_score"] < 65
        n_low = too_low.sum()
        spots = spots[~too_low]
        if n_low > 0:
            logger.info("  Removed %d spots (SSI < 65: not good/excellent)", n_low)

    n_final = len(spots)
    logger.info("  Filter result: %d -> %d spots (%d removed)",
                n_initial, n_final, n_initial - n_final)

    # Tiering: classify spots by infrastructure level
    import pandas as pd
    tier1_tags = {
        "tourism": ["viewpoint", "camp_site", "picnic_site", "wilderness_hut"],
        "amenity": ["shelter"],
        "man_made": ["observatory"],
        "leisure": ["nature_reserve", "bird_hide"],
    }
    is_tier1 = pd.Series(False, index=spots.index)
    for col, values in tier1_tags.items():
        if col in spots.columns:
            is_tier1 |= spots[col].isin(values)
    spots["tier"] = np.where(is_tier1, "destination", "candidate")
    n_dest = is_tier1.sum()
    logger.info("  Tiered: %d destinations + %d candidates", n_dest, n_final - n_dest)

    # Spatial deduplication: within 1km, keep only the highest-SSI spot
    if "ssi_score" in spots.columns and len(spots) > 1:
        from scipy.spatial import cKDTree
        spots = spots.reset_index(drop=True)
        spots_utm = spots.to_crs(epsg=32629)
        coords = np.column_stack([spots_utm.geometry.x, spots_utm.geometry.y])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=1000)

        to_remove = set()
        for i, j in pairs:
            if i in to_remove or j in to_remove:
                continue
            if spots.iloc[i]["ssi_score"] >= spots.iloc[j]["ssi_score"]:
                to_remove.add(j)
            else:
                to_remove.add(i)

        if to_remove:
            keep_mask = ~spots.index.isin(to_remove)
            spots = spots[keep_mask]
            logger.info("  Deduplication: removed %d spots within 1km of a better spot", len(to_remove))

    # Grid-based density cap: max 5 spots per 0.25 deg cell (~25km grid)
    if "ssi_score" in spots.columns and len(spots) > 100:
        cell_size = 0.25
        keep_indices = []
        lat_range = (spots.geometry.y.min(), spots.geometry.y.max())
        lon_range = (spots.geometry.x.min(), spots.geometry.x.max())

        for lat in np.arange(lat_range[0], lat_range[1], cell_size):
            for lon in np.arange(lon_range[0], lon_range[1], cell_size):
                in_cell = spots[
                    (spots.geometry.y >= lat) & (spots.geometry.y < lat + cell_size) &
                    (spots.geometry.x >= lon) & (spots.geometry.x < lon + cell_size)
                ]
                if len(in_cell) > 0:
                    top = in_cell.nlargest(min(5, len(in_cell)), "ssi_score")
                    keep_indices.extend(top.index.tolist())

        n_before_cap = len(spots)
        spots = spots.loc[keep_indices]
        n_removed = n_before_cap - len(spots)
        if n_removed > 0:
            logger.info("  Density cap: removed %d spots (max 5 per 25km cell)", n_removed)

    return spots.reset_index(drop=True)
