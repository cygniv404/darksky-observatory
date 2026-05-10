"""Probabilistic land cover scoring from ESA WorldCover + Hansen canopy.

Tree pixels (code 10) scored continuously via Hansen canopy cover:
score = 1 - canopy_fraction. Other classes have fixed scores.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)

# ESA WorldCover S3 base URL (public, no auth)
# ESA WorldCover — local first, HTTP fallback
_LOCAL_WC_DIR = Path(__file__).parent.parent / "input" / "portugal" / "worldcover"
ESA_WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# Hansen Global Forest Change v1.11 (2023) — continuous canopy cover (public)
# Hansen — local first, HTTP fallback
_LOCAL_HANSEN_DIR_LC = Path(__file__).parent.parent / "input" / "portugal" / "hansen"
HANSEN_BASE = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"

# Class definitions
LAND_COVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# Base suitability mapping for NON-tree classes: code → (score 0-1, label)
# Tree cover (10) is handled probabilistically via Hansen canopy cover.
SUITABILITY = {
    10: (None, "probabilistic"),  # Determined by Hansen canopy cover
    20: (0.8, "good"),            # shrubland — low, open above
    30: (1.0, "excellent"),       # grassland — fully open
    40: (0.6, "acceptable"),      # cropland — open but variable
    50: (0.0, "unsuitable"),      # built-up — lights/buildings
    60: (1.0, "excellent"),       # bare/sparse — no obstruction
    70: (0.5, "marginal"),        # snow/ice
    80: (0.0, "unsuitable"),      # water — can't stand there
    90: (0.4, "marginal"),        # wetland — open but inaccessible
    95: (0.0, "unsuitable"),      # mangroves
    100: (0.8, "good"),           # moss/lichen — open, high altitude
}

# Probabilistic tree cover thresholds for suitability labels
TREE_PROB_THRESHOLDS = {
    "high_suitability": 0.3,   # canopy < 30% → open woodland, good for stargazing
    "low_suitability": 0.8,    # canopy > 80% → dense forest, unsuitable
}


def _worldcover_tile_url(lat: float, lon: float) -> str:
    """Get WorldCover tile — local first, HTTP fallback."""
    tile_lat = int(np.floor(lat / 3) * 3)
    tile_lon = int(np.floor(lon / 3) * 3)

    ns = f"N{tile_lat:02d}" if tile_lat >= 0 else f"S{abs(tile_lat):02d}"
    ew = f"E{tile_lon:03d}" if tile_lon >= 0 else f"W{abs(tile_lon):03d}"

    tile_name = f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map"

    local = _LOCAL_WC_DIR / f"{tile_name}.tif"
    if local.exists():
        return str(local)
    return f"{ESA_WC_BASE}/{tile_name}.tif"


def _hansen_tile_url(lat: float, lon: float) -> str:
    """Get Hansen tile — local first, HTTP fallback."""
    tile_lat_upper = int(np.ceil(lat / 10) * 10)
    tile_lon_left = int(np.floor(lon / 10) * 10)
    lat_str = f"{abs(tile_lat_upper):02d}{'N' if tile_lat_upper >= 0 else 'S'}"
    lon_str = f"{abs(tile_lon_left):03d}{'E' if tile_lon_left >= 0 else 'W'}"
    filename = f"Hansen_GFC-2023-v1.11_treecover2000_{lat_str}_{lon_str}.tif"
    local = _LOCAL_HANSEN_DIR_LC / filename
    if local.exists():
        return str(local)
    return f"{HANSEN_BASE}/{filename}"


def _canopy_cover_to_score(canopy_fraction: float) -> float:
    """Convert continuous canopy cover fraction (0-1) to suitability score (0-1).

    Uses a linear model: score = 1 - canopy_fraction.
    This means:
        - 0% canopy → 1.0 score (open, excellent)
        - 30% canopy → 0.7 score (sparse woodland/montado, good)
        - 50% canopy → 0.5 score (moderate canopy, acceptable)
        - 80% canopy → 0.2 score (dense forest, poor)
        - 100% canopy → 0.0 score (closed canopy, unsuitable)

    Args:
        canopy_fraction: Hansen tree cover as fraction (0.0 to 1.0).

    Returns:
        Suitability score (0.0 to 1.0).
    """
    return float(np.clip(1.0 - canopy_fraction, 0.0, 1.0))


def _canopy_cover_to_label(canopy_fraction: float) -> str:
    """Assign a suitability label based on continuous canopy cover.

    Args:
        canopy_fraction: Hansen tree cover as fraction (0.0 to 1.0).

    Returns:
        Human-readable suitability label.
    """
    if canopy_fraction < TREE_PROB_THRESHOLDS["high_suitability"]:
        return "good"       # Sparse woodland, open enough for stargazing
    elif canopy_fraction < TREE_PROB_THRESHOLDS["low_suitability"]:
        return "moderate"   # Mixed/edge — don't filter, but reduced score
    else:
        return "unsuitable"  # Dense closed canopy


def get_tree_cover_at_points(
    lats: np.ndarray, lons: np.ndarray, indices: list[int]
) -> dict[int, float]:
    """Get Hansen continuous tree cover (0-1 fraction) for specified point indices.

    Uses HTTP range requests to public COG tiles on Google Cloud Storage.
    Groups points by 10-degree Hansen tile for efficiency.

    Args:
        lats: Array of latitudes for all spots.
        lons: Array of longitudes for all spots.
        indices: List of spot indices to query (only tree_cover pixels).

    Returns:
        Dict mapping spot index to canopy cover fraction (0.0 to 1.0).
    """
    canopy_values = {}

    if not indices:
        return canopy_values

    # Group by Hansen tile (10-degree blocks)
    hansen_tiles = {}
    for idx in indices:
        tile_lat_upper = int(np.ceil(lats[idx] / 10) * 10)
        tile_lon_left = int(np.floor(lons[idx] / 10) * 10)
        key = (tile_lat_upper, tile_lon_left)
        if key not in hansen_tiles:
            hansen_tiles[key] = []
        hansen_tiles[key].append(idx)

    logger.info("  Querying Hansen canopy cover for %d tree pixels across %d tiles...",
                len(indices), len(hansen_tiles))

    for (tile_lat, tile_lon), tile_indices in hansen_tiles.items():
        url = _hansen_tile_url(lats[tile_indices[0]], lons[tile_indices[0]])

        try:
            with rasterio.open(url) as hansen_src:
                for idx in tile_indices:
                    try:
                        row, col = hansen_src.index(lons[idx], lats[idx])
                        if 0 <= row < hansen_src.height and 0 <= col < hansen_src.width:
                            window = Window(col, row, 1, 1)
                            val = hansen_src.read(1, window=window)[0, 0]
                            if val < 255:  # 255 = nodata
                                canopy_values[idx] = float(val) / 100.0
                            else:
                                canopy_values[idx] = 0.0
                        else:
                            canopy_values[idx] = 0.0
                    except Exception:
                        canopy_values[idx] = 0.0
        except Exception as e:
            logger.warning("  Hansen tile failed (%s): %s", url.split("/")[-1], e)
            for idx in tile_indices:
                canopy_values[idx] = 0.5  # Conservative fallback for failed reads

    return canopy_values


def get_land_cover_at_points(spots: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Get land cover class and PROBABILISTIC suitability score at each spot.

    Two-stage approach:
        1. ESA WorldCover 10m provides the base class (tree/grass/built/etc.)
        2. For pixels classified as tree_cover (code 10), Hansen GFC provides
           the CONTINUOUS canopy cover percentage, converted to a suitability
           score instead of the old binary 0.0.

    This eliminates false negatives in montado, open woodland, forest edges,
    and post-fire clearings where WorldCover says "tree" but actual canopy
    is sparse enough for adequate sky visibility.

    Args:
        spots: GeoDataFrame with Point geometry (EPSG:4326).

    Returns:
        Tuple of (class_codes, suitability_scores):
            - class_codes: uint8 array of ESA WorldCover class per spot
            - suitability_scores: float array 0-1 per spot (continuous for trees)
    """
    n_spots = len(spots)
    class_codes = np.zeros(n_spots, dtype=np.uint8)
    suitability_scores = np.full(n_spots, np.nan)

    lons = spots.geometry.x.values
    lats = spots.geometry.y.values

    # --- Stage 1: Get ESA WorldCover class for all spots ---
    tile_keys = {}
    for i in range(n_spots):
        tile_lat = int(np.floor(lats[i] / 3) * 3)
        tile_lon = int(np.floor(lons[i] / 3) * 3)
        key = (tile_lat, tile_lon)
        if key not in tile_keys:
            tile_keys[key] = []
        tile_keys[key].append(i)

    logger.info("Querying land cover for %d spots across %d tiles...", n_spots, len(tile_keys))

    for (tile_lat, tile_lon), indices in tile_keys.items():
        url = _worldcover_tile_url(lats[indices[0]], lons[indices[0]])
        logger.info("  Tile %s: %d spots", url.split("/")[-1][:30], len(indices))

        try:
            with rasterio.open(url) as src:
                for idx in indices:
                    row, col = src.index(lons[idx], lats[idx])

                    # Bounds check
                    if 0 <= row < src.height and 0 <= col < src.width:
                        window = Window(col, row, 1, 1)
                        value = src.read(1, window=window)[0, 0]
                        class_codes[idx] = value

                        # Assign score for non-tree classes immediately
                        if value != 10:
                            score_entry = SUITABILITY.get(value, (0.5, "unknown"))
                            suitability_scores[idx] = score_entry[0] if score_entry[0] is not None else 0.5
                    else:
                        class_codes[idx] = 0
                        suitability_scores[idx] = np.nan

        except Exception as e:
            logger.warning("  Failed to read tile: %s", e)
            for idx in indices:
                class_codes[idx] = 0
                suitability_scores[idx] = np.nan

    # --- Stage 2: For tree_cover pixels, get Hansen continuous canopy cover ---
    tree_indices = [i for i in range(n_spots) if class_codes[i] == 10]

    if tree_indices:
        logger.info("  %d spots classified as tree_cover — refining with Hansen canopy...",
                    len(tree_indices))

        canopy_values = get_tree_cover_at_points(lats, lons, tree_indices)

        for idx in tree_indices:
            canopy_fraction = canopy_values.get(idx, 0.5)  # Conservative fallback
            suitability_scores[idx] = _canopy_cover_to_score(canopy_fraction)

        # Log the probabilistic distribution
        tree_scores = [suitability_scores[i] for i in tree_indices if not np.isnan(suitability_scores[i])]
        if tree_scores:
            tree_scores_arr = np.array(tree_scores)
            n_open = (tree_scores_arr >= 0.7).sum()
            n_moderate = ((tree_scores_arr >= 0.2) & (tree_scores_arr < 0.7)).sum()
            n_dense = (tree_scores_arr < 0.2).sum()
            logger.info("    Probabilistic tree cover results:")
            logger.info("      Open woodland (canopy <30%%): %d spots (score >= 0.7)", n_open)
            logger.info("      Mixed/edge (canopy 30-80%%): %d spots (score 0.2-0.7)", n_moderate)
            logger.info("      Dense forest (canopy >80%%): %d spots (score < 0.2)", n_dense)
            logger.info("    Previously ALL %d would have been scored 0.0 (binary)", len(tree_indices))

    # --- Summary ---
    valid = suitability_scores[~np.isnan(suitability_scores)]
    if len(valid) > 0:
        for code in sorted(LAND_COVER_CLASSES.keys()):
            count = (class_codes == code).sum()
            if count > 0:
                label = LAND_COVER_CLASSES.get(code, "?")
                if code == 10:
                    tree_valid = suitability_scores[class_codes == 10]
                    tree_valid = tree_valid[~np.isnan(tree_valid)]
                    mean_score = tree_valid.mean() if len(tree_valid) > 0 else 0
                    logger.info("    %s (%d): %d spots [probabilistic, mean_score=%.2f]",
                                label, code, count, mean_score)
                else:
                    logger.info("    %s (%d): %d spots", label, code, count)

        unsuitable = (valid == 0.0).sum()
        logger.info("  Summary: %d suitable (score>0), %d unsuitable (score=0)",
                    (valid > 0).sum(), unsuitable)
        logger.info("  Improvement: binary approach would score %d tree spots as 0.0;",
                    len(tree_indices))
        rescued = sum(1 for i in tree_indices
                      if not np.isnan(suitability_scores[i]) and suitability_scores[i] > 0.0)
        logger.info("    probabilistic approach rescues %d of them (canopy < 100%%)", rescued)

    return class_codes, suitability_scores


def get_tree_probability(lat: float, lon: float) -> float:
    """Get continuous tree cover probability (0-1) for a single point.

    Combines ESA WorldCover classification with Hansen continuous canopy
    cover to produce a probability of dense tree canopy at this location.

    Returns:
        - 0.0 if WorldCover says non-tree (grass, bare, crop, etc.)
        - Hansen canopy fraction (0-1) if WorldCover says tree_cover
        - This is the "tree probability" that replaces binary classification

    NOTE on Dynamic World:
        Google Dynamic World provides per-pixel class probabilities (ideal
        for this purpose) but requires Google Earth Engine authentication.
        No public COG endpoint exists. The earthengine-api package is not
        installed. If GEE access becomes available, use:
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterBounds(ee.Geometry.Point([lon, lat]))
                .sort('system:time_start', False)
                .first()
                .select('trees')
        to get the Dynamic World "trees" probability band (0-1).
    """
    # Step 1: Check WorldCover classification
    url = _worldcover_tile_url(lat, lon)
    try:
        with rasterio.open(url) as src:
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                window = Window(col, row, 1, 1)
                wc_class = src.read(1, window=window)[0, 0]
            else:
                return 0.0
    except Exception:
        return 0.5  # Unknown — conservative

    # If not classified as tree, tree probability is low
    if wc_class != 10:
        return 0.0

    # Step 2: Get continuous canopy cover from Hansen
    hansen_url = _hansen_tile_url(lat, lon)
    try:
        with rasterio.open(hansen_url) as hansen_src:
            row, col = hansen_src.index(lon, lat)
            if 0 <= row < hansen_src.height and 0 <= col < hansen_src.width:
                window = Window(col, row, 1, 1)
                val = hansen_src.read(1, window=window)[0, 0]
                if val < 255:
                    return float(val) / 100.0
    except Exception:
        pass

    return 0.5  # Fallback: assume moderate canopy if Hansen fails


def assess_land_cover(spots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add land cover columns to spots GeoDataFrame (probabilistic approach).

    Instead of binary suitable/unsuitable for tree pixels, uses continuous
    canopy cover from Hansen GFC to produce a probabilistic score.

    Args:
        spots: GeoDataFrame with Point geometry.

    Returns:
        GeoDataFrame with added columns:
            - land_cover_class: ESA WorldCover code (10-100)
            - land_cover_label: human-readable class name
            - land_cover_suitable: boolean (True = score > 0.2)
            - land_cover_score: 0-1 suitability score (CONTINUOUS for trees)
            - tree_canopy_fraction: Hansen canopy cover 0-1 (only for tree pixels)
    """
    spots = spots.copy()
    class_codes, scores = get_land_cover_at_points(spots)

    spots["land_cover_class"] = class_codes
    spots["land_cover_label"] = [LAND_COVER_CLASSES.get(c, "unknown") for c in class_codes]
    # Probabilistic threshold: score > 0.2 means canopy < 80%
    spots["land_cover_suitable"] = scores > 0.2
    spots["land_cover_score"] = scores

    # Add tree canopy fraction column for tree pixels (useful for downstream analysis)
    canopy_fractions = np.full(len(spots), np.nan)
    tree_mask = class_codes == 10
    canopy_fractions[tree_mask] = 1.0 - scores[tree_mask]  # Invert: score = 1 - canopy
    spots["tree_canopy_fraction"] = canopy_fractions

    return spots
