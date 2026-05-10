"""Dark-sky classification from VIIRS nighttime radiance.

Threshold at 3 nW/cm2/sr (Bortle 3-4 boundary). Input must be EPSG:4326.
"""

import logging

import geopandas as gpd
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 3.0  # nW/cm²/sr


def validate_geographic_coords(data: xr.DataArray) -> None:
    """Raise ValueError if coordinates are not geographic (lat/lon)."""
    y_min, y_max = float(data.coords["y"].min()), float(data.coords["y"].max())
    x_min, x_max = float(data.coords["x"].min()), float(data.coords["x"].max())

    if not (-90 <= y_min <= 90 and -90 <= y_max <= 90):
        raise ValueError(
            f"Y coordinates [{y_min:.2f}, {y_max:.2f}] are outside geographic bounds [-90, 90]. "
            f"Data appears to be in a projected CRS. Reproject to EPSG:4326 first."
        )
    if not (-180 <= x_min <= 180 and -180 <= x_max <= 180):
        raise ValueError(
            f"X coordinates [{x_min:.2f}, {x_max:.2f}] are outside geographic bounds [-180, 180]. "
            f"Data appears to be in a projected CRS. Reproject to EPSG:4326 first."
        )


def clean_radiance(radiance: xr.DataArray) -> xr.DataArray:
    """Set negative/fill values to NaN.

    Args:
        radiance: Raw radiance DataArray.

    Returns:
        Cleaned DataArray.
    """
    raw_count = int(radiance.count())
    cleaned = radiance.where(radiance >= 0)
    clean_count = int(cleaned.count())
    removed = raw_count - clean_count
    if removed > 0:
        logger.info(
            "Cleaned radiance: removed %d invalid pixels (negative/fill values). "
            "%d valid pixels remain.",
            removed,
            clean_count,
        )
    return cleaned


def classify_dark_sky(
    radiance: xr.DataArray,
    threshold: float = DEFAULT_THRESHOLD,
    drop_empty: bool = False,
) -> xr.DataArray:
    """Retain pixels below threshold; mask others as NaN.

    Args:
        radiance: 2D radiance DataArray (nW/cm2/sr).
        threshold: Maximum radiance for dark-sky classification.
        drop_empty: Remove all-NaN rows/cols.

    Returns:
        Masked DataArray.
    """
    cleaned = clean_radiance(radiance)
    dark_sky = cleaned.where(cleaned < threshold, drop=drop_empty)
    total_pixels = int(cleaned.count())
    dark_pixels = int(dark_sky.count())
    logger.info(
        "Classification: %d/%d pixels below %.1f nW/cm²/sr (%.1f%%)",
        dark_pixels,
        total_pixels,
        threshold,
        100 * dark_pixels / max(total_pixels, 1),
    )
    return dark_sky


def classify_temporal(
    radiance_years: list[xr.DataArray],
    threshold: float = DEFAULT_THRESHOLD,
    year_labels: list[str] | None = None,
) -> dict:
    """Multi-year temporal composite: median, stability, trend, consistency.

    Args:
        radiance_years: List of annual DataArrays (same grid shape).
        threshold: Dark-sky radiance threshold (nW/cm2/sr).
        year_labels: Labels per year.

    Returns:
        Dict with keys: median, dark_sky, stability, trend, consistency, stats.
    """
    if not year_labels:
        year_labels = [str(i) for i in range(len(radiance_years))]

    n_years = len(radiance_years)
    logger.info(
        "Temporal classification: %d years (%s), threshold=%.1f nW/cm²/sr",
        n_years,
        ", ".join(year_labels),
        threshold,
    )

    # Stack into a 3D array (years × y × x)
    stack = np.stack([clean_radiance(r).values for r in radiance_years])

    # Compute temporal statistics
    with np.errstate(all="ignore"):
        median_vals = np.nanmedian(stack, axis=0)
        std_vals = np.nanstd(stack, axis=0)

    # Trend: last year minus first year
    trend_vals = stack[-1] - stack[0]

    # Consistency: how many years was each pixel below threshold?
    dark_per_year = (stack < threshold).astype(np.float32)
    dark_per_year[np.isnan(stack)] = np.nan
    consistency_vals = np.nansum(dark_per_year, axis=0)

    # Build output DataArrays using the coordinate system from the first input
    coords = {"y": radiance_years[0].coords["y"], "x": radiance_years[0].coords["x"]}
    dims = ["y", "x"]

    median_da = xr.DataArray(median_vals, dims=dims, coords=coords,
                             attrs={"units": "nW/cm²/sr", "long_name": "Median Radiance"})
    stability_da = xr.DataArray(std_vals, dims=dims, coords=coords,
                                attrs={"units": "nW/cm²/sr", "long_name": "Temporal Std Dev"})
    trend_da = xr.DataArray(trend_vals, dims=dims, coords=coords,
                            attrs={"units": "nW/cm²/sr", "long_name": f"Trend ({year_labels[-1]} - {year_labels[0]})"})
    consistency_da = xr.DataArray(consistency_vals, dims=dims, coords=coords,
                                  attrs={"long_name": "Years dark", "max_value": n_years})

    # Classify using median (more robust than any single year)
    dark_sky = median_da.where(median_da < threshold)

    total_valid = int(median_da.count())
    dark_count = int(dark_sky.count())
    all_years_dark = int(np.nansum(consistency_vals == n_years))
    some_years_dark = int(np.nansum((consistency_vals > 0) & (consistency_vals < n_years)))

    logger.info("Temporal classification results:")
    logger.info("  Dark (median < %.1f): %d/%d pixels (%.1f%%)",
                threshold, dark_count, total_valid, 100 * dark_count / max(total_valid, 1))
    logger.info("  Dark ALL %d years (high reliability): %d pixels", n_years, all_years_dark)
    logger.info("  Dark SOME years (variable): %d pixels", some_years_dark)
    logger.info("  Mean stability (std): %.3f nW/cm²/sr", float(np.nanmean(std_vals)))

    stats = {
        "n_years": n_years,
        "year_labels": year_labels,
        "threshold": threshold,
        "total_valid_pixels": total_valid,
        "dark_pixels_median": dark_count,
        "dark_all_years": all_years_dark,
        "dark_some_years": some_years_dark,
        "mean_stability": float(np.nanmean(std_vals)),
    }

    return {
        "median": median_da,
        "dark_sky": dark_sky,
        "stability": stability_da,
        "trend": trend_da,
        "consistency": consistency_da,
        "stats": stats,
    }


def extract_dark_coordinates(dark_sky: xr.DataArray) -> gpd.GeoDataFrame:
    """Extract non-NaN pixels as Point geometries (EPSG:4326).

    Args:
        dark_sky: Masked DataArray from classify_dark_sky.

    Returns:
        GeoDataFrame with Point geometries.
    """
    validate_geographic_coords(dark_sky)
    values = dark_sky.values
    y_coords = dark_sky.coords["y"].values
    x_coords = dark_sky.coords["x"].values

    valid_mask = ~np.isnan(values)
    row_idx, col_idx = np.where(valid_mask)

    latitudes = y_coords[row_idx]
    longitudes = x_coords[col_idx]
    radiance_values = values[row_idx, col_idx]

    logger.info("Extracted %d dark-sky coordinate pairs.", len(latitudes))

    gdf = gpd.GeoDataFrame(
        {"latitude": latitudes, "longitude": longitudes, "radiance": radiance_values},
        geometry=gpd.points_from_xy(longitudes, latitudes),
        crs="EPSG:4326",
    )
    return gdf
