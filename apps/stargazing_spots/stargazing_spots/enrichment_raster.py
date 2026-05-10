"""Raster-native spatial join: sample classified raster at each OSM feature location.

O(n) pixel lookups vs O(n*m) vector intersection.
"""

import logging

import geopandas as gpd
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def raster_sample_join(
    dark_sky: xr.DataArray,
    features: gpd.GeoDataFrame,
    threshold: float = 3.0,
) -> gpd.GeoDataFrame:
    """Identify OSM features located within dark-sky pixels by direct raster sampling.

    For each feature point, finds the nearest pixel in the dark-sky raster
    and checks whether it's below the dark-sky threshold. Features on dark
    pixels are retained; those on bright or NaN pixels are dropped.

    Args:
        dark_sky: 2D DataArray of radiance values (NaN = excluded, values = radiance).
            Should be the output of classify_dark_sky or the median from classify_temporal.
        features: GeoDataFrame of OSM features with Point geometry (EPSG:4326).
        threshold: Radiance threshold — pixels below this are "dark".
            Only used if dark_sky contains raw radiance rather than pre-classified values.

    Returns:
        Filtered GeoDataFrame containing only features located on dark-sky pixels.
        Adds columns: 'pixel_radiance' (sampled value) and 'pixel_y_idx', 'pixel_x_idx'.
    """
    y_coords = dark_sky.coords["y"].values
    x_coords = dark_sky.coords["x"].values
    values = dark_sky.values

    # Ensure features are points in EPSG:4326
    points_only = features[features.geometry.geom_type == "Point"].copy()
    if points_only.crs and points_only.crs.to_epsg() != 4326:
        points_only = points_only.to_crs(epsg=4326)

    # Compute pixel resolution for bounds checking
    y_res = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 0.005
    x_res = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 0.005
    y_min, y_max = float(y_coords.min()) - y_res / 2, float(y_coords.max()) + y_res / 2
    x_min, x_max = float(x_coords.min()) - x_res / 2, float(x_coords.max()) + x_res / 2

    # Sample raster at each feature location
    radiance_values = []
    in_bounds = []

    for _, row in points_only.iterrows():
        lon, lat = row.geometry.x, row.geometry.y

        # Check if point falls within raster extent
        if lat < y_min or lat > y_max or lon < x_min or lon > x_max:
            radiance_values.append(np.nan)
            in_bounds.append(False)
            continue

        # Find nearest pixel (nearest-neighbor lookup)
        y_idx = int(np.argmin(np.abs(y_coords - lat)))
        x_idx = int(np.argmin(np.abs(x_coords - lon)))
        pixel_val = values[y_idx, x_idx]

        radiance_values.append(pixel_val)
        in_bounds.append(True)

    points_only["pixel_radiance"] = radiance_values
    points_only["in_raster_bounds"] = in_bounds

    # Filter: keep only features on dark pixels
    dark_features = points_only[
        points_only["in_raster_bounds"]
        & points_only["pixel_radiance"].notna()
        & (points_only["pixel_radiance"] < threshold)
    ].copy()

    # Drop helper columns
    dark_features = dark_features.drop(columns=["in_raster_bounds"])

    n_input = len(features)
    n_in_bounds = sum(in_bounds)
    n_dark = len(dark_features)
    n_outside = n_input - n_in_bounds
    n_bright = n_in_bounds - n_dark

    # Deduplicate by name + geometry proximity
    if "name" in dark_features.columns:
        pre_dedup = len(dark_features)
        dark_features = dark_features.drop_duplicates(subset=["name", "pixel_radiance"])
        n_dark = len(dark_features)
        if pre_dedup > n_dark:
            logger.info("  Deduplicated: %d → %d (removed %d duplicates)",
                        pre_dedup, n_dark, pre_dedup - n_dark)

    logger.info("Raster-native spatial join:")
    logger.info("  Input features: %d", n_input)
    if n_outside > 0:
        logger.info("  Outside raster extent: %d (skipped)", n_outside)
    logger.info("  On dark pixels (< %.1f nW): %d", threshold, n_dark)
    logger.info("  On bright/NaN pixels: %d (excluded)", n_bright)

    return dark_features


def raster_sample_with_context(
    dark_sky: xr.DataArray,
    features: gpd.GeoDataFrame,
    stability: xr.DataArray | None = None,
    trend: xr.DataArray | None = None,
    threshold: float = 3.0,
) -> gpd.GeoDataFrame:
    """Sample dark-sky raster with optional temporal context layers.

    Extends raster_sample_join by also sampling stability and trend rasters
    at each feature location, adding temporal metadata in one pass.

    Args:
        dark_sky: Radiance DataArray (classified or median).
        features: OSM features GeoDataFrame.
        stability: Optional temporal std deviation DataArray.
        trend: Optional trend DataArray (last year - first year).
        threshold: Dark-sky radiance threshold.

    Returns:
        GeoDataFrame with pixel_radiance, pixel_stability, pixel_trend columns.
    """
    dark_features = raster_sample_join(dark_sky, features, threshold)

    if stability is None and trend is None:
        return dark_features

    y_coords = dark_sky.coords["y"].values
    x_coords = dark_sky.coords["x"].values

    stab_values = []
    trend_values = []

    for _, row in dark_features.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        y_idx = int(np.argmin(np.abs(y_coords - lat)))
        x_idx = int(np.argmin(np.abs(x_coords - lon)))

        if stability is not None:
            stab_values.append(float(stability.values[y_idx, x_idx]))
        if trend is not None:
            trend_values.append(float(trend.values[y_idx, x_idx]))

    if stability is not None:
        dark_features["pixel_stability"] = stab_values
        dark_features["temporal_stability"] = dark_features["pixel_stability"].apply(
            lambda s: "high" if s < 0.5 else ("medium" if s < 1.5 else "low")
        )

    if trend is not None:
        # Note: with 3 years of data, this is inter-annual variability, not a
        # confirmed long-term trend (Kyba et al. 2017: need 5+ years).
        dark_features["pixel_trend"] = trend_values
        dark_features["trend_direction"] = dark_features["pixel_trend"].apply(
            lambda t: "darkening" if t < -1.0 else ("brightening" if t > 1.0 else "stable")
        )

    logger.info("  Added temporal context: stability=%s, trend=%s",
                "yes" if stability is not None else "no",
                "yes" if trend is not None else "no")

    return dark_features
