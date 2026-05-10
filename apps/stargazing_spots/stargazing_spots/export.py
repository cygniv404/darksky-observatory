"""Export processed dark-sky data in standard geospatial formats.

Supports:
    - GeoJSON (vector, for web applications)
    - GeoParquet (vector, for analytics)
    - Cloud Optimized GeoTIFF (raster, for cloud-native workflows)
"""

import logging
from datetime import UTC
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


def to_geojson(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    processing_params: dict | None = None,
) -> Path:
    """Export GeoDataFrame as GeoJSON with optional processing provenance.

    Embeds pipeline metadata directly in the GeoJSON so any consumer can
    trace how the file was produced and reproduce the results.

    Args:
        gdf: Processed spots GeoDataFrame.
        output_path: Destination file path.
        processing_params: Optional dict of processing parameters to embed.

    Returns:
        Path to written file.
    """
    import json
    from datetime import datetime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip OSM noise — keep only useful columns
    KEEP_COLUMNS = [
        "geometry", "name", "id", "@id",
        # Pipeline assessment results
        "predicted_sqm", "predicted_bortle", "predicted_brightness_mcd",
        "ssi_score", "ssi_class",
        "clear_night_fraction",
        "sky_view_factor",
        "land_cover_class", "land_cover_label", "land_cover_score",
        "terrain_slope_deg", "slope_score", "slope_class",
        "dem_elevation_m",
        "pixel_radiance", "pixel_stability", "pixel_trend", "trend_direction",
        "dark_confidence", "confidence_class",
        "tier", "access_type", "access_score", "night_accessible", "access_warnings",
        "stargazing_score",
        # Seasonal
        "ssi_summer", "ssi_winter", "ssi_spring", "ssi_autumn", "best_season",
        # Observation planning
        "mw_best_months", "mw_max_altitude_deg", "mw_peak_month",
        # OSM context (only if useful)
        "tourism", "natural", "leisure", "amenity", "man_made",
        "ele", "region",
    ]

    available = [c for c in KEEP_COLUMNS if c in gdf.columns]
    gdf_clean = gdf[available].copy()

    # Add explicit lat/lon columns for easy reading
    gdf_clean["lat"] = gdf_clean.geometry.y.round(6)
    gdf_clean["lon"] = gdf_clean.geometry.x.round(6)

    gdf_clean.to_file(output_path, driver="GeoJSON")

    # Strip null fields per-feature for compact output
    with open(output_path) as f:
        geojson = json.load(f)
    for feature in geojson["features"]:
        feature["properties"] = {
            k: v for k, v in feature["properties"].items() if v is not None
        }
    if processing_params:
        geojson["processing"] = {
            **processing_params,
            "generated_at": datetime.now(UTC).isoformat(),
            "pipeline_version": "0.7.0",
            "source_product": "VNP46A4v001",
        }
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    logger.info("Exported %d features to %s", len(gdf_clean), output_path)
    return output_path


def to_parquet(gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
    """Export GeoDataFrame as GeoParquet with Snappy compression.

    Args:
        gdf: Processed spots GeoDataFrame.
        output_path: Destination file path.

    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, compression="snappy")
    logger.info("Exported %d features to %s", len(gdf), output_path)
    return output_path


def to_cog(
    data: xr.DataArray,
    output_path: Path,
    crs: str = "EPSG:4326",
) -> Path:
    """Export raster data as a Cloud Optimized GeoTIFF (COG).

    COG format enables efficient range-request access from cloud storage (S3, GCS)
    without downloading the entire file. Uses internal tiling and overviews.

    Args:
        data: 2D DataArray with y/x coordinates (e.g., filtered radiance).
        output_path: Destination .tif path.
        crs: Coordinate reference system string.

    Returns:
        Path to written COG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_coords = data.coords["y"].values
    x_coords = data.coords["x"].values

    height, width = data.shape
    west, east = float(x_coords.min()), float(x_coords.max())
    south, north = float(y_coords.min()), float(y_coords.max())

    transform = from_bounds(west, south, east, north, width, height)

    values = data.values.astype(np.float32)
    values = np.where(np.isnan(values), -9999, values)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": -9999,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(values, 1)
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    logger.info("Exported COG (%dx%d) to %s", width, height, output_path)
    return output_path
