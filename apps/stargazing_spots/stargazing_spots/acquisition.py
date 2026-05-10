"""VIIRS VNP46A4 nighttime radiance acquisition via NASA Black Marble API.

Downloads annual composites (500m, nW/cm2/sr) with joblib caching.
"""

import logging
import os
from datetime import UTC
from pathlib import Path

import geopandas as gpd
import h5py
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from blackmarble.raster import bm_raster  # pip install blackmarblepy

logger = logging.getLogger(__name__)

QUALITY_FLAGS_REMOVE = [1, 255]
VIIRS_VARIABLE = "NearNadir_Composite_Snow_Free"


def get_earthdata_token() -> str:
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        raise OSError(
            "EARTHDATA_TOKEN not set. Register at https://urs.earthdata.nasa.gov/ "
            "and set the token as an environment variable."
        )
    return token


def _write_cache_metadata(cache_file: Path, params: dict) -> None:
    """Write a JSON sidecar with metadata about what produced this cache file."""
    import json
    from datetime import datetime

    meta_path = cache_file.with_suffix(".meta.json")
    metadata = {
        "cache_file": cache_file.name,
        "created_at": datetime.now(UTC).isoformat(),
        "parameters": params,
        "file_size_bytes": cache_file.stat().st_size if cache_file.exists() else 0,
        "library": "blackmarblepy",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.debug("Cache metadata written to %s", meta_path)


def _read_cache_metadata(cache_file: Path) -> dict | None:
    """Read cache metadata sidecar if it exists."""
    import json

    meta_path = cache_file.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def fetch_viirs_radiance(
    boundary_gdf: gpd.GeoDataFrame,
    product_id: str = "VNP46A4",
    date_range: str = "2023-01-01",
    cache_dir: Path = Path(__file__).parent.parent / "input" / "portugal" / "cache",
    no_cache: bool = False,
) -> xr.Dataset:
    """Download VIIRS radiance for a boundary, with joblib caching.

    Args:
        boundary_gdf: Area-of-interest polygon.
        product_id: NASA product identifier.
        date_range: Temporal query date string.
        cache_dir: Cache directory.
        no_cache: Force fresh download.

    Returns:
        xarray Dataset with radiance variable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{product_id}_{date_range}.joblib"

    cache_params = {
        "product_id": product_id,
        "date_range": date_range,
        "variable": VIIRS_VARIABLE,
        "quality_flags_removed": QUALITY_FLAGS_REMOVE,
        "boundary_bbox": list(boundary_gdf.total_bounds),
    }

    if cache_file.exists() and not no_cache:
        meta = _read_cache_metadata(cache_file)
        if meta:
            logger.info(
                "Loading cached VIIRS data: %s (created %s)",
                cache_file.name,
                meta.get("created_at", "unknown"),
            )
            cached_params = meta.get("parameters", {})
            if cached_params.get("quality_flags_removed") != QUALITY_FLAGS_REMOVE:
                logger.warning(
                    "Cache was created with different quality flags: %s vs current %s",
                    cached_params.get("quality_flags_removed"),
                    QUALITY_FLAGS_REMOVE,
                )
        else:
            logger.info("Loading cached VIIRS data: %s (no metadata sidecar)", cache_file.name)
        return joblib.load(cache_file)

    logger.info("Downloading VIIRS %s for %s...", product_id, date_range)
    token = get_earthdata_token()

    dataset = bm_raster(
        boundary_gdf,
        product_id=product_id,
        date_range=date_range,
        bearer=token,
        variable=VIIRS_VARIABLE,
        quality_flag_rm=QUALITY_FLAGS_REMOVE,
    )

    joblib.dump(dataset, cache_file)
    _write_cache_metadata(cache_file, cache_params)
    logger.info("Cached VIIRS data to %s", cache_file)
    return dataset


def extract_radiance_layer(dataset: xr.Dataset, time_slice: str) -> xr.DataArray:
    """Extract a single temporal slice of the radiance variable.

    Args:
        dataset: Full VIIRS dataset.
        time_slice: Date string to select (e.g., '2023-01-01').

    Returns:
        2D DataArray of radiance values (nW/cm2/sr).
    """
    return dataset[VIIRS_VARIABLE].sel(time=time_slice)


def parse_h5_tiles(h5_glob_pattern: str) -> pd.DataFrame:
    """Parse raw VIIRS HDF-EOS5 tiles directly for gap-filling or custom processing.

    Reads latitude/longitude grids and radiance from the sinusoidal-projected
    2400x2400 pixel tiles (h5 format from LAADS DAAC).

    Args:
        h5_glob_pattern: Glob pattern for .h5 files (e.g., './data/*.h5').

    Returns:
        DataFrame with columns: latitude, longitude, radiance, time.
    """
    import glob
    import re

    date_pattern = re.compile(r"\.A(\d{4})(\d{3})")
    frames = []

    for filepath in glob.glob(h5_glob_pattern):
        with h5py.File(filepath, "r") as h5:
            grid_path = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
            lat_1d = h5[f"{grid_path}/lat"][:]
            lon_1d = h5[f"{grid_path}/lon"][:]
            radiance = h5[f"{grid_path}/AllAngle_Composite_Snow_Free"][:]

            if lat_1d.ndim != 1 or lon_1d.ndim != 1 or radiance.shape != (2400, 2400):
                logger.warning("Skipping %s: unexpected dimensions", filepath)
                continue

            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
            tile_df = pd.DataFrame({
                "latitude": lat_2d.ravel(),
                "longitude": lon_2d.ravel(),
                "radiance": radiance.ravel(),
            }).dropna(subset=["radiance"])

            match = date_pattern.search(filepath)
            if match:
                year, doy = int(match.group(1)), int(match.group(2))
                tile_df["time"] = pd.to_datetime(f"{year}-{doy}", format="%Y-%j")
                frames.append(tile_df)
            else:
                logger.warning("No date in filename: %s", filepath)

    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "radiance", "time"])

    return pd.concat(frames, ignore_index=True)


def gap_fill_with_tiles(
    base_df: pd.DataFrame, tiles_df: pd.DataFrame, target_col: str = "radiance"
) -> pd.DataFrame:
    """Fill NaN values in base dataset using nearest-neighbor lookup from tile data.

    Uses scipy KDTree for efficient spatial matching between the base radiance
    grid and independently-parsed HDF5 tile data.

    Args:
        base_df: DataFrame with columns [y, x, <target_col>] where NaN = missing.
        tiles_df: DataFrame with [latitude, longitude, radiance] from parse_h5_tiles().
        target_col: Column name to fill.

    Returns:
        Updated base_df with NaN values filled from nearest tile pixel.
    """
    from scipy.spatial import KDTree

    null_mask = base_df[target_col].isnull()
    if not null_mask.any():
        logger.info("No gaps to fill.")
        return base_df

    tree = KDTree(tiles_df[["latitude", "longitude"]].values)
    null_coords = base_df.loc[null_mask, ["y", "x"]].values
    _, nearest_idx = tree.query(null_coords)

    base_df.loc[null_mask, target_col] = tiles_df.iloc[nearest_idx]["radiance"].values
    logger.info("Filled %d gaps using KDTree nearest-neighbor.", null_mask.sum())
    return base_df
