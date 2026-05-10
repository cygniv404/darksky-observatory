"""Sinusoidal to WGS84 reprojection for VIIRS HDF-EOS5 tiles.

MODIS sinusoidal grid: 36x18 tiles, 2400x2400 px each, ~500m at equator.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

logger = logging.getLogger(__name__)

# MODIS sinusoidal projection definition
MODIS_SINUSOIDAL_PROJ4 = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)
MODIS_SINUSOIDAL_CRS = CRS.from_proj4(MODIS_SINUSOIDAL_PROJ4)

# Full sinusoidal grid extent (meters)
GRID_X_MIN = -20015109.354
GRID_X_MAX = 20015109.354
GRID_Y_MIN = -10007554.677
GRID_Y_MAX = 10007554.677

# Standard tile dimensions
TILE_PIXELS = 2400
TILE_SIZE_M = (GRID_X_MAX - GRID_X_MIN) / 36  # ~1111950.5 m per tile in X


def tile_bounds_from_hv(h: int, v: int) -> tuple[float, float, float, float]:
    """Compute sinusoidal bounds (meters) for a given tile h/v index.

    The MODIS grid has 36 columns (h=0–35) and 18 rows (v=0–17).
    h increases eastward, v increases southward (from north pole).

    Args:
        h: Horizontal tile index (0–35).
        v: Vertical tile index (0–17).

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in sinusoidal meters.
    """
    tile_width = (GRID_X_MAX - GRID_X_MIN) / 36
    tile_height = (GRID_Y_MAX - GRID_Y_MIN) / 18

    x_min = GRID_X_MIN + h * tile_width
    x_max = x_min + tile_width
    y_max = GRID_Y_MAX - v * tile_height
    y_min = y_max - tile_height

    return x_min, y_min, x_max, y_max


def parse_tile_id(filename: str) -> tuple[int, int]:
    """Extract h and v indices from a VNP46 filename.

    Filename format: VNP46A4.A2023001.h17v05.002.2025161112612.h5

    Args:
        filename: HDF5 filename (just the basename).

    Returns:
        Tuple of (h, v) tile indices.
    """
    import re
    match = re.search(r"h(\d{2})v(\d{2})", filename)
    if not match:
        raise ValueError(f"Cannot parse tile ID from filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def read_h5_radiance(h5_path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read radiance data and bounds from a VNP46A4 HDF-EOS5 file.

    Extracts the NearNadir_Composite_Snow_Free variable and determines
    the tile's sinusoidal extent either from metadata or tile ID.

    Args:
        h5_path: Path to the .h5 file.

    Returns:
        Tuple of (radiance_array, (x_min, y_min, x_max, y_max)).
        radiance_array shape: (2400, 2400) or similar.
        Bounds are in sinusoidal meters.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        # Navigate to the data field
        grid_path = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
        if grid_path + "/NearNadir_Composite_Snow_Free" in f:
            data = f[f"{grid_path}/NearNadir_Composite_Snow_Free"][:]
        elif grid_path + "/AllAngle_Composite_Snow_Free" in f:
            data = f[f"{grid_path}/AllAngle_Composite_Snow_Free"][:]
        else:
            available = list(f[grid_path].keys()) if grid_path in f else []
            raise KeyError(
                f"No radiance variable found in {h5_path}. Available: {available}"
            )

        # Try to read bounds from metadata
        try:
            struct_meta = f["HDFEOS INFORMATION/StructMetadata.0"][()].decode()
            # Parse UpperLeftPointMtrs and LowerRightMtrs from the metadata string
            import re
            ul_match = re.search(r"UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)", struct_meta)
            lr_match = re.search(r"LowerRightMtrs=\(([-\d.]+),([-\d.]+)\)", struct_meta)
            if ul_match and lr_match:
                x_min = float(ul_match.group(1))
                y_max = float(ul_match.group(2))
                x_max = float(lr_match.group(1))
                y_min = float(lr_match.group(2))
                bounds = (x_min, y_min, x_max, y_max)
                logger.info("Read bounds from HDF-EOS5 metadata: %s", bounds)
            else:
                raise ValueError("Bounds not found in metadata")
        except (KeyError, ValueError):
            # Fallback: derive bounds from tile ID in filename
            h, v = parse_tile_id(h5_path.name)
            bounds = tile_bounds_from_hv(h, v)
            logger.info("Derived bounds from tile ID h%02dv%02d: %s", h, v, bounds)

    # Replace fill values with NaN
    data = data.astype(np.float32)
    data[data >= 65535] = np.nan  # Common fill value in VIIRS products
    data[data < 0] = np.nan

    logger.info("Read %s: shape=%s, valid pixels=%d", h5_path.name, data.shape, int(np.sum(~np.isnan(data))))
    return data, bounds


def reproject_sinusoidal_to_wgs84(
    h5_path: Path,
    output_path: Path,
    target_resolution: float = 0.004166,  # ~500m at equator in degrees
) -> Path:
    """Reproject a VIIRS HDF-EOS5 tile from sinusoidal to WGS84 (EPSG:4326).

    This is the core operation that blackmarblepy performs internally.
    Implementing it manually demonstrates understanding of:
        - HDF-EOS5 file structure
        - MODIS sinusoidal projection parameters
        - Rasterio-based raster reprojection with resampling

    The output is a standard GeoTIFF readable by any GIS software.

    Args:
        h5_path: Path to the input VNP46A4 .h5 file.
        output_path: Path for the output GeoTIFF (EPSG:4326).
        target_resolution: Output pixel size in degrees (~0.0042° ≈ 500m).

    Returns:
        Path to the written GeoTIFF file.

    Example:
        >>> reproject_sinusoidal_to_wgs84(
        ...     Path("VNP46A4.A2025001.h17v05.002.h5"),
        ...     Path("output/h17v05_wgs84.tif")
        ... )
    """
    h5_path = Path(h5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Read the raw data and sinusoidal bounds
    data, (x_min, y_min, x_max, y_max) = read_h5_radiance(h5_path)
    height, width = data.shape

    # Step 2: Define the source transform (sinusoidal projection)
    src_transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    src_crs = MODIS_SINUSOIDAL_CRS

    logger.info(
        "Source: %s, %dx%d pixels, sinusoidal bounds [%.0f, %.0f, %.0f, %.0f] m",
        h5_path.name, width, height, x_min, y_min, x_max, y_max,
    )

    # Step 3: Calculate the target transform (WGS84)
    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height,
        left=x_min, bottom=y_min, right=x_max, top=y_max,
        resolution=target_resolution,
    )

    logger.info(
        "Target: EPSG:4326, %dx%d pixels, resolution=%.6f°",
        dst_width, dst_height, target_resolution,
    )

    # Step 4: Perform the reprojection
    dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    reproject(
        source=data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Step 5: Write the output GeoTIFF
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": dst_width,
        "height": dst_height,
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dst_data, 1)

    logger.info("Reprojected to %s (%dx%d pixels)", output_path, dst_width, dst_height)
    return output_path
