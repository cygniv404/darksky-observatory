"""Vegetation-aware Sky View Factor from Copernicus DEM + Hansen canopy.

effective_SVF = terrain_SVF * (1 - canopy_fraction)

Terrain SVF: horizon-angle algorithm, 16 azimuths, 1.5km radius.
Canopy: Hansen GFC v1.11 treecover2000 (0-100%).
Both accessed via windowed COG reads (HTTP or local).
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)

# Copernicus DEM 30m — local first, HTTP fallback
_LOCAL_DEM_DIR = Path(__file__).parent.parent / "input" / "portugal" / "dem"
COP_DEM_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"

# Hansen Global Forest Change v1.11 (2023) — local first, HTTP fallback
_LOCAL_HANSEN_DIR = Path(__file__).parent.parent / "input" / "portugal" / "hansen"
HANSEN_BASE = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"

# SVF computation parameters
N_DIRECTIONS = 16  # azimuth rays (16 = good accuracy/speed balance)
MAX_RADIUS_PX = 50  # search radius in pixels (50 × 30m = 1.5km)
CELL_SIZE_M = 30.0  # Copernicus GLO-30 pixel size in meters


def _hansen_tile_path(lat: float, lon: float) -> str:
    """Get Hansen treecover tile — local file first, HTTP fallback.

    Tiles are 10°×10° blocks named by their UPPER-LEFT corner.
    """
    tile_lat_upper = int(np.ceil(lat / 10) * 10)
    tile_lon_left = int(np.floor(lon / 10) * 10)

    lat_str = f"{abs(tile_lat_upper):02d}{'N' if tile_lat_upper >= 0 else 'S'}"
    lon_str = f"{abs(tile_lon_left):03d}{'E' if tile_lon_left >= 0 else 'W'}"
    filename = f"Hansen_GFC-2023-v1.11_treecover2000_{lat_str}_{lon_str}.tif"

    # Check local first
    local_path = _LOCAL_HANSEN_DIR / filename
    if local_path.exists():
        return str(local_path)

    # Fallback to HTTP
    return f"{HANSEN_BASE}/{filename}"


def _get_canopy_cover_at_point(lat: float, lon: float, hansen_src) -> float:
    """Get tree cover percentage (0-100) at a single point from an open Hansen tile.

    Returns 0 if the point is outside the tile or has nodata.
    """
    try:
        row, col = hansen_src.index(lon, lat)
        if 0 <= row < hansen_src.height and 0 <= col < hansen_src.width:
            window = Window(col, row, 1, 1)
            val = hansen_src.read(1, window=window)[0, 0]
            if val < 255:  # 255 = nodata in Hansen
                return float(val) / 100.0  # Convert 0-100 to 0-1 fraction
    except Exception:
        pass
    return 0.0


def _dem_tile_path(lat: int, lon: int) -> str:
    """Get Copernicus DEM 30m tile — local file first, HTTP fallback.

    Checks input/portugal/dem/ for pre-downloaded tiles.
    Falls back to HTTP if not found locally.
    """
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"
    tile_name = f"Copernicus_DSM_COG_10_{lat_prefix}{lat_str}_00_{lon_prefix}{lon_str}_00_DEM"

    # Check local first
    local_path = _LOCAL_DEM_DIR / f"{tile_name}.tif"
    if local_path.exists():
        return str(local_path)

    # Fallback to HTTP
    return f"{COP_DEM_BASE}/{tile_name}/{tile_name}.tif"


def _compute_svf_from_dem(
    dem: np.ndarray,
    center_row: int,
    center_col: int,
    cell_size_m: float = CELL_SIZE_M,
    n_directions: int = N_DIRECTIONS,
    max_radius_px: int = MAX_RADIUS_PX,
) -> float:
    """Compute Sky View Factor at a single point from a DEM window.

    Casts rays in n_directions and finds the maximum horizon angle along each.
    SVF = 1 - mean(sin(max_horizon_angle)) per direction.

    Args:
        dem: 2D elevation array (meters) centered on the point.
        center_row: Row index of the observation point.
        center_col: Column index of the observation point.
        cell_size_m: Pixel size in meters.
        n_directions: Number of azimuth directions to sample.
        max_radius_px: Maximum search radius in pixels.

    Returns:
        SVF value (0-1). 1 = fully open sky, 0 = completely blocked.
    """
    z_observer = dem[center_row, center_col]
    if np.isnan(z_observer):
        return np.nan

    max_angles = np.zeros(n_directions)

    for i in range(n_directions):
        azimuth = 2 * np.pi * i / n_directions
        dx = np.sin(azimuth)  # column direction
        dy = -np.cos(azimuth)  # row direction (negative because row increases downward)

        max_angle = 0.0

        for step in range(1, max_radius_px + 1):
            col = center_col + int(round(dx * step))
            row = center_row + int(round(dy * step))

            if row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]:
                break

            z_target = dem[row, col]
            if np.isnan(z_target):
                continue

            distance_m = step * cell_size_m
            dz = z_target - z_observer
            angle = np.arctan2(dz, distance_m)

            if angle > max_angle:
                max_angle = angle

        max_angles[i] = max_angle

    # SVF formula: 1 - mean(sin(max_horizon_angle))
    # Only positive angles (terrain above observer) reduce SVF
    positive_angles = np.maximum(max_angles, 0)
    svf = 1.0 - np.mean(np.sin(positive_angles))

    return float(np.clip(svf, 0, 1))


def compute_svf_at_points(
    spots: gpd.GeoDataFrame,
    buffer_px: int = MAX_RADIUS_PX,
    include_canopy: bool = True,
) -> np.ndarray:
    """Compute vegetation-aware SVF for a set of geographic points.

    Combines terrain SVF (from Copernicus DEM) with canopy cover correction
    (from Hansen Global Forest Change 2023):

        effective_SVF = terrain_SVF × (1 - tree_cover_fraction)

    This handles both terrain obstruction (hills/valleys) AND vegetation
    obstruction (forest canopy blocking the sky dome).

    Args:
        spots: GeoDataFrame with Point geometry (EPSG:4326).
        buffer_px: Radius in pixels around each point for DEM horizon scan.
        include_canopy: If True, apply Hansen canopy cover correction.

    Returns:
        Tuple of (effective_svf, dem_elevation):
            - effective_svf: array of SVF values (0-1) per spot
            - dem_elevation: array of DEM-sampled elevation (meters) per spot
    """
    n_spots = len(spots)
    terrain_svf = np.full(n_spots, np.nan)
    dem_elevation = np.full(n_spots, np.nan)  # Also record DEM elevation
    canopy_cover = np.zeros(n_spots)

    lons = spots.geometry.x.values
    lats = spots.geometry.y.values

    # --- Step 1: Compute terrain SVF from DEM ---
    tile_keys = set()
    spot_tiles = []
    for i in range(n_spots):
        tile_lat = int(np.floor(lats[i]))
        tile_lon = int(np.floor(lons[i]))
        tile_keys.add((tile_lat, tile_lon))
        spot_tiles.append((tile_lat, tile_lon))

    logger.info("Computing SVF for %d spots across %d DEM tiles...", n_spots, len(tile_keys))

    for tile_lat, tile_lon in sorted(tile_keys):
        tile_mask = [(t == (tile_lat, tile_lon)) for t in spot_tiles]
        tile_indices = [i for i, m in enumerate(tile_mask) if m]

        if not tile_indices:
            continue

        url = _dem_tile_path(tile_lat, tile_lon)
        logger.info("  Tile N%02d W%03d: %d spots (%s)",
                    abs(tile_lat), abs(tile_lon), len(tile_indices), url.split("/")[-1])

        try:
            with rasterio.open(url) as src:
                for idx in tile_indices:
                    lon, lat = lons[idx], lats[idx]
                    row, col = src.index(lon, lat)

                    win_row = max(0, row - buffer_px)
                    win_col = max(0, col - buffer_px)
                    win_height = min(2 * buffer_px + 1, src.height - win_row)
                    win_width = min(2 * buffer_px + 1, src.width - win_col)

                    window = Window(win_col, win_row, win_width, win_height)
                    dem = src.read(1, window=window)

                    center_row = row - win_row
                    center_col = col - win_col

                    svf = _compute_svf_from_dem(dem, center_row, center_col)
                    terrain_svf[idx] = svf
                    # Record DEM elevation at spot (authoritative, not from OSM tag)
                    if 0 <= center_row < dem.shape[0] and 0 <= center_col < dem.shape[1]:
                        dem_elevation[idx] = dem[center_row, center_col]

        except Exception as e:
            logger.warning("  Failed to read tile N%d W%d: %s", tile_lat, abs(tile_lon), e)

    # --- Step 2: Get canopy cover from Hansen GFC ---
    if include_canopy:
        logger.info("Reading Hansen tree cover for canopy correction...")

        # Group by Hansen tile (10° tiles)
        hansen_tiles = {}
        for i in range(n_spots):
            tile_lat_upper = int(np.ceil(lats[i] / 10) * 10)
            tile_lon_left = int(np.floor(lons[i] / 10) * 10)
            key = (tile_lat_upper, tile_lon_left)
            if key not in hansen_tiles:
                hansen_tiles[key] = []
            hansen_tiles[key].append(i)

        for (tile_lat, tile_lon), indices in hansen_tiles.items():
            url = _hansen_tile_path(lats[indices[0]], lons[indices[0]])
            try:
                with rasterio.open(url) as hansen_src:
                    for idx in indices:
                        canopy_cover[idx] = _get_canopy_cover_at_point(
                            lats[idx], lons[idx], hansen_src
                        )
            except Exception as e:
                logger.warning("  Hansen tile failed: %s", e)

        mean_canopy = canopy_cover[canopy_cover > 0].mean() if (canopy_cover > 0).any() else 0
        logger.info("  Canopy cover: mean=%.0f%% (where >0), %d spots in forest (>30%%)",
                    mean_canopy * 100, (canopy_cover > 0.3).sum())

    # --- Step 3: Combine terrain SVF with canopy correction ---
    # effective_SVF = terrain_SVF × (1 - canopy_fraction)
    effective_svf = terrain_svf * (1 - canopy_cover)

    # Summary
    valid = effective_svf[~np.isnan(effective_svf)]
    if len(valid) > 0:
        logger.info("Effective SVF (terrain + canopy): mean=%.3f, range=%.3f–%.3f",
                    valid.mean(), valid.min(), valid.max())
        logger.info("  Open sky (SVF>0.85): %d spots", (valid > 0.85).sum())
        logger.info("  Moderate (0.5-0.85): %d spots", ((valid >= 0.5) & (valid <= 0.85)).sum())
        logger.info("  Obstructed (SVF<0.5): %d spots", (valid < 0.5).sum())

    return effective_svf, dem_elevation


def assess_sky_view(spots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add SVF column to spots GeoDataFrame.

    Args:
        spots: GeoDataFrame with Point geometry.

    Returns:
        GeoDataFrame with added 'sky_view_factor' column (0-1).
    """
    spots = spots.copy()
    spots["sky_view_factor"] = compute_svf_at_points(spots)
    return spots
