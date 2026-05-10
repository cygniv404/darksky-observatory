"""Terrain slope from Copernicus DEM using Horn (1981) method.

Slope scored for accessibility: flat (<5 deg) = 1.0, steep (>30 deg) = 0.0.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)

# Same DEM source as sky_view.py
_LOCAL_DEM_DIR_ACC = Path(__file__).parent.parent / "input" / "portugal" / "dem"
COP_DEM_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"


def _dem_tile_url(lat: int, lon: int) -> str:
    """Get DEM tile — local first, HTTP fallback."""
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"
    tile_name = f"Copernicus_DSM_COG_10_{lat_prefix}{lat_str}_00_{lon_prefix}{lon_str}_00_DEM"
    local = _LOCAL_DEM_DIR_ACC / f"{tile_name}.tif"
    if local.exists():
        return str(local)
    return f"{COP_DEM_BASE}/{tile_name}/{tile_name}.tif"


def _compute_slope_horn(dem: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
    """Compute slope in degrees using Horn's (1981) method.

    This is the standard method used by GDAL gdaldem, ArcGIS, and QGIS.
    Uses a 3×3 moving window to estimate the gradient in x and y directions.

    Args:
        dem: 2D elevation array (meters).
        cell_size_m: Pixel size in meters.

    Returns:
        2D array of slope in degrees.
    """
    # Pad edges for the 3×3 kernel
    padded = np.pad(dem, 1, mode='edge')

    # Horn's method: weighted finite differences
    # dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize)
    # dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize)
    # where the 3x3 window is:  a b c
    #                            d e f
    #                            g h i
    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell_size_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cell_size_m)

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad)


def compute_slope_at_points(
    spots: gpd.GeoDataFrame,
    buffer_px: int = 3,
) -> np.ndarray:
    """Compute terrain slope at each spot location from Copernicus DEM.

    Reads a small window around each point and computes slope using Horn's
    method. Returns the slope at the center pixel.

    Args:
        spots: GeoDataFrame with Point geometry (EPSG:4326).
        buffer_px: Radius in pixels for the DEM read window (3 = minimum for Horn).

    Returns:
        Array of slope values in degrees for each spot.
    """
    n_spots = len(spots)
    slopes = np.full(n_spots, np.nan)

    lons = spots.geometry.x.values
    lats = spots.geometry.y.values

    # Group by DEM tile
    tile_groups = {}
    for i in range(n_spots):
        tile_lat = int(np.floor(lats[i]))
        tile_lon = int(np.floor(lons[i]))
        key = (tile_lat, tile_lon)
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append(i)

    logger.info("Computing slope for %d spots across %d DEM tiles...", n_spots, len(tile_groups))

    for (tile_lat, tile_lon), indices in tile_groups.items():
        url = _dem_tile_url(tile_lat, tile_lon)

        try:
            with rasterio.open(url) as src:
                for idx in indices:
                    row, col = src.index(lons[idx], lats[idx])

                    # Read small window (need 3×3 minimum for Horn's method)
                    win_row = max(0, row - buffer_px)
                    win_col = max(0, col - buffer_px)
                    win_h = min(2 * buffer_px + 1, src.height - win_row)
                    win_w = min(2 * buffer_px + 1, src.width - win_col)

                    window = Window(win_col, win_row, win_w, win_h)
                    dem = src.read(1, window=window)

                    if dem.shape[0] < 3 or dem.shape[1] < 3:
                        continue

                    # Compute slope for the window
                    slope_grid = _compute_slope_horn(dem)

                    # Get center pixel slope
                    center_r = row - win_row
                    center_c = col - win_col
                    if 0 <= center_r < slope_grid.shape[0] and 0 <= center_c < slope_grid.shape[1]:
                        slopes[idx] = slope_grid[center_r, center_c]

        except Exception as e:
            logger.warning("  Failed tile N%d W%d: %s", tile_lat, abs(tile_lon), e)

    valid = slopes[~np.isnan(slopes)]
    if len(valid) > 0:
        logger.info("Slope computed: mean=%.1f°, max=%.1f°", valid.mean(), valid.max())
        logger.info("  Flat (<5°): %d spots", (valid < 5).sum())
        logger.info("  Gentle (5-15°): %d spots", ((valid >= 5) & (valid < 15)).sum())
        logger.info("  Steep (15-30°): %d spots", ((valid >= 15) & (valid < 30)).sum())
        logger.info("  Very steep (>30°): %d spots", (valid >= 30).sum())

    return slopes


def classify_slope(slope_degrees: float) -> tuple[float, str]:
    """Convert slope to suitability score and label.

    Args:
        slope_degrees: Terrain slope in degrees.

    Returns:
        Tuple of (score 0-1, label).
    """
    if np.isnan(slope_degrees):
        return (0.5, "unknown")
    if slope_degrees < 5:
        return (1.0, "flat")
    elif slope_degrees < 15:
        return (0.8, "gentle")
    elif slope_degrees < 30:
        return (0.4, "steep")
    else:
        return (0.0, "very_steep")


def assess_accessibility(spots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add slope and accessibility columns to spots GeoDataFrame.

    Args:
        spots: GeoDataFrame with Point geometry.

    Returns:
        GeoDataFrame with added columns:
            - terrain_slope_deg: slope in degrees
            - slope_score: 0-1 suitability
            - slope_class: flat/gentle/steep/very_steep
    """
    spots = spots.copy()
    slopes = compute_slope_at_points(spots)

    spots["terrain_slope_deg"] = np.round(slopes, 1)
    scores_and_labels = [classify_slope(s) for s in slopes]
    spots["slope_score"] = [s[0] for s in scores_and_labels]
    spots["slope_class"] = [s[1] for s in scores_and_labels]

    return spots
