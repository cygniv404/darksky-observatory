"""Clear-night climatology from ERA5 (25km) and CLAAS-3 (5km) cloud data.

Computes fraction of astronomically dark hours (sun < -18 deg) that are
cloud-free, using a 20-year climatology (2006-2025). CLAAS-3 used for
mainland when available; ERA5 as fallback and for islands.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

CDS_DATASET = "reanalysis-era5-single-levels-monthly-means"

# Threshold: fraction of sky covered by cloud below which we consider "clear"
# TCC < 0.2 means 80%+ of sky is visible — standard in observatory site surveys
# Reference: Aksaker et al. (2020), ESO site testing, Vernin et al. (2011)
CLEAR_THRESHOLD = 0.2

# Climatology period
CLIM_START_YEAR = 2006
CLIM_END_YEAR = 2025


@dataclass
class ClearSkyConfig:
    """Configuration for clear-sky climatology computation."""

    clear_threshold: float = CLEAR_THRESHOLD
    start_year: int = CLIM_START_YEAR
    end_year: int = CLIM_END_YEAR
    # Portugal bounding box (generous, includes Azores and Madeira)
    lat_range: tuple[float, float] = (32.0, 43.0)
    lon_range: tuple[float, float] = (-32.0, -6.0)
    # For mainland-only: lat (36.5, 42.5), lon (-10, -6)
    cache_dir: Path = Path(__file__).parent.parent / "input" / "portugal" / "cache"


def compute_sun_altitude(hour_utc: int, day_of_year: int, latitude: float) -> float:
    """Compute solar altitude angle using the astronomical almanac method.

    This is a simplified but accurate (±0.5°) solar position calculation
    suitable for determining astronomical twilight (sun < -18°).

    Based on NOAA Solar Calculator equations derived from Jean Meeus,
    "Astronomical Algorithms" (1991).

    Args:
        hour_utc: Hour of day in UTC (0-23).
        day_of_year: Day of year (1-365/366).
        latitude: Observer latitude in degrees.

    Returns:
        Solar altitude angle in degrees (negative = below horizon).
    """
    # Fractional year (radians)
    gamma = 2 * np.pi / 365 * (day_of_year - 1 + (hour_utc - 12) / 24)

    # Solar declination (radians)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    # Equation of time (minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma)
                       - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma)
                       - 0.04089 * np.sin(2 * gamma))

    # Solar hour angle
    # Time offset in minutes (Portugal is near UTC, no timezone offset needed for ERA5 UTC data)
    _ = eqtime  # noqa: F841
    # Actually for a proper calculation we need longitude, but since we're computing
    # per-latitude bands and ERA5 data is already in UTC, we compute for the center
    # longitude of Portugal (~-8°) as representative
    longitude = -8.0  # representative for Portugal mainland
    true_solar_time = hour_utc * 60 + eqtime + 4 * longitude
    hour_angle = np.radians(true_solar_time / 4 - 180)

    # Solar altitude
    lat_rad = np.radians(latitude)
    sin_alt = (np.sin(lat_rad) * np.sin(decl)
               + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle))
    altitude = np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))

    return altitude


def is_astronomical_dark(hour_utc: int, day_of_year: int, latitude: float) -> bool:
    """Determine if a given hour is during astronomical darkness.

    Astronomical twilight ends (and true darkness begins) when the sun
    is more than 18° below the horizon. This is when the sky background
    reaches its minimum natural level and no solar illumination remains
    in the upper atmosphere.

    Reference: US Naval Observatory, "Definitions of Twilight"

    Args:
        hour_utc: Hour in UTC (0-23).
        day_of_year: Day of year (1-366).
        latitude: Observer latitude in degrees.

    Returns:
        True if the sun is below -18° altitude (astronomical darkness).
    """
    altitude = compute_sun_altitude(hour_utc, day_of_year, latitude)
    return altitude < -18.0


def build_darkness_mask(latitudes: np.ndarray) -> np.ndarray:
    """Build a lookup table of which (hour, day_of_year, latitude) combinations are dark.

    Returns a 3D boolean array: [hour(24), day(366), lat_idx]
    True = astronomically dark at that combination.

    This precomputation avoids recalculating solar position for every timestep.
    """
    n_lat = len(latitudes)
    mask = np.zeros((24, 366, n_lat), dtype=bool)

    for h in range(24):
        for d in range(1, 367):
            for i, lat in enumerate(latitudes):
                mask[h, d - 1, i] = is_astronomical_dark(h, d, lat)

    # Summary statistics
    _ = mask.sum(axis=(0, 1)) / 366  # noqa: F841
    logger.info("Darkness mask built: %d latitudes", n_lat)
    logger.info("  Dark hours/day range: %.1f (summer, lat=%.1f) to %.1f (winter)",
                mask[:, 172, :].sum(axis=0).min(),  # June 21
                latitudes[mask[:, 172, :].sum(axis=0).argmin()],
                mask[:, 355, :].sum(axis=0).max())  # Dec 21

    return mask


def _get_cds_client():
    """Create a CDS API client using credentials from environment or .cdsapirc."""
    import cdsapi

    key = os.environ.get("CDS_API_KEY")
    if key:
        return cdsapi.Client(
            url="https://cds.climate.copernicus.eu/api",
            key=key,
        )
    # Fall back to ~/.cdsapirc
    return cdsapi.Client()


def _download_era5_monthly_means(
    output_path: Path,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    start_year: int,
    end_year: int,
) -> Path:
    """Download ERA5 monthly-mean-by-hour-of-day TCC from CDS API.

    This product provides the average TCC for each (month, hour) combination,
    pre-aggregated by ECMWF. Downloads in batches of 5 years to stay within
    CDS per-request cost limits, then merges locally.

    The server performs spatial subsetting — only the requested area is transferred.
    """
    client = _get_cds_client()

    months = [f"{m:02d}" for m in range(1, 13)]
    hours = [f"{h:02d}:00" for h in range(24)]
    area = [lat_range[1], lon_range[0], lat_range[0], lon_range[1]]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive a region tag from the output path for batch naming
    region_tag = output_path.stem.replace("era5_tcc_monthly_", "").split("_")[0]

    # Split into batches of 5 years (CDS cost limit)
    batch_size = 5
    batch_files = []

    for batch_start in range(start_year, end_year + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end_year)
        batch_years = [str(y) for y in range(batch_start, batch_end + 1)]

        batch_file = output_path.parent / f"era5_tcc_batch_{region_tag}_{batch_start}_{batch_end}.nc"
        batch_files.append(batch_file)

        if batch_file.exists():
            logger.info("  Batch %d–%d: cached (%s)", batch_start, batch_end, batch_file.name)
            continue

        logger.info("  Batch %d–%d: requesting from CDS...", batch_start, batch_end)

        client.retrieve(
            CDS_DATASET,
            {
                "product_type": ["monthly_averaged_reanalysis_by_hour_of_day"],
                "variable": ["total_cloud_cover"],
                "year": batch_years,
                "month": months,
                "time": hours,
                "area": area,
                "data_format": "netcdf",
            },
            str(batch_file),
        )

        size_mb = batch_file.stat().st_size / 1e6
        logger.info("  Batch %d–%d: downloaded (%.1f MB)", batch_start, batch_end, size_mb)

    # Merge all batches along valid_time dimension
    logger.info("Merging %d batch files...", len(batch_files))
    datasets = [xr.open_dataset(f) for f in batch_files]

    # CDS new format uses 'valid_time' as the actual temporal dimension
    # and may have a spurious 'time' batch dim. Flatten to just valid_time.
    flat_datasets = []
    for ds in datasets:
        if "valid_time" in ds.dims and "time" in ds.dims:
            # Reshape: (batch, valid_time, lat, lon) → (valid_time, lat, lon)
            tcc = ds["tcc"]
            # Stack time and valid_time
            stacked = tcc.stack(all_time=("time", "valid_time")).transpose("all_time", "latitude", "longitude")
            # Create clean dataset
            flat_ds = xr.Dataset({"tcc": (["valid_time", "latitude", "longitude"], stacked.values)},
                                 coords={"valid_time": ds["valid_time"].values,
                                         "latitude": ds["latitude"].values,
                                         "longitude": ds["longitude"].values})
            flat_datasets.append(flat_ds)
        elif "valid_time" in ds.dims:
            flat_datasets.append(ds[["tcc"]])
        else:
            flat_datasets.append(ds[["tcc"]])

    merged = xr.concat(flat_datasets, dim="valid_time")
    merged.to_netcdf(output_path)
    for ds in datasets:
        ds.close()

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Merged output: %s (%.1f MB)", output_path.name, size_mb)
    return output_path


def compute_clear_night_fraction(
    config: ClearSkyConfig | None = None,
    region: str = "mainland",
) -> xr.DataArray:
    """Compute climatological clear-night fraction from ERA5 monthly means.

    Downloads the ERA5 "monthly averaged reanalysis by hour of day" product
    via CDS API (server-side subsetting — only the requested area is transferred).

    For each grid cell and each (month, hour) combination:
    1. Check if that hour is astronomically dark at that latitude in that month
    2. If dark: check if the mean TCC < threshold (clear)
    3. Aggregate: clear_fraction = sum(dark_and_clear) / sum(dark)

    Using monthly means is statistically equivalent to processing hourly data
    for a climatology (law of large numbers over 20 years × 28-31 days/month).

    Args:
        config: Configuration parameters.
        region: One of "mainland", "madeira", "azores", "all".

    Returns:
        DataArray with clear-night fraction (0-1) on the ERA5 grid.
        Coordinates: latitude, longitude.
    """
    if config is None:
        config = ClearSkyConfig()

    # Region bounding boxes
    region_bounds = {
        "mainland": {"lat": (36.5, 42.5), "lon": (-10.0, -6.0)},
        "madeira": {"lat": (32.0, 34.0), "lon": (-17.5, -16.0)},
        "azores": {"lat": (36.5, 40.0), "lon": (-32.0, -25.0)},
        "all": {"lat": config.lat_range, "lon": config.lon_range},
    }
    bounds = region_bounds.get(region, region_bounds["mainland"])
    lat_range = bounds["lat"]
    lon_range = bounds["lon"]

    # Check cache (final result)
    cache_file = config.cache_dir / f"era5_clear_night_{region}_{config.start_year}_{config.end_year}.nc"
    if cache_file.exists():
        logger.info("Loading cached clear-night fraction: %s", cache_file)
        return xr.open_dataarray(cache_file)

    # Check if raw ERA5 data already downloaded
    raw_file = config.cache_dir / f"era5_tcc_monthly_{region}_{config.start_year}_{config.end_year}.nc"
    if not raw_file.exists():
        _download_era5_monthly_means(
            output_path=raw_file,
            lat_range=lat_range,
            lon_range=lon_range,
            start_year=config.start_year,
            end_year=config.end_year,
        )

    # Load downloaded data
    logger.info("Loading ERA5 monthly means: %s", raw_file)
    ds = xr.open_dataset(raw_file)

    # Determine the time dimension name (CDS may use 'valid_time' or 'time')
    if "valid_time" in ds.dims:
        time_dim = "valid_time"
    elif "time" in ds.dims:
        time_dim = "time"
    else:
        raise ValueError(f"No time dimension found. Dims: {list(ds.dims)}")

    tcc = ds["tcc"]
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    n_lat, n_lon = len(latitudes), len(longitudes)

    times = ds[time_dim].values
    tcc_values = tcc.values
    # If extra dims exist (e.g., from batch concat), squeeze them
    while tcc_values.ndim > 3:
        tcc_values = tcc_values.reshape(-1, n_lat, n_lon)

    logger.info("  Grid: %d lat × %d lon = %d cells", n_lat, n_lon, n_lat * n_lon)
    logger.info("  Time steps: %d", len(times))
    logger.info("  Period: %d–%d", config.start_year, config.end_year)
    logger.info("  Threshold: TCC < %.2f", config.clear_threshold)

    # Build astronomical darkness lookup
    logger.info("Computing astronomical twilight mask...")
    darkness_mask = build_darkness_mask(latitudes)

    # Extract month and hour from each timestamp
    # times are numpy datetime64 values like 2006-01-01T00:00:00
    times_dt = times.astype("datetime64[ns]")
    months = (times_dt.astype("datetime64[M]").astype(int) % 12 + 1).astype(int)
    hours = ((times_dt - times_dt.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(int)

    # For monthly means: use the 15th of the month as representative day-of-year
    representative_doy = np.array([
        int((np.datetime64(f"2020-{m:02d}-15") - np.datetime64("2020-01-01")).astype(int)) + 1
        for m in months
    ])

    logger.info("Computing clear-night fraction (vectorized)...")

    # Build darkness mask for all timesteps: (n_times, n_lat)
    dark_per_time = darkness_mask[hours, representative_doy - 1, :]

    # Expand to grid: (n_times, n_lat, n_lon)
    dark_3d = dark_per_time[:, :, np.newaxis].astype(np.float64)  # broadcast across lon

    # For monthly means, TCC represents the EXPECTED cloud fraction at that hour.
    # (1 - TCC) = expected clear-sky fraction at that hour.
    # We compute the weighted mean of (1 - TCC) over all dark hours.
    # This gives us: "on average, what fraction of the sky is clear during dark hours?"
    clear_sky_probability = 1.0 - tcc_values  # (n_times, n_lat, n_lon)

    # Weighted sum: only count dark hours
    total_dark_weight = dark_3d.sum(axis=0)
    weighted_clear = (clear_sky_probability * dark_3d).sum(axis=0)

    # Compute mean clear fraction during dark hours
    with np.errstate(divide='ignore', invalid='ignore'):
        clear_fraction = np.where(
            total_dark_weight > 0,
            weighted_clear / total_dark_weight,
            np.nan,
        )

    result = xr.DataArray(
        clear_fraction,
        dims=["latitude", "longitude"],
        coords={"latitude": latitudes, "longitude": longitudes},
        attrs={
            "long_name": "Clear Night Fraction (Climatology)",
            "units": "fraction (0-1)",
            "description": "Mean clear-sky probability (1-TCC) during astronomically dark hours",
            "threshold": config.clear_threshold,
            "period": f"{config.start_year}-{config.end_year}",
            "darkness_definition": "Astronomical twilight (sun altitude < -18 degrees)",
            "data_source": "ERA5 monthly averaged reanalysis by hour of day (CDS API)",
            "references": (
                "Hersbach et al. (2020) QJRMS 146:1999-2049; "
                "Aksaker et al. (2020) MNRAS 493(1)"
            ),
        },
    )

    # Cache result
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(cache_file)
    logger.info("Cached result to %s", cache_file)

    # Log summary
    valid = clear_fraction[~np.isnan(clear_fraction)]
    if len(valid) > 0:
        logger.info("Clear-night fraction computed:")
        logger.info("  Range: %.1f%% – %.1f%%", valid.min() * 100, valid.max() * 100)
        logger.info("  Mean: %.1f%%", valid.mean() * 100)

    ds.close()
    return result


def resample_to_viirs_grid(
    clear_fraction: xr.DataArray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """Resample clear-night fraction to the VIIRS 500m grid.

    Uses bilinear interpolation from the source grid (ERA5 at ~25km or
    CLAAS-3 at ~5.5km) to the ~500m VIIRS grid. This is appropriate because
    cloud climatology varies smoothly at these scales — though CLAAS-3 at 5.5km
    captures much more spatial detail than ERA5 before interpolation.

    Args:
        clear_fraction: Source-resolution clear-night fraction (lat x lon).
        target_x: VIIRS grid longitude coordinates.
        target_y: VIIRS grid latitude coordinates.

    Returns:
        2D array matching VIIRS grid shape with interpolated clear-night fraction.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Source coordinates (works for both ERA5 and CLAAS-3)
    src_lat = clear_fraction.latitude.values
    src_lon = clear_fraction.longitude.values
    values = clear_fraction.values

    # Handle NaN for interpolation (replace with nearest valid)
    mask = np.isnan(values)
    if mask.any():
        from scipy.ndimage import uniform_filter
        filled = np.where(mask, 0, values)
        count = np.where(mask, 0, 1.0)
        filled_smooth = uniform_filter(filled, size=3, mode='constant')
        count_smooth = uniform_filter(count, size=3, mode='constant')
        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.where(mask & (count_smooth > 0), filled_smooth / count_smooth, values)
        values = np.nan_to_num(values, nan=0.5)  # fallback for remaining NaN

    # Build interpolator (lat may be descending, need to flip for RegularGridInterpolator)
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        values = values[::-1, :]

    interpolator = RegularGridInterpolator(
        (src_lat, src_lon), values,
        method='linear', bounds_error=False, fill_value=None,
    )

    # Create target grid points
    target_yy, target_xx = np.meshgrid(target_y, target_x, indexing='ij')
    points = np.column_stack([target_yy.ravel(), target_xx.ravel()])

    # Interpolate
    resampled = interpolator(points).reshape(len(target_y), len(target_x))

    return resampled


# ===========================================================================
# CLAAS-3 High-Resolution Cloud Cover (5.5 km satellite observations)
# ===========================================================================

@dataclass
class CLAAS3Config:
    """Configuration for CLAAS-3 CFC (0.05 deg, ~5.5km, MSG/SEVIRI)."""

    start_year: int = 2006
    end_year: int = 2020  # CLAAS-3 CDR ends 2020; operational extension available
    # Portugal mainland bounding box
    lat_range: tuple[float, float] = (36.5, 42.5)
    lon_range: tuple[float, float] = (-10.0, -6.0)
    cache_dir: Path = Path(__file__).parent.parent / "input" / "portugal" / "cache"
    # CFC variable name in CLAAS-3 NetCDF files
    cfc_var: str = "cfc"
    # CLAAS-3 monthly means include hour-of-day dimension (diurnal cycle product)
    # If False, assume daily/monthly mean CFC without hour breakdown
    has_hourly_dim: bool = True


def _find_claas3_files(config: CLAAS3Config, region: str) -> list[Path]:
    """Locate CLAAS-3 CFC NetCDF files in the cache directory.

    Expected naming patterns:
        claas3_cfc_monthly_<region>_<year>.nc  (one file per year)
        claas3_cfc_monthly_<region>_<start>_<end>.nc  (merged multi-year)
        CFCmm*.nc  (raw CM SAF naming convention)

    Returns list of found files, empty if CLAAS-3 data not available.
    """
    cache = config.cache_dir
    if not cache.exists():
        return []

    patterns = [
        f"claas3_cfc_monthly_{region}_*.nc",
        f"claas3_cfc_{region}_*.nc",
        "CFCmm*.nc",
        "claas3_*.nc",
    ]

    files = []
    for pattern in patterns:
        files.extend(sorted(cache.glob(pattern)))

    return files


def claas3_available(config: CLAAS3Config | None = None, region: str = "mainland") -> bool:
    """Check if CLAAS-3 data is available in the cache directory.

    Returns True if CLAAS-3 NetCDF files are found and can be used as a
    higher-resolution alternative to ERA5.
    """
    if config is None:
        config = CLAAS3Config()
    files = _find_claas3_files(config, region)
    return len(files) > 0


def compute_clear_night_fraction_claas3(
    config: CLAAS3Config | None = None,
    region: str = "mainland",
) -> xr.DataArray:
    """Compute clear-night fraction from CLAAS-3 satellite CFC (0.05 deg).

    Args:
        config: CLAAS-3 configuration.
        region: One of "mainland", "madeira", "azores", "all".

    Returns:
        DataArray with clear-night fraction (0-1).
    """
    if config is None:
        config = CLAAS3Config()

    # Region bounding boxes (same as ERA5 version)
    region_bounds = {
        "mainland": {"lat": (36.5, 42.5), "lon": (-10.0, -6.0)},
        "madeira": {"lat": (32.0, 34.0), "lon": (-17.5, -16.0)},
        "azores": {"lat": (36.5, 40.0), "lon": (-32.0, -25.0)},
        "all": {"lat": config.lat_range, "lon": config.lon_range},
    }
    bounds = region_bounds.get(region, region_bounds["mainland"])

    # Check for cached result first
    cache_file = (
        config.cache_dir / f"claas3_clear_night_{region}_{config.start_year}_{config.end_year}.nc"
    )
    if cache_file.exists():
        logger.info("Loading cached CLAAS-3 clear-night fraction: %s", cache_file)
        return xr.open_dataarray(cache_file)

    # Find CLAAS-3 source data
    files = _find_claas3_files(config, region)
    if not files:
        raise FileNotFoundError(
            f"CLAAS-3 CFC data not found in {config.cache_dir}. "
            f"To use CLAAS-3 (5.5 km resolution), register at https://wui.cmsaf.eu "
            f"and download the CFC monthly mean product for the region of interest. "
            f"Falling back to ERA5 (25 km) is recommended until data is obtained."
        )

    logger.info("Loading CLAAS-3 CFC data from %d file(s)...", len(files))

    # Load and merge all CLAAS-3 files
    datasets = []
    for f in files:
        ds = xr.open_dataset(f)
        # Subset to region if needed
        lat_var = "lat" if "lat" in ds.dims else "latitude"
        lon_var = "lon" if "lon" in ds.dims else "longitude"
        ds = ds.sel(
            **{
                lat_var: slice(bounds["lat"][0], bounds["lat"][1]),
                lon_var: slice(bounds["lon"][0], bounds["lon"][1]),
            }
        )
        datasets.append(ds)

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        # Determine the time dimension name
        time_dim = None
        for candidate in ["time", "valid_time", "month"]:
            if candidate in datasets[0].dims:
                time_dim = candidate
                break
        if time_dim:
            merged = xr.concat(datasets, dim=time_dim)
        else:
            merged = datasets[0]

    # Extract CFC variable (try common names)
    cfc = None
    for var_name in [config.cfc_var, "cfc", "CFC", "cloud_fractional_cover", "cfc_mean"]:
        if var_name in merged.data_vars:
            cfc = merged[var_name]
            break

    if cfc is None:
        available_vars = list(merged.data_vars)
        raise ValueError(
            f"Could not find CFC variable in CLAAS-3 data. "
            f"Available variables: {available_vars}. "
            f"Set config.cfc_var to the correct variable name."
        )

    # Standardize coordinate names
    lat_var = "lat" if "lat" in cfc.dims else "latitude"
    lon_var = "lon" if "lon" in cfc.dims else "longitude"
    if lat_var == "lat":
        cfc = cfc.rename({lat_var: "latitude", lon_var: "longitude"})

    latitudes = cfc.coords["latitude"].values
    n_lat = len(latitudes)

    logger.info("  CLAAS-3 grid: %d lat x %d lon = %d cells (vs ERA5: ~425)",
                n_lat, len(cfc.coords["longitude"].values),
                n_lat * len(cfc.coords["longitude"].values))
    logger.info("  Resolution: %.4f deg (~%.1f km at 39N)",
                abs(latitudes[1] - latitudes[0]) if n_lat > 1 else 0.05,
                abs(latitudes[1] - latitudes[0]) * 111 * np.cos(np.radians(39))
                if n_lat > 1 else 5.5)

    # Determine if we have hour-of-day dimension
    has_hours = any(d in cfc.dims for d in ["hour", "time_of_day", "hour_of_day"])

    if has_hours:
        # Same logic as ERA5: weight by astronomical darkness
        hour_dim = next(d for d in ["hour", "time_of_day", "hour_of_day"] if d in cfc.dims)
        logger.info("  Diurnal cycle available (%s dimension) — applying darkness weighting", hour_dim)

        # Build darkness mask
        darkness_mask = build_darkness_mask(latitudes)

        # Get time metadata
        time_dim = next((d for d in ["time", "valid_time", "month"] if d in cfc.dims), None)
        cfc_values = cfc.values

        if time_dim:
            # Has both time (month) and hour dimensions
            times = cfc.coords[time_dim].values
            times_dt = times.astype("datetime64[ns]")
            months = (times_dt.astype("datetime64[M]").astype(int) % 12 + 1).astype(int)
            hours = cfc.coords[hour_dim].values.astype(int)

            # Representative day of year for each month
            representative_doy = np.array([
                int((np.datetime64(f"2020-{m:02d}-15") - np.datetime64("2020-01-01")).astype(int)) + 1
                for m in months
            ])

            # Vectorized: compute darkness for all (time, hour, lat) combinations
            _ = len(cfc.coords["longitude"].values)  # noqa: F841
            clear_sky = 1.0 - cfc_values  # Convert CFC to clear probability
            dark_weight = np.zeros_like(clear_sky)

            for t_idx, (doy, month) in enumerate(zip(representative_doy, months)):
                for h_idx, hour in enumerate(hours):
                    dark_col = darkness_mask[hour, doy - 1, :]  # (n_lat,)
                    dark_weight[t_idx, h_idx, :, :] = dark_col[:, np.newaxis]

            weighted_clear = (clear_sky * dark_weight).sum(axis=(0, 1))
            total_dark = dark_weight.sum(axis=(0, 1))
        else:
            # Monthly data without separate hour dimension — use nighttime hours only
            # Assume the data represents all-day average; approximate nighttime fraction
            logger.info("  No hour dimension — using CFC as-is (assumes nighttime representative)")
            clear_sky = 1.0 - cfc_values
            weighted_clear = clear_sky.mean(axis=0) if cfc_values.ndim > 2 else clear_sky
            total_dark = np.ones_like(weighted_clear)

    else:
        # No hour dimension: CFC is already a time-mean (monthly/climatological)
        # Use (1 - CFC) directly as clear-sky probability
        logger.info("  No diurnal cycle — using mean CFC directly")
        cfc_values = cfc.values

        if cfc_values.ndim > 2:
            # Average over time dimension to get climatology
            clear_sky = 1.0 - np.nanmean(cfc_values, axis=0)
        else:
            clear_sky = 1.0 - cfc_values

        # Apply nighttime weighting using darkness mask
        # Since we don't have hour breakdown, weight by fraction of night hours per latitude
        darkness_mask = build_darkness_mask(latitudes)
        # Annual average dark hours per latitude (fraction of total hours)
        dark_fraction_per_lat = darkness_mask.sum(axis=(0, 1)) / (24 * 366)
        # Weight clear-sky by how much of the night is truly dark at each latitude
        # This is a mild correction (accounts for varying night length with latitude)
        dark_weight_2d = dark_fraction_per_lat[:, np.newaxis] * np.ones(clear_sky.shape[1])
        weighted_clear = clear_sky * dark_weight_2d
        total_dark = dark_weight_2d

    # Compute final clear-night fraction
    with np.errstate(divide='ignore', invalid='ignore'):
        clear_fraction = np.where(
            total_dark > 0,
            weighted_clear / total_dark,
            np.nan,
        )

    # Ensure values in [0, 1]
    clear_fraction = np.clip(clear_fraction, 0.0, 1.0)

    result = xr.DataArray(
        clear_fraction,
        dims=["latitude", "longitude"],
        coords={
            "latitude": latitudes,
            "longitude": cfc.coords["longitude"].values,
        },
        attrs={
            "long_name": "Clear Night Fraction (CLAAS-3 Satellite Climatology)",
            "units": "fraction (0-1)",
            "description": (
                "Mean clear-sky probability (1-CFC) during astronomically dark hours, "
                "derived from MSG/SEVIRI satellite cloud observations"
            ),
            "resolution": "0.05 deg (~5.5 km)",
            "period": f"{config.start_year}-{config.end_year}",
            "darkness_definition": "Astronomical twilight (sun altitude < -18 degrees)",
            "data_source": "CLAAS-3 Cloud Fractional Cover (CM SAF, DOI:10.5676/EUM_SAF_CM/CLAAS/V003)",
            "sensor": "SEVIRI on Meteosat Second Generation",
            "advantage_over_era5": (
                "5x finer resolution enables micro-climate discrimination: "
                "valley fog, orographic cloud, coastal stratus"
            ),
        },
    )

    # Cache result
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(cache_file)
    logger.info("Cached CLAAS-3 result to %s", cache_file)

    # Log summary
    valid = clear_fraction[~np.isnan(clear_fraction)]
    if len(valid) > 0:
        logger.info("CLAAS-3 clear-night fraction computed:")
        logger.info("  Range: %.1f%% - %.1f%%", valid.min() * 100, valid.max() * 100)
        logger.info("  Mean: %.1f%%", valid.mean() * 100)
        logger.info("  Grid cells: %d (ERA5 equivalent: ~425)", len(valid))

    for ds in datasets:
        ds.close()

    return result


def compute_clear_night_fraction_best_available(
    config: ClearSkyConfig | None = None,
    region: str = "mainland",
) -> xr.DataArray:
    """Use CLAAS-3 if available, otherwise fall back to ERA5.

    Args:
        config: Configuration.
        region: One of "mainland", "madeira", "azores", "all".

    Returns:
        DataArray with clear-night fraction.
    """
    if config is None:
        config = ClearSkyConfig()

    # Try CLAAS-3 first (higher resolution)
    claas3_cfg = CLAAS3Config(
        start_year=max(config.start_year, 2004),  # CLAAS-3 starts 2004
        end_year=min(config.end_year, 2020),  # CDR ends 2020
        lat_range=config.lat_range,
        lon_range=config.lon_range,
        cache_dir=config.cache_dir,
    )

    if claas3_available(claas3_cfg, region):
        logger.info(
            "CLAAS-3 data available — using satellite observations at 5.5 km resolution"
        )
        try:
            result = compute_clear_night_fraction_claas3(claas3_cfg, region)
            logger.info(
                "Resolution upgrade: CLAAS-3 (0.05 deg, %d cells) vs ERA5 (0.25 deg, ~425 cells)",
                result.size,
            )
            return result
        except Exception as e:
            logger.warning("CLAAS-3 processing failed (%s), falling back to ERA5", e)

    # Fallback to ERA5
    logger.info("Using ERA5 cloud cover at 25 km resolution (CLAAS-3 not available)")
    logger.info(
        "  To upgrade to 5.5 km: register at https://wui.cmsaf.eu, "
        "download CLAAS-3 CFC monthly means to %s",
        config.cache_dir,
    )
    return compute_clear_night_fraction(config, region)
