"""PSF skyglow propagation model.

FFT convolution of VIIRS radiance with a Garstang/Duriscoe PSF kernel to
predict clear-sky SQM at each observation point. CALIB_FACTOR fitted against
27 ground-truth points; systematic uncertainty +/-0.5 mag vs full RT.
"""

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)

NATURAL_SKY_MCDM2 = 0.171  # Natural zenith brightness (mcd/m^2), Falchi 2016
NATURAL_SKY_SQM = 22.0  # mag/arcsec^2 at solar minimum

# Fitted by least-squares against 27 ground-truth points (MAE = 0.48 mag)
CALIB_FACTOR = 0.04237

# Atmospheric extinction (Portuguese atmosphere, tau_aerosol ~0.15)
EXTINCTION_RATE = 0.0187  # per km


@dataclass
class PropagationConfig:
    """Configuration for the skyglow propagation model."""

    integration_radius_km: float = 200.0
    psf_exponent: float = 2.5  # Walker/Garstang: d^(-2.5)
    pixel_size_km: float = 0.42  # VIIRS VNP46A4 at ~39°N
    softening_km: float = 1.0  # Near-field softening (Duriscoe 2018)
    extinction_rate: float = EXTINCTION_RATE  # Atmospheric extinction (per km)
    elevation_correction: bool = True


def build_psf_kernel(
    radius_km: float = 200.0,
    pixel_size_km: float = 0.42,
    exponent: float = 2.5,
    softening_km: float = 1.0,
    extinction_rate: float = EXTINCTION_RATE,
) -> np.ndarray:
    """Build 2D PSF kernel: (d + d0)^(-alpha) * exp(-beta*d).

    Self-pixel retained (physically significant at suburban sites).

    Args:
        radius_km: Integration radius.
        pixel_size_km: Pixel size in km.
        exponent: Power-law exponent.
        softening_km: Duriscoe (2018) near-field softening d0.
        extinction_rate: Atmospheric extinction (per km).

    Returns:
        2D kernel array (odd dimensions, centered).
    """
    radius_px = int(radius_km / pixel_size_km)
    size = 2 * radius_px + 1

    y = np.arange(-radius_px, radius_px + 1) * pixel_size_km
    x = np.arange(-radius_px, radius_px + 1) * pixel_size_km
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)

    psf = (d + softening_km) ** (-exponent) * np.exp(-extinction_rate * d)
    psf[d > radius_km] = 0.0

    return psf


def compute_sky_brightness(
    radiance: np.ndarray,
    config: PropagationConfig | None = None,
) -> np.ndarray:
    """Convolve radiance with PSF kernel to predict artificial sky brightness.

    Args:
        radiance: 2D VIIRS radiance (nW/cm2/sr). NaN = nodata.
        config: Propagation model parameters.

    Returns:
        2D array of artificial sky brightness (mcd/m2).
    """
    if config is None:
        config = PropagationConfig()

    logger.info("Building PSF kernel (radius=%.0fkm, exponent=%.1f, softening=%.1fkm)...",
                config.integration_radius_km, config.psf_exponent, config.softening_km)
    psf = build_psf_kernel(
        radius_km=config.integration_radius_km,
        pixel_size_km=config.pixel_size_km,
        exponent=config.psf_exponent,
        softening_km=config.softening_km,
        extinction_rate=config.extinction_rate,
    )
    logger.info("  Kernel size: %d × %d pixels", psf.shape[0], psf.shape[1])

    radiance_clean = np.where(np.isnan(radiance), 0.0, radiance)
    radiance_clean = np.maximum(radiance_clean, 0.0)

    pixel_area_km2 = config.pixel_size_km ** 2

    logger.info("Running FFT convolution...")
    raw_integral = fftconvolve(
        radiance_clean * pixel_area_km2,
        psf,
        mode='same'
    )
    raw_integral = np.maximum(raw_integral, 0.0)

    sky_brightness = CALIB_FACTOR * raw_integral
    sky_brightness[np.isnan(radiance)] = np.nan

    valid = sky_brightness[~np.isnan(sky_brightness)]
    logger.info("  Artificial brightness range: %.4f – %.2f mcd/m²", valid.min(), valid.max())
    logger.info("  Mean: %.4f mcd/m² (%.1f%% of natural)",
                valid.mean(), 100 * valid.mean() / NATURAL_SKY_MCDM2)

    return sky_brightness


def brightness_to_sqm(artificial_mcdm2: np.ndarray) -> np.ndarray:
    """Convert artificial sky brightness to predicted SQM reading.

    SQM measures total sky brightness (artificial + natural background).
    The natural sky at a pristine site is 0.171 mcd/m² (22.0 mag/arcsec²).

    Standard formula (from lightpollutionmap.info, Falchi):
        SQM = log10(total_brightness_mcd / 108000000) / (-0.4)
        Simplifies to: SQM = -2.5 × log10(S_cd) + 12.583

    Verification: natural = 0.171 mcd/m² = 1.71×10⁻⁴ cd/m²
        → SQM = -2.5×log10(1.71e-4) + 12.583 = 22.0 ✓

    Args:
        artificial_mcdm2: Predicted artificial sky brightness in mcd/m².

    Returns:
        Array of predicted SQM values (mag/arcsec²). Higher = darker.
    """
    total_mcdm2 = artificial_mcdm2 + NATURAL_SKY_MCDM2
    total_cdm2 = total_mcdm2 / 1000.0  # mcd → cd

    with np.errstate(divide='ignore', invalid='ignore'):
        sqm = -2.5 * np.log10(total_cdm2) + 12.583

    return sqm


def sqm_to_bortle(sqm: np.ndarray) -> np.ndarray:
    """Classify SQM readings into Bortle scale classes.

    Based on correlations from Bortle (2001) and observational data.

    Args:
        sqm: Array of SQM values (mag/arcsec²).

    Returns:
        Array of Bortle class integers (1-9).
    """
    bortle = np.full_like(sqm, 9, dtype=np.int8)
    bortle[sqm >= 18.4] = 7
    bortle[sqm >= 18.9] = 6
    bortle[sqm >= 19.5] = 5
    bortle[sqm >= 20.4] = 4
    bortle[sqm >= 21.0] = 3
    bortle[sqm >= 21.5] = 2
    bortle[sqm >= 21.7] = 1
    bortle[np.isnan(sqm)] = 0
    return bortle


def classify_stargazing_quality(sqm: float) -> str:
    """Human-readable stargazing quality from SQM value.

    Based on Bortle scale correlations:
        excellent (Bortle 1-2): Zodiacal light visible, Milky Way casts shadows
        good (Bortle 3): Milky Way with structure, M33 visible
        acceptable (Bortle 4): Milky Way obvious, some light domes on horizon
        poor (Bortle 5-6): Milky Way weak or invisible
        urban (Bortle 7+): Only bright stars visible
    """
    if np.isnan(sqm):
        return "unknown"
    if sqm >= 21.5:
        return "excellent"
    elif sqm >= 21.0:
        return "good"
    elif sqm >= 20.4:
        return "acceptable"
    elif sqm >= 19.5:
        return "poor"
    else:
        return "urban"


def apply_elevation_correction(sqm: np.ndarray, elevation_m: np.ndarray) -> np.ndarray:
    """Apply elevation correction to SQM predictions.

    Higher altitude = less atmosphere above = less scattering.
    Empirical correction from Falchi et al. (2016):
        ~0.1-0.2 mag improvement per 1000m elevation.

    Args:
        sqm: Predicted SQM values at sea level.
        elevation_m: Elevation in meters.

    Returns:
        Corrected SQM values (higher = darker = better).
    """
    # Conservative estimate: 0.13 mag/1000m (between Falchi's 0.1-0.2 range)
    correction = 0.13 * elevation_m / 1000.0
    return sqm + correction


def assess_spots(
    spots: gpd.GeoDataFrame,
    radiance: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    config: PropagationConfig | None = None,
) -> gpd.GeoDataFrame:
    """Run full skyglow propagation assessment on a set of spots.

    Computes the PSF-convolved sky brightness map, then samples it at
    each spot location to predict the SQM reading and Bortle class.

    Args:
        spots: GeoDataFrame with Point geometry and optional 'ele' column.
        radiance: Full 2D radiance grid (median, ALL pixels including bright).
        x_coords: 1D array of x (longitude) coordinates.
        y_coords: 1D array of y (latitude) coordinates.
        config: Propagation model configuration.

    Returns:
        GeoDataFrame with added columns:
            - predicted_brightness_mcd (artificial sky brightness)
            - predicted_sqm (mag/arcsec²)
            - predicted_bortle (1-9)
            - stargazing_suitability (excellent/good/acceptable/poor/urban)
    """
    if config is None:
        config = PropagationConfig()

    spots = spots.copy()

    # Step 1: Compute full sky brightness map via PSF convolution
    sky_brightness = compute_sky_brightness(radiance, config)

    # Step 2: Convert to SQM
    sqm_map = brightness_to_sqm(sky_brightness)

    # Step 3: Sample at spot locations
    spot_cols = np.array([int(np.argmin(np.abs(x_coords - geom.x))) for geom in spots.geometry])
    spot_rows = np.array([int(np.argmin(np.abs(y_coords - geom.y))) for geom in spots.geometry])

    brightness_values = np.array([
        sky_brightness[r, c] if 0 <= r < sky_brightness.shape[0] and 0 <= c < sky_brightness.shape[1] else np.nan
        for r, c in zip(spot_rows, spot_cols)
    ])

    sqm_values = np.array([
        sqm_map[r, c] if 0 <= r < sqm_map.shape[0] and 0 <= c < sqm_map.shape[1] else np.nan
        for r, c in zip(spot_rows, spot_cols)
    ])

    # Step 4: Elevation correction
    if config.elevation_correction and 'ele' in spots.columns:
        elevation = pd.to_numeric(spots['ele'], errors='coerce').fillna(0).values
        sqm_values = apply_elevation_correction(sqm_values, elevation)
        logger.info("Applied elevation correction (mean ele: %.0fm)", np.mean(elevation[elevation > 0]))

    # Step 5: Classify
    bortle_values = np.array([int(sqm_to_bortle(np.array([s]))[0]) for s in sqm_values])
    quality_values = [classify_stargazing_quality(s) for s in sqm_values]

    # Store results
    spots['predicted_brightness_mcd'] = np.round(brightness_values, 2)
    spots['predicted_sqm'] = np.round(sqm_values, 2)
    spots['predicted_bortle'] = bortle_values
    spots['stargazing_suitability'] = quality_values

    # Log summary
    for quality in ['excellent', 'good', 'acceptable', 'poor', 'urban']:
        count = quality_values.count(quality)
        if count > 0:
            logger.info("  %s: %d spots", quality, count)

    return spots
