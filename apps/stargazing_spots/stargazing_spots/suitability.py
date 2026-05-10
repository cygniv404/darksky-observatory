"""Stargazing Suitability Index (SSI): 6-factor weighted linear combination.

Weights: darkness 0.30, clear_sky 0.25, SVF 0.15, land_cover 0.15,
slope 0.10, elevation 0.05. Output: 0-100 score per spot.
"""

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SSIWeights:
    """SSI component weights. Must sum to 1.0."""

    darkness: float = 0.30
    clear_sky: float = 0.25
    sky_openness: float = 0.15
    land_cover: float = 0.15
    slope: float = 0.10
    elevation: float = 0.05

    def __post_init__(self):
        total = (self.darkness + self.clear_sky + self.sky_openness
                 + self.land_cover + self.slope + self.elevation)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")


def normalize_darkness(predicted_sqm: np.ndarray) -> np.ndarray:
    """Normalize SQM prediction to [0, 1] suitability score.

    Normalizes over the dark-site range (20.0–22.0) for discrimination
    among candidate stargazing sites:

        SQM 22.0 (pristine) → 1.0
        SQM 21.0 (good)     → 0.5
        SQM 20.0 (marginal) → 0.0

    With the softened PSF (Duriscoe d₀=1km), the model is unbiased
    (MAE 0.32 mag, bias -0.02) so no bias correction is needed.
    The range 20.0-22.0 provides good discrimination for dark sites
    while correctly scoring suburban pixels (SQM 18-20) at zero.

    Note: The +0.5 mag systematic bias (vs full Garstang RT) is implicitly
    absorbed here — a "score=1.0" spot has predicted SQM 22.0 but real sky
    ~21.5, and "score=0.5" predicts 21.0 but real sky ~20.5.
    """
    return np.clip((predicted_sqm - 20.0) / (22.0 - 20.0), 0, 1)


def normalize_clear_sky(clear_fraction: np.ndarray) -> np.ndarray:
    """Normalize clear-night fraction to [0, 1].

    Already in [0, 1] from the ERA5 computation.
    Apply slight power scaling to reward consistently clear sites.
    """
    return np.clip(clear_fraction, 0, 1)


def normalize_sky_openness(svf: np.ndarray) -> np.ndarray:
    """Normalize Sky View Factor to [0, 1] suitability.

    SVF 1.0 (fully open)    → 1.0
    SVF 0.85 (minor hills)  → 0.75 (still good)
    SVF 0.5 (obstructed)    → 0.0

    Linear between 0.5 and 1.0; below 0.5 is unsuitable.
    """
    return np.clip((svf - 0.5) / 0.5, 0, 1)


def normalize_elevation(elevation_m: np.ndarray) -> np.ndarray:
    """Normalize elevation to [0, 1] bonus.

    0m      → 0.0 (sea level, more atmosphere)
    500m    → 0.33
    1500m+  → 1.0 (significant atmospheric reduction)

    Capped at 1500m for Portugal context (max ~2000m Serra da Estrela).
    """
    return np.clip(elevation_m / 1500.0, 0, 1)


def compute_ssi(
    spots: gpd.GeoDataFrame,
    weights: SSIWeights | None = None,
) -> gpd.GeoDataFrame:
    """Compute the Stargazing Suitability Index for each spot.

    Expects the following columns to already exist in spots (from prior
    assessment modules):
        - predicted_sqm: from skyglow.py
        - clear_night_fraction: from cloud_cover.py (resampled to spots)
        - sky_view_factor: from sky_view.py
        - land_cover_score: from land_cover.py
        - slope_score: from accessibility.py
        - ele: elevation from OSM (optional)

    Missing columns are handled gracefully (weight redistributed).

    Args:
        spots: GeoDataFrame with assessment columns.
        weights: Component weights.

    Returns:
        GeoDataFrame with added 'ssi_score' (0-100) and 'ssi_class' columns.
    """
    if weights is None:
        weights = SSIWeights()

    spots = spots.copy()
    n = len(spots)

    # Normalize each criterion
    import pandas as pd

    # 1. Darkness (from predicted_sqm)
    if "predicted_sqm" in spots.columns:
        s_dark = normalize_darkness(spots["predicted_sqm"].values)
        w_dark = weights.darkness
    else:
        logger.warning("No predicted_sqm column — skipping darkness criterion")
        s_dark = np.full(n, 0.5)
        w_dark = 0.0

    # 2. Clear sky (from cloud_cover resampled to spot locations)
    if "clear_night_fraction" in spots.columns:
        s_clear = normalize_clear_sky(spots["clear_night_fraction"].values)
        w_clear = weights.clear_sky
    else:
        logger.warning("No clear_night_fraction column — skipping clear sky criterion")
        s_clear = np.full(n, 0.5)
        w_clear = 0.0

    # 3. Sky openness (from SVF)
    if "sky_view_factor" in spots.columns:
        s_svf = normalize_sky_openness(spots["sky_view_factor"].values)
        w_svf = weights.sky_openness
    else:
        logger.warning("No sky_view_factor column — skipping sky openness criterion")
        s_svf = np.full(n, 0.5)
        w_svf = 0.0

    # 4. Land cover
    if "land_cover_score" in spots.columns:
        s_land = np.clip(spots["land_cover_score"].values, 0, 1)
        w_land = weights.land_cover
    else:
        logger.warning("No land_cover_score column — skipping land cover criterion")
        s_land = np.full(n, 0.5)
        w_land = 0.0

    # 5. Slope
    if "slope_score" in spots.columns:
        s_slope = np.clip(spots["slope_score"].values, 0, 1)
        w_slope = weights.slope
    else:
        logger.warning("No slope_score column — skipping slope criterion")
        s_slope = np.full(n, 0.5)
        w_slope = 0.0

    # 6. Elevation
    if "ele" in spots.columns:
        elevation = pd.to_numeric(spots["ele"], errors="coerce").fillna(0).values
        s_elev = normalize_elevation(elevation)
        w_elev = weights.elevation
    else:
        s_elev = np.zeros(n)
        w_elev = 0.0

    # Redistribute weights for missing criteria
    total_w = w_dark + w_clear + w_svf + w_land + w_slope + w_elev
    if total_w < 0.99:
        logger.warning("Total weight = %.2f (some criteria missing). Renormalizing.", total_w)
        if total_w > 0:
            scale = 1.0 / total_w
            w_dark *= scale
            w_clear *= scale
            w_svf *= scale
            w_land *= scale
            w_slope *= scale
            w_elev *= scale

    # Weighted linear combination
    ssi = (
        w_dark * s_dark
        + w_clear * s_clear
        + w_svf * s_svf
        + w_land * s_land
        + w_slope * s_slope
        + w_elev * s_elev
    ) * 100

    # Handle NaN (any NaN input → NaN output)
    nan_mask = (
        np.isnan(spots.get("predicted_sqm", pd.Series(dtype=float)).values)
        if "predicted_sqm" in spots.columns
        else np.zeros(n, dtype=bool)
    )
    ssi[nan_mask] = np.nan

    spots["ssi_score"] = np.round(ssi, 1)

    # Classification
    spots["ssi_class"] = spots["ssi_score"].apply(
        lambda s: (
            "excellent" if s >= 80 else
            "good" if s >= 65 else
            "acceptable" if s >= 50 else
            "marginal" if s >= 35 else
            "poor"
        ) if not np.isnan(s) else "unknown"
    )

    # Log summary
    for cls in ["excellent", "good", "acceptable", "marginal", "poor"]:
        count = (spots["ssi_class"] == cls).sum()
        if count > 0:
            mean_score = spots[spots["ssi_class"] == cls]["ssi_score"].mean()
            logger.info("  %s: %d spots (mean SSI %.1f)", cls, count, mean_score)

    # Log component contributions for top spots
    top5 = spots.nlargest(5, "ssi_score")
    logger.info("Top 5 by SSI:")
    for _, row in top5.iterrows():
        logger.info("  SSI %.1f | %s", row["ssi_score"], row.get("name", "?"))

    return spots
