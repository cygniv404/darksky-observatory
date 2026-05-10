"""Per-pixel confidence for dark-sky classification.

Combines distance from threshold (sigmoid), temporal stability, and
year-over-year consistency into a 0-100 confidence score.
"""

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceWeights:
    """Weights for the confidence model components."""

    distance: float = 0.4
    stability: float = 0.3
    consistency: float = 0.3

    def __post_init__(self):
        total = self.distance + self.stability + self.consistency
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


def _sigmoid_distance_score(radiance: np.ndarray, threshold: float, steepness: float = 2.0) -> np.ndarray:
    """Compute distance-from-threshold confidence using a sigmoid curve.

    Pixels well below threshold → score near 1.0
    Pixels near threshold → score near 0.5
    Pixels above threshold → score near 0.0

    The sigmoid provides a smooth transition rather than a hard cutoff,
    reflecting that confidence degrades gradually near the boundary.

    Args:
        radiance: Per-pixel radiance values.
        threshold: Classification threshold.
        steepness: Controls how quickly confidence drops near threshold.

    Returns:
        Array of scores in [0, 1].
    """
    normalized_distance = (threshold - radiance) / threshold
    return 1 / (1 + np.exp(-steepness * normalized_distance * 5))


def _stability_score(std: np.ndarray, threshold: float) -> np.ndarray:
    """Convert temporal standard deviation to a stability confidence score.

    Low std (stable across years) → high score
    High std (variable) → low score

    Normalized by threshold so the scale is meaningful:
    std = 0 → score = 1.0 (perfectly stable)
    std = threshold → score = 0.0 (varies by the entire threshold range)

    Args:
        std: Per-pixel standard deviation across years.
        threshold: Classification threshold for normalization.

    Returns:
        Array of scores in [0, 1].
    """
    normalized = np.clip(std / threshold, 0, 1)
    return 1.0 - normalized


def _consistency_score(years_dark: np.ndarray, n_years: int) -> np.ndarray:
    """Convert years-dark count to a consistency confidence score.

    Dark in all years → 1.0
    Dark in no years → 0.0

    Args:
        years_dark: Per-pixel count of years classified as dark.
        n_years: Total number of years in the analysis.

    Returns:
        Array of scores in [0, 1].
    """
    return years_dark / max(n_years, 1)


def compute_confidence_layer(
    median_radiance: xr.DataArray,
    stability: xr.DataArray | None = None,
    consistency: xr.DataArray | None = None,
    threshold: float = 3.0,
    n_years: int = 3,
    weights: ConfidenceWeights | None = None,
) -> xr.DataArray:
    """Compute a per-pixel confidence layer for dark-sky classification.

    Combines distance from threshold, temporal stability, and classification
    consistency into a single confidence score per pixel.

    Args:
        median_radiance: Median radiance DataArray (from temporal composite).
        stability: Temporal std deviation DataArray. If None, stability weight is redistributed.
        consistency: Years-dark count DataArray. If None, consistency weight is redistributed.
        threshold: Dark-sky classification threshold.
        n_years: Number of years in temporal analysis.
        weights: Model component weights.

    Returns:
        DataArray with confidence values in [0, 100] (percentage).
        Only computed for pixels where median < threshold (dark pixels).
    """
    if weights is None:
        weights = ConfidenceWeights()

    radiance_vals = median_radiance.values

    # Distance score (always available)
    dist_score = _sigmoid_distance_score(radiance_vals, threshold)

    # Stability score (if temporal data available)
    if stability is not None:
        stab_score = _stability_score(stability.values, threshold)
        w_stab = weights.stability
    else:
        stab_score = np.ones_like(radiance_vals) * 0.5
        w_stab = 0.0

    # Consistency score (if temporal data available)
    if consistency is not None:
        cons_score = _consistency_score(consistency.values, n_years)
        w_cons = weights.consistency
    else:
        cons_score = np.ones_like(radiance_vals) * 0.5
        w_cons = 0.0

    # Redistribute weights if temporal data missing
    w_dist = weights.distance
    if w_stab == 0 and w_cons == 0:
        w_dist = 1.0
    elif w_stab == 0:
        w_dist += weights.stability / 2
        w_cons += weights.stability / 2
    elif w_cons == 0:
        w_dist += weights.consistency / 2
        w_stab += weights.consistency / 2

    # Combined confidence
    confidence = (w_dist * dist_score + w_stab * stab_score + w_cons * cons_score) * 100

    # Mask: only dark pixels get confidence scores
    dark_mask = radiance_vals < threshold
    confidence[~dark_mask] = np.nan
    confidence[np.isnan(radiance_vals)] = np.nan

    result = xr.DataArray(
        confidence,
        dims=median_radiance.dims,
        coords=median_radiance.coords,
        attrs={
            "units": "percent",
            "long_name": "Dark-Sky Classification Confidence",
            "valid_range": "0-100",
            "weights": f"distance={w_dist:.2f}, stability={w_stab:.2f}, consistency={w_cons:.2f}",
        },
    )

    valid_confidence = confidence[~np.isnan(confidence)]
    if len(valid_confidence) > 0:
        logger.info("Confidence layer computed:")
        logger.info("  Mean confidence: %.1f%%", np.mean(valid_confidence))
        logger.info("  High (>75%%): %d pixels", np.sum(valid_confidence > 75))
        logger.info("  Medium (50-75%%): %d pixels", np.sum((valid_confidence >= 50) & (valid_confidence <= 75)))
        logger.info("  Low (<50%%): %d pixels", np.sum(valid_confidence < 50))

    return result


def enrich_spots_with_confidence(
    spots: gpd.GeoDataFrame,
    confidence_layer: xr.DataArray,
) -> gpd.GeoDataFrame:
    """Add per-spot confidence scores by sampling the confidence raster.

    Args:
        spots: GeoDataFrame of stargazing spots with Point geometry.
        confidence_layer: Output of compute_confidence_layer().

    Returns:
        GeoDataFrame with added 'dark_confidence' (0-100) and 'confidence_class' columns.
    """
    y_coords = confidence_layer.coords["y"].values
    x_coords = confidence_layer.coords["x"].values
    values = confidence_layer.values

    scores = []
    for _, row in spots.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        y_idx = int(np.argmin(np.abs(y_coords - lat)))
        x_idx = int(np.argmin(np.abs(x_coords - lon)))
        scores.append(float(values[y_idx, x_idx]))

    spots = spots.copy()
    spots["dark_confidence"] = scores
    spots["confidence_class"] = spots["dark_confidence"].apply(
        lambda c: "high" if c > 75 else ("medium" if c >= 50 else "low")
    )

    high = (spots["confidence_class"] == "high").sum()
    medium = (spots["confidence_class"] == "medium").sum()
    low = (spots["confidence_class"] == "low").sum()
    logger.info("Spot confidence: %d high, %d medium, %d low", high, medium, low)

    return spots
