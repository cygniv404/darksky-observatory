"""Multi-temporal radiance change detection (2023-2025).

Pairwise and overall trend classification. With 3 years, results represent
inter-annual variability, not confirmed trends (Kyba et al. 2017).
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class ChangeStats:
    """Summary statistics for a change detection analysis."""

    period: str
    total_valid_pixels: int
    brightening_pixels: int
    stable_pixels: int
    darkening_pixels: int
    mean_change: float
    median_change: float
    max_brightening: float
    max_darkening: float

    @property
    def brightening_pct(self) -> float:
        return 100 * self.brightening_pixels / max(self.total_valid_pixels, 1)

    @property
    def darkening_pct(self) -> float:
        return 100 * self.darkening_pixels / max(self.total_valid_pixels, 1)

    @property
    def stable_pct(self) -> float:
        return 100 * self.stable_pixels / max(self.total_valid_pixels, 1)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "total_valid_pixels": self.total_valid_pixels,
            "brightening_pixels": self.brightening_pixels,
            "brightening_pct": round(self.brightening_pct, 2),
            "stable_pixels": self.stable_pixels,
            "stable_pct": round(self.stable_pct, 2),
            "darkening_pixels": self.darkening_pixels,
            "darkening_pct": round(self.darkening_pct, 2),
            "mean_change_nw": round(self.mean_change, 4),
            "median_change_nw": round(self.median_change, 4),
            "max_brightening_nw": round(self.max_brightening, 2),
            "max_darkening_nw": round(self.max_darkening, 2),
        }


def compute_change(
    radiance_earlier: xr.DataArray,
    radiance_later: xr.DataArray,
) -> xr.DataArray:
    """Compute per-pixel radiance change between two time periods.

    Args:
        radiance_earlier: Radiance grid from the earlier period.
        radiance_later: Radiance grid from the later period.

    Returns:
        DataArray of change values (later - earlier).
        Positive = brightening (more light pollution).
        Negative = darkening (less light pollution).

    Raises:
        ValueError: If grids have different shapes.
    """
    if radiance_earlier.shape != radiance_later.shape:
        raise ValueError(
            f"Shape mismatch: earlier={radiance_earlier.shape}, later={radiance_later.shape}. "
            f"Both grids must cover the same spatial extent."
        )

    change = radiance_later - radiance_earlier
    change.attrs = {
        "units": "nW/cm²/sr",
        "long_name": "Radiance Change",
        "description": "Positive = brightening, Negative = darkening",
    }
    return change


def classify_change(
    change: xr.DataArray,
    threshold: float = 1.0,
) -> xr.DataArray:
    """Classify change into categories: brightening, stable, darkening.

    Args:
        change: Per-pixel change DataArray (output of compute_change).
        threshold: Minimum absolute change magnitude to classify as changed (nW/cm²/sr).

    Returns:
        DataArray with integer classes:
            0 = no data
            1 = darkening (decreased by more than threshold)
            2 = stable (within ±threshold)
            3 = brightening (increased by more than threshold)
    """
    values = change.values
    classes = np.full_like(values, 0, dtype=np.int8)

    valid = ~np.isnan(values)
    classes[valid & (values > threshold)] = 3
    classes[valid & (values < -threshold)] = 1
    classes[valid & (np.abs(values) <= threshold)] = 2

    result = xr.DataArray(
        classes,
        dims=change.dims,
        coords=change.coords,
        attrs={
            "long_name": "Change Classification",
            "flag_values": "0, 1, 2, 3",
            "flag_meanings": "no_data darkening stable brightening",
            "threshold_nw": threshold,
        },
    )
    return result


def compute_change_stats(
    change: xr.DataArray, threshold: float = 1.0, period: str = ""
) -> ChangeStats:
    """Compute summary statistics for a change detection result."""
    values = change.values[~np.isnan(change.values)]
    total = len(values)

    if total == 0:
        return ChangeStats(
            period=period,
            total_valid_pixels=0,
            brightening_pixels=0,
            stable_pixels=0,
            darkening_pixels=0,
            mean_change=0.0,
            median_change=0.0,
            max_brightening=0.0,
            max_darkening=0.0,
        )

    return ChangeStats(
        period=period,
        total_valid_pixels=total,
        brightening_pixels=int(np.sum(values > threshold)),
        stable_pixels=int(np.sum(np.abs(values) <= threshold)),
        darkening_pixels=int(np.sum(values < -threshold)),
        mean_change=float(np.mean(values)),
        median_change=float(np.median(values)),
        max_brightening=float(np.max(values)),
        max_darkening=float(np.min(values)),
    )


def run_change_detection(
    radiance_2023: xr.DataArray,
    radiance_2024: xr.DataArray,
    radiance_2025: xr.DataArray,
    threshold: float = 1.0,
) -> dict:
    """Run full multi-year change detection analysis.

    Computes pairwise changes (2023→2024, 2024→2025) and overall trend (2023→2025).

    Args:
        radiance_2023: Annual composite for 2023.
        radiance_2024: Annual composite for 2024.
        radiance_2025: Annual composite for 2025.
        threshold: Change magnitude threshold (nW/cm²/sr).

    Returns:
        Dictionary with change maps, classifications, and statistics for each pair.
    """
    logger.info("Running multi-temporal change detection (2023 → 2024 → 2025)")
    logger.info("Change threshold: ±%.1f nW/cm²/sr", threshold)

    results = {}

    pairs = [
        ("2023_to_2024", radiance_2023, radiance_2024),
        ("2024_to_2025", radiance_2024, radiance_2025),
        ("2023_to_2025", radiance_2023, radiance_2025),
    ]

    for label, earlier, later in pairs:
        logger.info("Computing change: %s", label)
        change = compute_change(earlier, later)
        classified = classify_change(change, threshold)
        stats = compute_change_stats(change, threshold, period=label)

        results[label] = {
            "change_map": change,
            "classified": classified,
            "stats": stats,
        }

        logger.info("  %s:", label)
        logger.info("    Brightening: %d px (%.1f%%)", stats.brightening_pixels, stats.brightening_pct)
        logger.info("    Stable:      %d px (%.1f%%)", stats.stable_pixels, stats.stable_pct)
        logger.info("    Darkening:   %d px (%.1f%%)", stats.darkening_pixels, stats.darkening_pct)
        logger.info("    Mean change: %+.3f nW/cm²/sr", stats.mean_change)

    # Trend classification: consistent brightening across both periods
    change_23_24 = results["2023_to_2024"]["change_map"]
    change_24_25 = results["2024_to_2025"]["change_map"]

    consistent_brightening = (change_23_24 > threshold) & (change_24_25 > threshold)
    consistent_darkening = (change_23_24 < -threshold) & (change_24_25 < -threshold)

    n_consistent_bright = int(consistent_brightening.sum())
    n_consistent_dark = int(consistent_darkening.sum())

    logger.info("Trend analysis (consistent across both periods):")
    logger.info("  Consistently brightening: %d pixels", n_consistent_bright)
    logger.info("  Consistently darkening:   %d pixels", n_consistent_dark)

    results["trend"] = {
        "consistent_brightening": consistent_brightening,
        "consistent_darkening": consistent_darkening,
        "n_consistent_brightening": n_consistent_bright,
        "n_consistent_darkening": n_consistent_dark,
    }

    return results
