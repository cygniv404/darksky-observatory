"""VIIRS VNP46A4 quality flag analysis.

Flags 1 (poor) and 255 (fill) removed at ingestion. Remaining pixels are
good (flag 0) or gap-filled (flag 2).
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class QualityStats:
    """Quality flag distribution summary."""

    total_pixels: int
    good_pixels: int
    gap_filled_pixels: int
    poor_pixels: int
    fill_pixels: int
    nan_pixels: int

    @property
    def good_pct(self) -> float:
        return 100 * self.good_pixels / max(self.total_pixels, 1)

    @property
    def gap_filled_pct(self) -> float:
        return 100 * self.gap_filled_pixels / max(self.total_pixels, 1)

    @property
    def usable_pct(self) -> float:
        """Percentage of pixels that are either good or gap-filled (usable for analysis)."""
        return 100 * (self.good_pixels + self.gap_filled_pixels) / max(self.total_pixels, 1)

    def to_dict(self) -> dict:
        return {
            "total_pixels": self.total_pixels,
            "good_pixels": self.good_pixels,
            "good_pct": round(self.good_pct, 2),
            "gap_filled_pixels": self.gap_filled_pixels,
            "gap_filled_pct": round(self.gap_filled_pct, 2),
            "poor_pixels": self.poor_pixels,
            "fill_pixels": self.fill_pixels,
            "nan_pixels": self.nan_pixels,
            "usable_pct": round(self.usable_pct, 2),
        }


def analyze_quality(radiance: xr.DataArray) -> QualityStats:
    """Analyze the quality characteristics of a radiance DataArray.

    Since blackmarblepy already removes flags 1 and 255, we infer quality from
    the remaining pixel values:
        - Valid pixels (non-NaN, >= 0): good or gap-filled (indistinguishable post-filtering)
        - NaN pixels: were removed by quality filtering (poor/fill) or are outside coverage
        - Negative pixels: calibration artifacts (shouldn't exist after clean_radiance)

    For a more detailed breakdown, raw HDF5 files with QA bands would be needed.

    Args:
        radiance: DataArray (may contain NaN for filtered/missing pixels).

    Returns:
        QualityStats with the distribution breakdown.
    """
    values = radiance.values
    total = values.size
    nan_count = int(np.isnan(values).sum())
    valid = values[~np.isnan(values)]
    negative_count = int(np.sum(valid < 0))
    usable_count = int(np.sum(valid >= 0))

    stats = QualityStats(
        total_pixels=total,
        good_pixels=usable_count,
        gap_filled_pixels=0,
        poor_pixels=0,
        fill_pixels=nan_count - negative_count,
        nan_pixels=nan_count,
    )

    logger.info("Quality analysis:")
    logger.info("  Total pixels in grid: %d", total)
    logger.info("  Usable (valid, non-negative): %d (%.1f%%)", usable_count, stats.good_pct)
    logger.info("  Removed by QA filtering: %d (%.1f%%)", nan_count, 100 * nan_count / max(total, 1))
    if negative_count > 0:
        logger.info("  Negative/artifact pixels: %d", negative_count)

    return stats


def quality_report(radiance: xr.DataArray, year: str = "") -> str:
    """Generate a human-readable quality report for inclusion in logs or notebooks.

    Args:
        radiance: Input radiance DataArray.
        year: Optional year label for the report header.

    Returns:
        Formatted multi-line string report.
    """
    stats = analyze_quality(radiance)
    header = f"Quality Report{f' ({year})' if year else ''}"

    report = f"""
{'=' * 50}
{header}
{'=' * 50}

Grid dimensions: {radiance.shape[0]} × {radiance.shape[1]} = {stats.total_pixels:,} total pixels

Pixel Classification:
  ┌─────────────────────┬───────────┬─────────┐
  │ Category            │ Count     │ Percent │
  ├─────────────────────┼───────────┼─────────┤
  │ Usable (good + gap) │ {stats.good_pixels:>9,} │ {stats.good_pct:>5.1f}%  │
  │ Removed (QA filter) │ {stats.nan_pixels:>9,} │ {100*stats.nan_pixels/max(stats.total_pixels,1):>5.1f}%  │
  └─────────────────────┴───────────┴─────────┘

QA Flags Applied:
  • Flag 1 (poor quality, ≤3 obs): REMOVED at ingestion
  • Flag 255 (fill/no data):        REMOVED at ingestion
  • Flag 0 (good, >3 obs):          RETAINED
  • Flag 2 (gap-filled):            RETAINED

Note: After blackmarblepy filtering, good and gap-filled pixels are
indistinguishable. Temporal stability analysis (std dev across years)
provides an alternative confidence metric.
{'=' * 50}
"""
    return report
