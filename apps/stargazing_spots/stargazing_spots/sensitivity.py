"""Threshold sensitivity analysis and validation-calibrated threshold selection.

Evaluates classification at multiple thresholds via Otsu, elbow analysis,
and ground-truth detection rate.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

PIXEL_AREA_KM2 = 0.25  # Each 500m × 500m pixel ≈ 0.25 km²

BORTLE_ANNOTATIONS = {
    0.5: "Bortle 1 (Excellent dark site)",
    1.0: "Bortle 2 (Typical dark site)",
    2.0: "Bortle 3 (Rural sky)",
    3.0: "Bortle 3-4 (Rural/suburban transition)",
    5.0: "Bortle 4-5 (Suburban sky)",
    10.0: "Bortle 5-6 (Bright suburban)",
}

DEFAULT_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]


@dataclass
class ThresholdResult:
    """Result for a single threshold value."""

    threshold: float
    dark_pixels: int
    dark_area_km2: float
    dark_pct: float
    bortle_class: str


def run_sensitivity(
    radiance: xr.DataArray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Run classification at multiple thresholds and compare results.

    For each threshold, computes:
        - Number of dark pixels
        - Approximate dark-sky area in km²
        - Percentage of total valid pixels
        - Corresponding Bortle class

    Args:
        radiance: Cleaned radiance DataArray (should have negatives/fill already removed).
        thresholds: List of thresholds to evaluate. Defaults to 0.5–10.0 range.

    Returns:
        DataFrame with one row per threshold, columns:
            threshold, dark_pixels, dark_area_km2, dark_pct, bortle_class, marginal_gain_pct
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    values = radiance.values
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    total_valid = len(valid_values)

    logger.info("Sensitivity analysis: %d thresholds, %d valid pixels", len(thresholds), total_valid)

    results = []
    prev_dark = 0

    for t in sorted(thresholds):
        dark_count = int(np.sum(valid_values < t))
        dark_area = dark_count * PIXEL_AREA_KM2
        dark_pct = 100 * dark_count / max(total_valid, 1)

        # Find nearest Bortle annotation
        bortle = ""
        for bortle_thresh, label in sorted(BORTLE_ANNOTATIONS.items()):
            if t <= bortle_thresh:
                bortle = label
                break
        if not bortle:
            bortle = "Bortle 6+ (Urban)"

        # Marginal gain: how many MORE pixels vs previous threshold
        marginal = dark_count - prev_dark
        marginal_pct = 100 * marginal / max(total_valid, 1)
        prev_dark = dark_count

        results.append({
            "threshold_nw": t,
            "dark_pixels": dark_count,
            "dark_area_km2": round(dark_area, 1),
            "dark_pct": round(dark_pct, 2),
            "bortle_class": bortle,
            "marginal_gain_pixels": marginal,
            "marginal_gain_pct": round(marginal_pct, 2),
        })

    df = pd.DataFrame(results)

    # Log summary
    logger.info("Sensitivity results:")
    logger.info("  %-8s %-12s %-12s %-10s %s", "Thresh", "Dark px", "Area (km²)", "% total", "Bortle")
    for _, row in df.iterrows():
        marker = " ◄" if row["threshold_nw"] == 3.0 else ""
        logger.info(
            "  %-8.1f %-12d %-12.0f %-10.1f %s%s",
            row["threshold_nw"],
            row["dark_pixels"],
            row["dark_area_km2"],
            row["dark_pct"],
            row["bortle_class"],
            marker,
        )

    return df


def find_optimal_threshold(sensitivity_df: pd.DataFrame) -> float:
    """Identify the threshold at the elbow point (maximum marginal gain drop-off).

    The "elbow" is where increasing the threshold stops giving meaningful
    additional dark-sky area — diminishing returns beyond this point.

    Args:
        sensitivity_df: Output of run_sensitivity().

    Returns:
        Threshold value at the elbow point.
    """
    gains = sensitivity_df["marginal_gain_pct"].values
    if len(gains) < 3:
        return 3.0

    # Find where the second derivative is most negative (biggest deceleration)
    second_derivative = np.diff(gains, n=2)
    if len(second_derivative) == 0:
        return 3.0

    elbow_idx = int(np.argmin(second_derivative)) + 1
    optimal = float(sensitivity_df.iloc[elbow_idx]["threshold_nw"])
    logger.info("Elbow analysis suggests optimal threshold: %.1f nW/cm²/sr", optimal)
    return optimal


def otsu_threshold(radiance: xr.DataArray, n_bins: int = 256, max_val: float = 20.0) -> float:
    """Derive dark-sky threshold using Otsu's method (automatic bimodal split).

    Otsu's method finds the threshold that minimizes the intra-class variance
    of a bimodal distribution. Applied to nighttime radiance, it separates the
    "dark" population (natural background) from the "bright" population
    (artificial light).

    This is the same algorithm used in image segmentation — here applied to
    satellite radiometry. Used in Liu et al. (2019) for VIIRS urban extraction.

    Args:
        radiance: Cleaned radiance DataArray.
        n_bins: Number of histogram bins for the computation.
        max_val: Cap radiance at this value to focus on the dark/bright boundary.
            Extreme urban values (>20 nW) distort the histogram.

    Returns:
        Optimal threshold in nW/cm²/sr.
    """
    values = radiance.values[~np.isnan(radiance.values)]
    values = values[values <= max_val]

    hist, bin_edges = np.histogram(values, bins=n_bins, range=(0, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 3.0

    # Otsu's algorithm: find threshold that maximizes between-class variance
    best_threshold = 0.0
    best_variance = 0.0

    cumsum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centers)
    global_mean = cum_mean[-1] / total

    for i in range(1, n_bins):
        w0 = cumsum[i] / total
        w1 = 1.0 - w0

        if w0 == 0 or w1 == 0:
            continue

        mean0 = cum_mean[i] / cumsum[i]
        mean1 = (cum_mean[-1] - cum_mean[i]) / (total - cumsum[i])

        between_variance = w0 * w1 * (mean0 - mean1) ** 2

        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = bin_centers[i]

    logger.info(
        "Otsu's method: optimal threshold = %.2f nW/cm²/sr "
        "(maximizes between-class variance of dark/bright populations)",
        best_threshold,
    )
    return best_threshold


def load_all_validation_sites(validation_dir: "Path | str") -> "gpd.GeoDataFrame":
    """Load validation sites from the consolidated stargazing_sites.csv file.

    The file contains columns: WKT, name, region, expected_sqm, sqm_source,
    sqm_type, description.

    sqm_type values:
        - "measured": Real ground-truth from papers/IDA (primary validation)
        - "certification_threshold": Starlight Foundation >= 21.0 (secondary)
        - "estimated": From Falchi atlas, not independent

    Args:
        validation_dir: Path to the validation/ directory containing stargazing_sites.csv.

    Returns:
        GeoDataFrame with all validation sites and metadata columns.
    """
    import geopandas as gpd
    from pathlib import Path
    from shapely import wkt as _wkt

    validation_dir = Path(validation_dir)
    csv_file = validation_dir / "stargazing_sites.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Validation file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if "WKT" not in df.columns:
        raise ValueError(f"Expected 'WKT' column in {csv_file}")

    df["geometry"] = df["WKT"].apply(_wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    logger.info(
        "Loaded %d validation sites: %d measured, %d certification_threshold, %d estimated",
        len(gdf),
        (gdf["sqm_type"] == "measured").sum(),
        (gdf["sqm_type"] == "certification_threshold").sum(),
        (gdf["sqm_type"] == "estimated").sum(),
    )
    return gdf


def validation_calibrated_threshold(
    radiance: xr.DataArray,
    certified_sites: "gpd.GeoDataFrame",
    thresholds: list[float] | None = None,
    buffer_m: float = 750,
) -> tuple[float, pd.DataFrame]:
    """Find the threshold that maximizes detection of known dark-sky sites.

    This is a validation-calibrated approach: we use ground-truth certified
    dark-sky locations to find which threshold best captures them while
    minimizing over-classification.

    Analogous to ROC analysis — for each threshold, compute:
        - True positive rate: % of certified sites correctly classified as dark
        - Specificity proxy: % of total area classified as dark (lower = more selective)

    The optimal threshold maximizes site detection while remaining selective.

    Sites that fall outside the raster coverage (NaN pixels) are automatically
    excluded from the analysis.

    Args:
        radiance: Cleaned radiance DataArray.
        certified_sites: GeoDataFrame with Point geometry of known dark-sky locations.
        thresholds: Thresholds to evaluate. Defaults to fine-grained 0.5–10.0 range.
        buffer_m: Buffer in meters around each site for pixel sampling.

    Returns:
        Tuple of (optimal_threshold, results_dataframe).
    """
    import geopandas as gpd

    if thresholds is None:
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]

    y_coords = radiance.coords["y"].values
    x_coords = radiance.coords["x"].values
    values = radiance.values

    # Check raster bounds to filter out-of-coverage sites
    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    x_min, x_max = float(x_coords.min()), float(x_coords.max())

    # Sample radiance at each site location
    site_radiances = []
    site_names = []
    for _, row in certified_sites.iterrows():
        lon, lat = row.geometry.x, row.geometry.y

        # Skip sites outside raster extent
        if lat < y_min or lat > y_max or lon < x_min or lon > x_max:
            continue

        y_idx = int(np.argmin(np.abs(y_coords - lat)))
        x_idx = int(np.argmin(np.abs(x_coords - lon)))
        pixel_val = values[y_idx, x_idx]
        site_radiances.append(pixel_val)
        site_names.append(row.get("name", "Unknown"))

    site_radiances = np.array(site_radiances)
    valid_sites = ~np.isnan(site_radiances)
    n_valid = int(valid_sites.sum())
    n_total = len(site_radiances)
    n_skipped = len(certified_sites) - n_total

    logger.info("Validation-calibrated threshold analysis:")
    logger.info("  Input sites: %d, within raster coverage: %d, with valid data: %d",
                len(certified_sites), n_total, n_valid)
    if n_skipped > 0:
        logger.info("  Skipped %d sites outside raster coverage (islands/other regions)", n_skipped)
    logger.info("  Site radiance range: %.2f – %.2f nW/cm²/sr",
                np.nanmin(site_radiances), np.nanmax(site_radiances))
    logger.info("  Site radiance mean: %.2f, median: %.2f",
                np.nanmean(site_radiances), np.nanmedian(site_radiances))

    # For each threshold, compute detection rate
    valid_values = values[~np.isnan(values)]
    total_pixels = len(valid_values)

    results = []
    for t in thresholds:
        detected = int(np.sum(site_radiances[valid_sites] < t))
        detection_rate = 100 * detected / max(n_valid, 1)
        dark_pct = 100 * np.sum(valid_values < t) / total_pixels

        # F-score balancing detection rate and selectivity
        selectivity = 100 - dark_pct  # higher = more selective
        if detection_rate + selectivity > 0:
            f_score = 2 * (detection_rate * selectivity) / (detection_rate + selectivity)
        else:
            f_score = 0

        results.append({
            "threshold_nw": t,
            "sites_detected": detected,
            "detection_rate_pct": round(detection_rate, 1),
            "dark_area_pct": round(dark_pct, 1),
            "selectivity_pct": round(selectivity, 1),
            "f_score": round(f_score, 2),
        })

    results_df = pd.DataFrame(results)

    # Optimal = lowest threshold that achieves >= 75% site detection
    # (We want the most selective threshold that still captures most known sites)
    high_detection = results_df[results_df["detection_rate_pct"] >= 75]
    if len(high_detection) > 0:
        best_idx = high_detection["threshold_nw"].idxmin()
    else:
        best_idx = results_df["detection_rate_pct"].idxmax()
    optimal = float(results_df.loc[best_idx, "threshold_nw"])

    logger.info("  Validation results:")
    for _, row in results_df.iterrows():
        marker = " ◄ optimal" if row["threshold_nw"] == optimal else ""
        logger.info(
            "    %.1f nW: %d/%d sites detected (%.0f%%), area=%.0f%%, F=%.1f%s",
            row["threshold_nw"], row["sites_detected"], n_valid,
            row["detection_rate_pct"], row["dark_area_pct"], row["f_score"], marker,
        )

    logger.info("  Validation-calibrated optimal threshold: %.1f nW/cm²/sr", optimal)
    return optimal, results_df


def full_threshold_justification(
    radiance: xr.DataArray,
    certified_sites: "gpd.GeoDataFrame",
) -> dict:
    """Run all three threshold derivation methods and compare results.

    Produces a complete justification for the chosen threshold by combining:
        1. Otsu's method (data-driven, automatic)
        2. Validation-calibrated (ground-truth optimized, using all available sites)
        3. Elbow analysis (diminishing returns)

    If all three converge, the threshold is strongly justified.
    If they diverge, the analysis reveals interesting trade-offs to discuss.

    Args:
        radiance: Cleaned radiance DataArray.
        certified_sites: GeoDataFrame of known dark-sky locations
            (can be combined output of load_all_validation_sites).

    Returns:
        Dictionary with all three derived thresholds and analysis details.
    """
    logger.info("=" * 50)
    logger.info("FULL THRESHOLD JUSTIFICATION")
    logger.info("=" * 50)

    # Method 1: Otsu
    otsu = otsu_threshold(radiance)

    # Method 2: Validation-calibrated
    val_optimal, val_df = validation_calibrated_threshold(radiance, certified_sites)

    # Method 3: Elbow from sensitivity
    sens_df = run_sensitivity(radiance)
    elbow = find_optimal_threshold(sens_df)

    logger.info("")
    logger.info("THRESHOLD CONVERGENCE:")
    logger.info("  Otsu (data-driven):         %.2f nW/cm²/sr", otsu)
    logger.info("  Validation (ground-truth):   %.1f nW/cm²/sr", val_optimal)
    logger.info("  Elbow (diminishing returns): %.1f nW/cm²/sr", elbow)
    logger.info("  Literature (Falchi 2016):    3.0 nW/cm²/sr")
    logger.info("")

    mean_threshold = np.mean([otsu, val_optimal, elbow, 3.0])
    logger.info("  Mean of all methods: %.2f nW/cm²/sr", mean_threshold)

    return {
        "otsu_threshold": otsu,
        "validation_threshold": val_optimal,
        "elbow_threshold": elbow,
        "literature_threshold": 3.0,
        "mean_threshold": round(mean_threshold, 2),
        "sensitivity_df": sens_df,
        "validation_df": val_df,
    }
