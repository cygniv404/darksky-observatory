"""Automated validation against ground-truth (published SQM, TESS, darkskysites.com, expert spots)."""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from shapely import wkt

logger = logging.getLogger(__name__)

VALIDATION_DIR = Path(__file__).parent.parent.parent.parent.parent / "validation" / "portugal" / "portugal"


def _load_published_sites():
    """Load 7 measured SQM sites from published papers."""
    csv_path = VALIDATION_DIR / "stargazing_sites.csv"
    if not csv_path.exists():
        return None
    sites = pd.read_csv(csv_path)
    measured = sites[sites["sqm_type"] == "measured"].copy()
    measured["geometry"] = measured["WKT"].apply(wkt.loads)
    return gpd.GeoDataFrame(measured, geometry="geometry", crs="EPSG:4326")


def _load_tess_data():
    """Load TESS photometer station data from tess_station_summary.json."""
    tess_path = VALIDATION_DIR / "tess-ida" / "tess_station_summary.json"
    if not tess_path.exists():
        return None

    with open(tess_path) as f:
        data = json.load(f)

    stations = []
    for name, info in data["stations"].items():
        stations.append((
            name,
            info["lat"],
            info["lon"],
            info["p95_sqm"],
            info["max_sqm"],
        ))
    return stations


def _load_darkskysites_grid():
    """Load darkskysites.com reference grid."""
    grid_path = VALIDATION_DIR / "darkskysites" / "portugal_grid_sqm_2012_2025.json"
    if not grid_path.exists():
        return None
    with open(grid_path) as f:
        data = json.load(f)
    return data["data"]


def _load_expert_spots():
    """Load expert-recommended stargazing spots."""
    expert_path = VALIDATION_DIR / "expert_recommended_spots.csv"
    if not expert_path.exists():
        return None
    sites = pd.read_csv(expert_path)
    sites["geometry"] = sites["WKT"].apply(wkt.loads)
    return gpd.GeoDataFrame(sites, geometry="geometry", crs="EPSG:4326")


def _sample_sqm_map(sqm_map, x_coords, y_coords, lat, lon):
    """Sample predicted SQM at a geographic coordinate."""
    col = int(np.argmin(np.abs(x_coords - lon)))
    row = int(np.argmin(np.abs(y_coords - lat)))
    if 0 <= row < sqm_map.shape[0] and 0 <= col < sqm_map.shape[1]:
        return float(sqm_map[row, col])
    return np.nan


def run_validation(
    sqm_map: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    spots: gpd.GeoDataFrame | None = None,
) -> dict:
    """Run full validation against all available data sources.

    Args:
        sqm_map: 2D predicted SQM array from PSF convolution.
        x_coords: Longitude coordinates of the SQM grid.
        y_coords: Latitude coordinates of the SQM grid.
        spots: Optional pipeline output spots for detection rate check.

    Returns:
        Dict with validation metrics per source and overall summary.
    """
    report = {
        "sources": {},
        "overall": {},
        "flagged_spots": [],
    }

    # --- 1. Published SQM sites (primary ground truth) ---
    published = _load_published_sites()
    if published is not None and len(published) > 0:
        preds, expected = [], []
        for _, row in published.iterrows():
            pred = _sample_sqm_map(sqm_map, x_coords, y_coords,
                                   row.geometry.y, row.geometry.x)
            if not np.isnan(pred):
                preds.append(pred)
                expected.append(row["expected_sqm"])

        if preds:
            errors = np.array(preds) - np.array(expected)
            report["sources"]["published_sqm"] = {
                "n_sites": len(preds),
                "mae": round(float(np.mean(np.abs(errors))), 3),
                "bias": round(float(np.mean(errors)), 3),
                "max_error": round(float(np.max(np.abs(errors))), 3),
                "within_0.5": int((np.abs(errors) <= 0.5).sum()),
                "description": "7 sites with peer-reviewed SQM measurements",
            }
            logger.info("Published sites: MAE=%.3f, bias=%+.3f (%d sites)",
                        report["sources"]["published_sqm"]["mae"],
                        report["sources"]["published_sqm"]["bias"], len(preds))

    # --- 2. TESS photometer stations (20 stations, 6.4M readings) ---
    tess = _load_tess_data()
    if tess:
        preds, measured_p95, measured_max = [], [], []
        for name, lat, lon, p95_sqm, max_sqm in tess:
            pred = _sample_sqm_map(sqm_map, x_coords, y_coords, lat, lon)
            if not np.isnan(pred):
                preds.append(pred)
                measured_p95.append(p95_sqm)
                measured_max.append(max_sqm)

        if preds:
            errors_p95 = np.array(preds) - np.array(measured_p95)
            errors_max = np.array(preds) - np.array(measured_max)
            corr_p95 = float(np.corrcoef(preds, measured_p95)[0, 1]) if len(preds) > 2 else 0.0
            report["sources"]["tess_stations"] = {
                "n_stations": len(preds),
                "total_readings": 6409730,
                "mae_vs_p95": round(float(np.mean(np.abs(errors_p95))), 3),
                "bias_vs_p95": round(float(np.mean(errors_p95)), 3),
                "correlation_vs_p95": round(corr_p95, 4),
                "mae_vs_max": round(float(np.mean(np.abs(errors_max))), 3),
                "bias_vs_max": round(float(np.mean(errors_max)), 3),
                "description": "TESS Stars4All TESS-IDA network (20 stations, 6.4M readings, 2019-2025, vs P95)",
            }
            logger.info("TESS stations: MAE=%.3f vs P95, r=%.3f (%d stations, 6.4M readings)",
                        report["sources"]["tess_stations"]["mae_vs_p95"],
                        corr_p95, len(preds))

    # --- 3. darkskysites.com reference grid ---
    dss_grid = _load_darkskysites_grid()
    if dss_grid:
        preds, refs = [], []
        for pt in dss_grid:
            annual = {r["year"]: r["sqm"] for r in pt["annual"]}
            ref_sqm = annual.get(2024, annual.get(2023, None))
            if ref_sqm is None:
                continue
            pred = _sample_sqm_map(sqm_map, x_coords, y_coords, pt["lat"], pt["lon"])
            if not np.isnan(pred):
                preds.append(pred)
                refs.append(ref_sqm)

        if len(preds) > 10:
            errors = np.array(preds) - np.array(refs)
            corr, _ = pearsonr(preds, refs)
            report["sources"]["darkskysites_grid"] = {
                "n_points": len(preds),
                "mae": round(float(np.mean(np.abs(errors))), 3),
                "bias": round(float(np.mean(errors)), 3),
                "correlation": round(float(corr), 4),
                "within_0.5": int((np.abs(errors) <= 0.5).sum()),
                "within_1.0": int((np.abs(errors) <= 1.0).sum()),
                "pct_within_0.5": round(100 * (np.abs(errors) <= 0.5).sum() / len(errors), 1),
                "description": "darkskysites.com VIIRS-based reference grid (0.1 deg spacing)",
            }
            logger.info("darkskysites grid: MAE=%.3f, r=%.3f (%d points)",
                        report["sources"]["darkskysites_grid"]["mae"],
                        corr, len(preds))

    # --- 4. Expert spots detection rate ---
    expert = _load_expert_spots()
    if expert is not None and spots is not None and len(spots) > 0:
        from scipy.spatial import cKDTree

        spots_utm = spots.to_crs(epsg=32629)
        expert_utm = expert.to_crs(epsg=32629)

        spot_coords = np.column_stack([spots_utm.geometry.x, spots_utm.geometry.y])
        tree = cKDTree(spot_coords)

        expert_coords = np.column_stack([expert_utm.geometry.x, expert_utm.geometry.y])
        distances, _ = tree.query(expert_coords)
        distances_km = distances / 1000

        report["sources"]["expert_detection"] = {
            "n_expert_spots": len(expert),
            "detected_5km": int((distances_km < 5).sum()),
            "detected_10km": int((distances_km < 10).sum()),
            "pct_detected_10km": round(100 * (distances_km < 10).sum() / len(expert), 1),
            "mean_distance_km": round(float(distances_km.mean()), 1),
            "description": "Expert-recommended spots (Dark Sky Alqueva, astronomy clubs, tourism)",
        }
        logger.info("Expert detection: %d/%d within 10km (%.0f%%)",
                    report["sources"]["expert_detection"]["detected_10km"],
                    len(expert),
                    report["sources"]["expert_detection"]["pct_detected_10km"])

    # --- 5. Flag suspicious output spots ---
    if spots is not None and dss_grid and "predicted_sqm" in spots.columns:
        dss_lookup = {}
        for pt in dss_grid:
            annual = {r["year"]: r["sqm"] for r in pt["annual"]}
            ref = annual.get(2024, annual.get(2023))
            if ref:
                key = (round(pt["lat"], 1), round(pt["lon"], 1))
                dss_lookup[key] = ref

        flagged = []
        for _, spot in spots.iterrows():
            key = (round(spot.geometry.y, 1), round(spot.geometry.x, 1))
            ref = dss_lookup.get(key)
            if ref and abs(spot["predicted_sqm"] - ref) > 1.0:
                flagged.append({
                    "name": spot.get("name", "?"),
                    "predicted_sqm": round(float(spot["predicted_sqm"]), 2),
                    "reference_sqm": ref,
                    "divergence": round(float(spot["predicted_sqm"] - ref), 2),
                })

        report["flagged_spots"] = flagged
        if flagged:
            logger.warning("%d spots diverge >1 mag from darkskysites reference", len(flagged))

    # --- Overall summary ---
    all_maes = []
    for src in report["sources"].values():
        if "mae" in src:
            all_maes.append(src["mae"])
        elif "mae_vs_p95" in src:
            all_maes.append(src["mae_vs_p95"])
        elif "mae_vs_max" in src:
            all_maes.append(src["mae_vs_max"])

    report["overall"] = {
        "validation_sources": len(report["sources"]),
        "mean_mae_across_sources": round(float(np.mean(all_maes)), 3) if all_maes else None,
        "total_validation_points": sum(
            src.get("n_sites", src.get("n_stations", src.get("n_points", 0)))
            for src in report["sources"].values()
        ),
        "model_interpretation": "clear-sky potential predictor (best-night estimate)",
    }

    return report


def save_validation_report(report: dict, output_path: Path):
    """Save validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved: %s", output_path)
