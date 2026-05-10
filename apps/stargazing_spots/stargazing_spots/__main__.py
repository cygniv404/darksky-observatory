"""Stargazing spots pipeline CLI entry point.

Usage:
    python -m stargazing_spots [--from-cache PATH] [--output-dir PATH] [--threshold FLOAT]

Runs the full pipeline:
    1. Acquire VIIRS radiance (or load from cached parquet)
    2. Classify dark-sky pixels
    3. Enrich with OSM features
    4. Export outputs (GeoJSON, COG, STAC)
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from app root (apps/stargazing_spots/.env)
load_dotenv(Path(__file__).parent.parent / ".env")

from datetime import UTC

import geopandas as gpd
import pandas as pd
import xarray as xr

from stargazing_spots.assess import filter_unsuitable, run_full_assessment
from stargazing_spots.change_detection import run_change_detection
from stargazing_spots.enrichment import (
    fetch_osm_features,
    filter_relevant_features,
    score_spots,
)
from stargazing_spots.enrichment_raster import raster_sample_with_context
from stargazing_spots.export import to_cog, to_geojson, to_parquet
from stargazing_spots.processing import (
    classify_dark_sky,
    classify_temporal,
    extract_dark_coordinates,
)
from stargazing_spots.quality import analyze_quality
from stargazing_spots.sensitivity import find_optimal_threshold, run_sensitivity
from stargazing_spots.stac_catalog import create_darksky_catalog
from stargazing_spots.uncertainty import compute_confidence_layer, enrich_spots_with_confidence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_cached_radiance(cache_path: Path) -> xr.DataArray:
    """Load radiance data from either a joblib cache or parquet file.

    Supports:
        - .joblib files (output of blackmarblepy bm_raster, xarray Dataset)
        - .parquet files (pre-processed multi-indexed DataFrame)
    """
    import joblib as jl

    cache_path = Path(cache_path)
    logger.info("Loading cached radiance from %s", cache_path)

    if cache_path.suffix == ".joblib":
        ds = jl.load(cache_path)
        from stargazing_spots.processing import clean_radiance
        layer = ds["NearNadir_Composite_Snow_Free"].squeeze("time", drop=True)
        return clean_radiance(layer)

    return _load_from_parquet(cache_path)


def _load_from_parquet(parquet_path: Path) -> xr.DataArray:
    """Load pre-processed light pollution data from parquet and reconstruct as DataArray.

    The parquet is a multi-indexed DataFrame (y, x) with a radiance column.
    We pivot it into a 2D grid suitable for xarray processing.
    """
    import numpy as np

    logger.info("Loading from parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    # Identify the radiance column
    if "portugal_2023" in df.columns:
        value_col = "portugal_2023"
    elif "portugal_2022" in df.columns:
        value_col = "portugal_2022"
    else:
        value_col = [c for c in df.columns if c not in ("time",)][0]

    # Reset index if y/x are in the index
    if "y" in df.index.names:
        df = df.reset_index()

    logger.info("DataFrame: %d rows, value_col='%s'", len(df), value_col)

    # Get unique sorted coordinates
    y_vals = np.sort(df["y"].unique())[::-1]  # descending (north to south)
    x_vals = np.sort(df["x"].unique())

    logger.info("Grid dimensions: %d y × %d x", len(y_vals), len(x_vals))

    # Build index mappings for fast placement
    y_idx = {v: i for i, v in enumerate(y_vals)}
    x_idx = {v: i for i, v in enumerate(x_vals)}

    # Create empty grid and fill
    grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float32)
    rows = df["y"].map(y_idx).values
    cols = df["x"].map(x_idx).values
    values = df[value_col].values.astype(np.float32)
    grid[rows, cols] = values

    data_array = xr.DataArray(
        grid,
        dims=["y", "x"],
        coords={"y": y_vals, "x": x_vals},
        attrs={"units": "nW/cm²/sr", "long_name": "Nighttime Radiance", "crs": "EPSG:4326"},
    )
    non_null = int(data_array.count())
    logger.info("Reconstructed DataArray: shape=%s, non-null pixels=%d", data_array.shape, non_null)
    return data_array


class _ArgsRef:
    """Simple container to pass optional flags into run_pipeline."""
    run_sensitivity: bool = False


def run_pipeline(
    radiance: xr.DataArray,
    threshold: float = 3.0,
    place_name: str = "Portugal",
    buffer_m: float = 500,
    output_dir: Path = Path("output/portugal"),
    skip_osm: bool = False,
    preloaded_features: gpd.GeoDataFrame | None = None,
    temporal_layers: list[xr.DataArray] | None = None,
    temporal_labels: list[str] | None = None,
    args_ref: _ArgsRef | None = None,
    region: str = "mainland",
) -> None:
    """Execute the full pipeline."""
    import json
    import time
    from datetime import datetime

    run_start = time.perf_counter()
    timings = {}
    metrics = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous run outputs (spots are regenerated fresh each run)
    for old_file in output_dir.glob("spots.*"):
        old_file.unlink()
        logger.info("Cleaned previous output: %s", old_file.name)

    # Step 0: Quality assessment
    logger.info("=" * 60)
    logger.info("STEP 0: Data quality assessment")
    logger.info("=" * 60)
    qa_stats = analyze_quality(radiance)
    logger.info("  Usable pixels: %d (%.1f%%)", qa_stats.good_pixels, qa_stats.good_pct)

    if args_ref is None:
        args_ref = _ArgsRef()

    # Optional: sensitivity analysis
    if args_ref.run_sensitivity:
        logger.info("=" * 60)
        logger.info("STEP 0b: Threshold sensitivity analysis")
        logger.info("=" * 60)
        from stargazing_spots.processing import clean_radiance as _clean
        sens_df = run_sensitivity(_clean(radiance))
        sens_path = output_dir / "sensitivity_analysis.csv"
        sens_df.to_csv(sens_path, index=False)
        logger.info("Sensitivity results saved to %s", sens_path)
        optimal = find_optimal_threshold(sens_df)
        logger.info("Elbow-suggested threshold: %.1f (using configured: %.1f)", optimal, threshold)

    # Step 1: Classification
    t1 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("STEP 1: Dark-sky classification (threshold=%.1f nW/cm²/sr)", threshold)
    logger.info("=" * 60)

    temporal_result = None
    if temporal_layers and len(temporal_layers) >= 2:
        labels = temporal_labels or [str(i) for i in range(len(temporal_layers))]
        logger.info("Using TEMPORAL composite (%d years: %s)", len(temporal_layers), ", ".join(labels))
        temporal_result = classify_temporal(temporal_layers, threshold=threshold, year_labels=labels)
        dark_sky_full = temporal_result["dark_sky"]
        # Fill NaN extent for COG export
        dark_sky_for_cog = temporal_result["median"].where(temporal_result["median"] < threshold)
    else:
        logger.info("Using SINGLE-YEAR classification")
        dark_sky_full = classify_dark_sky(radiance, threshold=threshold, drop_empty=False)
        dark_sky_for_cog = dark_sky_full

    # Drop empty rows/cols for efficient coordinate extraction
    dark_sky_compact = dark_sky_full.dropna(dim="y", how="all").dropna(dim="x", how="all")
    dark_coords = extract_dark_coordinates(dark_sky_compact)
    logger.info("Result: %d dark-sky pixel coordinates extracted", len(dark_coords))

    timings["step1_classification_s"] = round(time.perf_counter() - t1, 3)
    metrics["dark_pixels"] = len(dark_coords)
    metrics["total_valid_pixels"] = int(dark_sky_for_cog.count()) if not temporal_result else temporal_result["stats"]["total_valid_pixels"]
    metrics["dark_pct"] = round(100 * metrics["dark_pixels"] / max(metrics["total_valid_pixels"], 1), 2)

    # Step 2: Export COGs
    t2 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("STEP 2: Export Cloud Optimized GeoTIFF(s)")
    logger.info("=" * 60)
    cog_path = output_dir / "portugal_darksky.tif"
    to_cog(dark_sky_for_cog, cog_path)

    if temporal_result:
        to_cog(temporal_result["stability"], output_dir / "temporal_stability.tif")
        to_cog(temporal_result["trend"], output_dir / "temporal_trend.tif")

    # Compute confidence layer
    confidence_layer = None
    if temporal_result:
        confidence_layer = compute_confidence_layer(
            median_radiance=temporal_result["median"],
            stability=temporal_result["stability"],
            consistency=temporal_result["consistency"],
            threshold=threshold,
            n_years=len(temporal_layers) if temporal_layers else 1,
        )
        to_cog(confidence_layer, output_dir / "dark_confidence.tif")
    else:
        from stargazing_spots.processing import clean_radiance as _cr
        confidence_layer = compute_confidence_layer(
            median_radiance=_cr(radiance), threshold=threshold
        )
        to_cog(confidence_layer, output_dir / "dark_confidence.tif")

    timings["step2_cog_export_s"] = round(time.perf_counter() - t2, 3)

    # Step 3: OSM enrichment
    t3 = time.perf_counter()
    if not skip_osm:
        logger.info("=" * 60)
        logger.info("STEP 3: OSM enrichment (place=%s)", place_name)
        logger.info("=" * 60)

        if preloaded_features is not None:
            logger.info("Using pre-loaded features (%d entries)", len(preloaded_features))
            filtered = preloaded_features
        else:
            osm_features = fetch_osm_features(place_name)
            filtered = filter_relevant_features(osm_features)

        # Primary method: raster-native spatial join (fast, scientifically correct)
        logger.info("Using raster-native spatial join (direct pixel sampling)")
        raster_for_join = temporal_result["median"] if temporal_result else dark_sky_for_cog
        spots = raster_sample_with_context(
            dark_sky=raster_for_join,
            features=filtered,
            stability=temporal_result["stability"] if temporal_result else None,
            trend=temporal_result["trend"] if temporal_result else None,
            threshold=threshold,
        )

        named_spots = spots[spots["name"].notna()].copy()
        logger.info("Named spots on dark pixels: %d", len(named_spots))

        scored = score_spots(named_spots)

        # Add confidence scores
        if confidence_layer is not None:
            scored = enrich_spots_with_confidence(scored, confidence_layer)

        # Step 3b: Multi-criteria assessment (PSF + cloud + SVF + land cover + slope → SSI)
        logger.info("--- Multi-criteria stargazing suitability assessment ---")
        full_radiance = temporal_result["median"].values if temporal_result else radiance.values
        x_coords = (temporal_result["median"] if temporal_result else radiance).coords["x"].values
        y_coords = (temporal_result["median"] if temporal_result else radiance).coords["y"].values

        scored = run_full_assessment(scored, full_radiance, x_coords, y_coords, region=region)

        # Filter physically unsuitable spots (forest, water, etc.)
        n_before_filter = len(scored)
        scored = filter_unsuitable(scored)
        n_filtered = n_before_filter - len(scored)
        if n_filtered > 0:
            logger.info("Filtered %d unsuitable spots (forest/water/urban)", n_filtered)

        # Log summary
        if "ssi_class" in scored.columns:
            for cls in ["excellent", "good", "acceptable"]:
                count = (scored["ssi_class"] == cls).sum()
                if count > 0:
                    metrics[f"spots_ssi_{cls}"] = int(count)

        if "predicted_sqm" in scored.columns:
            logger.info("Top 5 spots by SSI:")
            for _, row in scored.nlargest(5, "ssi_score").iterrows():
                logger.info(
                    "  SSI %.1f | SQM %.2f | %s",
                    row.get("ssi_score", 0),
                    row.get("predicted_sqm", 0),
                    row.get("name", "?"),
                )

        # Step 4: Export vector outputs
        logger.info("=" * 60)
        logger.info("STEP 4: Export vector outputs")
        logger.info("=" * 60)
        geojson_path = output_dir / "spots.geojson"
        parquet_path = output_dir / "spots.parquet"
        processing_params = {
            "threshold_nw_cm2_sr": threshold,
            "buffer_m": buffer_m,
            "place_name": place_name,
            "total_dark_pixels": len(dark_coords),
            "spots_count": len(scored),
            "crs": "EPSG:4326",
        }
        to_geojson(scored, geojson_path, processing_params=processing_params)
        to_parquet(scored, parquet_path)
        metrics["spots_count"] = len(scored)
        metrics["input_features"] = len(filtered)
        if "temporal_stability" in scored.columns:
            metrics["spots_high_stability"] = int((scored["temporal_stability"] == "high").sum())
            metrics["spots_brightening"] = int((scored["trend_direction"] == "brightening").sum())
    else:
        logger.info("STEP 3: OSM enrichment SKIPPED (--skip-osm)")
        geojson_path = output_dir / "spots.geojson"
        cog_path = output_dir / "portugal_darksky.tif"

    timings["step3_enrichment_s"] = round(time.perf_counter() - t3, 3)

    # Step 5: Generate STAC catalog
    logger.info("=" * 60)
    logger.info("STEP 5: Generate STAC catalog")
    logger.info("=" * 60)
    stac_dir = output_dir / "stac"
    geojson_for_stac = geojson_path if geojson_path.exists() else None
    cog_for_stac = cog_path if cog_path.exists() else None
    create_darksky_catalog(stac_dir, geojson_path=geojson_for_stac, cog_path=cog_for_stac)

    timings["step5_stac_s"] = round(time.perf_counter() - t3, 3)  # includes step4+5
    total_elapsed = round(time.perf_counter() - run_start, 3)
    timings["total_s"] = total_elapsed

    # Generate run summary
    run_summary = {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "pipeline_version": "0.7.0",
        "parameters": {
            "threshold_nw": threshold,
            "place_name": place_name,
            "temporal_years": temporal_labels if temporal_layers else None,
            "skip_osm": skip_osm,
        },
        "metrics": metrics,
        "timings": timings,
        "output_dir": str(output_dir.resolve()),
    }

    summary_path = output_dir / "run_metadata.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE (%.1fs total)", total_elapsed)
    logger.info("=" * 60)
    logger.info("  Dark pixels:  %d (%.1f%%)", metrics.get("dark_pixels", 0), metrics.get("dark_pct", 0))
    if "spots_count" in metrics:
        logger.info("  Spots found:  %d", metrics["spots_count"])
    logger.info("  Timing: classification=%.1fs, export=%.1fs, enrichment=%.1fs",
                timings.get("step1_classification_s", 0),
                timings.get("step2_cog_export_s", 0),
                timings.get("step3_enrichment_s", 0))
    logger.info("  Summary: %s", summary_path)


def main():
    parser = argparse.ArgumentParser(description="Dark-sky classification pipeline")
    parser.add_argument(
        "--from-cache",
        type=Path,
        help="Path to pre-processed parquet file (skips satellite download)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/portugal"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Radiance threshold in nW/cm²/sr (default: 3.0)",
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip OSM enrichment (only run classification + COG export)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run threshold sensitivity analysis and export results",
    )
    parser.add_argument(
        "--change-detection",
        nargs=3,
        metavar=("CACHE_2023", "CACHE_2024", "CACHE_2025"),
        help="Run multi-temporal change detection with 3 cached parquet/joblib files",
    )
    parser.add_argument(
        "--from-geojson",
        type=Path,
        help="Use pre-computed OSM features GeoJSON instead of downloading fresh",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Process all Portugal regions (mainland + Madeira + Azores) and merge results",
    )
    parser.add_argument(
        "--place",
        type=str,
        default="Portugal",
        help="Place name for OSM feature query",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=500,
        help="Buffer distance in meters for spatial join",
    )

    args = parser.parse_args()

    REGIONS = {
        "mainland": {"suffix": "", "years": ["2023", "2024", "2025"]},
        "madeira": {"suffix": "_madeira", "years": ["2023", "2024", "2025"]},
        "azores": {"suffix": "_azores", "years": ["2023", "2024", "2025"]},
    }

    if args.all_regions:
        _run_all_regions(args, REGIONS)
    else:
        _run_single(args)


def _run_single(args):
    """Original single-region pipeline execution."""
    if args.from_cache:
        radiance = load_cached_radiance(args.from_cache)
    else:
        from stargazing_spots.acquisition import extract_radiance_layer, fetch_viirs_radiance

        boundary_path = Path(__file__).parent.parent / "input" / "portugal" / "boundary" / "gadm41_PRT_1.json.zip"
        if not boundary_path.exists():
            logger.error("Boundary file not found: %s", boundary_path)
            raise SystemExit(1)

        boundary_gdf = gpd.read_file(str(boundary_path))
        dataset = fetch_viirs_radiance(boundary_gdf, date_range="2023-01-01")
        radiance = extract_radiance_layer(dataset, "2023-01-01")

    preloaded = None
    if args.from_geojson:
        logger.info("Loading pre-computed features from %s", args.from_geojson)
        preloaded = gpd.read_file(str(args.from_geojson))
        logger.info("Loaded %d features", len(preloaded))

    temporal_layers = None
    temporal_labels = None
    if args.change_detection:
        temporal_labels = ["2023", "2024", "2025"]
        temporal_layers = [load_cached_radiance(Path(p)) for p in args.change_detection]

    args_ref = _ArgsRef()
    args_ref.run_sensitivity = args.sensitivity

    run_pipeline(
        radiance=radiance,
        threshold=args.threshold,
        place_name=args.place,
        buffer_m=args.buffer,
        output_dir=args.output_dir,
        skip_osm=args.skip_osm,
        preloaded_features=preloaded,
        temporal_layers=temporal_layers,
        temporal_labels=temporal_labels,
        args_ref=args_ref,
    )

    if args.change_detection:
        _run_change_detection(args)


def _run_all_regions(args, regions: dict):
    """Process all Portugal regions and merge results."""
    import json
    import time

    cache_dir = Path(__file__).parent.parent / "input" / "portugal" / "cache"
    all_spots = []
    region_metrics = {}
    total_start = time.perf_counter()

    # Clean previous merged output
    for old_file in args.output_dir.glob("spots_all.*"):
        old_file.unlink()

    preloaded = None
    if args.from_geojson:
        logger.info("Loading pre-computed features from %s", args.from_geojson)
        preloaded = gpd.read_file(str(args.from_geojson))
        logger.info("Loaded %d features", len(preloaded))

    for region_name, region_info in regions.items():
        suffix = region_info["suffix"]
        years = region_info["years"]

        logger.info("")
        logger.info("*" * 60)
        logger.info("REGION: %s", region_name.upper())
        logger.info("*" * 60)

        # Load temporal data for this region
        temporal_files = [cache_dir / f"VNP46A4_{y}{suffix}.joblib" for y in years]
        missing = [f for f in temporal_files if not f.exists()]
        if missing:
            logger.warning("Skipping %s — missing cache files: %s", region_name, missing)
            continue

        temporal_layers = [load_cached_radiance(f) for f in temporal_files]
        radiance = temporal_layers[-1]  # Use latest year as primary

        args_ref = _ArgsRef()
        args_ref.run_sensitivity = False  # Only run sensitivity for first region

        region_output = args.output_dir / region_name
        run_pipeline(
            radiance=radiance,
            threshold=args.threshold,
            place_name=args.place,
            buffer_m=args.buffer,
            output_dir=region_output,
            skip_osm=args.skip_osm,
            preloaded_features=preloaded,
            temporal_layers=temporal_layers,
            temporal_labels=years,
            args_ref=args_ref,
            region=region_name,
        )

        # Load region spots and tag with region name
        region_geojson = region_output / "spots.geojson"
        if region_geojson.exists():
            region_spots = gpd.read_file(region_geojson)
            region_spots["region"] = region_name
            all_spots.append(region_spots)
            region_metrics[region_name] = {
                "spots_count": len(region_spots),
            }
            logger.info("  %s: %d spots", region_name, len(region_spots))
        else:
            region_metrics[region_name] = {"spots_count": 0}

        # Run change detection per region
        _run_change_detection_for_layers(
            temporal_layers, region_output / "change_detection"
        )

    # Merge all regions into unified output
    logger.info("")
    logger.info("=" * 60)
    logger.info("MERGING ALL REGIONS")
    logger.info("=" * 60)

    if all_spots:
        merged = gpd.GeoDataFrame(pd.concat(all_spots, ignore_index=True))
        merged_path = args.output_dir / "spots_all.geojson"
        processing_params = {
            "threshold_nw_cm2_sr": args.threshold,
            "regions": list(regions.keys()),
            "temporal_years": ["2023", "2024", "2025"],
            "total_spots": len(merged),
            "per_region": region_metrics,
        }
        to_geojson(merged, merged_path, processing_params=processing_params)
        to_parquet(merged, args.output_dir / "spots_all.parquet")

        logger.info("Merged output: %d total spots across %d regions", len(merged), len(all_spots))
        for region, m in region_metrics.items():
            logger.info("  %s: %d spots", region, m["spots_count"])
    else:
        logger.warning("No spots found in any region")

    total_elapsed = time.perf_counter() - total_start

    # Write unified run metadata
    run_summary = {
        "run_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "pipeline_version": "0.7.0",
        "mode": "all_regions",
        "parameters": {
            "threshold_nw": args.threshold,
            "temporal_years": ["2023", "2024", "2025"],
            "regions": list(regions.keys()),
        },
        "metrics": {
            "total_spots": len(merged) if all_spots else 0,
            "per_region": region_metrics,
        },
        "total_time_s": round(total_elapsed, 2),
    }
    with open(args.output_dir / "run_metadata.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL REGIONS COMPLETE (%.1fs total)", total_elapsed)
    logger.info("=" * 60)
    logger.info("  Total spots: %d", len(merged) if all_spots else 0)
    logger.info("  Output: %s", args.output_dir / "spots_all.geojson")


def _run_change_detection_for_layers(layers: list, output_dir: Path):
    """Run change detection for a set of temporal layers."""
    import json

    if len(layers) < 3:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_change_detection(layers[0], layers[1], layers[2], threshold=1.0)

    to_cog(results["2023_to_2025"]["change_map"], output_dir / "change_2023_to_2025.tif")
    stats_output = {
        key: results[key]["stats"].to_dict()
        for key in ["2023_to_2024", "2024_to_2025", "2023_to_2025"]
    }
    stats_output["trend"] = {
        "consistent_brightening_pixels": results["trend"]["n_consistent_brightening"],
        "consistent_darkening_pixels": results["trend"]["n_consistent_darkening"],
    }
    with open(output_dir / "change_stats.json", "w") as f:
        json.dump(stats_output, f, indent=2)


def _run_change_detection(args):
    """Run change detection from CLI args (single region)."""

    logger.info("=" * 60)
    logger.info("CHANGE DETECTION: 2023 → 2024 → 2025")
    logger.info("=" * 60)

    paths = [Path(p) for p in args.change_detection]
    layers = [load_cached_radiance(p) for p in paths]
    _run_change_detection_for_layers(layers, args.output_dir / "change_detection")


if __name__ == "__main__":
    main()
