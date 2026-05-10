.PHONY: install test lint run validate clean docker-build docker-run

# === Setup ===
install:
	uv sync
	cd apps/stargazing_spots && uv sync

download-data:
	cd apps/stargazing_spots && uv run python scripts/download_terrain_data.py

# === Development ===
test:
	uv run python -m pytest apps/stargazing_spots/tests/ -v

lint:
	uv run ruff check apps/stargazing_spots/stargazing_spots/ apps/stargazing_spots/tests/
	uv run ruff format --check apps/stargazing_spots/stargazing_spots/

format:
	uv run ruff format apps/stargazing_spots/stargazing_spots/ apps/stargazing_spots/tests/

# === Pipeline ===
run:
	cd apps/stargazing_spots && uv run python -m stargazing_spots \
		--all-regions \
		--from-geojson input/portugal/osm/osm_features_portugal.geojson \
		--output-dir output/portugal \
		--threshold 3.0

validate:
	cd apps/stargazing_spots && uv run python -c "\
		from stargazing_spots.validate import run_validation, save_validation_report; \
		from stargazing_spots.skyglow import compute_sky_brightness, brightness_to_sqm, PropagationConfig; \
		import numpy as np, joblib; \
		from pathlib import Path; \
		import geopandas as gpd; \
		years = [joblib.load(f'input/portugal/cache/VNP46A4_{y}.joblib') for y in ['2023','2024','2025']]; \
		radiance = np.nanmedian(np.stack([d['NearNadir_Composite_Snow_Free'].values[0] for d in years]), axis=0); \
		x = years[0].coords['x'].values; y = years[0].coords['y'].values; \
		sky = compute_sky_brightness(np.where(radiance<0, np.nan, radiance), PropagationConfig()); \
		sqm = brightness_to_sqm(sky); \
		spots = gpd.read_file('output/portugal/mainland/spots.geojson'); \
		report = run_validation(sqm, x, y, spots); \
		save_validation_report(report, Path('output/portugal/validation_report.json')); \
		print(f'MAE: {report[\"overall\"][\"mean_mae_across_sources\"]} mag')"

# === Docker ===
docker-build:
	docker compose build stargazing-spots

docker-run:
	docker compose run --rm stargazing-spots

docker-test:
	docker compose run --rm stargazing-spots-test

# === Cleanup ===
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
