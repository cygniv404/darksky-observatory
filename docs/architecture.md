# System Architecture

## Repository Structure

```
darksky-observatory/
├── apps/stargazing_spots/       # Python pipeline (23 modules)
├── Makefile
├── docker-compose.yml
├── pyproject.toml               # uv workspace monorepo
└── docs/
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: ACQUISITION                                                    │
│  acquisition.py                                                          │
│  NASA LP DAAC → blackmarblepy → joblib cache → xarray Dataset            │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────┐
│  STAGE 2-3: PROCESSING + CLASSIFICATION                                  │
│  processing.py                                                           │
│  Temporal composite (median + stability + trend) → dark-sky mask         │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────┐
│  STAGE 4: ENRICHMENT                                                     │
│  enrichment.py + enrichment_raster.py                                    │
│  OSM features → raster-native pixel sampling O(n)                        │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────┐
│  STAGE 5-6: SKYGLOW + CLOUD COVER                                        │
│  skyglow.py + cloud_cover.py                                             │
│  PSF convolution (FFT) → SQM prediction                                 │
│  CLAAS-3 (mainland) / ERA5 (islands) → clear-night fraction              │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────┐
│  STAGE 7-9: ASSESSMENT (parallel)                                        │
│  assess.py → sky_view.py + land_cover.py + accessibility.py              │
│  SVF (DEM + Hansen) | WorldCover + Hansen | DEM slope                    │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────┐
│  STAGE 10-11: SUITABILITY                                                │
│  suitability.py                                                          │
│  SSI (6-factor WLC) → dedup (1km) → density cap (5/cell)                │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────┐
│  STAGE 12-14: TIERING + PLANNING + CHANGE DETECTION                      │
│  access_check.py + observation_planning.py + change_detection.py         │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────────────┐
│  STAGE 15-16: EXPORT + VALIDATION                                        │
│  export.py + stac_catalog.py + validate.py                               │
│  GeoJSON | Parquet | COG | STAC Catalog                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Role |
|--------|------|
| `acquisition` | VIIRS download via blackmarblepy, joblib caching |
| `processing` | Temporal composite, classification, coordinate extraction |
| `enrichment` | OSM feature download and filtering |
| `enrichment_raster` | Raster-native pixel sampling of OSM features |
| `skyglow` | PSF convolution (FFT), SQM prediction, Bortle classification |
| `cloud_cover` | CLAAS-3 / ERA5 clear-night fraction |
| `sky_view` | Vegetation-aware SVF (DEM + Hansen) |
| `land_cover` | WorldCover + Hansen probabilistic scoring |
| `accessibility` | DEM slope |
| `suitability` | SSI weighted linear combination |
| `assess` | Assessment orchestrator (parallel SVF/land_cover/slope) |
| `access_check` | Nighttime access classification, island detection |
| `observation_planning` | Milky Way season, best months |
| `change_detection` | Multi-temporal trend analysis |
| `uncertainty` | Per-pixel confidence scoring |
| `sensitivity` | Threshold analysis (Otsu + validation-calibrated) |
| `quality` | Data quality assessment |
| `validate` | Automated validation against ground-truth |
| `export` | GeoJSON, Parquet, COG output |
| `stac_catalog` | STAC catalog generation |
| `reprojection` | CRS utilities |

## Data Formats

| Stage | Format |
|-------|--------|
| Input | HDF-EOS5 (.h5), sinusoidal projection |
| Cached | joblib pickle (xarray Dataset) |
| Classification | xarray DataArray, exported as COG |
| Vector | GeoDataFrame → GeoJSON + GeoParquet |
| Metadata | STAC JSON |

## Infrastructure

| Component | Technology |
|-----------|-----------|
| Package management | uv workspace (pyproject.toml) |
| Build orchestration | Makefile |
| Containerization | docker-compose.yml |
| Environment config | apps/stargazing_spots/.env |
