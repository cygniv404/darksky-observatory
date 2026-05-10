# Getting Started

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (optional)
- ~5 GB disk space

## Quick Start

```bash
git clone https://github.com/cygniv404/darksky-observatory.git
cd darksky-observatory
uv sync
cd apps/stargazing_spots && uv sync && cd ../..
make test
```

## Environment Setup

```bash
cp apps/stargazing_spots/.env.example apps/stargazing_spots/.env
```

| Variable | Source | Purpose |
|----------|--------|---------|
| `EARTHDATA_TOKEN` | [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/) | VIIRS data |
| `CDS_API_KEY` | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) | ERA5 cloud |

## Data Acquisition

**Automatic** (first run): Copernicus DEM, ESA WorldCover, Hansen GFC, ERA5 cloud cover.

**One-time setup:**

1. VIIRS: set `EARTHDATA_TOKEN`, pipeline downloads and caches (~50 MB, 5-10 min).
2. GADM boundary: download from gadm.org, save as `input/portugal/boundary/gadm41_PRT_1.json.zip`.
3. OSM features: export from Overpass Turbo (~2 min) or let pipeline query directly (~30 min).
4. CLAAS-3 (optional): register at wui.cmsaf.eu, order CFC monthly product, place in `input/portugal/cache/claas3/`.

## Running the Pipeline

```bash
make run                  # Full pipeline (all regions)

# Manual:
cd apps/stargazing_spots
uv run python -m stargazing_spots \
  --all-regions \
  --from-geojson input/portugal/osm/osm_features_portugal.geojson \
  --output-dir output/portugal \
  --threshold 3.0
```

First run: ~12 min (DEM HTTP reads). With cached terrain: ~36s.

## Validation

```bash
make validate
```

## Docker

```bash
make docker-build
make docker-run
```

## Multi-Country

```bash
mkdir -p apps/stargazing_spots/input/spain/{boundary,cache,osm}
# Add GADM boundary + OSM features for the target country
uv run python -m stargazing_spots --place "Spain" --output-dir output/spain
```

Satellite data (VIIRS, DEM, WorldCover, ERA5) covers all of Europe. Only boundary and OSM features are country-specific.
