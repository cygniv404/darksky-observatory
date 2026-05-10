# Validation Data — Portugal

Ground-truth datasets used to validate the PSF skyglow model and pipeline output.

## Data Sources

| File/Directory | Points | Type | Description |
|----------------|--------|------|-------------|
| `stargazing_sites.csv` | 33 | SQM measurements | Published + certified dark-sky sites |
| `expert_recommended_spots.csv` | 20 | Locations | Astrotourism-recommended spots |
| `darkskysites/` | 1,920 | SQM grid | darkskysites.com VIIRS-based reference model |
| `tess-ida/` | 20 stations | Continuous monitoring | Stars4All TESS photometer network (6.4M readings) |

## stargazing_sites.csv

33 sites across mainland Portugal, organized by validation tier:

| Tier | Count | Description | Use |
|------|-------|-------------|-----|
| `measured` | 7 | Peer-reviewed SQM field measurements | Primary validation (MAE computation) |
| `certification_threshold` | 10 | Starlight Foundation certified >= 21.0 | Binary detection validation |
| `estimated` | 16 | Falchi 2016 World Atlas derived | Reference only (not independent) |

**Columns:** WKT, name, region, expected_sqm, sqm_source, sqm_type, description

**Sources:** Lima et al. 2016, IDA certification 2018, Barbosa et al. 2024, Lima 2015 PhD, Dark Sky Alqueva monitoring.

## expert_recommended_spots.csv

20 locations recommended by Portuguese astrotourism organizations and astronomy clubs.

**Columns:** WKT, name, region, recommended_by, description

**Sources:** Dark Sky Alqueva, Visit Portugal, Portuguese astronomy clubs, travel guides.

**Validation use:** Proximity detection — what % of expert-recommended spots have a pipeline output spot within 5km / 10km.

## darkskysites/

VIIRS-based reference sky brightness model from darkskysites.com (LUMIX engine).

- `portugal_grid_sqm_2012_2025.json` — 1,920 points at 0.1° spacing, annual SQM 2012-2025
- `README.md` — API details and download methodology

**Validation use:** Cross-model comparison (our simplified PSF vs. full Garstang RT). Not independent ground-truth but tests consistency with established models.

## tess-ida/

TESS (Telescope Encoder and Sky Sensor) photometer data from the Stars4All European network.

- `IDA/` — Monthly .dat files for 20 Portuguese stations (2019-2025)
- `adm/tessida.db` — SQLite database of all readings
- `tess_station_summary.json` — Per-station statistics (readings, median/max/P95 SQM)
- `.env` — Stars4All NextCloud URL for data download

**Validation use:** Continuous instrumental ground-truth (6.4M nighttime readings). Best-night max SQM compared against PSF predictions.

See `tess-ida/README.md` for station details and download instructions.
