# TESS-IDA Validation Data

20 Portuguese TESS photometer stations (Stars4All network), 6.4M nighttime readings (2019-2025). Range: SQM 17.7 (Porto) to 23.0 (Alentejo).

## Pipeline usage

The pipeline only reads `tess_station_summary.json` for validation. The raw `.dat` files and SQLite DB are not required to run the pipeline.

## Downloading raw data (optional)

To reproduce the station summary or run custom analysis on the raw photometer readings:

1. Copy `.env.example` to `.env`
2. Request the IDA_URL from the Stars4All project (contact: rafael08@ucm.es or jzamorano@fis.ucm.es)
3. Download station directories via WebDAV:

```bash
cp .env.example .env
# Edit .env — set IDA_URL to the NextCloud shared link you receive

# Download a single station:
curl -O "${IDA_URL}/stars201/stars201_2023-07.dat"

# Or bulk-download all stations with wget:
wget -r -np -nH --cut-dirs=4 "${IDA_URL}"
```

The URL provides access to a NextCloud WebDAV share containing monthly `.dat` files per station.

## Stations

| Station | Lat | Lon | Readings | Median SQM | P95 SQM |
|---------|-----|-----|----------|-----------|---------|
| stars201 | 38.756 | -6.985 | 620,914 | 19.79 | 21.01 |
| stars218 | 39.110 | -6.689 | 768,358 | 20.40 | 21.35 |
| stars271 | 38.743 | -7.210 | 463,094 | 18.83 | 20.39 |
| stars288 | 38.220 | -6.632 | 1,252,597 | 20.56 | 21.48 |
| stars316 | 38.758 | -7.003 | 78,498 | 19.98 | 20.98 |
| stars319 | 38.458 | -7.528 | 3,685 | 20.37 | 21.13 |
| stars320 | 38.463 | -7.493 | 13,429 | 20.16 | 20.91 |
| stars321 | 38.405 | -7.450 | 15,153 | 20.70 | 21.26 |
| stars322 | 38.425 | -7.430 | 67,988 | 20.65 | 21.25 |
| stars323 | 38.481 | -7.528 | 20,019 | 20.21 | 20.84 |
| stars324 | 38.522 | -7.508 | 3,103 | 20.24 | 20.82 |
| stars325 | 38.418 | -7.408 | 71,461 | 20.73 | 21.28 |
| stars327 | 38.277 | -6.540 | 123,980 | 19.19 | 20.33 |
| stars328 | 38.277 | -6.920 | 21,340 | 19.33 | 20.27 |
| stars332 | 38.632 | -7.158 | 133,407 | 19.86 | 20.57 |
| stars384 | 40.270 | -6.104 | 128,928 | 20.47 | 21.42 |
| stars387 | 40.164 | -6.099 | 96,519 | 20.49 | 21.55 |
| stars40 | 41.177 | -8.606 | 608,077 | 17.68 | 18.27 |
| stars41 | 41.177 | -8.606 | 638,159 | 17.88 | 18.50 |
| stars66 | 39.781 | -6.016 | 526,474 | 20.35 | 21.52 |

## File structure

```
tess-ida/
├── .env.example                # Template — needs IDA_URL from Stars4All
├── tess_station_summary.json   # Pre-computed stats (used by pipeline)
├── README.md
├── adm/
│   └── tessida.db              # SQLite DB (not committed)
└── IDA/
    └── stars*/                  # Monthly .dat files (not committed)
```

## .dat format

IDA standard: tab-separated, one row per 5-minute measurement.

```
Date(UTC)	Time(UTC)	Temp(C)	Voltage	MSAS(mag/arcsec2)
2019-01-01	00:05:00	8.3	4.12	19.45
```
