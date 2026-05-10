# Data Sources

## Dataset Inventory

| Dataset | Provider | Product | Resolution | Access |
|---------|----------|---------|-----------|--------|
| Nighttime Lights | NASA LP DAAC | VNP46A4v001 | 500m annual | Earthdata Login (free) |
| Cloud (mainland) | CM SAF / EUMETSAT | CLAAS-3 CFC | 5 km | wui.cmsaf.eu (free) |
| Cloud (islands) | ECMWF / C3S | ERA5 TCC | 0.25 deg | CDS API (free) |
| Elevation | ESA / Copernicus | GLO-30 DEM | 30m | AWS S3 (public) |
| Forest Canopy | U. Maryland | Hansen GFC v1.11 | 30m | GCS (public) |
| Land Cover | ESA | WorldCover v200 | 10m | AWS S3 (public) |
| Points of Interest | OSM Foundation | Overpass API | Vector | Public |
| Admin Boundaries | UC Davis | GADM v4.1 | Vector | gadm.org |
| Validation (TESS) | Stars4All | TESS-IDA network | Point | api.stars4all.eu |
| Validation (grid) | darkskysites.com | LUMIX model | 0.1 deg | Web |
| Validation (SQM) | Literature | Field measurements | Point | Papers |

## OSM Tags Queried

```
tourism = camp_site | viewpoint | picnic_site | wilderness_hut
natural = peak | hill | ridge | saddle | heath | grassland | bare_rock | scree | cliff
leisure = nature_reserve | picnic_site
amenity = shelter
man_made = observatory
landuse = meadow | grass
```

## Licensing

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| NASA VIIRS | Public domain | Yes |
| Copernicus DEM | Copernicus Data Policy | Yes |
| ESA WorldCover | CC-BY 4.0 | Yes |
| Hansen GFC | CC-BY 4.0 | Yes |
| CLAAS-3 (CM SAF) | CM SAF data policy | Research only |
| ERA5 | Copernicus License | Yes |
| OpenStreetMap | ODbL 1.0 | Yes (share-alike) |

## Authentication

| Dataset | Registration URL |
|---------|-----------------|
| VIIRS VNP46A4 | https://urs.earthdata.nasa.gov/ |
| CLAAS-3 | https://wui.cmsaf.eu |
| ERA5 | https://cds.climate.copernicus.eu/ |

All other datasets are publicly accessible without authentication. The pipeline fetches them automatically on first run.
