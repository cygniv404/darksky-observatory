"""STAC (SpatioTemporal Asset Catalog) generation for processed dark-sky data.

Generates a standards-compliant STAC catalog describing the pipeline outputs,
enabling discovery and access through STAC-compatible tools and APIs.

Reference: https://stacspec.org/
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pystac
from pystac import (
    Asset,
    Catalog,
    Collection,
    Extent,
    Item,
    MediaType,
    SpatialExtent,
    TemporalExtent,
)

logger = logging.getLogger(__name__)

PORTUGAL_BBOX = [-9.52, 36.96, -6.19, 42.15]
SOURCE_COLLECTION_URL = (
    "https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/"
    "products/VNP46A4"
)


def create_darksky_catalog(
    output_dir: Path,
    geojson_path: Path | None = None,
    cog_path: Path | None = None,
    processing_date: str = "2023-01-01",
) -> Catalog:
    """Generate a STAC catalog for the dark-sky observatory outputs.

    Creates a catalog with a single collection containing one item per processing run.
    Links to the source NASA LP DAAC VNP46A4 product for provenance.

    Args:
        output_dir: Directory to write the catalog JSON files.
        geojson_path: Path to the output spots GeoJSON (optional asset).
        cog_path: Path to the output COG file (optional asset).
        processing_date: Date string of the source VIIRS composite.

    Returns:
        pystac.Catalog object (also written to disk).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = Catalog(
        id="darksky-observatory",
        description=(
            "Dark-sky locations identified from NASA VIIRS VNP46A4 nighttime radiance. "
            "Pixels with radiance < 3 nW/cm²/sr are classified as dark-sky and "
            "cross-referenced with OpenStreetMap for accessible stargazing sites."
        ),
    )

    spatial_extent = SpatialExtent(bboxes=[PORTUGAL_BBOX])
    temporal_extent = TemporalExtent(
        intervals=[[datetime(2023, 1, 1, tzinfo=timezone.utc), None]]
    )
    extent = Extent(spatial=spatial_extent, temporal=temporal_extent)

    collection = Collection(
        id="portugal-darksky",
        description=(
            "Dark-sky classification for mainland Portugal and islands, "
            "derived from VIIRS VNP46A4 monthly nighttime radiance composites."
        ),
        extent=extent,
        license="MIT",
    )
    collection.add_link(
        pystac.Link(
            rel="derived_from",
            target=SOURCE_COLLECTION_URL,
            title="NASA VIIRS VNP46A4 (source product)",
        )
    )

    item = Item(
        id=f"portugal-darksky-{processing_date}",
        geometry={
            "type": "Polygon",
            "coordinates": [[
                [PORTUGAL_BBOX[0], PORTUGAL_BBOX[1]],
                [PORTUGAL_BBOX[2], PORTUGAL_BBOX[1]],
                [PORTUGAL_BBOX[2], PORTUGAL_BBOX[3]],
                [PORTUGAL_BBOX[0], PORTUGAL_BBOX[3]],
                [PORTUGAL_BBOX[0], PORTUGAL_BBOX[1]],
            ]],
        },
        bbox=PORTUGAL_BBOX,
        datetime=datetime.fromisoformat(processing_date).replace(tzinfo=timezone.utc),
        properties={
            "processing:software": "darksky-pipeline v0.1.0",
            "processing:threshold_nw": 3.0,
            "processing:buffer_m": 500,
            "processing:spots_count": 2951,
            "eo:instrument": "VIIRS",
            "eo:platform": "Suomi NPP",
            "sat:product": "VNP46A4",
        },
    )

    if geojson_path and Path(geojson_path).exists():
        item.add_asset(
            "spots",
            Asset(
                href=str(geojson_path),
                media_type=MediaType.GEOJSON,
                title="Dark-sky stargazing spots",
                roles=["data"],
            ),
        )

    if cog_path and Path(cog_path).exists():
        item.add_asset(
            "radiance_cog",
            Asset(
                href=str(cog_path),
                media_type=MediaType.COG,
                title="Filtered nighttime radiance (< 3 nW/cm²/sr)",
                roles=["data"],
            ),
        )

    collection.add_item(item)
    catalog.add_child(collection)

    catalog.normalize_hrefs(str(output_dir))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    logger.info("STAC catalog written to %s", output_dir)
    return catalog
