"""OSM-based enrichment of dark-sky locations.

Cross-references satellite-derived dark-sky coordinates with OpenStreetMap
features to identify accessible stargazing sites.
"""

import logging

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Tags to QUERY from OSM (what we download)
DEFAULT_OSM_TAGS = {
    "leisure": ["nature_reserve", "picnic_site"],
    "tourism": ["camp_site", "viewpoint", "picnic_site", "wilderness_hut"],
    "natural": ["peak", "hill", "ridge", "saddle", "heath", "grassland", "bare_rock", "scree", "cliff"],
    "landuse": ["meadow", "grass"],
    "man_made": ["observatory"],
    "amenity": ["shelter"],
}

# Relevance filters applied AFTER download
RELEVANCE_FILTERS = {
    "natural": ["peak", "hill", "ridge", "saddle", "heath", "grassland", "bare_rock", "scree", "cliff", None],
    "landuse": ["meadow", "grass", "recreation_ground", "greenfield", "village_green", None],
    "leisure": ["nature_reserve", "picnic_table", None],
    "tourism": ["viewpoint", "picnic_site", "camp_site", "wilderness_hut", None],
    "man_made": ["observatory", None],
    "amenity": ["shelter", None],
    "access": ["yes", "permissive", None],
}

# Tags that DISQUALIFY a feature
EXCLUSION_TAGS = {
    "lit": ["yes"],
    "access": ["private", "no", "customers"],
    "natural": ["wood", "water", "wetland"],
    "landuse": ["forest", "industrial", "commercial", "retail", "residential"],
}


def fetch_osm_features(
    place_name: str, tags: dict | None = None
) -> gpd.GeoDataFrame:
    """Download OpenStreetMap features for a named place."""
    if tags is None:
        tags = DEFAULT_OSM_TAGS

    logger.info("Fetching OSM features for '%s'...", place_name)
    gdf = ox.features_from_place(place_name, tags)
    logger.info("Retrieved %d features.", len(gdf))
    return gdf


def filter_relevant_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply relevance filters and exclusions for stargazing-appropriate features."""
    # Stage 1: Relevance inclusion (OR across categories)
    masks = []
    for col, values in RELEVANCE_FILTERS.items():
        if col not in gdf.columns:
            continue
        non_null_values = [v for v in values if v is not None]
        if None in values:
            masks.append(gdf[col].isin(non_null_values) | gdf[col].isna())
        else:
            masks.append(gdf[col].isin(non_null_values))

    if not masks:
        included = gdf
    else:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
        included = gdf[combined]

    logger.info("Inclusion filter: %d -> %d features.", len(gdf), len(included))

    # Stage 2: Exclusion
    exclusion_mask = ~included.index.isin(included.index)
    for col, bad_values in EXCLUSION_TAGS.items():
        if col in included.columns:
            excluded_by_col = included[col].isin(bad_values)
            exclusion_mask = exclusion_mask | excluded_by_col
            n_excluded = excluded_by_col.sum()
            if n_excluded > 0:
                logger.info("  Excluded %d features by %s=%s", n_excluded, col, bad_values)

    filtered = included[~exclusion_mask]
    n_excluded_total = len(included) - len(filtered)
    if n_excluded_total > 0:
        logger.info("Exclusion filter: removed %d features (lit, private, forest, water).", n_excluded_total)
    logger.info("Final: %d stargazing-relevant features.", len(filtered))

    return filtered


def _estimate_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
    """Estimate the appropriate UTM zone EPSG code from a GeoDataFrame's centroid."""
    centroid = gdf.geometry.union_all().centroid
    lon = centroid.x
    lat = centroid.y
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone_number
    return 32700 + zone_number


def spatial_join(
    features: gpd.GeoDataFrame,
    dark_coords: gpd.GeoDataFrame,
    buffer_m: float = 500,
) -> gpd.GeoDataFrame:
    """Identify OSM features within buffer distance of dark-sky pixels."""
    utm_epsg = _estimate_utm_epsg(dark_coords)
    logger.info("Using UTM projection EPSG:%d for metric buffering.", utm_epsg)

    features_proj = features.to_crs(epsg=utm_epsg)
    dark_proj = dark_coords.to_crs(epsg=utm_epsg)

    dark_proj["buffer"] = dark_proj.geometry.buffer(buffer_m)
    dark_buffered = dark_proj.set_geometry("buffer")

    joined = gpd.sjoin(features_proj, dark_buffered, how="inner", predicate="within")
    joined = joined.set_geometry("geometry")

    cols_to_drop = [c for c in joined.columns if c in ("index_right", "buffer", "geometry_right")]
    joined = joined.drop(columns=cols_to_drop, errors="ignore")

    result = joined.to_crs(epsg=4326)
    points_only = result[result.geometry.geom_type == "Point"]

    if "osmid" in points_only.columns:
        points_only = points_only.drop_duplicates(subset=["osmid"])
    else:
        points_only = points_only.drop_duplicates(subset=["geometry"])

    logger.info(
        "Spatial join: %d unique features within %dm of dark pixels.",
        len(points_only),
        buffer_m,
    )
    return points_only


def score_spots(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign a stargazing quality score to each identified spot."""
    scores = []
    for _, row in gdf.iterrows():
        score = 0.0

        ele = row.get("ele")
        if ele is not None and str(ele).replace(".", "").isdigit():
            score += float(ele) / 100

        tourism = row.get("tourism", "")
        if tourism in ("viewpoint", "camp_site", "picnic_site"):
            score += 5

        leisure = row.get("leisure", "")
        if leisure in ("nature_reserve", "park"):
            score += 3

        access_val = row.get("access", "")
        if access_val in ("yes", "permissive"):
            score += 2

        scores.append(score)

    gdf = gdf.copy()
    gdf["stargazing_score"] = scores
    return gdf.sort_values("stargazing_score", ascending=False)


def enrich_with_temporal(
    spots: gpd.GeoDataFrame,
    stability: "xr.DataArray",
    trend: "xr.DataArray",
    consistency: "xr.DataArray",
    n_years: int = 3,
    threshold: float = 3.0,
) -> gpd.GeoDataFrame:
    """Add temporal stability metadata to each spot."""
    import numpy as np

    spots = spots.copy()

    y_coords = stability.coords["y"].values
    x_coords = stability.coords["x"].values

    stab_vals = []
    trend_vals = []
    consistency_vals = []

    for _, row in spots.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        y_idx = int(np.argmin(np.abs(y_coords - lat)))
        x_idx = int(np.argmin(np.abs(x_coords - lon)))

        stab_vals.append(float(stability.values[y_idx, x_idx]))
        trend_vals.append(float(trend.values[y_idx, x_idx]))
        consistency_vals.append(int(consistency.values[y_idx, x_idx]))

    spots["temporal_std"] = stab_vals
    spots["temporal_trend"] = trend_vals
    spots["years_dark"] = consistency_vals

    spots["temporal_stability"] = spots["temporal_std"].apply(
        lambda s: "high" if s < 0.5 else ("medium" if s < 1.5 else "low")
    )

    spots["trend_direction"] = spots["temporal_trend"].apply(
        lambda t: "darkening" if t < -threshold else ("brightening" if t > threshold else "stable")
    )

    high = (spots["temporal_stability"] == "high").sum()
    medium = (spots["temporal_stability"] == "medium").sum()
    low = (spots["temporal_stability"] == "low").sum()
    brightening = (spots["trend_direction"] == "brightening").sum()
    darkening = (spots["trend_direction"] == "darkening").sum()

    logger.info("Temporal enrichment complete:")
    logger.info("  Stability: %d high, %d medium, %d low", high, medium, low)
    logger.info("  Trend: %d stable, %d brightening, %d darkening",
                len(spots) - brightening - darkening, brightening, darkening)
    logger.info("  Dark all %d years: %d spots", n_years, (spots["years_dark"] == n_years).sum())

    return spots
