"""Nighttime accessibility from OSM tags.

Classifies spots as drive_up, short_walk, hike_required, or boat_required
based on feature type and access/barrier tags.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tags indicating nighttime accessibility
POSITIVE_ACCESS_TAGS = {
    "tourism": ["camp_site", "viewpoint", "picnic_site", "wilderness_hut"],
    "amenity": ["shelter"],
    "leisure": ["nature_reserve"],
}

# Tags indicating restricted/problematic access
NEGATIVE_ACCESS_TAGS = {
    "access": ["private", "no", "customers", "permit"],
    "barrier": ["gate", "lift_gate", "bollard", "fence"],
    "lit": ["yes"],  # Lit = light pollution at the spot itself
}

# Feature types and their typical accessibility
FEATURE_ACCESS_TYPE = {
    # Drive-up (roadside, parking expected)
    "viewpoint": "drive_up",
    "camp_site": "drive_up",
    "picnic_site": "drive_up",
    "wilderness_hut": "drive_up",
    "observatory": "drive_up",
    "shelter": "short_walk",
    # May require walking
    "peak": "short_walk",
    "hill": "short_walk",
    "ridge": "hike_required",
    "saddle": "hike_required",
    # Open terrain (varies)
    "grassland": "unknown",
    "heath": "unknown",
    "bare_rock": "short_walk",
    "cliff": "short_walk",
}


def assess_access(spots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assess nighttime accessibility for each spot.

    Uses OSM tags already on the features to determine:
    - Is it accessible at night? (no gate, no private tag)
    - What type of access? (drive-up, short walk, hike)
    - Any warnings? (lit, restricted)

    Args:
        spots: GeoDataFrame with OSM feature tags.

    Returns:
        GeoDataFrame with added columns:
            - access_type: "drive_up" | "short_walk" | "hike_required" | "unknown"
            - night_accessible: True/False/None (None = unknown)
            - access_warnings: list of warning strings
            - access_score: 0-1 (1 = definitely accessible at night)
    """
    spots = spots.copy()
    n = len(spots)

    access_types = []
    night_accessible = []
    access_warnings_list = []
    access_scores = []

    for _, row in spots.iterrows():
        warnings = []
        access_type = "unknown"
        accessible = True  # assume accessible unless evidence says otherwise
        score = 0.7  # default: probably accessible

        # Determine access type from feature tags
        for tag_col, type_map in [("tourism", FEATURE_ACCESS_TYPE),
                                   ("natural", FEATURE_ACCESS_TYPE),
                                   ("man_made", FEATURE_ACCESS_TYPE),
                                   ("amenity", FEATURE_ACCESS_TYPE)]:
            if tag_col in row.index and pd.notna(row.get(tag_col)):
                val = str(row[tag_col])
                if val in type_map:
                    access_type = type_map[val]
                    break

        # Check for positive indicators
        for tag_col, good_vals in POSITIVE_ACCESS_TAGS.items():
            if tag_col in row.index and pd.notna(row.get(tag_col)):
                if str(row[tag_col]) in good_vals:
                    score = max(score, 0.9)

        # Check for negative indicators
        for tag_col, bad_vals in NEGATIVE_ACCESS_TAGS.items():
            if tag_col in row.index and pd.notna(row.get(tag_col)):
                if str(row[tag_col]) in bad_vals:
                    if tag_col == "access":
                        accessible = False
                        score = 0.0
                        warnings.append(f"access={row[tag_col]}")
                    elif tag_col == "barrier":
                        accessible = False
                        score = 0.1
                        warnings.append(f"barrier={row[tag_col]} (may be locked at night)")
                    elif tag_col == "lit":
                        warnings.append("lit=yes (local light pollution)")
                        score = min(score, 0.5)

        # Detect offshore islands: DEM=0 + wetland/water land cover
        dem_elev = row.get("dem_elevation_m") if "dem_elevation_m" in row.index else None
        lc_class = row.get("land_cover_class") if "land_cover_class" in row.index else None
        if dem_elev is not None and dem_elev == 0 and lc_class in (80, 90):
            access_type = "boat_required"
            score = 0.3
            warnings.append("offshore island (boat access only)")

        # Adjust score by access type
        if access_type == "drive_up":
            score = min(score + 0.1, 1.0)
        elif access_type == "hike_required":
            score = min(score, 0.5)
            warnings.append("may require hiking (no nearby road)")

        access_types.append(access_type)
        night_accessible.append(accessible)
        access_warnings_list.append("; ".join(warnings) if warnings else "")
        access_scores.append(round(score, 2))

    spots["access_type"] = access_types
    spots["night_accessible"] = night_accessible
    spots["access_warnings"] = access_warnings_list
    spots["access_score"] = access_scores

    # Summary
    logger.info("Accessibility assessment:")
    for atype in ["drive_up", "short_walk", "hike_required", "boat_required", "unknown"]:
        count = access_types.count(atype)
        if count > 0:
            logger.info("  %s: %d spots", atype, count)

    n_restricted = sum(1 for a in night_accessible if not a)
    n_lit = sum(1 for w in access_warnings_list if "lit=yes" in w)
    if n_restricted > 0:
        logger.info("  Restricted access: %d spots", n_restricted)
    if n_lit > 0:
        logger.info("  Lit (local light pollution): %d spots", n_lit)

    return spots
