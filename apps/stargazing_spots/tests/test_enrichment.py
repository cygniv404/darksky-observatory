"""Tests for OSM enrichment and scoring."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from stargazing_spots.enrichment import score_spots, spatial_join


@pytest.fixture
def sample_features():
    """Create mock OSM features."""
    return gpd.GeoDataFrame(
        {
            "name": ["Viewpoint A", "Park B", "Camp C", "City Center"],
            "tourism": ["viewpoint", None, "camp_site", None],
            "leisure": [None, "park", None, None],
            "ele": ["800", None, "300", "50"],
            "access": ["yes", "permissive", None, "private"],
        },
        geometry=[
            Point(-8.0, 40.0),
            Point(-8.1, 40.1),
            Point(-8.2, 40.2),
            Point(-8.5, 40.5),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_dark_coords():
    """Create mock dark-sky pixel centroids near the first 3 features."""
    return gpd.GeoDataFrame(
        {"radiance": [1.0, 1.5, 2.0]},
        geometry=[
            Point(-8.001, 40.001),
            Point(-8.101, 40.101),
            Point(-8.201, 40.201),
        ],
        crs="EPSG:4326",
    )


def test_score_spots_ordering(sample_features):
    scored = score_spots(sample_features)
    assert "stargazing_score" in scored.columns
    # Viewpoint A should score highest (viewpoint + elevation + access)
    assert scored.iloc[0]["name"] == "Viewpoint A"


def test_score_spots_values(sample_features):
    scored = score_spots(sample_features)
    scores = scored["stargazing_score"].values
    # All scores should be non-negative
    assert all(s >= 0 for s in scores)


def test_spatial_join_finds_nearby(sample_features, sample_dark_coords):
    result = spatial_join(sample_features, sample_dark_coords, buffer_m=5000)
    # First 3 features are near dark coords; City Center is far away
    names = result["name"].tolist()
    assert "Viewpoint A" in names
    assert "City Center" not in names


def test_spatial_join_empty_result(sample_features):
    far_coords = gpd.GeoDataFrame(
        {"radiance": [1.0]},
        geometry=[Point(10.0, 60.0)],  # Far from Portugal
        crs="EPSG:4326",
    )
    result = spatial_join(sample_features, far_coords, buffer_m=500)
    assert len(result) == 0
