"""Tests for dark-sky classification and coordinate extraction."""

import numpy as np
import pytest
import xarray as xr

from stargazing_spots.processing import classify_dark_sky, extract_dark_coordinates


@pytest.fixture
def sample_radiance():
    """Create a synthetic 10x10 radiance grid with known dark/bright distribution."""
    np.random.seed(42)
    data = np.random.uniform(0, 10, size=(10, 10))
    data[0:3, 0:3] = 1.5  # 9 dark pixels (below threshold)
    data[5, 5] = np.nan  # one missing pixel

    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords={
            "y": np.linspace(42.0, 37.0, 10),
            "x": np.linspace(-9.5, -6.0, 10),
        },
    )


def test_classify_dark_sky_threshold(sample_radiance):
    result = classify_dark_sky(sample_radiance, threshold=3.0)
    assert int(result.count()) >= 9  # at least our known dark pixels


def test_classify_dark_sky_zero_threshold(sample_radiance):
    result = classify_dark_sky(sample_radiance, threshold=0.0)
    assert int(result.count()) == 0


def test_classify_dark_sky_high_threshold(sample_radiance):
    result = classify_dark_sky(sample_radiance, threshold=100.0)
    valid_count = int(sample_radiance.count())
    assert int(result.count()) == valid_count


def test_extract_dark_coordinates_schema(sample_radiance):
    dark = classify_dark_sky(sample_radiance, threshold=3.0)
    gdf = extract_dark_coordinates(dark)

    assert "geometry" in gdf.columns
    assert "latitude" in gdf.columns
    assert "longitude" in gdf.columns
    assert "radiance" in gdf.columns
    assert gdf.crs.to_epsg() == 4326


def test_extract_dark_coordinates_values(sample_radiance):
    dark = classify_dark_sky(sample_radiance, threshold=3.0)
    gdf = extract_dark_coordinates(dark)

    assert all(gdf["radiance"] < 3.0)
    assert all(gdf["latitude"] >= 37.0)
    assert all(gdf["latitude"] <= 42.0)
    assert all(gdf["longitude"] >= -9.5)
    assert all(gdf["longitude"] <= -6.0)


def test_classify_preserves_nan(sample_radiance):
    result = classify_dark_sky(sample_radiance, threshold=100.0)
    original_nan_count = int(sample_radiance.isnull().sum())
    result_nan_count = int(result.isnull().sum())
    assert result_nan_count >= original_nan_count
