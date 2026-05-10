"""Tests for the Multi-Criteria Stargazing Suitability Index."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from stargazing_spots.suitability import (
    SSIWeights,
    compute_ssi,
    normalize_clear_sky,
    normalize_darkness,
    normalize_elevation,
    normalize_sky_openness,
)


class TestSSIWeights:
    """Tests for weight configuration."""

    def test_default_weights_sum_to_one(self):
        w = SSIWeights()
        total = w.darkness + w.clear_sky + w.sky_openness + w.land_cover + w.slope + w.elevation
        assert abs(total - 1.0) < 0.01

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            SSIWeights(darkness=0.5, clear_sky=0.5, sky_openness=0.5,
                       land_cover=0.1, slope=0.1, elevation=0.1)


class TestNormalizeDarkness:
    """Tests for SQM → suitability normalization."""

    def test_pristine_sky(self):
        """SQM 22.0 should score 1.0."""
        score = normalize_darkness(np.array([22.0]))
        assert abs(score[0] - 1.0) < 0.01

    def test_marginal_sky(self):
        """SQM 20.0 should score 0.0."""
        score = normalize_darkness(np.array([20.0]))
        assert abs(score[0] - 0.0) < 0.01

    def test_urban_scores_zero(self):
        """SQM below 20.0 should clip to 0."""
        score = normalize_darkness(np.array([18.0, 17.0, 15.0]))
        assert all(score == 0.0)

    def test_monotonic(self):
        """Higher SQM should always give higher score."""
        sqm = np.array([20.0, 20.5, 21.0, 21.5, 22.0])
        scores = normalize_darkness(sqm)
        assert all(np.diff(scores) >= 0)


class TestNormalizeClearSky:
    """Tests for cloud fraction normalization."""

    def test_always_clear(self):
        score = normalize_clear_sky(np.array([1.0]))
        assert score[0] == 1.0

    def test_always_cloudy(self):
        score = normalize_clear_sky(np.array([0.0]))
        assert score[0] == 0.0

    def test_clips_above_one(self):
        score = normalize_clear_sky(np.array([1.5]))
        assert score[0] == 1.0


class TestNormalizeSkyOpenness:
    """Tests for SVF normalization."""

    def test_fully_open(self):
        score = normalize_sky_openness(np.array([1.0]))
        assert score[0] == 1.0

    def test_obstructed(self):
        """SVF 0.5 should score 0."""
        score = normalize_sky_openness(np.array([0.5]))
        assert score[0] == 0.0

    def test_below_threshold_clips(self):
        score = normalize_sky_openness(np.array([0.3]))
        assert score[0] == 0.0


class TestNormalizeElevation:
    """Tests for elevation bonus."""

    def test_sea_level(self):
        score = normalize_elevation(np.array([0.0]))
        assert score[0] == 0.0

    def test_high_altitude(self):
        score = normalize_elevation(np.array([1500.0]))
        assert score[0] == 1.0

    def test_caps_at_max(self):
        score = normalize_elevation(np.array([3000.0]))
        assert score[0] == 1.0


class TestComputeSSI:
    """Tests for the full SSI computation."""

    @pytest.fixture
    def sample_spots(self):
        """Create a minimal GeoDataFrame with all required columns."""
        return gpd.GeoDataFrame({
            "name": ["dark_flat_open", "urban_cloudy", "forest_steep"],
            "geometry": [Point(-7.5, 38.5), Point(-9.1, 38.7), Point(-8.0, 40.0)],
            "predicted_sqm": [21.5, 18.0, 21.0],
            "clear_night_fraction": [0.65, 0.40, 0.50],
            "sky_view_factor": [0.95, 0.90, 0.60],
            "land_cover_score": [1.0, 0.0, 0.3],
            "slope_score": [1.0, 0.8, 0.4],
            "ele": [400, 50, 800],
        }, crs="EPSG:4326")

    def test_returns_ssi_columns(self, sample_spots):
        result = compute_ssi(sample_spots)
        assert "ssi_score" in result.columns
        assert "ssi_class" in result.columns

    def test_dark_open_scores_higher(self, sample_spots):
        result = compute_ssi(sample_spots)
        # The dark/flat/open spot should score highest
        assert result.iloc[0]["ssi_score"] > result.iloc[1]["ssi_score"]
        assert result.iloc[0]["ssi_score"] > result.iloc[2]["ssi_score"]

    def test_urban_scores_low(self, sample_spots):
        result = compute_ssi(sample_spots)
        # Urban spot should be marginal or poor
        assert result.iloc[1]["ssi_class"] in ["marginal", "poor"]

    def test_scores_in_valid_range(self, sample_spots):
        result = compute_ssi(sample_spots)
        assert all(result["ssi_score"] >= 0)
        assert all(result["ssi_score"] <= 100)

    def test_classification_thresholds(self, sample_spots):
        result = compute_ssi(sample_spots)
        for _, row in result.iterrows():
            score = row["ssi_score"]
            cls = row["ssi_class"]
            if score >= 80:
                assert cls == "excellent"
            elif score >= 65:
                assert cls == "good"
            elif score >= 50:
                assert cls == "acceptable"
            elif score >= 35:
                assert cls == "marginal"
            else:
                assert cls == "poor"
