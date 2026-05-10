"""Tests for observation planning (Milky Way, moon phases)."""

import pytest

from stargazing_spots.observation_planning import (
    milky_way_season,
    moon_free_nights_per_month,
    observation_summary,
)


class TestMilkyWaySeason:
    """Tests for Milky Way visibility calculation."""

    def test_portugal_has_visibility(self):
        """Portugal (39°N) should have Milky Way visibility."""
        result = milky_way_season(39.0)
        assert len(result["visible_months"]) > 0
        assert result["max_altitude"] > 0

    def test_best_months_subset_of_visible(self):
        result = milky_way_season(39.0)
        assert all(m in result["visible_months"] for m in result["best_months"])

    def test_peak_month_in_best(self):
        result = milky_way_season(39.0)
        assert result["peak_month"] in result["best_months"]

    def test_summer_is_best(self):
        """June-September should be in best months for mid-latitudes."""
        result = milky_way_season(39.0)
        assert 8 in result["best_months"]  # August

    def test_southern_latitude_higher_altitude(self):
        """Lower latitude → GC reaches higher altitude."""
        north = milky_way_season(42.0)
        south = milky_way_season(33.0)
        assert south["max_altitude"] > north["max_altitude"]

    def test_extreme_latitude_no_visibility(self):
        """At 70°N, GC may not rise."""
        result = milky_way_season(70.0)
        assert result["max_altitude"] <= 0 or len(result["visible_months"]) == 0


class TestMoonFreeNights:
    """Tests for moon-free night calculation."""

    def test_returns_reasonable_count(self):
        n = moon_free_nights_per_month()
        assert 8 <= n <= 14  # between last quarter and first quarter


class TestObservationSummary:
    """Tests for complete observation summary."""

    def test_returns_all_keys(self):
        summary = observation_summary(39.0)
        assert "milky_way" in summary
        assert "moon_free_nights_per_month" in summary
        assert "best_overall" in summary
        assert "seasonal_notes" in summary

    def test_seasonal_notes_complete(self):
        summary = observation_summary(39.0)
        notes = summary["seasonal_notes"]
        assert "summer_jun_sep" in notes
        assert "winter_dec_feb" in notes
