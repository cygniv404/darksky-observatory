"""Tests for astronomical twilight and cloud cover computation."""

import numpy as np
import pytest

from stargazing_spots.cloud_cover import (
    compute_sun_altitude,
    is_astronomical_dark,
    build_darkness_mask,
)


class TestSunAltitude:
    """Tests for solar position calculation."""

    def test_midday_positive(self):
        """Sun should be above horizon at noon in Portugal."""
        alt = compute_sun_altitude(hour_utc=12, day_of_year=172, latitude=39.0)
        assert alt > 0

    def test_midnight_negative(self):
        """Sun should be below horizon at midnight in Portugal."""
        alt = compute_sun_altitude(hour_utc=0, day_of_year=172, latitude=39.0)
        assert alt < 0

    def test_summer_higher_than_winter(self):
        """Noon altitude should be higher in summer than winter."""
        summer = compute_sun_altitude(hour_utc=12, day_of_year=172, latitude=39.0)
        winter = compute_sun_altitude(hour_utc=12, day_of_year=355, latitude=39.0)
        assert summer > winter

    def test_equator_higher_than_poles(self):
        """At equinox noon, equator has higher sun than 60°N."""
        equator = compute_sun_altitude(hour_utc=12, day_of_year=80, latitude=0.0)
        high_lat = compute_sun_altitude(hour_utc=12, day_of_year=80, latitude=60.0)
        assert equator > high_lat


class TestAstronomicalDark:
    """Tests for astronomical darkness determination."""

    def test_midnight_winter_is_dark(self):
        """Midnight in December at 39°N should be astronomically dark."""
        assert is_astronomical_dark(hour_utc=0, day_of_year=355, latitude=39.0)

    def test_noon_is_not_dark(self):
        """Noon is never astronomically dark."""
        assert not is_astronomical_dark(hour_utc=12, day_of_year=172, latitude=39.0)

    def test_summer_has_fewer_dark_hours(self):
        """Summer should have fewer dark hours than winter at 39°N."""
        summer_dark = sum(1 for h in range(24) if is_astronomical_dark(h, 172, 39.0))
        winter_dark = sum(1 for h in range(24) if is_astronomical_dark(h, 355, 39.0))
        assert summer_dark < winter_dark

    def test_equator_consistent_darkness(self):
        """Near equator, dark hours should be ~similar year-round."""
        summer_dark = sum(1 for h in range(24) if is_astronomical_dark(h, 172, 5.0))
        winter_dark = sum(1 for h in range(24) if is_astronomical_dark(h, 355, 5.0))
        assert abs(summer_dark - winter_dark) <= 2


class TestDarknessMask:
    """Tests for the precomputed darkness lookup table."""

    def test_mask_shape(self):
        lats = np.array([38.0, 39.0, 40.0])
        mask = build_darkness_mask(lats)
        assert mask.shape == (24, 366, 3)

    def test_mask_dtype(self):
        lats = np.array([39.0])
        mask = build_darkness_mask(lats)
        assert mask.dtype == bool

    def test_noon_always_false(self):
        """Hour 12 should never be dark at mid-latitudes."""
        lats = np.array([39.0])
        mask = build_darkness_mask(lats)
        assert not mask[12, :, 0].any()

    def test_some_hours_dark(self):
        """There should be SOME dark hours every day."""
        lats = np.array([39.0])
        mask = build_darkness_mask(lats)
        daily_dark = mask[:, :, 0].sum(axis=0)
        assert daily_dark.min() >= 4  # at least 4 dark hours even in summer
