"""Tests for the PSF skyglow propagation model."""

import numpy as np
import pytest

from stargazing_spots.skyglow import (
    build_psf_kernel,
    compute_sky_brightness,
    brightness_to_sqm,
    PropagationConfig,
    CALIB_FACTOR,
    EXTINCTION_RATE,
)


class TestBuildPSFKernel:
    """Tests for PSF kernel construction."""

    def test_kernel_shape_is_odd(self):
        psf = build_psf_kernel(radius_km=10, pixel_size_km=0.5)
        assert psf.shape[0] % 2 == 1
        assert psf.shape[1] % 2 == 1

    def test_kernel_center_not_zero(self):
        """Softened PSF should have finite value at center (self-pixel)."""
        psf = build_psf_kernel(radius_km=10, pixel_size_km=0.5, softening_km=1.0)
        center = psf.shape[0] // 2
        assert psf[center, center] > 0, "Center should NOT be zeroed (self-pixel contribution)"

    def test_kernel_center_value(self):
        """At d=0 with softening d₀=1: PSF = (0+1)^(-2.5) × exp(0) = 1.0."""
        psf = build_psf_kernel(radius_km=10, pixel_size_km=0.5, softening_km=1.0, exponent=2.5)
        center = psf.shape[0] // 2
        assert abs(psf[center, center] - 1.0) < 0.01

    def test_kernel_decays_with_distance(self):
        """PSF should decrease monotonically from center."""
        psf = build_psf_kernel(radius_km=50, pixel_size_km=1.0)
        center = psf.shape[0] // 2
        # Check along a row
        assert psf[center, center] > psf[center, center + 5]
        assert psf[center, center + 5] > psf[center, center + 20]

    def test_kernel_zero_beyond_radius(self):
        """PSF should be zero outside the integration radius."""
        psf = build_psf_kernel(radius_km=10, pixel_size_km=0.5)
        # Corner pixel is at distance sqrt(2) * radius > radius
        assert psf[0, 0] == 0.0

    def test_kernel_symmetric(self):
        """PSF kernel should be radially symmetric."""
        psf = build_psf_kernel(radius_km=20, pixel_size_km=1.0)
        center = psf.shape[0] // 2
        # Compare opposite directions
        assert abs(psf[center, center + 5] - psf[center, center - 5]) < 1e-10
        assert abs(psf[center + 5, center] - psf[center - 5, center]) < 1e-10

    def test_extinction_reduces_far_field(self):
        """Atmospheric extinction should make far pixels dimmer."""
        psf_no_ext = build_psf_kernel(radius_km=50, pixel_size_km=1.0, extinction_rate=0.0)
        psf_with_ext = build_psf_kernel(radius_km=50, pixel_size_km=1.0, extinction_rate=0.02)
        center = psf_no_ext.shape[0] // 2
        # At 30km, extinction should reduce PSF
        assert psf_with_ext[center, center + 30] < psf_no_ext[center, center + 30]


class TestComputeSkyBrightness:
    """Tests for the full PSF convolution."""

    @pytest.fixture
    def dark_radiance(self):
        """A 50x50 grid of zeros (perfectly dark)."""
        return np.zeros((50, 50))

    @pytest.fixture
    def point_source(self):
        """A 50x50 grid with a single bright pixel in the center."""
        r = np.zeros((50, 50))
        r[25, 25] = 100.0  # bright point source
        return r

    def test_dark_grid_gives_zero_brightness(self, dark_radiance):
        config = PropagationConfig(integration_radius_km=10, pixel_size_km=1.0)
        sky = compute_sky_brightness(dark_radiance, config)
        assert np.nanmax(sky) < 0.001

    def test_point_source_creates_halo(self, point_source):
        config = PropagationConfig(integration_radius_km=20, pixel_size_km=1.0)
        sky = compute_sky_brightness(point_source, config)
        # Brightness should be highest near the source
        assert sky[25, 25] > sky[25, 30]
        assert sky[25, 30] > sky[25, 40]

    def test_nan_pixels_contribute_zero(self):
        """NaN pixels (ocean) should not contribute to sky brightness."""
        r = np.full((50, 50), np.nan)
        r[25, 25] = 0.0  # single valid dark pixel
        config = PropagationConfig(integration_radius_km=10, pixel_size_km=1.0)
        sky = compute_sky_brightness(r, config)
        # Only the valid pixel region should have a result
        assert not np.isnan(sky[25, 25])

    def test_linearity(self):
        """Doubling radiance should double brightness (linear model)."""
        r1 = np.ones((30, 30)) * 5.0
        r2 = np.ones((30, 30)) * 10.0
        config = PropagationConfig(integration_radius_km=10, pixel_size_km=1.0)
        sky1 = compute_sky_brightness(r1, config)
        sky2 = compute_sky_brightness(r2, config)
        ratio = sky2[15, 15] / sky1[15, 15]
        assert abs(ratio - 2.0) < 0.1


class TestBrightnessToSQM:
    """Tests for mcd/m² to SQM conversion."""

    def test_natural_sky(self):
        """Zero artificial brightness → natural sky = 22.0 mag/arcsec²."""
        sqm = brightness_to_sqm(np.array([0.0]))
        assert abs(sqm[0] - 22.0) < 0.1

    def test_brighter_is_lower_sqm(self):
        """More artificial light → lower SQM number."""
        sqm = brightness_to_sqm(np.array([0.01, 0.1, 1.0, 10.0]))
        assert sqm[0] > sqm[1] > sqm[2] > sqm[3]

    def test_urban_brightness(self):
        """10 mcd/m² artificial → SQM ~17-18 (urban)."""
        sqm = brightness_to_sqm(np.array([10.0]))
        assert 17.0 < sqm[0] < 18.5


class TestCalibration:
    """Tests for calibration constant validity."""

    def test_calibration_factor_positive(self):
        assert CALIB_FACTOR > 0

    def test_calibration_factor_reasonable(self):
        """Factor should be in range that produces mcd/m² from typical integrals."""
        # Typical dark site raw integral: 0.3-2.0
        # Expected artificial: 0.01-0.1 mcd/m²
        art_dark = CALIB_FACTOR * 1.0
        assert 0.001 < art_dark < 1.0

    def test_extinction_rate_positive(self):
        assert EXTINCTION_RATE > 0
        assert EXTINCTION_RATE < 0.1  # reasonable range
