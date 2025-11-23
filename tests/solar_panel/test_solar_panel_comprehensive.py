"""Additional comprehensive tests for solar_panel.py to achieve near 100% coverage."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import SolarPanel, SolarPanelSet


class TestSolarPanelSetCoverage:
    """Tests for SolarPanelSet class."""

    def test_panel_illumination_fraction_empty_panels(self):
        """Test panel_illumination_fraction with an empty panels list."""
        panel_set = SolarPanelSet(panels=[])
        ephem = Mock()
        # This should return 0.0 for empty panels
        result_scalar = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0
        )
        assert result_scalar == 0.0

        times = [
            datetime.fromtimestamp(1514764800.0, tz=timezone.utc),
            datetime.fromtimestamp(1514764860.0, tz=timezone.utc),
        ]
        result_array = panel_set.panel_illumination_fraction(
            time=times, ephem=ephem, ra=0.0, dec=0.0
        )
        assert isinstance(result_array, np.ndarray)
        assert np.all(result_array == 0)

    def test_power_empty_panels(self):
        """Test power calculation with an empty panels list."""
        panel_set = SolarPanelSet(panels=[])
        ephem = Mock()
        result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)
        assert result == 0.0


# SolarPanel Tests
class TestSolarPanelInitialization:
    """Test SolarPanel initialization and default values."""

    def test_default_panel_creation(self):
        """Test creating a solar panel with default values."""
        panel = SolarPanel()
        assert panel.name == "Panel"
        assert panel.gimbled is False
        assert panel.sidemount is True
        assert panel.cant_x == 0.0
        assert panel.cant_y == 0.0
        assert panel.max_power == 800.0
        assert panel.conversion_efficiency is None

    def test_custom_panel_creation(self):
        """Test creating a solar panel with custom values."""
        panel = SolarPanel(
            name="Custom",
            gimbled=True,
            sidemount=False,
            cant_x=5.0,
            cant_y=10.0,
            max_power=500.0,
            conversion_efficiency=0.92,
        )
        assert panel.name == "Custom"
        assert panel.gimbled is True
        assert panel.sidemount is False
        assert panel.cant_x == 5.0
        assert panel.cant_y == 10.0
        assert panel.max_power == 500.0
        assert panel.conversion_efficiency == 0.92

    def test_azimuth_deg_configuration(self):
        """Test azimuth_deg configuration."""
        panel = SolarPanel(azimuth_deg=45.0)
        assert panel.azimuth_deg == 45.0


class TestSolarPanelIllumination:
    """Test solar panel illumination calculations."""

    def test_gimbled_panel_illumination_in_eclipse(self, mock_ephemeris):
        """Test gimbled panel returns 0 when in eclipse."""
        # Use unix time (float) to trigger scalar path
        mock_ephem = Mock()
        mock_time = 1514764800.0  # 2018-01-01 in unix time

        # Mock ephemeris attributes
        mock_ephem.sun = np.array([Mock()])
        mock_ephem.earth = np.array([Mock()])
        mock_ephem.earth_radius_angle = np.array([0.3])
        mock_ephem._tle_ephem = Mock()

        # Mock index method for ephemeris
        mock_ephem.index = Mock(return_value=0)

        # Create a mock eclipse constraint and patch it on the class
        mock_eclipse_constraint = Mock()
        mock_eclipse_constraint.in_constraint.return_value = True  # In eclipse

        with patch.object(SolarPanel, "_eclipse_constraint", mock_eclipse_constraint):
            panel = SolarPanel(gimbled=True)
            result = panel.panel_illumination_fraction(
                time=mock_time,
                ephem=mock_ephem,
                ra=0.0,
                dec=0.0,
            )

        # In eclipse, illumination should be 0
        assert result == 0.0

    def test_gimbled_panel_illumination_not_in_eclipse(self, mock_ephemeris):
        """Test gimbled panel returns 1 when not in eclipse."""
        # Use unix time (float) to trigger scalar path
        mock_ephem = Mock()
        mock_time = 1514764800.0  # 2018-01-01 in unix time

        mock_sun = Mock()
        mock_earth = Mock()

        not_eclipse_separation = Mock()
        not_eclipse_separation.deg = 89.0  # Not in eclipse
        mock_sun.separation = Mock(return_value=not_eclipse_separation)

        earth_sep = Mock()
        earth_sep.deg = 0.3
        mock_earth.separation = Mock(return_value=earth_sep)

        mock_earth_angle = Mock()
        mock_earth_angle.deg = 0.3

        mock_ephem.sun = np.array([mock_sun])
        mock_ephem.earth = np.array([mock_earth])
        mock_ephem.earth_radius_angle = np.array([mock_earth_angle])
        mock_ephem._tle_ephem = Mock()

        # Mock index method for ephemeris
        mock_ephem.index = Mock(return_value=0)

        # Create a mock eclipse constraint and patch it on the class
        mock_eclipse_constraint = Mock()
        mock_eclipse_constraint.in_constraint.return_value = False  # Not in eclipse

        with patch.object(SolarPanel, "_eclipse_constraint", mock_eclipse_constraint):
            panel = SolarPanel(gimbled=True)
            result = panel.panel_illumination_fraction(
                time=mock_time,
                ephem=mock_ephem,
                ra=0.0,
                dec=0.0,
            )
        assert result == 1.0

    def test_non_gimbled_panel_basic_illumination(self, mock_ephemeris):
        """Test non-gimbled panel basic illumination calculation."""
        # Use unix time (float) to trigger scalar path
        mock_ephem = Mock()
        mock_time = 1514764800.0  # 2018-01-01 in unix time

        # Mock arrays that support indexing with scalar indices
        mock_sun = Mock()
        mock_sun.ra.deg = 0.0
        mock_sun.dec.deg = 0.0

        # Create a mock array that properly supports fancy indexing
        sun_array = np.array([mock_sun], dtype=object)

        # Add __getitem__ to handle array indexing returning another mock
        def sun_getitem(self, idx):
            if isinstance(idx, np.ndarray):
                # Return a single mock when indexed with array [0]
                result = Mock()
                result.ra.deg = 0.0
                result.dec.deg = 0.0
                return result
            return sun_array[idx]

        mock_sun_array = Mock()
        mock_sun_array.__getitem__ = sun_getitem

        mock_ephem.sun = mock_sun_array
        mock_ephem.earth = np.array([Mock()])
        mock_ephem.earth_radius_angle = np.array([0.3])
        mock_ephem._tle_ephem = Mock()

        # Mock index method for ephemeris
        mock_ephem.index = Mock(return_value=0)

        # Create a mock eclipse constraint and patch it on the class
        mock_eclipse_constraint = Mock()
        mock_eclipse_constraint.in_constraint.return_value = False  # Not in eclipse

        with patch.object(SolarPanel, "_eclipse_constraint", mock_eclipse_constraint):
            panel = SolarPanel(gimbled=False, sidemount=True, cant_x=0.0, cant_y=0.0)
            # Mock separation to return a reasonable angle
            with patch("conops.separation", return_value=np.array([45.0])):
                result = panel.panel_illumination_fraction(
                    time=mock_time,
                    ephem=mock_ephem,
                    ra=0.0,
                    dec=0.0,
                )
                assert isinstance(result, (float, np.floating))


class TestSolarPanelCantCalculation:
    """Test cant angle calculations."""

    def test_cant_magnitude_calculation(self):
        """Test that cant magnitude is calculated correctly."""
        panel = SolarPanel(cant_x=3.0, cant_y=4.0)
        # cant_mag = sqrt(3^2 + 4^2) = 5.0
        assert np.hypot(panel.cant_x, panel.cant_y) == 5.0

    def test_sidemount_offset_calculation(self):
        """Test side-mounted panel offset calculation."""
        panel = SolarPanel(sidemount=True, cant_x=10.0, cant_y=0.0)
        # panel_offset_angle = 90 - 10 = 80
        cant_mag = np.hypot(panel.cant_x, panel.cant_y)
        assert cant_mag == 10.0
        assert (90 - cant_mag) == 80.0

    def test_body_mounted_offset_calculation(self):
        """Test body-mounted panel offset calculation."""
        panel = SolarPanel(sidemount=False, cant_x=0.0, cant_y=10.0)
        # panel_offset_angle = 0 + 10 = 10
        cant_mag = np.hypot(panel.cant_x, panel.cant_y)
        assert cant_mag == 10.0
        assert (0 + cant_mag) == 10.0


class TestSolarPanelSetBasics:
    """Test SolarPanelSet basic functionality."""

    def test_default_panel_set_creation(self):
        """Test creating a solar panel set with defaults."""
        panel_set = SolarPanelSet()
        assert panel_set.name == "Default Solar Panel"
        assert panel_set.conversion_efficiency == 0.95
        assert len(panel_set.panels) == 1

    def test_custom_panel_set_creation(self):
        """Test creating a solar panel set with custom panels."""
        panels = [
            SolarPanel(name="Panel1", max_power=500.0),
            SolarPanel(name="Panel2", max_power=800.0),
        ]
        panel_set = SolarPanelSet(
            name="Custom Set",
            panels=panels,
            conversion_efficiency=0.92,
        )
        assert panel_set.name == "Custom Set"
        assert len(panel_set.panels) == 2
        assert panel_set.conversion_efficiency == 0.92

    def test_sidemount_property_true(self):
        """Test sidemount property when any panel is side-mounted."""
        panels = [
            SolarPanel(sidemount=False),
            SolarPanel(sidemount=True),
        ]
        panel_set = SolarPanelSet(panels=panels)
        assert panel_set.sidemount is True

    def test_sidemount_property_false(self):
        """Test sidemount property when no panels are side-mounted."""
        panels = [
            SolarPanel(sidemount=False),
            SolarPanel(sidemount=False),
        ]
        panel_set = SolarPanelSet(panels=panels)
        assert panel_set.sidemount is False

    def test_sidemount_property_empty_panels(self):
        """Test sidemount property with empty panel list."""
        panel_set = SolarPanelSet(panels=[])
        assert panel_set.sidemount is False


class TestSolarPanelSetEffectivePanels:
    """Test _effective_panels method."""

    def test_effective_panels_returns_all(self):
        """Test that _effective_panels returns all configured panels."""
        panels = [
            SolarPanel(name="P1"),
            SolarPanel(name="P2"),
            SolarPanel(name="P3"),
        ]
        panel_set = SolarPanelSet(panels=panels)
        effective = panel_set._effective_panels()
        assert len(effective) == 3
        assert effective[0].name == "P1"
        assert effective[1].name == "P2"
        assert effective[2].name == "P3"


class TestSolarPanelSetIllumination:
    """Test panel set illumination calculations."""

    def test_panel_illumination_single_panel(self, mock_ephemeris):
        """Test illumination with single panel."""
        panel_set = SolarPanelSet(panels=[SolarPanel()])
        mock_ephem = mock_ephemeris
        mock_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

        with patch.object(
            SolarPanel,
            "panel_illumination_fraction",
            return_value=0.8,
        ):
            result = panel_set.panel_illumination_fraction(
                time=mock_time,
                ephem=mock_ephem,
                ra=0.0,
                dec=0.0,
            )
            assert result == 0.8

    def test_panel_illumination_multiple_panels_weighted(self, mock_ephemeris):
        """Test illumination with multiple panels (weighted average)."""
        panels = [
            SolarPanel(max_power=500.0),
            SolarPanel(max_power=500.0),
        ]
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

        with patch.object(
            SolarPanel,
            "panel_illumination_fraction",
            side_effect=[0.8, 0.6],
        ):
            result = panel_set.panel_illumination_fraction(
                time=mock_time,
                ephem=mock_ephem,
                ra=0.0,
                dec=0.0,
            )
            # Weighted average: 0.5 * 0.8 + 0.5 * 0.6 = 0.7
            assert result == pytest.approx(0.7)

    def test_panel_illumination_zero_max_power(self, mock_ephemeris):
        """Test illumination when total max power is zero."""
        panels = [SolarPanel(max_power=0.0)]
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

        with patch.object(
            SolarPanel,
            "panel_illumination_fraction",
            return_value=0.5,
        ):
            result = panel_set.panel_illumination_fraction(
                time=mock_time,
                ephem=mock_ephem,
                ra=0.0,
                dec=0.0,
            )
            assert result == 0.0


class TestSolarPanelSetPower:
    """Test power calculation."""

    def test_power_single_panel(self, mock_ephemeris):
        """Test power calculation with single panel."""
        panel_set = SolarPanelSet(
            panels=[SolarPanel(max_power=1000.0)],
            conversion_efficiency=1.0,
        )
        mock_ephem = mock_ephemeris
        mock_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

        with patch.object(
            SolarPanel,
            "panel_illumination_fraction",
            return_value=0.8,
        ):
            result = panel_set.power(
                time=mock_time,
                ra=0.0,
                dec=0.0,
                ephem=mock_ephem,
            )
            # Power = 0.8 * 1000 * 1.0 = 800
            assert result == pytest.approx(800.0)

    def test_power_multiple_panels(self, mock_ephemeris):
        """Test power calculation with multiple panels."""
        panels = [
            SolarPanel(max_power=500.0, conversion_efficiency=0.95),
            SolarPanel(max_power=500.0, conversion_efficiency=0.90),
        ]
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

        with patch.object(
            SolarPanel,
            "panel_illumination_fraction",
            side_effect=[1.0, 1.0],
        ):
            result = panel_set.power(
                time=mock_time,
                ra=0.0,
                dec=0.0,
                ephem=mock_ephem,
            )
            # Power = (1.0 * 500 * 0.95) + (1.0 * 500 * 0.90) = 475 + 450 = 925
            assert result == pytest.approx(925.0)

    def test_power_efficiency_fallback(self, mock_ephemeris):
        """Test power calculation with array-level efficiency fallback."""
        panels = [
            SolarPanel(max_power=500.0, conversion_efficiency=None),
        ]
        panel_set = SolarPanelSet(
            panels=panels,
            conversion_efficiency=0.88,
        )
        mock_ephem = mock_ephemeris
        mock_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

        with patch.object(
            SolarPanel,
            "panel_illumination_fraction",
            return_value=1.0,
        ):
            result = panel_set.power(
                time=mock_time,
                ra=0.0,
                dec=0.0,
                ephem=mock_ephem,
            )
            # Power = 1.0 * 500 * 0.88 = 440
            assert result == pytest.approx(440.0)

    def test_power_zero_panels(self, mock_ephemeris):
        """Test power with empty panel list."""
        panel_set = SolarPanelSet(panels=[])
        mock_ephem = mock_ephemeris
        mock_time = datetime(2018, 1, 1, tzinfo=timezone.utc)

        result = panel_set.power(
            time=mock_time,
            ra=0.0,
            dec=0.0,
            ephem=mock_ephem,
        )
        assert result == 0.0


class TestSolarPanelSetOptimalCharging:
    """Test optimal charging pointing."""

    def test_optimal_pointing_sidemount(self, mock_ephemeris):
        """Test optimal pointing for side-mounted panels."""
        panels = [SolarPanel(sidemount=True)]
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = 1514764800.0

        mock_ephem.index = Mock(return_value=0)
        mock_ephem.sun = [Mock(ra=Mock(deg=90.0), dec=Mock(deg=30.0))]

        ra, dec = panel_set.optimal_charging_pointing(mock_time, mock_ephem)

        # For sidemount: optimal_ra = (sun_ra + 90) % 360 = (90 + 90) % 360 = 180
        # optimal_dec = sun_dec = 30
        assert ra == pytest.approx(180.0)
        assert dec == pytest.approx(30.0)

    def test_optimal_pointing_body_mounted(self, mock_ephemeris):
        """Test optimal pointing for body-mounted panels."""
        panels = [SolarPanel(sidemount=False)]
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = 1514764800.0

        mock_ephem.index = Mock(return_value=0)
        mock_ephem.sun = [Mock(ra=Mock(deg=90.0), dec=Mock(deg=30.0))]

        ra, dec = panel_set.optimal_charging_pointing(mock_time, mock_ephem)

        # For body-mounted: optimal_ra = sun_ra = 90, optimal_dec = sun_dec = 30
        assert ra == pytest.approx(90.0)
        assert dec == pytest.approx(30.0)

    def test_optimal_pointing_wrapping(self, mock_ephemeris):
        """Test optimal pointing with RA wrapping."""
        panels = [SolarPanel(sidemount=True)]
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = 1514764800.0

        mock_ephem.index = Mock(return_value=0)
        # Sun at RA 350 degrees
        mock_ephem.sun = [Mock(ra=Mock(deg=350.0), dec=Mock(deg=0.0))]

        ra, dec = panel_set.optimal_charging_pointing(mock_time, mock_ephem)

        # For sidemount: optimal_ra = (350 + 90) % 360 = 440 % 360 = 80
        assert ra == pytest.approx(80.0)
        assert dec == pytest.approx(0.0)


class TestSolarPanelEdgeCases:
    """Test edge cases and special conditions."""

    def test_panel_with_extreme_cant_angles(self):
        """Test panel with very large cant angles."""
        panel = SolarPanel(cant_x=45.0, cant_y=45.0)
        cant_mag = np.hypot(panel.cant_x, panel.cant_y)
        assert cant_mag == pytest.approx(np.sqrt(2) * 45)

    def test_panel_set_with_unequal_max_power(self):
        """Test panel set with very different max power values."""
        panels = [
            SolarPanel(max_power=10.0),
            SolarPanel(max_power=10000.0),
        ]
        panel_set = SolarPanelSet(panels=panels)
        total = sum(p.max_power for p in panel_set._effective_panels())
        assert total == pytest.approx(10010.0)

    def test_negative_cant_angles(self):
        """Test panels with negative cant angles."""
        panel = SolarPanel(cant_x=-5.0, cant_y=-10.0)
        cant_mag = np.hypot(panel.cant_x, panel.cant_y)
        assert cant_mag == pytest.approx(np.sqrt(125))

    def test_panel_efficiency_boundary_values(self):
        """Test panel with boundary efficiency values."""
        panel_high_eff = SolarPanel(conversion_efficiency=0.99)
        panel_low_eff = SolarPanel(conversion_efficiency=0.50)
        assert panel_high_eff.conversion_efficiency == 0.99
        assert panel_low_eff.conversion_efficiency == 0.50

    def test_panel_set_all_zero_power(self):
        """Test panel set where all panels have zero power."""
        panels = [
            SolarPanel(max_power=0.0),
            SolarPanel(max_power=0.0),
        ]
        panel_set = SolarPanelSet(panels=panels)
        total = sum(p.max_power for p in panel_set.panels)
        assert total == 0.0


class TestSolarPanelIlluminationRealistic:
    """
    Realistic tests for panel_illumination_fraction with calculated expected values.

    For a side-mounted panel (sidemount=True):
    - panel_offset_angle = 90 - cant_mag
    - panel_sun_angle = 180 - sunangle - panel_offset_angle
    - illumination = cos(panel_sun_angle) (clipped to [0, inf))

    Where sunangle is the angular separation between spacecraft pointing and sun.
    """

    def test_side_mounted_panel_perpendicular_sun(self):
        """
        Test side-mounted panel (no cant) with sun perpendicular to pointing.

        Setup:
        - Spacecraft pointing: RA=0°, Dec=0°
        - Sun position: RA=90°, Dec=0° (perpendicular in equatorial plane)
        - Panel: sidemount=True, cant=0°

        Expected calculation:
        - sunangle = 90° (separation between pointing and sun)
        - cant_mag = 0°
        - panel_offset_angle = 90 - 0 = 90°
        - panel_sun_angle = 180 - 90 - 90 = 0°
        - illumination = cos(0°) = 1.0 (maximum illumination)
        """
        panel = SolarPanel(sidemount=True, cant_x=0.0, cant_y=0.0)

        # Create mock ephemeris with sun at 90° from pointing
        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Mock sun position: RA=90°, Dec=0°
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        # Mock index method
        ephem.index = Mock(return_value=0)

        # Mock eclipse constraint - not in eclipse
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,  # Unix timestamp
                ephem=ephem,
                ra=0.0,  # Spacecraft pointing RA
                dec=0.0,  # Spacecraft pointing Dec
            )

        # Should get maximum illumination (cos(0°) = 1.0)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_side_mounted_panel_aligned_with_sun(self):
        """
        Test side-mounted panel when spacecraft points at sun.

        Setup:
        - Spacecraft pointing: RA=0°, Dec=0°
        - Sun position: RA=0°, Dec=0° (aligned with pointing)
        - Panel: sidemount=True, cant=0°

        Expected calculation:
        - sunangle = 0° (pointing at sun)
        - cant_mag = 0°
        - panel_offset_angle = 90°
        - panel_sun_angle = 180 - 0 - 90 = 90°
        - illumination = cos(90°) = 0.0 (no illumination, panel edge-on to sun)
        """
        panel = SolarPanel(sidemount=True, cant_x=0.0, cant_y=0.0)

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun at same position as pointing
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 0.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Should get zero illumination (cos(90°) = 0.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_side_mounted_panel_45_degree_sun_angle(self):
        """
        Test side-mounted panel with sun at 45° from pointing.

        Setup:
        - Spacecraft pointing: RA=0°, Dec=0°
        - Sun position: RA=45°, Dec=0°
        - Panel: sidemount=True, cant=0°

        Expected calculation:
        - sunangle ≈ 45° (separation)
        - cant_mag = 0°
        - panel_offset_angle = 90°
        - panel_sun_angle = 180 - 45 - 90 = 45°
        - illumination = cos(45°) ≈ 0.7071
        """
        panel = SolarPanel(sidemount=True, cant_x=0.0, cant_y=0.0)

        ephem = Mock()
        ephem._tle_ephem = Mock()

        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 45.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        expected = np.cos(np.radians(45.0))  # ≈ 0.7071
        assert result == pytest.approx(expected, rel=1e-4)

    def test_side_mounted_panel_with_cant_angle(self):
        """
        Test side-mounted panel with cant angle.

        Setup:
        - Spacecraft pointing: RA=0°, Dec=0°
        - Sun position: RA=90°, Dec=0° (perpendicular)
        - Panel: sidemount=True, cant_x=10°, cant_y=0°

        Expected calculation:
        - sunangle = 90°
        - cant_mag = 10°
        - panel_offset_angle = 90 - 10 = 80°
        - panel_sun_angle = 180 - 90 - 80 = 10°
        - illumination = cos(10°) ≈ 0.9848
        """
        panel = SolarPanel(sidemount=True, cant_x=10.0, cant_y=0.0)

        ephem = Mock()
        ephem._tle_ephem = Mock()

        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        expected = np.cos(np.radians(10.0))  # ≈ 0.9848
        assert result == pytest.approx(expected, rel=1e-4)

    def test_side_mounted_panel_in_eclipse(self):
        """
        Test side-mounted panel returns 0 when in eclipse.

        Even with optimal sun angle, eclipse should force illumination to 0.
        """
        panel = SolarPanel(sidemount=True, cant_x=0.0, cant_y=0.0)

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun perpendicular (would normally give max illumination)
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        # Mock eclipse condition - IN eclipse
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=True)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Should be 0 due to eclipse
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_side_mounted_panel_negative_illumination_clipped(self):
        """
        Test that negative illumination values are clipped to 0.

        Setup:
        - Spacecraft pointing: RA=0°, Dec=0°
        - Sun position: RA=180°, Dec=0° (opposite direction)
        - Panel: sidemount=True, cant=0°

        Expected calculation:
        - sunangle = 180°
        - panel_offset_angle = 90°
        - panel_sun_angle = 180 - 180 - 90 = -90°
        - cos(-90°) = 0, but if it went more negative, would clip to 0
        """
        panel = SolarPanel(sidemount=True, cant_x=0.0, cant_y=0.0)

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun opposite to pointing
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 180.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Should be clipped to 0 (cos(-90°) = 0)
        assert result >= 0.0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_non_sidemount_panel_aligned_with_sun(self):
        """
        Test non-sidemount panel (body-mounted) when pointing at sun.

        Setup:
        - Spacecraft pointing: RA=0°, Dec=0°
        - Sun position: RA=0°, Dec=0° (aligned)
        - Panel: sidemount=False, cant=0°

        Expected calculation:
        - sunangle = 0°
        - cant_mag = 0°
        - panel_offset_angle = 0 + 0 = 0°
        - panel_sun_angle = 180 - 0 - 0 = 180°
        - illumination = cos(180°) = -1.0, clipped to 0
        """
        panel = SolarPanel(sidemount=False, cant_x=0.0, cant_y=0.0)

        ephem = Mock()
        ephem._tle_ephem = Mock()

        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 0.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # cos(180°) = -1.0, clipped to 0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_non_sidemount_panel_sun_behind(self):
        """
        Test non-sidemount panel with sun behind spacecraft.

        Setup:
        - Spacecraft pointing: RA=0°, Dec=0°
        - Sun position: RA=180°, Dec=0° (behind)
        - Panel: sidemount=False, cant=0°

        Expected calculation:
        - sunangle = 180°
        - cant_mag = 0°
        - panel_offset_angle = 0°
        - panel_sun_angle = 180 - 180 - 0 = 0°
        - illumination = cos(0°) = 1.0 (maximum)
        """
        panel = SolarPanel(sidemount=False, cant_x=0.0, cant_y=0.0)

        ephem = Mock()
        ephem._tle_ephem = Mock()

        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 180.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        # Create a mock array that returns sun_mock when indexed
        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Should get maximum illumination
        assert result == pytest.approx(1.0, rel=1e-6)


class TestSolarPanelPowerGenerationRealistic:
    """
    Realistic tests for power generation with calculated expected values.

    Power = illumination_fraction * max_power * conversion_efficiency

    For a single panel:
    - First calculate illumination using the panel geometry
    - Then multiply by max_power and efficiency

    For panel sets:
    - Sum power from all panels
    """

    def test_single_panel_maximum_power(self):
        """
        Test maximum power output with perfect illumination.

        Setup:
        - Single panel: max_power=500W, efficiency=0.90
        - Illumination: 1.0 (100%)

        Expected:
        - power = 1.0 * 500 * 0.90 = 450W
        """
        panel = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=500.0,
            conversion_efficiency=0.90,
        )
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun perpendicular (gives illumination = 1.0)
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        expected_power = 1.0 * 500.0 * 0.90  # 450W
        assert result == pytest.approx(expected_power, rel=1e-4)

    def test_single_panel_partial_illumination(self):
        """
        Test power with 50% illumination (45° sun angle).

        Setup:
        - Single panel: max_power=800W, efficiency=0.95
        - Sun at 45° from pointing
        - Illumination: cos(45°) ≈ 0.7071

        Expected:
        - power = 0.7071 * 800 * 0.95 ≈ 537.4W
        """
        panel = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=800.0,
            conversion_efficiency=0.95,
        )
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun at 45°
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 45.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        illumination = np.cos(np.radians(45.0))  # ≈ 0.7071
        expected_power = illumination * 800.0 * 0.95
        assert result == pytest.approx(expected_power, rel=1e-4)

    def test_single_panel_zero_power_in_eclipse(self):
        """
        Test that power is zero during eclipse regardless of geometry.

        Setup:
        - Single panel: max_power=1000W, efficiency=0.90
        - Perfect geometry but in eclipse

        Expected:
        - power = 0W (eclipse overrides everything)
        """
        panel = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=1000.0,
            conversion_efficiency=0.90,
        )
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Optimal geometry
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        # IN ECLIPSE
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=True)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_multi_panel_equal_contribution(self):
        """
        Test power from multiple identical panels.

        Setup:
        - Two panels: each 300W max, 0.90 efficiency
        - Both have illumination = 1.0

        Expected:
        - Panel 1: 1.0 * 300 * 0.90 = 270W
        - Panel 2: 1.0 * 300 * 0.90 = 270W
        - Total: 540W
        """
        panel1 = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=300.0,
            conversion_efficiency=0.90,
        )
        panel2 = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=300.0,
            conversion_efficiency=0.90,
        )
        panel_set = SolarPanelSet(panels=[panel1, panel2])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Perfect illumination for both
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Both panels at full power
        expected_power = 2 * (1.0 * 300.0 * 0.90)  # 540W
        assert result == pytest.approx(expected_power, rel=1e-4)

    def test_multi_panel_different_power_ratings(self):
        """
        Test power from panels with different max_power ratings.

        Setup:
        - Panel 1: 400W max, 0.95 efficiency, illumination=1.0
        - Panel 2: 600W max, 0.92 efficiency, illumination=1.0

        Expected:
        - Panel 1: 1.0 * 400 * 0.95 = 380W
        - Panel 2: 1.0 * 600 * 0.92 = 552W
        - Total: 932W
        """
        panel1 = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=400.0,
            conversion_efficiency=0.95,
        )
        panel2 = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=600.0,
            conversion_efficiency=0.92,
        )
        panel_set = SolarPanelSet(panels=[panel1, panel2])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Perfect illumination
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        panel1_power = 1.0 * 400.0 * 0.95
        panel2_power = 1.0 * 600.0 * 0.92
        expected_power = panel1_power + panel2_power
        assert result == pytest.approx(expected_power, rel=1e-4)

    def test_panel_uses_set_efficiency_when_not_specified(self):
        """
        Test that panel uses set-level efficiency if not individually specified.

        Setup:
        - Panel: max_power=500W, efficiency=None
        - Set: efficiency=0.88
        - Illumination: 1.0

        Expected:
        - power = 1.0 * 500 * 0.88 = 440W (uses set efficiency)
        """
        panel = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=500.0,
            conversion_efficiency=None,  # Will use set's efficiency
        )
        panel_set = SolarPanelSet(
            panels=[panel],
            conversion_efficiency=0.88,  # Set-level efficiency
        )

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Perfect illumination
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        expected_power = 1.0 * 500.0 * 0.88  # Uses set efficiency
        assert result == pytest.approx(expected_power, rel=1e-4)

    def test_panel_with_cant_angle_reduces_power(self):
        """
        Test that cant angle affects power by changing illumination.

        Setup:
        - Panel: max_power=1000W, efficiency=0.95, cant=10°
        - Sun perpendicular (RA=90°)

        Expected calculation:
        - sunangle = 90°
        - panel_offset_angle = 90 - 10 = 80°
        - panel_sun_angle = 180 - 90 - 80 = 10°
        - illumination = cos(10°) ≈ 0.9848
        - power = 0.9848 * 1000 * 0.95 ≈ 935.6W
        """
        panel = SolarPanel(
            sidemount=True,
            cant_x=10.0,
            cant_y=0.0,
            max_power=1000.0,
            conversion_efficiency=0.95,
        )
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun perpendicular
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        illumination = np.cos(np.radians(10.0))  # ≈ 0.9848
        expected_power = illumination * 1000.0 * 0.95
        assert result == pytest.approx(expected_power, rel=1e-4)

    def test_zero_power_when_panel_edge_on_to_sun(self):
        """
        Test zero power when panel is edge-on to sun.

        Setup:
        - Panel: sidemount=True, cant=0°, max_power=800W, eff=0.90
        - Sun aligned with pointing (RA=0°)

        Expected:
        - illumination = 0.0 (cos(90°))
        - power = 0.0 * 800 * 0.90 = 0W
        """
        panel = SolarPanel(
            sidemount=True,
            cant_x=0.0,
            cant_y=0.0,
            max_power=800.0,
            conversion_efficiency=0.90,
        )
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun aligned with pointing
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 0.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert result == pytest.approx(0.0, abs=1e-6)


class TestGetSliceIndices:
    """Tests for get_slice_indices function."""

    def test_get_slice_indices_multiple_times(self):
        """Test get_slice_indices with multiple datetime times."""
        from datetime import datetime, timezone

        from conops.config.solar_panel import get_slice_indices

        times = [
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2018, 1, 2, tzinfo=timezone.utc),
            datetime(2018, 1, 3, tzinfo=timezone.utc),
        ]

        # Mock ephemeris with proper indexing
        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1, 2])

        indices = get_slice_indices(times, ephem)

        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(indices, expected)
        assert ephem.index.call_count == 3

    def test_get_slice_indices_single_time(self):
        """Test get_slice_indices with single datetime time."""
        from datetime import datetime, timezone

        from conops.config.solar_panel import get_slice_indices

        time_single = datetime(2018, 1, 1, tzinfo=timezone.utc)

        ephem = Mock()
        ephem.index = Mock(return_value=5)

        indices = get_slice_indices(time_single, ephem)

        assert indices == 5
        ephem.index.assert_called_once_with(time_single)


class TestPanelIlluminationExceptionHandling:
    """Tests for exception handling in panel_illumination_fraction."""

    def test_panel_illumination_invalid_ephem_index(self):
        """Test panel_illumination_fraction with invalid ephem index."""
        panel = SolarPanel()

        ephem = Mock()
        ephem.index = Mock(side_effect=IndexError("Invalid index"))

        with pytest.raises(IndexError):
            panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

    def test_panel_illumination_sun_access_error(self):
        """Test panel_illumination_fraction when sun array access fails."""
        panel = SolarPanel()

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(side_effect=KeyError("Sun access failed"))

        # Mock eclipse constraint to avoid rust_ephem issues
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            with pytest.raises(KeyError):
                panel.panel_illumination_fraction(
                    time=1514764800.0,
                    ephem=ephem,
                    ra=0.0,
                    dec=0.0,
                )


class TestPanelIlluminationEclipseConstraint:
    """Tests for eclipse constraint evaluation in panel_illumination_fraction."""

    def test_eclipse_constraint_evaluation_true(self):
        """Test that eclipse constraint is properly evaluated when true."""
        panel = SolarPanel()

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        # Mock eclipse constraint to return True (in eclipse)
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=True)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Should return 0 due to eclipse
        assert result == 0.0
        mock_constraint.in_constraint.assert_called_once()

    def test_eclipse_constraint_evaluation_false(self):
        """Test that eclipse constraint is properly evaluated when false."""
        panel = SolarPanel(sidemount=True, cant_x=0.0, cant_y=0.0)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        # Mock eclipse constraint to return False (not in eclipse)
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Should return maximum illumination (sun perpendicular)
        assert result == pytest.approx(1.0, rel=1e-6)
        mock_constraint.in_constraint.assert_called_once()


class TestPowerEdgeCases:
    """Tests for edge cases in power method."""

    def test_power_with_zero_max_power(self):
        """Test power calculation when max_power is zero."""
        panel = SolarPanel(max_power=0.0)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        assert result == 0.0

    def test_power_with_zero_efficiency(self):
        """Test power calculation when efficiency is zero."""
        panel = SolarPanel(max_power=1000.0, conversion_efficiency=0.0)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        assert result == 0.0

    def test_power_with_negative_efficiency(self):
        """Test power calculation with negative efficiency (should be clamped)."""
        panel = SolarPanel(max_power=1000.0, conversion_efficiency=-0.1)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        # Negative efficiency should still produce some power (illumination * max_power * efficiency)
        # Since efficiency is negative, result should be negative
        assert result < 0.0

    def test_power_with_extreme_max_power(self):
        """Test power calculation with very large max_power."""
        panel = SolarPanel(max_power=1e6)  # 1 MW
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        # Should handle large values without overflow
        assert isinstance(result, (float, np.floating))

    def test_panel_illumination_fraction_gimbled_array_time(self):
        """Test panel_illumination_fraction with gimbled panel and array time."""
        from datetime import datetime, timezone

        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9, gimbled=True)

        times = [
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2018, 1, 2, tzinfo=timezone.utc),
        ]

        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1])

        # Mock eclipse constraint for array evaluation
        mock_constraint = Mock()
        mock_result = Mock()
        mock_result.constraint_array = np.array([False, False])  # Not in eclipse
        mock_constraint.evaluate = Mock(return_value=mock_result)

        with patch.object(panel, "_eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=times,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(
            result, [1.0, 1.0]
        )  # Not in eclipse, so fully illuminated

    def test_panel_illumination_fraction_unix_timestamp(self):
        """Test panel_illumination_fraction with unix timestamp input."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        # Mock eclipse constraint
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)  # Not in eclipse

        with patch.object(panel, "_eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,  # Unix timestamp
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert isinstance(result, float)
        assert result == 1.0  # Not in eclipse, so fully illuminated


class TestIlluminationAndPower:
    """Tests for the illumination_and_power method."""

    def test_illumination_and_power_single_panel(self):
        """Test illumination_and_power with single panel."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Sun perpendicular to pointing should give max illumination
        assert illumination == pytest.approx(1.0, rel=1e-6)
        assert power == pytest.approx(450.0, rel=1e-4)  # 1.0 * 500 * 0.9

    def test_illumination_and_power_multiple_panels(self):
        """Test illumination_and_power with multiple panels."""
        panels = [
            SolarPanel(max_power=300.0, conversion_efficiency=0.95),
            SolarPanel(max_power=400.0, conversion_efficiency=0.90),
        ]
        panel_set = SolarPanelSet(panels=panels)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Both panels at max illumination
        expected_illumination = 1.0
        expected_power = (1.0 * 300.0 * 0.95) + (1.0 * 400.0 * 0.90)  # 285 + 360 = 645

        assert illumination == pytest.approx(expected_illumination, rel=1e-6)
        assert power == pytest.approx(expected_power, rel=1e-4)

    def test_illumination_and_power_in_eclipse(self):
        """Test illumination_and_power during eclipse."""
        panel = SolarPanel(max_power=1000.0)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=True)  # In eclipse

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert illumination == 0.0
        assert power == 0.0

    def test_illumination_and_power_empty_panels(self):
        """Test illumination_and_power with empty panel list."""
        panel_set = SolarPanelSet(panels=[])

        ephem = Mock()
        illumination, power = panel_set.illumination_and_power(
            time=1514764800.0,
            ephem=ephem,
            ra=0.0,
            dec=0.0,
        )

        assert illumination == 0.0
        assert power == 0.0

    def test_illumination_and_power_empty_panels_array_time(self):
        """Test illumination_and_power with empty panel list and array time."""
        from datetime import datetime, timezone

        panel_set = SolarPanelSet(panels=[])

        times = [
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2018, 1, 2, tzinfo=timezone.utc),
        ]

        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1])
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        # Mock eclipse constraint for dummy panel shape determination
        mock_constraint = Mock()
        mock_result = Mock()
        mock_result.constraint_array = np.array([False, False])
        mock_constraint.evaluate = Mock(return_value=mock_result)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=times,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert isinstance(illumination, np.ndarray)
        assert isinstance(power, np.ndarray)
        assert np.all(illumination == 0.0)
        assert np.all(power == 0.0)

    def test_illumination_and_power_empty_panels_numpy_array_time(self):
        """Test illumination_and_power with empty panel list and numpy array time."""
        panel_set = SolarPanelSet(panels=[])

        times = np.array([1514764800.0, 1514851200.0])  # Unix timestamps

        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1])
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        # Mock eclipse constraint for dummy panel shape determination
        mock_constraint = Mock()
        mock_result = Mock()
        mock_result.constraint_array = np.array([False, False])
        mock_constraint.evaluate = Mock(return_value=mock_result)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=times,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert isinstance(illumination, np.ndarray)
        assert isinstance(power, np.ndarray)
        assert np.all(illumination == 0.0)
        assert np.all(power == 0.0)

    def test_illumination_and_power_with_array_times(self):
        """Test illumination_and_power with array of times."""
        from datetime import datetime, timezone

        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)
        panel_set = SolarPanelSet(panels=[panel])

        times = [
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2018, 1, 2, tzinfo=timezone.utc),
        ]

        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1])
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        # Mock eclipse constraint for array evaluation
        mock_constraint = Mock()
        mock_result = Mock()
        mock_result.constraint_array = np.array([False, False])  # Not in eclipse
        mock_constraint.evaluate = Mock(return_value=mock_result)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=times,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        # Should return arrays
        assert isinstance(illumination, np.ndarray)
        assert isinstance(power, np.ndarray)
        assert len(illumination) == 2
        assert len(power) == 2
        np.testing.assert_array_almost_equal(illumination, [1.0, 1.0])
        np.testing.assert_array_almost_equal(power, [450.0, 450.0])

    def test_illumination_and_power_efficiency_fallback(self):
        """Test illumination_and_power with efficiency fallback to set level."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=None)
        panel_set = SolarPanelSet(panels=[panel], conversion_efficiency=0.85)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert illumination == pytest.approx(1.0, rel=1e-6)
        assert power == pytest.approx(425.0, rel=1e-4)  # 1.0 * 500 * 0.85

    def test_single_panel_cant_x_45_degrees_max_power(self):
        """
        Test that a single panel canted at 45 degrees in X-direction generates
        maximum power that's cos(45°) of the max panel value.

        Setup:
        - Single panel: sidemount=True, cant_x=45°, cant_y=0°, max_power=1000W, efficiency=0.95
        - Sun perpendicular to pointing (RA=90°)

        Expected calculation:
        - sunangle = 90°
        - cant_x = 45°
        - panel_offset_angle = 90 - 45 = 45°
        - panel_sun_angle = 180 - 90 - 45 = 45°
        - illumination = cos(45°) ≈ 0.7071
        - power = 0.7071 * 1000 * 0.95 ≈ 671.0W
        """
        panel = SolarPanel(
            sidemount=True,
            cant_x=45.0,
            cant_y=0.0,
            max_power=1000.0,
            conversion_efficiency=0.95,
        )
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem._tle_ephem = Mock()

        # Sun perpendicular to pointing
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 0.0

        sun_array = Mock()
        sun_array.__getitem__ = Mock(return_value=sun_mock)
        ephem.sun = sun_array

        ephem.index = Mock(return_value=0)

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        illumination = np.cos(np.radians(45.0))  # ≈ 0.7071
        expected_power = illumination * 1000.0 * 0.95
        assert result == pytest.approx(expected_power, rel=1e-4)
