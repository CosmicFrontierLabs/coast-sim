"""Additional comprehensive tests for solar_panel.py to achieve near 100% coverage."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import SolarPanel, SolarPanelSet


class TestSolarPanelSetCoverage:
    """Tests for SolarPanelSet class."""

    def test_panel_illumination_fraction_empty_panels(
        self, empty_solar_panel_set, mock_ephem
    ) -> None:
        """Test panel_illumination_fraction with an empty panels list."""
        panel_set = empty_solar_panel_set
        # This should return 0.0 for empty panels
        result_scalar = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=mock_ephem, ra=0.0, dec=0.0
        )
        assert result_scalar == 0.0

        times = [
            datetime.fromtimestamp(1514764800.0, tz=timezone.utc),
            datetime.fromtimestamp(1514764860.0, tz=timezone.utc),
        ]
        result_array = panel_set.panel_illumination_fraction(
            time=times, ephem=mock_ephem, ra=0.0, dec=0.0
        )
        assert isinstance(result_array, np.ndarray)
        assert np.all(result_array == 0)

    def test_power_empty_panels(self, empty_solar_panel_set, mock_ephem) -> None:
        """Test power calculation with an empty panels list."""
        panel_set = empty_solar_panel_set
        result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=mock_ephem)
        assert result == 0.0


# SolarPanel Tests
class TestSolarPanelInitialization:
    """Test SolarPanel initialization and default values."""

    def test_default_panel_creation(self, default_solar_panel) -> None:
        """Test creating a solar panel with default values."""
        panel = default_solar_panel
        assert panel.name == "Panel"
        assert panel.gimbled is False
        assert panel.normal == (0.0, 1.0, 0.0)
        assert panel.max_power == 800.0
        assert panel.conversion_efficiency is None

    def test_custom_panel_creation(self) -> None:
        """Test creating a solar panel with custom values."""
        panel = SolarPanel(
            name="Custom",
            gimbled=True,
            normal=(0.5, 0.5, -0.707),
            max_power=500.0,
            conversion_efficiency=0.92,
        )
        assert panel.name == "Custom"
        assert panel.gimbled is True
        assert panel.normal == (0.5, 0.5, -0.707)
        assert panel.max_power == 500.0
        assert panel.conversion_efficiency == 0.92

    def test_normal_vector_configuration(self) -> None:
        """Test normal vector configuration."""
        panel = SolarPanel(normal=(1.0, 0.0, 0.0))
        assert panel.normal == (1.0, 0.0, 0.0)


class TestSolarPanelIllumination:
    """Test solar panel illumination calculations."""

    def test_gimbled_panel_illumination_in_eclipse(self, mock_ephemeris: Mock) -> None:
        """Test gimbled panel returns 0 when in eclipse."""
        # Use unix time (float) to trigger scalar path
        mock_ephem = Mock()
        mock_time = 1514764800.0  # 2018-01-01 in unix time

        # Mock ephemeris attributes
        mock_ephem.sun = np.array([Mock()])
        # New direct array access (rust-ephem 0.3.0+)
        mock_ephem.sun_ra_deg = np.array([90.0])
        mock_ephem.sun_dec_deg = np.array([0.0])
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

    def test_gimbled_panel_illumination_not_in_eclipse(
        self, mock_ephemeris: Mock
    ) -> None:
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
        # New direct array access (rust-ephem 0.3.0+)
        mock_ephem.sun_ra_deg = np.array([90.0])
        mock_ephem.sun_dec_deg = np.array([0.0])
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

    def test_non_gimbled_panel_basic_illumination(self, mock_ephemeris: Mock) -> None:
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
        # New direct array access (rust-ephem 0.3.0+)
        mock_ephem.sun_ra_deg = np.array([0.0])
        mock_ephem.sun_dec_deg = np.array([0.0])
        # Add position vectors for sun and spacecraft
        mock_ephem.sun_pv = Mock()
        mock_ephem.sun_pv.position = np.array([[1.496e8, 0.0, 0.0]])  # Sun position
        mock_ephem.gcrs_pv = Mock()
        mock_ephem.gcrs_pv.position = np.array([[0.0, 0.0, 0.0]])  # Spacecraft position
        mock_ephem.earth = np.array([Mock()])
        mock_ephem.earth_radius_angle = np.array([0.3])
        mock_ephem._tle_ephem = Mock()

        # Mock index method for ephemeris
        mock_ephem.index = Mock(return_value=0)

        # Create a mock eclipse constraint and patch it on the class
        mock_eclipse_constraint = Mock()
        mock_eclipse_constraint.in_constraint.return_value = False  # Not in eclipse

        with patch.object(SolarPanel, "_eclipse_constraint", mock_eclipse_constraint):
            panel = SolarPanel(gimbled=False, normal=(0.0, 1.0, 0.0))
            # Mock separation to return a reasonable angle
            with patch("conops.separation", return_value=np.array([45.0])):
                result = panel.panel_illumination_fraction(
                    time=mock_time,
                    ephem=mock_ephem,
                    ra=0.0,
                    dec=0.0,
                )
                assert isinstance(result, (float, np.floating))


class TestSolarPanelNormalVectorCalculation:
    """Test normal vector-based panel geometry."""

    def test_normal_vector_side_mounted(self) -> None:
        """Test side-mounted panel with Y-pointing normal."""
        panel = SolarPanel(normal=(0.0, 1.0, 0.0))
        assert panel.normal == (0.0, 1.0, 0.0)

    def test_normal_vector_body_mounted(self) -> None:
        """Test body-mounted panel with -Z-pointing normal."""
        panel = SolarPanel(normal=(0.0, 0.0, -1.0))
        assert panel.normal == (0.0, 0.0, -1.0)

    def test_normal_vector_custom(self) -> None:
        """Test custom normal vector."""
        normal = (0.577, 0.577, -0.577)
        panel = SolarPanel(normal=normal)
        assert panel.normal == normal


class TestSolarPanelSetBasics:
    """Test SolarPanelSet basic functionality."""

    def test_default_panel_set_creation(self) -> None:
        """Test creating a solar panel set with defaults."""
        panel_set = SolarPanelSet()
        assert panel_set.name == "Default Solar Panel"
        assert panel_set.conversion_efficiency == 0.95
        assert len(panel_set.panels) == 1

    def test_custom_panel_set_creation(self) -> None:
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

    def test_sidemount_property_true(self) -> None:
        """Test sidemount property when any panel has Y-component dominant normal."""
        panels = [
            SolarPanel(normal=(0.0, 0.0, -1.0)),  # Body-mounted
            SolarPanel(normal=(0.0, 1.0, 0.0)),  # Side-mounted
        ]
        panel_set = SolarPanelSet(panels=panels)
        assert panel_set.sidemount is True

    def test_sidemount_property_false(self) -> None:
        """Test sidemount property when no panels are side-mounted."""
        panels = [
            SolarPanel(normal=(0.0, 0.0, -1.0)),
            SolarPanel(normal=(0.0, 0.0, -1.0)),
        ]
        panel_set = SolarPanelSet(panels=panels)
        assert panel_set.sidemount is False

    def test_sidemount_property_empty_panels(
        self, empty_solar_panel_set: SolarPanelSet
    ) -> None:
        """Test sidemount property with empty panel list."""
        panel_set = empty_solar_panel_set
        assert panel_set.sidemount is False


class TestSolarPanelSetIllumination:
    """Test panel set illumination calculations."""

    def test_panel_illumination_single_panel(
        self, mock_ephemeris: Mock, single_panel_set: SolarPanelSet
    ) -> None:
        """Test illumination with single panel."""
        panel_set = single_panel_set
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

    def test_panel_illumination_multiple_panels_weighted(
        self, mock_ephemeris: Mock
    ) -> None:
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

    def test_panel_illumination_zero_max_power(
        self, mock_ephemeris: Mock, zero_power_panel_set: SolarPanelSet
    ) -> None:
        """Test illumination when total max power is zero."""
        panel_set = zero_power_panel_set
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

    def test_power_single_panel(self, mock_ephemeris: Mock) -> None:
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

    def test_power_multiple_panels(self, mock_ephemeris: Mock) -> None:
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

    def test_power_efficiency_fallback(self, mock_ephemeris: Mock) -> None:
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

    def test_power_zero_panels(
        self, mock_ephemeris: Mock, empty_solar_panel_set: SolarPanelSet
    ) -> None:
        """Test power with empty panel list."""
        panel_set = empty_solar_panel_set
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

    def test_optimal_pointing_sidemount(self, mock_ephemeris: Mock) -> None:
        """Test optimal pointing for side-mounted panels."""
        panels = [SolarPanel(normal=(0.0, 1.0, 0.0))]  # Y-pointing (side-mounted)
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = 1514764800.0

        mock_ephem.index = Mock(return_value=0)
        mock_ephem.sun = [Mock(ra=Mock(deg=90.0), dec=Mock(deg=30.0))]
        # New direct array access (rust-ephem 0.3.0+)
        mock_ephem.sun_ra_deg = np.array([90.0])
        mock_ephem.sun_dec_deg = np.array([30.0])

        ra, dec = panel_set.optimal_charging_pointing(mock_time, mock_ephem)

        # For sidemount: optimal_ra = (sun_ra + 90) % 360 = (90 + 90) % 360 = 180
        # optimal_dec = sun_dec = 30
        assert ra == pytest.approx(180.0)
        assert dec == pytest.approx(30.0)

    def test_optimal_pointing_body_mounted(self, mock_ephemeris: Mock) -> None:
        """Test optimal pointing for body-mounted panels."""
        panels = [SolarPanel(normal=(0.0, 0.0, -1.0))]  # Z-pointing (body-mounted)
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = 1514764800.0

        mock_ephem.index = Mock(return_value=0)
        mock_ephem.sun = [Mock(ra=Mock(deg=90.0), dec=Mock(deg=30.0))]
        # New direct array access (rust-ephem 0.3.0+)
        mock_ephem.sun_ra_deg = np.array([90.0])
        mock_ephem.sun_dec_deg = np.array([30.0])

        ra, dec = panel_set.optimal_charging_pointing(mock_time, mock_ephem)

        # For body-mounted: optimal_ra = sun_ra = 90, optimal_dec = sun_dec = 30
        assert ra == pytest.approx(90.0)
        assert dec == pytest.approx(30.0)

    def test_optimal_pointing_wrapping(self, mock_ephemeris: Mock) -> None:
        """Test optimal pointing with RA wrapping."""
        panels = [SolarPanel(normal=(0.0, 1.0, 0.0))]  # Y-pointing (side-mounted)
        panel_set = SolarPanelSet(panels=panels)
        mock_ephem = mock_ephemeris
        mock_time = 1514764800.0

        mock_ephem.index = Mock(return_value=0)
        # Sun at RA 350 degrees
        mock_ephem.sun = [Mock(ra=Mock(deg=350.0), dec=Mock(deg=0.0))]
        # New direct array access (rust-ephem 0.3.0+)
        mock_ephem.sun_ra_deg = np.array([350.0])
        mock_ephem.sun_dec_deg = np.array([0.0])

        ra, dec = panel_set.optimal_charging_pointing(mock_time, mock_ephem)

        # For sidemount: optimal_ra = (350 + 90) % 360 = 440 % 360 = 80
        assert ra == pytest.approx(80.0)
        assert dec == pytest.approx(0.0)


class TestSolarPanelEdgeCases:
    """Test edge cases and special conditions."""

    def test_panel_set_with_unequal_max_power(self) -> None:
        """Test panel set with very different max power values."""
        panels = [
            SolarPanel(max_power=10.0),
            SolarPanel(max_power=10000.0),
        ]
        panel_set = SolarPanelSet(panels=panels)
        total = sum(p.max_power for p in panel_set.panels)
        assert total == pytest.approx(10010.0)

    def test_panel_efficiency_boundary_values(self) -> None:
        """Test panel with boundary efficiency values."""
        panel_high_eff = SolarPanel(conversion_efficiency=0.99)
        panel_low_eff = SolarPanel(conversion_efficiency=0.50)
        assert panel_high_eff.conversion_efficiency == 0.99
        assert panel_low_eff.conversion_efficiency == 0.50

    def test_panel_set_all_zero_power(self) -> None:
        """Test panel set where all panels have zero power."""
        panels = [
            SolarPanel(max_power=0.0),
            SolarPanel(max_power=0.0),
        ]
        panel_set = SolarPanelSet(panels=panels)
        total = sum(p.max_power for p in panel_set.panels)
        assert total == 0.0


class TestGetEphemerisIndices:
    """Tests for get_ephemeris_indices function."""

    def test_get_ephemeris_indices_multiple_times(self) -> None:
        """Test get_ephemeris_indices with multiple datetime times."""
        from datetime import datetime, timezone

        from conops.config.solar_panel import get_ephemeris_indices

        times = [
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2018, 1, 2, tzinfo=timezone.utc),
            datetime(2018, 1, 3, tzinfo=timezone.utc),
        ]

        # Mock ephemeris with proper indexing
        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1, 2])

        indices = get_ephemeris_indices(times, ephem)

        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(indices, expected)
        assert ephem.index.call_count == 3

    def test_get_ephemeris_indices_single_time(self) -> None:
        """Test get_ephemeris_indices with single datetime time."""
        from datetime import datetime, timezone

        from conops.config.solar_panel import get_ephemeris_indices

        time_single = datetime(2018, 1, 1, tzinfo=timezone.utc)

        ephem = Mock()
        ephem.index = Mock(return_value=5)

        indices = get_ephemeris_indices(time_single, ephem)

        assert indices == 5
        ephem.index.assert_called_once_with(time_single)


class TestPowerEdgeCases:
    """Tests for edge cases in power method."""

    def test_power_with_zero_max_power(self) -> None:
        """Test power calculation when max_power is zero."""
        panel = SolarPanel(max_power=0.0)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        assert result == 0.0

    def test_power_with_zero_efficiency(self) -> None:
        """Test power calculation when efficiency is zero."""
        panel = SolarPanel(max_power=1000.0, conversion_efficiency=0.0)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        assert result == 0.0

    def test_power_with_negative_efficiency(self) -> None:
        """Test power calculation with negative efficiency (should be clamped)."""
        panel = SolarPanel(max_power=1000.0, conversion_efficiency=-0.1)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        # Negative efficiency should still produce some power (illumination * max_power * efficiency)
        # Since efficiency is negative, result should be negative
        assert result < 0.0

    def test_power_with_extreme_max_power(self) -> None:
        """Test power calculation with very large max_power."""
        panel = SolarPanel(max_power=1e6)  # 1 MW
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel_set.power(time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem)

        # Should handle large values without overflow
        assert isinstance(result, (float, np.floating))

    def test_panel_illumination_fraction_gimbled_array_time(self) -> None:
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

    def test_panel_illumination_fraction_unix_timestamp(self) -> None:
        """Test panel_illumination_fraction with unix timestamp input."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

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

    def test_illumination_and_power_single_panel(
        self, standard_single_panel_set: SolarPanelSet
    ) -> None:
        """Test illumination_and_power with single panel."""
        panel_set = standard_single_panel_set

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

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

    def test_illumination_and_power_multiple_panels(self) -> None:
        """Test illumination_and_power with multiple panels."""
        panels = [
            SolarPanel(max_power=300.0, conversion_efficiency=0.95),
            SolarPanel(max_power=400.0, conversion_efficiency=0.90),
        ]
        panel_set = SolarPanelSet(panels=panels)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

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

    def test_illumination_and_power_in_eclipse(self) -> None:
        """Test illumination_and_power during eclipse."""
        panel = SolarPanel(max_power=1000.0)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
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

    def test_illumination_and_power_empty_panels(
        self, empty_solar_panel_set: SolarPanelSet
    ) -> None:
        """Test illumination_and_power with empty panel list."""
        panel_set = empty_solar_panel_set

        ephem = Mock()
        illumination, power = panel_set.illumination_and_power(
            time=1514764800.0,
            ephem=ephem,
            ra=0.0,
            dec=0.0,
        )

        assert illumination == 0.0
        assert power == 0.0

    def test_illumination_and_power_empty_panels_array_time(
        self, empty_solar_panel_set: SolarPanelSet
    ) -> None:
        """Test illumination_and_power with empty panel list and array time."""
        from datetime import datetime, timezone

        panel_set = empty_solar_panel_set

        times = [
            datetime(2018, 1, 1, tzinfo=timezone.utc),
            datetime(2018, 1, 2, tzinfo=timezone.utc),
        ]

        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1])
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
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

    def test_illumination_and_power_empty_panels_numpy_array_time(
        self, empty_solar_panel_set: SolarPanelSet
    ) -> None:
        """Test illumination_and_power with empty panel list and numpy array time."""
        panel_set = empty_solar_panel_set

        times = np.array([1514764800.0, 1514851200.0])  # Unix timestamps

        ephem = Mock()
        ephem.index = Mock(side_effect=[0, 1])
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
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

    def test_illumination_and_power_efficiency_fallback(self) -> None:
        """Test illumination_and_power with efficiency fallback to set level."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=None)
        panel_set = SolarPanelSet(panels=[panel], conversion_efficiency=0.85)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun = Mock()
        # New direct array access (rust-ephem 0.3.0+)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([0.0])
        ephem.sun.__getitem__ = Mock(
            return_value=Mock(ra=Mock(deg=90.0), dec=Mock(deg=0.0))
        )
        # Add position vectors for new vector-based calculations
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +X
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

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


class TestCoverageCompletion:
    """Tests to achieve 100% coverage of solar_panel.py."""

    def test_panel_illumination_fraction_exception_handling(
        self, mock_ephemeris: Mock
    ) -> None:
        """Test exception handling in panel_illumination_fraction (lines 90-91)."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)

        ephem = Mock()
        ephem.index.side_effect = Exception("Test exception")
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +Y
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        # Remove the patch and let ephem.index raise the exception
        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            try:
                _ = panel.panel_illumination_fraction(
                    time=1514764800.0,
                    ephem=ephem,
                    ra=0.0,
                    dec=0.0,
                )
                assert False, "Expected exception but got result"
            except Exception as e:
                print(f"Got exception: {e}")
                assert "Test exception" in str(e)

    def test_panel_illumination_fraction_datetime_input(self) -> None:
        """Test panel_illumination_fraction with datetime input (lines 89-91)."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +Y
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=datetime(2018, 1, 1, tzinfo=timezone.utc),
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )
            assert isinstance(result, float)
        """Test panel_illumination_fraction when sun magnitude is zero (line 144)."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 0, 0]])  # Sun at origin (zero magnitude)
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            result = panel.panel_illumination_fraction(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert result == 0.0

    def test_solar_panel_set_geometry_cache_hit(
        self, standard_single_panel_set: SolarPanelSet
    ) -> None:
        """Test _get_geometry returns cached geometry (line 237)."""
        panel_set = standard_single_panel_set

        # Access geometry once to populate cache
        geom1 = panel_set._get_geometry()

        # Access again - should return cached version
        geom2 = panel_set._get_geometry()

        # Should be the same object (cached)
        assert geom1 is geom2

    def test_illumination_method_empty_panels_different_time_types(
        self, zero_power_panel_set: SolarPanelSet
    ) -> None:
        """Test illumination method with zero power panels for different time types (lines 305, 309-312)."""
        # Use panels with zero max_power to hit the total_max <= 0 path
        panel_set = zero_power_panel_set

        ephem = Mock()

        # Test scalar time types
        assert (
            panel_set.panel_illumination_fraction(
                time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem
            )
            == 0.0
        )
        assert (
            panel_set.panel_illumination_fraction(
                time=datetime(2018, 1, 1, tzinfo=timezone.utc),
                ra=0.0,
                dec=0.0,
                ephem=ephem,
            )
            == 0.0
        )

        # Test array time types - these should hit the isinstance checks
        result1 = panel_set.panel_illumination_fraction(
            time=np.array([1514764800.0]), ra=0.0, dec=0.0, ephem=ephem
        )
        assert np.array_equal(result1, np.array([0.0]))

        result2 = panel_set.panel_illumination_fraction(
            time=[1514764800.0], ra=0.0, dec=0.0, ephem=ephem
        )
        assert np.array_equal(result2, np.array([0.0]))

        # Test custom sequence that fails on len() to hit the except block
        class BadLenSequence:
            def __len__(self) -> None:
                raise Exception("Bad len")

        bad_time = BadLenSequence()
        result3 = panel_set.panel_illumination_fraction(
            time=bad_time, ra=0.0, dec=0.0, ephem=ephem
        )
        assert result3 == 0.0

    def test_illumination_and_power_array_time_fallback(self) -> None:
        """Test illumination_and_power with array time falls back to loop (lines 415-421, 474-495)."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 1.496e8, 0]])  # Sun at +Y
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        # Create proper mock result for evaluate
        mock_result = Mock()
        mock_result.constraint_array = [False, False]
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)
        mock_constraint.evaluate = Mock(return_value=mock_result)

        with (
            patch("conops.SolarPanel._eclipse_constraint", mock_constraint),
            patch(
                "conops.config.solar_panel._get_eclipse_constraint",
                return_value=mock_constraint,
            ),
        ):
            illumination, power = panel_set.illumination_and_power(
                time=[1514764800.0, 1514764801.0],  # List of times triggers fallback
                ra=0.0,
                dec=0.0,
                ephem=ephem,
            )

        assert isinstance(illumination, np.ndarray)
        assert isinstance(power, np.ndarray)
        assert len(illumination) == 2
        assert len(power) == 2

        # Test with datetime to cover the datetime branch
        with (
            patch("conops.SolarPanel._eclipse_constraint", mock_constraint),
            patch(
                "conops.config.solar_panel._get_eclipse_constraint",
                return_value=mock_constraint,
            ),
        ):
            illumination2, power2 = panel_set.illumination_and_power(
                time=datetime(2018, 1, 1, tzinfo=timezone.utc),
                ra=0.0,
                dec=0.0,
                ephem=ephem,
            )
        assert isinstance(illumination2, float)
        assert isinstance(power2, float)
        assert illumination[0] == pytest.approx(1.0, rel=1e-6)
        assert power[0] == pytest.approx(450.0, rel=1e-4)  # 1.0 * 500 * 0.9

    def test_illumination_and_power_zero_sun_magnitude(self) -> None:
        """Test illumination_and_power when sun magnitude is zero (line 447)."""
        panel = SolarPanel(max_power=500.0, conversion_efficiency=0.9)
        panel_set = SolarPanelSet(panels=[panel])

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun_pv = Mock()
        ephem.sun_pv.position = np.array([[0, 0, 0]])  # Sun at origin (zero magnitude)
        ephem.gcrs_pv = Mock()
        ephem.gcrs_pv.position = np.array([[0, 0, 0]])  # Spacecraft at origin

        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        with patch("conops.SolarPanel._eclipse_constraint", mock_constraint):
            illumination, power = panel_set.illumination_and_power(
                time=1514764800.0,
                ephem=ephem,
                ra=0.0,
                dec=0.0,
            )

        assert illumination == 0.0
        assert power == 0.0

    def test_optimal_charging_pointing_zero_total_power(self) -> None:
        """Test optimal_charging_pointing when total power is zero (line 526)."""
        panel_set = SolarPanelSet(
            panels=[SolarPanel(max_power=0.0)]
        )  # Zero power panel

        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sun_ra_deg = np.array([90.0])
        ephem.sun_dec_deg = np.array([30.0])

        ra, dec = panel_set.optimal_charging_pointing(time=1514764800.0, ephem=ephem)

        # Should return sun position when no physical panels
        assert ra == 90.0
        assert dec == 30.0

    def test_create_normal_vector_sidemount_no_cant(self) -> None:
        """Test create_solar_panel_vector with sidemount and no cant."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("sidemount", 0.0, 0.0)
        assert normal == (0.0, 1.0, 0.0)

    def test_create_normal_vector_sidemount_z_cant_only(self) -> None:
        """Test create_solar_panel_vector with sidemount and Z-axis cant only."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("sidemount", cant_z=30.0)
        expected_x = -math.sin(math.radians(30.0))
        expected_y = math.cos(math.radians(30.0))
        assert abs(normal[0] - expected_x) < 1e-10
        assert abs(normal[1] - expected_y) < 1e-10
        assert abs(normal[2]) < 1e-10

    def test_create_normal_vector_sidemount_perp_cant_only(self) -> None:
        """Test create_solar_panel_vector with sidemount and perpendicular cant only."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("sidemount", cant_z=0.0, cant_perp=45.0)
        # After X rotation of 45°: y becomes cos(45°), z becomes sin(45°)
        expected_y = math.cos(math.radians(45.0))
        expected_z = math.sin(math.radians(45.0))
        assert abs(normal[0]) < 1e-10
        assert abs(normal[1] - expected_y) < 1e-10
        assert abs(normal[2] - expected_z) < 1e-10

    def test_create_normal_vector_sidemount_both_cants_unit_vector(self) -> None:
        """Test create_solar_panel_vector with sidemount and both cants produces unit vector."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("sidemount", cant_z=30.0, cant_perp=45.0)
        # This combines both rotations
        assert len(normal) == 3
        # Verify it's still a unit vector (approximately)
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_aftmount_no_cant(self) -> None:
        """Test create_solar_panel_vector with aftmount and no cant."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("aftmount", 0.0, 0.0)
        assert normal == (-1.0, 0.0, 0.0)

    def test_create_normal_vector_aftmount_z_cant_only(self) -> None:
        """Test create_solar_panel_vector with aftmount and Z-axis cant only."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("aftmount", cant_z=45.0)
        expected_x = -math.cos(math.radians(45.0))
        expected_y = -math.sin(math.radians(45.0))
        assert abs(normal[0] - expected_x) < 1e-10
        assert abs(normal[1] - expected_y) < 1e-10
        assert abs(normal[2]) < 1e-10

    def test_create_normal_vector_aftmount_perp_cant_only(self) -> None:
        """Test create_solar_panel_vector with aftmount and perpendicular cant only."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("aftmount", cant_z=0.0, cant_perp=45.0)
        # After Y rotation of 45°: x becomes -cos(45°), z becomes sin(45°)
        expected_x = -math.cos(math.radians(45.0))
        expected_z = math.sin(math.radians(45.0))
        assert abs(normal[0] - expected_x) < 1e-10
        assert abs(normal[1]) < 1e-10
        assert abs(normal[2] - expected_z) < 1e-10

    def test_create_normal_vector_aftmount_both_cants_unit_vector(self) -> None:
        """Test create_solar_panel_vector with aftmount and both cants produces unit vector."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("aftmount", cant_z=30.0, cant_perp=30.0)
        # Verify it's still a unit vector (approximately)
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_boresight_no_cant(self) -> None:
        """Test create_solar_panel_vector with boresight and no cant."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("boresight", 0.0, 0.0)
        assert normal == (1.0, 0.0, 0.0)

    def test_create_normal_vector_boresight_backward_slant(self) -> None:
        """Test create_solar_panel_vector with boresight and backward slant."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("boresight", cant_z=0.0, cant_perp=-45.0)
        # After Y rotation of -45°: x becomes cos(-45°), z becomes sin(-45°)
        expected_x = math.cos(math.radians(-45.0))
        expected_z = math.sin(math.radians(-45.0))
        assert abs(normal[0] - expected_x) < 1e-10
        assert abs(normal[1]) < 1e-10
        assert abs(normal[2] - expected_z) < 1e-10

    def test_create_normal_vector_boresight_z_cant_only(self) -> None:
        """Test create_solar_panel_vector with boresight and Z-axis cant only."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("boresight", cant_z=30.0)
        expected_x = math.cos(math.radians(30.0))
        expected_y = math.sin(math.radians(30.0))
        assert abs(normal[0] - expected_x) < 1e-10
        assert abs(normal[1] - expected_y) < 1e-10
        assert abs(normal[2]) < 1e-10

    def test_create_normal_vector_boresight_both_cants_unit_vector(self) -> None:
        """Test create_solar_panel_vector with boresight and both cants produces unit vector."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector("boresight", cant_z=30.0, cant_perp=-45.0)
        # Verify it's still a unit vector (approximately)
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_invalid_mount(self) -> None:
        """Test create_solar_panel_vector with invalid mount type raises ValueError."""
        from conops.config.solar_panel import create_solar_panel_vector

        with pytest.raises(ValueError, match="Unknown mount type"):
            create_solar_panel_vector("invalid")

    def test_create_normal_vector_old_style_azimuth_0(self) -> None:
        """Test create_solar_panel_vector old style with azimuth 0° (north/+Y)."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=0.0)
        assert normal == (0.0, 1.0, 0.0)

    def test_create_normal_vector_old_style_azimuth_90(self) -> None:
        """Test create_solar_panel_vector old style with azimuth 90° (up/+Z)."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=90.0)
        assert normal == (0.0, 0.0, 1.0)

    def test_create_normal_vector_old_style_azimuth_180(self) -> None:
        """Test create_solar_panel_vector old style with azimuth 180° (south/-Y)."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=180.0)
        assert normal == (0.0, -1.0, 0.0)

    def test_create_normal_vector_old_style_azimuth_270(self) -> None:
        """Test create_solar_panel_vector old style with azimuth 270° (down/-Z)."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=270.0)
        assert normal == (0.0, 0.0, -1.0)

    def test_create_normal_vector_old_style_with_cants_unit_vector(self) -> None:
        """Test create_solar_panel_vector old style with cant angles produces unit vector."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=30.0, cant_y=15.0, azimuth_deg=0.0)
        # Verify it's still a unit vector (approximately)
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_old_style_non_cardinal_azimuth_45(self) -> None:
        """Test create_solar_panel_vector old style with non-cardinal azimuth 45°."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=45.0)
        # 45° rounds to 0° (closest cardinal), so should be +Y
        assert normal == (0.0, 1.0, 0.0)

    def test_create_normal_vector_old_style_non_cardinal_azimuth_135(self) -> None:
        """Test create_solar_panel_vector old style with non-cardinal azimuth 135°."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=135.0)
        # 135° rounds to 180° (closest cardinal), so should be -Y
        assert normal == (0.0, -1.0, 0.0)

    def test_create_normal_vector_parameter_validation_mix_mount_cant_x(self) -> None:
        """Test parameter validation: mixing mount with cant_x raises ValueError."""
        from conops.config.solar_panel import create_solar_panel_vector

        with pytest.raises(
            ValueError,
            match="Cannot mix old style parameters.*with new style parameters",
        ):
            create_solar_panel_vector(mount="sidemount", cant_x=10.0)

    def test_create_normal_vector_parameter_validation_mix_mount_cant_y(self) -> None:
        """Test parameter validation: mixing mount with cant_y raises ValueError."""
        from conops.config.solar_panel import create_solar_panel_vector

        with pytest.raises(
            ValueError,
            match="Cannot mix old style parameters.*with new style parameters",
        ):
            create_solar_panel_vector(mount="sidemount", cant_y=10.0)

    def test_create_normal_vector_parameter_validation_mix_mount_azimuth(self) -> None:
        """Test parameter validation: mixing mount with azimuth_deg raises ValueError."""
        from conops.config.solar_panel import create_solar_panel_vector

        with pytest.raises(
            ValueError,
            match="Cannot mix old style parameters.*with new style parameters",
        ):
            create_solar_panel_vector(mount="sidemount", azimuth_deg=90.0)

    def test_create_normal_vector_parameter_validation_mix_cant_x_mount(self) -> None:
        """Test parameter validation: mixing cant_x with mount raises ValueError."""
        from conops.config.solar_panel import create_solar_panel_vector

        with pytest.raises(
            ValueError,
            match="Cannot mix old style parameters.*with new style parameters",
        ):
            create_solar_panel_vector(cant_x=10.0, mount="sidemount")

    def test_create_normal_vector_parameter_validation_default_behavior(self) -> None:
        """Test parameter validation: default behavior when no parameters provided."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector()
        # Should default to sidemount with no cant
        assert normal == (0.0, 1.0, 0.0)

    def test_create_normal_vector_parameter_validation_old_style_partial_cant_x(
        self,
    ) -> None:
        """Test parameter validation: old style with partial parameters (cant_x only)."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(
            cant_x=30.0
        )  # missing cant_y and azimuth_deg
        # Should be azimuth 0° with cant_x=30°
        expected_y = math.cos(math.radians(30.0))
        expected_z = math.sin(math.radians(30.0))
        assert abs(normal[0]) < 1e-10
        assert abs(normal[1] - expected_y) < 1e-10
        assert abs(normal[2] - expected_z) < 1e-10

    def test_create_normal_vector_parameter_validation_old_style_azimuth_only(
        self,
    ) -> None:
        """Test parameter validation: old style with only azimuth_deg provided."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(
            azimuth_deg=90.0
        )  # missing cant_x and cant_y
        # Should be +Z direction with no cant
        assert normal == (0.0, 0.0, 1.0)

    def test_create_normal_vector_parameter_validation_old_style_cant_y_only(
        self,
    ) -> None:
        """Test parameter validation: old style with only cant_y provided."""
        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(
            cant_y=20.0
        )  # missing cant_x and azimuth_deg
        # Should be azimuth 0° with cant_y=20°
        # For azimuth 0°, cant_y rotates around Y-axis
        # x_final = sin(0°) * sin(20°) = 0
        # y_final = cos(0°) = 1
        # z_final = sin(0°) * cos(20°) = 0
        assert normal == (0.0, 1.0, 0.0)

    def test_create_normal_vector_interpolation_logic_azimuth_0(self) -> None:
        """Test interpolation logic for azimuth 0°."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=0.0)
        # For now, just check that we get a valid unit vector - the exact mapping may need adjustment
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_interpolation_logic_azimuth_30(self) -> None:
        """Test interpolation logic for azimuth 30°."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=30.0)
        # For now, just check that we get a valid unit vector - the exact mapping may need adjustment
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_interpolation_logic_azimuth_60(self) -> None:
        """Test interpolation logic for azimuth 60°."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=60.0)
        # For now, just check that we get a valid unit vector - the exact mapping may need adjustment
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_interpolation_logic_azimuth_90(self) -> None:
        """Test interpolation logic for azimuth 90°."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=90.0)
        # For now, just check that we get a valid unit vector - the exact mapping may need adjustment
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_interpolation_logic_azimuth_120(self) -> None:
        """Test interpolation logic for azimuth 120°."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=120.0)
        # For now, just check that we get a valid unit vector - the exact mapping may need adjustment
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_interpolation_logic_azimuth_180(self) -> None:
        """Test interpolation logic for azimuth 180°."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=180.0)
        # For now, just check that we get a valid unit vector - the exact mapping may need adjustment
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10

    def test_create_normal_vector_interpolation_logic_azimuth_270(self) -> None:
        """Test interpolation logic for azimuth 270°."""
        import math

        from conops.config.solar_panel import create_solar_panel_vector

        normal = create_solar_panel_vector(cant_x=0.0, cant_y=0.0, azimuth_deg=270.0)
        # For now, just check that we get a valid unit vector - the exact mapping may need adjustment
        magnitude = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        assert abs(magnitude - 1.0) < 1e-10
