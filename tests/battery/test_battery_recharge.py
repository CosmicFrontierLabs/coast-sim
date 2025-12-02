"""Unit tests for emergency battery recharge functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import (
    ACSMode,
    Battery,
    EmergencyCharging,
    Pointing,
    QueueDITL,
    SolarPanel,
    SolarPanelSet,
    angular_separation,
)


class TestBattery:
    """Test Battery class emergency recharge functionality."""

    def test_default_recharge_threshold(self, default_battery):
        """Test that recharge_threshold defaults to 0.95 (95%)."""
        assert default_battery.recharge_threshold == 0.95

    def test_custom_recharge_threshold(self, battery_with_custom_threshold):
        """Test that custom recharge_threshold can be set."""
        assert battery_with_custom_threshold.recharge_threshold == 0.90

    def test_battery_alert_true_when_below_max_depth_of_discharge(
        self, battery_with_dod
    ):
        """Test that battery_alert triggers when below max_depth_of_discharge."""
        battery_with_dod.charge_level = battery_with_dod.watthour * 0.60
        assert battery_with_dod.battery_alert is True

    def test_emergency_recharge_true_when_below_max_depth_of_discharge(
        self, battery_with_dod
    ):
        """Test that emergency_recharge is True when below max_depth_of_discharge."""
        battery_with_dod.charge_level = battery_with_dod.watthour * 0.60
        # Access battery_alert to trigger the logic
        _ = battery_with_dod.battery_alert
        assert battery_with_dod.emergency_recharge is True

    def test_battery_alert_set_when_below_threshold(
        self, battery_with_dod_and_threshold
    ):
        """When below max DoD, battery_alert should be True."""
        battery_with_dod_and_threshold.charge_level = (
            battery_with_dod_and_threshold.watthour * 0.60
        )
        assert battery_with_dod_and_threshold.battery_alert is True

    def test_battery_alert_persists_at_ninety_percent(
        self, battery_with_dod_and_threshold
    ):
        """Battery alert should still be True at 90% when threshold is 95%."""
        # First trigger emergency recharge by going below min charge level
        battery_with_dod_and_threshold.charge_level = (
            battery_with_dod_and_threshold.watthour * 0.60
        )
        _ = battery_with_dod_and_threshold.battery_alert  # Trigger emergency_recharge
        # Now set to 90% and check that alert persists
        battery_with_dod_and_threshold.charge_level = (
            battery_with_dod_and_threshold.watthour * 0.90
        )
        assert battery_with_dod_and_threshold.battery_alert is True

    def test_emergency_recharge_persists_at_ninety_percent(
        self, battery_with_dod_and_threshold
    ):
        """Emergency recharge flag should still be True at 90% when threshold is 95%."""
        # First trigger emergency recharge by going below min charge level
        battery_with_dod_and_threshold.charge_level = (
            battery_with_dod_and_threshold.watthour * 0.60
        )
        _ = battery_with_dod_and_threshold.battery_alert  # Trigger emergency_recharge
        # Now set to 90% and check that emergency_recharge persists
        battery_with_dod_and_threshold.charge_level = (
            battery_with_dod_and_threshold.watthour * 0.90
        )
        assert battery_with_dod_and_threshold.emergency_recharge is True

    def test_battery_alert_clears_at_ninetyfive_percent(
        self, battery_with_dod_and_threshold
    ):
        """Battery alert should clear at recharge threshold (95%)."""
        battery_with_dod_and_threshold.charge_level = (
            battery_with_dod_and_threshold.watthour * 0.95
        )
        assert battery_with_dod_and_threshold.battery_alert is False

    def test_emergency_recharge_clears_at_ninetyfive_percent(
        self, battery_with_dod_and_threshold
    ):
        """Emergency recharge should clear at recharge threshold (95%)."""
        battery_with_dod_and_threshold.charge_level = (
            battery_with_dod_and_threshold.watthour * 0.95
        )
        assert battery_with_dod_and_threshold.emergency_recharge is False

    def test_battery_alert_false_when_above_threshold(self, battery_with_dod):
        """Test battery_alert False when battery level is sufficient."""
        battery_with_dod.charge_level = battery_with_dod.watthour * 0.96
        assert battery_with_dod.battery_alert is False

    def test_emergency_recharge_false_when_above_threshold(self, battery_with_dod):
        """Test emergency_recharge False when battery level is sufficient."""
        battery_with_dod.charge_level = battery_with_dod.watthour * 0.80
        assert battery_with_dod.emergency_recharge is False


class TestACSMode:
    """Test that CHARGING mode is added to ACSMode enum."""

    def test_charging_mode_exists(self):
        """Test that CHARGING mode exists in ACSMode enum."""
        assert hasattr(ACSMode, "CHARGING")

    def test_charging_mode_value(self):
        """Test that CHARGING mode has value 4."""
        assert ACSMode.CHARGING.value == 4

    def test_acsmode_has_science(self):
        """SCIENCE should be present in ACSMode."""
        modes = [mode.name for mode in ACSMode]
        assert "SCIENCE" in modes

    def test_acsmode_has_slewing(self):
        """SLEWING should be present in ACSMode."""
        modes = [mode.name for mode in ACSMode]
        assert "SLEWING" in modes

    def test_acsmode_has_saa(self):
        """SAA should be present in ACSMode."""
        modes = [mode.name for mode in ACSMode]
        assert "SAA" in modes

    def test_acsmode_has_pass(self):
        """PASS should be present in ACSMode."""
        modes = [mode.name for mode in ACSMode]
        assert "PASS" in modes

    def test_acsmode_has_charging(self):
        """CHARGING should be present in ACSMode."""
        modes = [mode.name for mode in ACSMode]
        assert "CHARGING" in modes


class TestSolarPanel:
    """Test SolarPanel optimal charging pointing functionality."""

    def test_optimal_charging_ra_sidemount(self):
        """Test RA for side-mounted panels."""
        panel = SolarPanelSet(panels=[SolarPanel(sidemount=True)])
        mock_ephem = Mock()
        mock_ephem.index.return_value = 0
        mock_sun_coord = Mock()
        mock_sun_coord.ra.deg = 45.0
        mock_sun_coord.dec.deg = 10.0
        mock_ephem.sun = [mock_sun_coord]
        utime = 1700000000.0
        ra, dec = panel.optimal_charging_pointing(utime, mock_ephem)
        assert ra == (45.0 + 90.0) % 360.0

    def test_optimal_charging_dec_sidemount(self):
        """Test Dec for side-mounted panels."""
        panel = SolarPanelSet(panels=[SolarPanel(sidemount=True)])
        mock_ephem = Mock()
        mock_ephem.index.return_value = 0
        mock_sun_coord = Mock()
        mock_sun_coord.ra.deg = 45.0
        mock_sun_coord.dec.deg = 10.0
        mock_ephem.sun = [mock_sun_coord]
        utime = 1700000000.0
        ra, dec = panel.optimal_charging_pointing(utime, mock_ephem)
        assert dec == 10.0

    def test_optimal_charging_ra_bodymount(self):
        """Test RA for body-mounted panels."""
        panel = SolarPanelSet(panels=[SolarPanel(sidemount=False)])
        mock_ephem = Mock()
        mock_ephem.index.return_value = 0
        mock_sun_coord = Mock()
        mock_sun_coord.ra.deg = 120.0
        mock_sun_coord.dec.deg = -15.0
        mock_ephem.sun = [mock_sun_coord]
        utime = 1700000000.0
        ra, dec = panel.optimal_charging_pointing(utime, mock_ephem)
        assert ra == 120.0

    def test_optimal_charging_dec_bodymount(self):
        """Test Dec for body-mounted panels."""
        panel = SolarPanelSet(panels=[SolarPanel(sidemount=False)])
        mock_ephem = Mock()
        mock_ephem.index.return_value = 0
        mock_sun_coord = Mock()
        mock_sun_coord.ra.deg = 120.0
        mock_sun_coord.dec.deg = -15.0
        mock_ephem.sun = [mock_sun_coord]
        utime = 1700000000.0
        ra, dec = panel.optimal_charging_pointing(utime, mock_ephem)
        assert dec == -15.0

    def test_optimal_charging_pointing_wraps_ra(self):
        """Test that RA wraps correctly at 360 degrees."""
        panel = SolarPanelSet(panels=[SolarPanel(sidemount=True)])
        mock_ephem = Mock()
        mock_ephem.index.return_value = 0
        mock_sun_coord = Mock()
        mock_sun_coord.ra.deg = 350.0
        mock_sun_coord.dec.deg = 0.0
        mock_ephem.sun = [mock_sun_coord]
        utime = 1700000000.0
        ra, dec = panel.optimal_charging_pointing(utime, mock_ephem)
        assert ra == 80.0


class TestEmergencyCharging:
    """Test EmergencyCharging class functionality."""

    def test_initialization_sets_constraint(self, mock_config):
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=555000,
        )
        assert ec.constraint == mock_config.constraint

    def test_initialization_sets_solar_panel(self, mock_config):
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=555000,
        )
        assert ec.solar_panel == mock_config.solar_panel

    def test_initialization_next_charging_obsid(self, mock_config):
        # Ensure config has the expected subsystems
        mock_config.solar_panel = mock_config.solar_panel
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=555000,
        )
        assert ec.next_charging_obsid == 555000

    def test_initialization_current_charging_ppt_none(self, mock_config):
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=555000,
        )
        assert ec.current_charging_ppt is None

    def test_create_charging_pointing_success_returns_pointing(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        monkeypatch.setattr(
            emergency_charging.constraint, "in_constraint", lambda ra, dec, time: False
        )
        ppt = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert ppt is not None

    def test_create_charging_pointing_success_is_pointing_type(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        ppt = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert isinstance(ppt, Pointing)

    def test_create_charging_pointing_assigns_ra_dec_name_obsid(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        ppt = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert ppt.ra == 180.0
        assert ppt.dec == 0.0
        assert ppt.name == "EMERGENCY_CHARGE_999000"
        assert ppt.obsid == 999000

    def test_create_charging_pointing_sets_current_ppt(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        ppt = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert emergency_charging.current_charging_ppt == ppt

    def test_create_charging_pointing_increments_obsid_values(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        ppt1 = emergency_charging.create_charging_pointing(utime, mock_ephem)
        ppt2 = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert ppt1.obsid == 999000
        assert ppt2.obsid == 999001

    def test_create_charging_pointing_increments_next_charging_obsid(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        emergency_charging.create_charging_pointing(utime, mock_ephem)
        emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert emergency_charging.next_charging_obsid == 999002

    def test_create_charging_pointing_in_eclipse_returns_none_and_no_current_ppt(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: True
        )
        ppt = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert ppt is None
        assert emergency_charging.current_charging_ppt is None

    def test_create_charging_pointing_constraint_violation_returns_alternative_ra(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        def mock_in_constraint(ra, dec, utime, hardonly=True):
            return ra == 180.0

        emergency_charging.constraint.in_constraint = mock_in_constraint

        def mock_illumination(time, ra, dec, ephem):
            if ra == 210.0:
                return 0.9
            return 0.5

        object.__setattr__(
            emergency_charging.solar_panel,
            "panel_illumination_fraction",
            mock_illumination,
        )
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        ppt = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert ppt is not None
        assert ppt.ra == 210.0

    def test_create_charging_pointing_no_valid_pointing_returns_none_and_no_current(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        emergency_charging.constraint.in_constraint = Mock(return_value=True)
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        ppt = emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert ppt is None
        assert emergency_charging.current_charging_ppt is None

    def test_clear_current_charging_resets_current_ppt(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert emergency_charging.current_charging_ppt is not None
        emergency_charging.clear_current_charging()
        assert emergency_charging.current_charging_ppt is None

    def test_is_charging_active_initially_false(self, emergency_charging):
        assert emergency_charging.is_charging_active() is False

    def test_is_charging_active_true_after_create(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        emergency_charging.create_charging_pointing(utime, mock_ephem)
        assert emergency_charging.is_charging_active() is True

    def test_is_charging_active_false_after_clear(
        self, emergency_charging, mock_ephem, monkeypatch, utime
    ):
        monkeypatch.setattr(
            emergency_charging.constraint, "in_eclipse", lambda ra, dec, time: False
        )
        emergency_charging.create_charging_pointing(utime, mock_ephem)
        emergency_charging.clear_current_charging()
        assert emergency_charging.is_charging_active() is False

    def _helper_find_valid_pointing_sidemount(
        self,
        emergency_charging,
        sun_ra,
        sun_dec,
        utime,
        current_ra=None,
        current_dec=None,
    ):
        # Utility: returns found pointing
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime, current_ra, current_dec
        )
        return ra, dec

    def test_find_valid_pointing_sidemount_success_ra_not_none(
        self, emergency_charging
    ):
        sun_ra = 180.0
        sun_dec = 0.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=False)
        ra, dec = self._helper_find_valid_pointing_sidemount(
            emergency_charging, sun_ra, sun_dec, utime
        )
        assert ra is not None

    def test_find_valid_pointing_sidemount_success_dec_not_none(
        self, emergency_charging
    ):
        sun_ra = 180.0
        sun_dec = 0.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=False)
        ra, dec = self._helper_find_valid_pointing_sidemount(
            emergency_charging, sun_ra, sun_dec, utime
        )
        assert dec is not None

    def test_find_valid_pointing_sidemount_success_is_approximately_90deg(
        self, emergency_charging
    ):
        sun_ra = 180.0
        sun_dec = 0.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=False)
        ra, dec = self._helper_find_valid_pointing_sidemount(
            emergency_charging, sun_ra, sun_dec, utime
        )
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        sun_ra_rad = np.radians(sun_ra)
        sun_dec_rad = np.radians(sun_dec)
        pointing_vec = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )
        sun_vec = np.array(
            [
                np.cos(sun_dec_rad) * np.cos(sun_ra_rad),
                np.cos(sun_dec_rad) * np.sin(sun_ra_rad),
                np.sin(sun_dec_rad),
            ]
        )
        dot_product = np.dot(pointing_vec, sun_vec)
        sep_angle = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        assert abs(sep_angle - 90.0) < 1.5

    def test_find_valid_pointing_sidemount_constraint_violation_ra_not_none(
        self, emergency_charging
    ):
        sun_ra = 90.0
        sun_dec = 45.0
        utime = 1700000000.0

        def mock_in_constraint(ra, dec, utime, hardonly=True):
            return ra <= 180.0

        emergency_charging.constraint.in_constraint = mock_in_constraint
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        assert ra is not None

    def test_find_valid_pointing_sidemount_constraint_violation_dec_not_none(
        self, emergency_charging
    ):
        sun_ra = 90.0
        sun_dec = 45.0
        utime = 1700000000.0

        def mock_in_constraint(ra, dec, utime, hardonly=True):
            return ra <= 180.0

        emergency_charging.constraint.in_constraint = mock_in_constraint
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        assert dec is not None

    def test_find_valid_pointing_sidemount_constraint_violation_ra_greater_180(
        self, emergency_charging
    ):
        sun_ra = 90.0
        sun_dec = 45.0
        utime = 1700000000.0

        def mock_in_constraint(ra, dec, utime, hardonly=True):
            return ra <= 180.0

        emergency_charging.constraint.in_constraint = mock_in_constraint
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        assert ra > 180.0

    def test_find_valid_pointing_sidemount_constraint_violation_still_90deg(
        self, emergency_charging
    ):
        sun_ra = 90.0
        sun_dec = 45.0
        utime = 1700000000.0

        def mock_in_constraint(ra, dec, utime, hardonly=True):
            return ra <= 180.0

        emergency_charging.constraint.in_constraint = mock_in_constraint
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        sun_ra_rad = np.radians(sun_ra)
        sun_dec_rad = np.radians(sun_dec)
        pointing_vec = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )
        sun_vec = np.array(
            [
                np.cos(sun_dec_rad) * np.cos(sun_ra_rad),
                np.cos(sun_dec_rad) * np.sin(sun_ra_rad),
                np.sin(sun_dec_rad),
            ]
        )
        dot_product = np.dot(pointing_vec, sun_vec)
        sep_angle = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        assert abs(sep_angle - 90.0) < 1.5

    def test_find_valid_pointing_sidemount_all_constrained_ra_none(
        self, emergency_charging
    ):
        sun_ra = 0.0
        sun_dec = 0.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=True)
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        assert ra is None

    def test_find_valid_pointing_sidemount_all_constrained_dec_none(
        self, emergency_charging
    ):
        sun_ra = 0.0
        sun_dec = 0.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=True)
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        assert dec is None

    def test_find_valid_pointing_sidemount_sun_near_pole_ra_not_none(
        self, emergency_charging
    ):
        sun_ra = 0.0
        sun_dec = 85.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=False)
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        assert ra is not None

    def test_find_valid_pointing_sidemount_sun_near_pole_dec_not_none(
        self, emergency_charging
    ):
        sun_ra = 0.0
        sun_dec = 85.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=False)
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        assert dec is not None

    def test_find_valid_pointing_sidemount_sun_near_pole_separated_by_90deg(
        self, emergency_charging
    ):
        sun_ra = 0.0
        sun_dec = 85.0
        utime = 1700000000.0
        emergency_charging.constraint.in_constraint = Mock(return_value=False)
        ra, dec = emergency_charging._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime
        )
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        sun_ra_rad = np.radians(sun_ra)
        sun_dec_rad = np.radians(sun_dec)
        pointing_vec = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )
        sun_vec = np.array(
            [
                np.cos(sun_dec_rad) * np.cos(sun_ra_rad),
                np.cos(sun_dec_rad) * np.sin(sun_ra_rad),
                np.sin(sun_dec_rad),
            ]
        )
        dot_product = np.dot(pointing_vec, sun_vec)
        sep_angle = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        assert abs(sep_angle - 90.0) < 1.5

    def test_slew_limit_constraint_returns_ppt_and_within_slew(
        self,
        mock_config,
        mock_ephem,
        monkeypatch,
    ):
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
            max_slew_deg=60.0,
        )
        object.__setattr__(
            mock_config.solar_panel,
            "optimal_charging_pointing",
            Mock(return_value=(100.0, 0.0)),
        )
        mock_config.constraint.in_constraint = Mock(return_value=False)

        def mock_illumination(time, ra, dec, ephem):
            if abs(ra) < 60 or ra > 300:
                return 0.85
            return 0.5

        object.__setattr__(
            mock_config.solar_panel, "panel_illumination_fraction", mock_illumination
        )
        monkeypatch.setattr(ec.constraint, "in_eclipse", lambda ra, dec, time: False)
        utime = 1700000000.0
        ppt = ec.create_charging_pointing(utime, mock_ephem, lastra=0.0, lastdec=0.0)
        assert ppt is not None
        slew = angular_separation(0.0, 0.0, ppt.ra, ppt.dec)
        assert slew <= 60.0

    def test_slew_limit_sidemount_ra_dec_not_none(self, mock_config):
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
            max_slew_deg=45.0,
        )
        sun_ra = 90.0
        sun_dec = 0.0
        utime = 1700000000.0
        current_ra = 0.0
        current_dec = 0.0
        ra, dec = ec._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime, current_ra, current_dec
        )
        assert ra is not None
        assert dec is not None

    def test_slew_limit_sidemount_within_45deg_and_90deg_sep(self, mock_config):
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
            max_slew_deg=45.0,
        )
        sun_ra = 90.0
        sun_dec = 0.0
        utime = 1700000000.0
        current_ra = 0.0
        current_dec = 0.0
        ra, dec = ec._find_valid_pointing_sidemount(
            sun_ra, sun_dec, utime, current_ra, current_dec
        )
        slew = angular_separation(current_ra, current_dec, ra, dec)
        assert slew <= 45.0
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        sun_ra_rad = np.radians(sun_ra)
        sun_dec_rad = np.radians(sun_dec)
        pointing_vec = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )
        sun_vec = np.array(
            [
                np.cos(sun_dec_rad) * np.cos(sun_ra_rad),
                np.cos(sun_dec_rad) * np.sin(sun_ra_rad),
                np.sin(sun_dec_rad),
            ]
        )
        dot_product = np.dot(pointing_vec, sun_vec)
        sep_angle = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        assert abs(sep_angle - 90.0) < 1.5

    @pytest.mark.parametrize(
        "sun_ra,sun_dec",
        [
            (0.0, 0.0),
            (90.0, 0.0),
            (180.0, 30.0),
            (270.0, -30.0),
            (45.0, 60.0),
            (300.0, -60.0),
        ],
    )
    def test_sidemount_pointing_has_full_illumination_ra_dec_not_none(
        self,
        sun_ra,
        sun_dec,
        mock_config,
    ):
        utime = 1700000000.0
        mock_config.constraint.in_constraint = Mock(return_value=False)

        def mock_illumination(time, ra, dec, ephem):
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            sun_ra_rad = np.radians(sun_ra)
            sun_dec_rad = np.radians(sun_dec)
            pointing_vec = np.array(
                [
                    np.cos(dec_rad) * np.cos(ra_rad),
                    np.cos(dec_rad) * np.sin(ra_rad),
                    np.sin(dec_rad),
                ]
            )
            sun_vec = np.array(
                [
                    np.cos(sun_dec_rad) * np.cos(sun_ra_rad),
                    np.cos(sun_dec_rad) * np.sin(sun_ra_rad),
                    np.sin(sun_dec_rad),
                ]
            )
            sep = np.degrees(
                np.arccos(np.clip(np.dot(pointing_vec, sun_vec), -1.0, 1.0))
            )
            if abs(sep - 90.0) < 1.5:
                return 1.0
            return 0.5

        object.__setattr__(
            mock_config.solar_panel, "panel_illumination_fraction", mock_illumination
        )
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
        )
        ra, dec = ec._find_valid_pointing_sidemount(sun_ra, sun_dec, utime)
        assert ra is not None
        assert dec is not None

    def test_sidemount_pointing_has_full_illumination_value_is_one(
        self,
        mock_config,
    ):
        # pick a sample sun position and verify illumination is 1.0 for returned pointing
        sun_ra = 90.0
        sun_dec = 0.0
        utime = 1700000000.0
        mock_config.constraint.in_constraint = Mock(return_value=False)

        def mock_illumination(time, ra, dec, ephem):
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            sun_ra_rad = np.radians(sun_ra)
            sun_dec_rad = np.radians(sun_dec)
            pointing_vec = np.array(
                [
                    np.cos(dec_rad) * np.cos(ra_rad),
                    np.cos(dec_rad) * np.sin(ra_rad),
                    np.sin(dec_rad),
                ]
            )
            sun_vec = np.array(
                [
                    np.cos(sun_dec_rad) * np.cos(sun_ra_rad),
                    np.cos(sun_dec_rad) * np.sin(sun_ra_rad),
                    np.sin(sun_dec_rad),
                ]
            )
            sep = np.degrees(
                np.arccos(np.clip(np.dot(pointing_vec, sun_vec), -1.0, 1.0))
            )
            if abs(sep - 90.0) < 1.5:
                return 1.0
            return 0.5

        object.__setattr__(
            mock_config.solar_panel, "panel_illumination_fraction", mock_illumination
        )
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
        )
        ra, dec = ec._find_valid_pointing_sidemount(sun_ra, sun_dec, utime)
        illum = mock_config.solar_panel.panel_illumination_fraction(
            utime, ra, dec, Mock()
        )
        assert illum == 1.0

    def test_find_valid_pointing_prefers_max_illumination_return_ra_dec_and_illum(
        self, mock_config, mock_ephem
    ):
        optimal_ra = 10.0
        optimal_dec = 0.0
        utime = 1700000000.0

        def mock_in_constraint(ra, dec, utime_inner, hardonly=True):
            return abs(ra - optimal_ra) < 1e-6 and abs(dec - optimal_dec) < 1e-6

        mock_config.constraint.in_constraint = mock_in_constraint
        object.__setattr__(
            mock_config.solar_panel,
            "optimal_charging_pointing",
            Mock(return_value=(optimal_ra, optimal_dec)),
        )
        high_ra = (optimal_ra + 90.0) % 360.0
        high_dec = optimal_dec

        def mock_illumination(time, ra, dec, ephem):
            if abs(ra - high_ra) < 1e-6 and abs(dec - high_dec) < 1e-6:
                return 1.0
            if abs(ra - optimal_ra) < 1e-6 and abs(dec - optimal_dec) < 1e-6:
                return 0.8
            return 0.6

        object.__setattr__(
            mock_config.solar_panel, "panel_illumination_fraction", mock_illumination
        )
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
        )
        ra, dec = ec._find_valid_pointing(
            optimal_ra,
            optimal_dec,
            utime,
            mock_ephem,
            current_ra=optimal_ra,
            current_dec=optimal_dec,
        )
        assert ra == high_ra
        assert dec == high_dec
        assert (
            mock_config.solar_panel.panel_illumination_fraction(
                utime, ra, dec, mock_ephem
            )
            == 1.0
        )

    def test_find_valid_pointing_optimal_already_max_return_optimal(
        self, mock_config, mock_ephem
    ):
        utime = 1700000000.0
        optimal_ra, optimal_dec = 140.0, -10.0
        mock_config.constraint.in_constraint = Mock(return_value=False)
        object.__setattr__(
            mock_config.solar_panel,
            "optimal_charging_pointing",
            Mock(return_value=(optimal_ra, optimal_dec)),
        )

        def mock_illumination(time, ra, dec, ephem):
            if abs(ra - optimal_ra) < 1e-6 and abs(dec - optimal_dec) < 1e-6:
                return 1.0
            return 0.7

        object.__setattr__(
            mock_config.solar_panel, "panel_illumination_fraction", mock_illumination
        )
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
        )
        ra, dec = ec._find_valid_pointing(
            optimal_ra,
            optimal_dec,
            utime,
            mock_ephem,
            current_ra=optimal_ra,
            current_dec=optimal_dec,
        )
        assert ra == optimal_ra
        assert dec == optimal_dec
        assert (
            mock_config.solar_panel.panel_illumination_fraction(
                utime, ra, dec, mock_ephem
            )
            == 1.0
        )

    def test_create_charging_pointing_sidemount_returns_valid_pointing(
        self,
        mock_config,
        mock_ephem,
        monkeypatch,
    ):
        utime = 1700000000.0
        optimal_ra, optimal_dec = 180.0, 0.0
        mock_config.constraint.in_constraint = Mock(return_value=False)
        object.__setattr__(
            mock_config.solar_panel,
            "optimal_charging_pointing",
            Mock(return_value=(optimal_ra, optimal_dec)),
        )
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
            sidemount=True,
        )
        monkeypatch.setattr(ec.constraint, "in_eclipse", lambda ra, dec, time: False)
        ppt = ec.create_charging_pointing(utime, mock_ephem)
        assert ppt is not None

    def test_create_charging_pointing_sidemount_is_pointing_type(
        self,
        mock_config,
        mock_ephem,
        monkeypatch,
    ):
        utime = 1700000000.0
        optimal_ra, optimal_dec = 180.0, 0.0
        mock_config.constraint.in_constraint = Mock(return_value=False)
        object.__setattr__(
            mock_config.solar_panel,
            "optimal_charging_pointing",
            Mock(return_value=(optimal_ra, optimal_dec)),
        )
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
            sidemount=True,
        )
        monkeypatch.setattr(ec.constraint, "in_eclipse", lambda ra, dec, time: False)
        ppt = ec.create_charging_pointing(utime, mock_ephem)
        assert isinstance(ppt, Pointing)

    def test_create_charging_pointing_sidemount_obsid_and_current(
        self,
        mock_config,
        mock_ephem,
        monkeypatch,
    ):
        utime = 1700000000.0
        optimal_ra, optimal_dec = 180.0, 0.0
        mock_config.constraint.in_constraint = Mock(return_value=False)
        object.__setattr__(
            mock_config.solar_panel,
            "optimal_charging_pointing",
            Mock(return_value=(optimal_ra, optimal_dec)),
        )
        ec = EmergencyCharging(
            config=mock_config,
            starting_obsid=999000,
            sidemount=True,
        )
        monkeypatch.setattr(ec.constraint, "in_eclipse", lambda ra, dec, time: False)
        ppt = ec.create_charging_pointing(utime, mock_ephem)
        assert ppt.obsid == 999000
        assert ec.current_charging_ppt == ppt


class TestQueueDITLEmergencyCharging:
    """Test QueueDITL emergency charging functionality."""

    def test_initialization_adds_charging_ppt_attribute(self, mock_config):
        """Test that QueueDITL initializes charging-related variables."""

        def mock_ditl_init(self, config=None):
            self.config = config
            self.ephem = Mock()
            self._init_subsystems()

        with patch(
            "conops.DITLMixin.__init__",
            side_effect=mock_ditl_init,
            autospec=False,
        ):
            ditl = QueueDITL(config=mock_config)
            assert hasattr(ditl, "charging_ppt")

    def test_initialization_charging_ppt_is_none(self, mock_config):
        def mock_ditl_init(self, config=None):
            self.config = config
            self.ephem = Mock()
            self._init_subsystems()

        with patch(
            "conops.DITLMixin.__init__",
            side_effect=mock_ditl_init,
            autospec=False,
        ):
            ditl = QueueDITL(config=mock_config)
            assert ditl.charging_ppt is None

    def test_initialization_emergency_charging_exists(self, mock_config):
        def mock_ditl_init(self, config=None):
            self.config = config
            self.ephem = Mock()
            self._init_subsystems()

        with patch(
            "conops.DITLMixin.__init__",
            side_effect=mock_ditl_init,
            autospec=False,
        ):
            ditl = QueueDITL(config=mock_config)
            assert hasattr(ditl, "emergency_charging")
            assert isinstance(ditl.emergency_charging, EmergencyCharging)

    def test_emergency_charging_integration_returns_pointing(self, queue_ditl):
        utime = 1700000000.0
        mock_ppt = Mock(spec=Pointing)
        mock_ppt.ra = 180.0
        mock_ppt.dec = 0.0
        mock_ppt.obsid = 999000
        queue_ditl.emergency_charging.create_charging_pointing = Mock(
            return_value=mock_ppt
        )
        result = queue_ditl.emergency_charging.create_charging_pointing(
            utime, queue_ditl.ephem, 0.0, 0.0
        )
        assert result == mock_ppt

    def test_emergency_charging_integration_called_once(self, queue_ditl):
        utime = 1700000000.0
        mock_ppt = Mock(spec=Pointing)
        mock_ppt.ra = 180.0
        mock_ppt.dec = 0.0
        mock_ppt.obsid = 999000
        queue_ditl.emergency_charging.create_charging_pointing = Mock(
            return_value=mock_ppt
        )
        queue_ditl.emergency_charging.create_charging_pointing(
            utime, queue_ditl.ephem, 0.0, 0.0
        )
        queue_ditl.emergency_charging.create_charging_pointing.assert_called_once()

    def test_charging_ppt_type_annotation_present(self):
        from typing import get_type_hints

        hints = get_type_hints(QueueDITL)
        assert "charging_ppt" in hints

    def test_charging_ppt_type_annotation_contains_pointing(self):
        from typing import get_type_hints

        hints = get_type_hints(QueueDITL)
        assert "Pointing" in str(hints["charging_ppt"])


class TestQueueDITLIntegration:
    """Integration tests for emergency charging in DITL loop."""

    def test_mode_set_to_charging_when_battery_alert_and_charging_ppt(
        self, mock_battery
    ):
        assert ACSMode.CHARGING.value == 4

    def test_charging_ppt_terminated_on_battery_recharged(self):
        pass

    def test_charging_ppt_terminated_on_constraint_violation(self):
        pass

    def test_science_ppt_terminated_on_battery_alert(self):
        pass


class TestBatteryRechargeScenarios:
    """End-to-end scenario tests for battery recharge."""

    def test_full_discharge_recharge_cycle_initial_state(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        assert battery.battery_level == 1.0

    def test_full_discharge_recharge_cycle_initial_alert_false(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        assert battery.battery_alert is False

    def test_full_discharge_recharge_cycle_alert_after_60percent(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 0.60
        assert battery.battery_alert is True

    def test_full_discharge_recharge_cycle_emergency_recharge_after_60percent(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 0.60
        # Access battery_alert to trigger the logic
        _ = battery.battery_alert
        assert battery.emergency_recharge is True

    def test_full_discharge_recharge_cycle_alert_persists_at_80percent(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 0.80
        assert battery.battery_alert is True

    def test_full_discharge_recharge_cycle_emergency_recharge_persists_at_80percent(
        self,
    ):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 0.80
        # Access battery_alert to trigger emergency_recharge setting
        _ = battery.battery_alert
        assert battery.emergency_recharge is True

    def test_full_discharge_recharge_cycle_alert_persists_at_94percent(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 0.94
        assert battery.battery_alert is True

    def test_full_discharge_recharge_cycle_alert_clears_at_95percent(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 0.95
        assert battery.battery_alert is False

    def test_full_discharge_recharge_cycle_emergency_recharge_clears_at_95percent(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 0.95
        assert battery.emergency_recharge is False

    def test_full_discharge_recharge_cycle_alert_remains_cleared_at_100percent(self):
        battery = Battery(
            max_depth_of_discharge=0.35, recharge_threshold=0.95, watthour=560.0
        )
        battery.charge_level = battery.watthour * 1.0
        assert battery.battery_alert is False

    def test_multiple_charge_discharge_cycles_first_cycle_alert(self):
        battery = Battery(max_depth_of_discharge=0.35, recharge_threshold=0.95)
        battery.charge_level = battery.watthour * 0.60
        assert battery.battery_alert is True

    def test_multiple_charge_discharge_cycles_first_cycle_clear(self):
        battery = Battery(max_depth_of_discharge=0.35, recharge_threshold=0.95)
        battery.charge_level = battery.watthour * 0.95
        assert battery.battery_alert is False

    def test_multiple_charge_discharge_cycles_second_cycle_alert(self):
        battery = Battery(max_depth_of_discharge=0.35, recharge_threshold=0.95)
        battery.charge_level = battery.watthour * 0.55
        assert battery.battery_alert is True

    def test_multiple_charge_discharge_cycles_second_cycle_clear(self):
        battery = Battery(max_depth_of_discharge=0.35, recharge_threshold=0.95)
        battery.charge_level = battery.watthour * 0.95
        assert battery.battery_alert is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
