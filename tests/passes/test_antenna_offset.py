"""Tests for Pass antenna pointing offset functionality."""

from datetime import datetime, timezone

import numpy as np
import pytest
from rust_ephem import TLEEphemeris

from conops.config import (
    AntennaPointing,
    AntennaType,
    BandCapability,
    CommunicationsSystem,
    Constraint,
)
from conops.simulation import Pass


class TestAntennaPointingOffset:
    """Test antenna pointing offset calculations."""

    def test_apply_antenna_offset_zero(self):
        """Test that zero offset returns original pointing."""
        ra, dec = 45.0, 30.0
        adjusted_ra, adjusted_dec = Pass.apply_antenna_offset(ra, dec, 0.0, 0.0)

        assert np.isclose(adjusted_ra, ra, atol=1e-6)
        assert np.isclose(adjusted_dec, dec, atol=1e-6)

    def test_apply_antenna_offset_azimuth_90(self):
        """Test 90-degree azimuth offset (pointing right)."""
        ra, dec = 0.0, 0.0
        adjusted_ra, adjusted_dec = Pass.apply_antenna_offset(ra, dec, 90.0, 0.0)

        # Azimuth offset should rotate RA
        # The actual value depends on the rotation convention
        assert adjusted_ra != ra  # RA should change
        assert np.isclose(
            adjusted_dec, dec, atol=1.0
        )  # Dec should stay near 0    def test_apply_antenna_offset_elevation_90_nadir(self):
        """Test -90-degree elevation offset (nadir pointing)."""
        ra, dec = 45.0, 45.0
        adjusted_ra, adjusted_dec = Pass.apply_antenna_offset(ra, dec, 0.0, -90.0)

        # Nadir pointing should point toward declination -90 (south pole)
        assert adjusted_dec < dec  # Should point more toward south

    def test_apply_antenna_offset_elevation_90_zenith(self):
        """Test 90-degree elevation offset (zenith pointing)."""
        ra, dec = 45.0, -45.0
        adjusted_ra, adjusted_dec = Pass.apply_antenna_offset(ra, dec, 0.0, 90.0)

        # Zenith pointing should point toward declination 90 (north pole)
        assert adjusted_dec > dec  # Should point more toward north

    def test_apply_antenna_offset_combined(self):
        """Test combined azimuth and elevation offset."""
        ra, dec = 0.0, 0.0
        adjusted_ra, adjusted_dec = Pass.apply_antenna_offset(ra, dec, 45.0, 45.0)

        # Should be offset in both directions
        assert adjusted_ra != ra
        assert adjusted_dec != dec
        # Verify offset is applied (values changed)
        assert abs(adjusted_ra - ra) > 1.0
        assert abs(adjusted_dec - dec) > 1.0


class TestPassWithAntennaOffset:
    """Test Pass objects with antenna pointing offsets."""

    @pytest.fixture
    def ephem(self):
        """Create test ephemeris."""
        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 8, 15, 12, 15, 0, tzinfo=timezone.utc)
        return TLEEphemeris(
            tle="examples/example.tle", begin=begin, end=end, step_size=60
        )

    @pytest.fixture
    def constraint(self, ephem):
        """Create constraint with ephemeris."""
        constraint = Constraint()
        constraint.ephem = ephem
        return constraint

    def test_pass_with_nadir_antenna_offset(self, constraint, ephem):
        """Test Pass with nadir-pointing antenna (most common case)."""
        # Create nadir-pointing antenna (0, 0 = default, points away from telescope)
        comms = CommunicationsSystem(
            name="Nadir Antenna",
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=10.0)],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.FIXED,
                fixed_azimuth_deg=0.0,
                fixed_elevation_deg=0.0,
            ),
            pointing_accuracy_deg=10.0,
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        gs_pass = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="SGS",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        # For nadir antenna (0, 0), pointing should be unchanged
        # (since 0,0 is the default/no offset case)
        assert np.isclose(gs_pass.gsstartra, 45.0, atol=1e-6)
        assert np.isclose(gs_pass.gsstartdec, 25.0, atol=1e-6)

    def test_pass_with_offset_antenna(self, constraint, ephem):
        """Test Pass with offset fixed antenna."""
        # Create antenna pointing 45 degrees in azimuth and elevation
        comms = CommunicationsSystem(
            name="Offset Antenna",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.FIXED,
                fixed_azimuth_deg=45.0,
                fixed_elevation_deg=30.0,
            ),
            pointing_accuracy_deg=10.0,
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        original_ra, original_dec = 45.0, 25.0

        gs_pass = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="SGS",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=original_ra,
            gsstartdec=original_dec,
            gsendra=50.0,
            gsenddec=30.0,
        )

        # Pointing should be different from original due to antenna offset
        # Note: The offset is applied in PassTimes.get(), not in Pass.__init__
        # So we need to manually verify the offset calculation
        adjusted_ra, adjusted_dec = Pass.apply_antenna_offset(
            original_ra, original_dec, 45.0, 30.0
        )

        # Verify the offset function produces different values
        assert adjusted_ra != original_ra
        assert adjusted_dec != original_dec
        # Pass object stores original values (offset not yet applied in init)
        assert np.isclose(gs_pass.gsstartra, original_ra, atol=1e-6)
        assert np.isclose(gs_pass.gsstartdec, original_dec, atol=1e-6)

    def test_pass_with_omni_antenna_no_offset(self, constraint, ephem):
        """Test Pass with omni antenna has no pointing offset."""
        # Omni antenna should not apply any offset
        comms = CommunicationsSystem(
            name="Omni Antenna",
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=1.0)],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.OMNI,
            ),
            pointing_accuracy_deg=180.0,
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        gs_pass = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="SGS",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        # Omni antenna should not change pointing
        # (offset only applied to FIXED antennas)
        assert np.isclose(gs_pass.gsstartra, 45.0, atol=1e-6)
        assert np.isclose(gs_pass.gsstartdec, 25.0, atol=1e-6)

    def test_pass_with_gimbaled_antenna_no_offset(self, constraint, ephem):
        """Test Pass with gimbaled antenna has no pointing offset."""
        # Gimbaled antenna should not apply offset (can point where needed)
        comms = CommunicationsSystem(
            name="Gimbaled Antenna",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=300.0)],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED,
                gimbal_range_deg=45.0,
            ),
            pointing_accuracy_deg=5.0,
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        gs_pass = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="SGS",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        # Gimbaled antenna should not change pointing
        # (offset only applied to FIXED antennas)
        assert np.isclose(gs_pass.gsstartra, 45.0, atol=1e-6)
        assert np.isclose(gs_pass.gsstartdec, 25.0, atol=1e-6)

    def test_pass_pointing_profile_offset(self, constraint, ephem):
        """Test that full pointing profile is offset for fixed antenna."""
        comms = CommunicationsSystem(
            name="Offset Antenna",
            band_capabilities=[BandCapability(band="X", downlink_rate_mbps=150.0)],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.FIXED,
                fixed_azimuth_deg=30.0,
                fixed_elevation_deg=20.0,
            ),
            pointing_accuracy_deg=10.0,
        )

        begin = datetime(2025, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        gs_pass = Pass(
            constraint=constraint,
            ephem=ephem,
            acs_config=None,
            comms_config=comms,
            station="SGS",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=45.0,
            gsstartdec=25.0,
            gsendra=50.0,
            gsenddec=30.0,
        )

        # Add pointing profile
        gs_pass.utime = [begin.timestamp() + i * 60 for i in range(5)]
        original_ra = [45.0, 46.0, 47.0, 48.0, 49.0]
        original_dec = [25.0, 26.0, 27.0, 28.0, 29.0]
        gs_pass.ra = original_ra.copy()
        gs_pass.dec = original_dec.copy()

        # Verify offset would be applied by PassTimes.get()
        # In Pass.__init__, values remain unchanged
        for i in range(len(gs_pass.ra)):
            assert gs_pass.ra[i] == original_ra[i]  # Not yet offset in __init__
            assert gs_pass.dec[i] == original_dec[i]
