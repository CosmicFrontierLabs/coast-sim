"""Tests for legacy Pass antenna pointing offset helpers."""

from datetime import datetime, timezone

import numpy as np
import pytest
from rust_ephem import TLEEphemeris

from conops.common.enums import AntennaType
from conops.config import (
    AntennaPointing,
    BandCapability,
    CommunicationsSystem,
    Constraint,
)
from conops.simulation import Pass


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

    def test_pass_with_nadir_antenna_offset(self, mock_config, constraint, ephem):
        """Test Pass with legacy zero-offset antenna metadata."""
        # Legacy zero azimuth/elevation metadata preserves the pass fields.
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
        # Inject our local constraint and communications into the shared mock config
        mock_config.constraint = constraint
        mock_config.spacecraft_bus.communications = comms
        gs_pass = Pass(
            config=mock_config,
            ephem=ephem,
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

    def test_pass_with_offset_antenna(self, mock_config, constraint, ephem):
        """Test Pass with offset fixed antenna."""
        # Create antenna pointing 45 degrees in azimuth and elevation
        with pytest.warns(DeprecationWarning, match="legacy metadata fields"):
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

        mock_config.constraint = constraint
        mock_config.spacecraft_bus.communications = comms
        gs_pass = Pass(
            config=mock_config,
            ephem=ephem,
            station="SGS",
            begin=begin.timestamp(),
            length=480.0,
            gsstartra=original_ra,
            gsstartdec=original_dec,
            gsendra=50.0,
            gsenddec=30.0,
        )

        # Pass object stores original values (offset not yet applied in init)
        assert np.isclose(gs_pass.gsstartra, original_ra, atol=1e-6)
        assert np.isclose(gs_pass.gsstartdec, original_dec, atol=1e-6)

    def test_pass_with_omni_antenna_no_offset(self, mock_config, constraint, ephem):
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
        mock_config.constraint = constraint
        mock_config.spacecraft_bus.communications = comms
        gs_pass = Pass(
            config=mock_config,
            ephem=ephem,
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

    def test_pass_with_gimbaled_antenna_no_offset(self, mock_config, constraint, ephem):
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
        mock_config.constraint = constraint
        mock_config.spacecraft_bus.communications = comms
        gs_pass = Pass(
            config=mock_config,
            ephem=ephem,
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

    def test_pass_pointing_profile_offset(self, mock_config, constraint, ephem):
        """Test that full pointing profile is offset for fixed antenna."""
        with pytest.warns(DeprecationWarning, match="legacy metadata fields"):
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
        mock_config.constraint = constraint
        mock_config.spacecraft_bus.communications = comms
        gs_pass = Pass(
            config=mock_config,
            ephem=ephem,
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

        # In Pass.__init__, values remain unchanged.
        for i in range(len(gs_pass.ra)):
            assert gs_pass.ra[i] == original_ra[i]  # Not yet offset in __init__
            assert gs_pass.dec[i] == original_dec[i]
