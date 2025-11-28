"""Tests for Pass communications functionality."""

import pytest

from conops import Pass
from conops.config import (
    AntennaPointing,
    AntennaType,
    BandCapability,
    CommunicationsSystem,
    Polarization,
)


class TestPassCommunications:
    """Test Pass with CommunicationsSystem integration."""

    def test_pass_with_no_comms_config(self, mock_constraint, mock_acs_config):
        """Test Pass without communications configuration."""
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=None,
            station="SGS",
            begin=1514764800.0,
            length=480.0,
        )
        assert p.comms_config is None

    def test_pass_with_comms_config(self, mock_constraint, mock_acs_config):
        """Test Pass with communications configuration."""
        comms = CommunicationsSystem(
            name="S-band",
            band_capabilities=[
                BandCapability(band="S", uplink_rate_mbps=1.0, downlink_rate_mbps=5.0)
            ],
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=480.0,
        )
        assert p.comms_config is comms
        assert p.comms_config.name == "S-band"


class TestPassPointingError:
    """Test Pass.pointing_error method."""

    def test_pointing_error_zero(self, basic_pass):
        """Test pointing error when spacecraft is perfectly aligned."""
        error = basic_pass.pointing_error(
            spacecraft_ra=10.0,
            spacecraft_dec=20.0,
            target_ra=10.0,
            target_dec=20.0,
        )
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_pointing_error_nonzero(self, basic_pass):
        """Test pointing error with misalignment."""
        error = basic_pass.pointing_error(
            spacecraft_ra=10.0,
            spacecraft_dec=20.0,
            target_ra=15.0,
            target_dec=20.0,
        )
        # Should be approximately 5 degrees (small angle approximation)
        assert error > 0.0
        assert error < 10.0  # Reasonable range


class TestPassCanCommunicate:
    """Test Pass.can_communicate method."""

    def test_can_communicate_no_comms_config(self, basic_pass):
        """Test communication check with no comms config (should always return True)."""
        basic_pass.utime = [1514764800.0, 1514764900.0]
        basic_pass.ra = [10.0, 12.0]
        basic_pass.dec = [20.0, 22.0]

        # Without comms config, communication is always possible
        assert basic_pass.can_communicate(0.0, 0.0)
        assert basic_pass.can_communicate(90.0, 0.0)

    def test_can_communicate_with_fixed_antenna(self, mock_constraint, mock_acs_config):
        """Test communication with fixed antenna and pointing accuracy."""
        comms = CommunicationsSystem(
            name="Fixed S-band",
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=5.0)],
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.FIXED),
            pointing_accuracy_deg=5.0,
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=480.0,
        )
        p.utime = [1514764800.0, 1514764900.0]
        p.ra = [10.0, 12.0]
        p.dec = [20.0, 22.0]

        # Within pointing accuracy (spacecraft pointing near target)
        assert p.can_communicate(10.5, 20.5)

        # Outside pointing accuracy
        assert not p.can_communicate(50.0, 50.0)

    def test_can_communicate_with_omni_antenna(self, mock_constraint, mock_acs_config):
        """Test communication with omnidirectional antenna."""
        comms = CommunicationsSystem(
            name="Omni S-band",
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=1.0)],
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=480.0,
        )
        p.utime = [1514764800.0, 1514764900.0]
        p.ra = [10.0, 12.0]
        p.dec = [20.0, 22.0]

        # Omni antenna can always communicate
        assert p.can_communicate(0.0, 0.0)
        assert p.can_communicate(90.0, 0.0)
        assert p.can_communicate(180.0, 0.0)


class TestPassDataRates:
    """Test Pass data rate methods."""

    def test_get_data_rate_no_comms(self, basic_pass):
        """Test data rate with no comms config."""
        assert basic_pass.get_data_rate("S", "downlink") == 0.0
        assert basic_pass.get_data_rate("X", "uplink") == 0.0

    def test_get_downlink_rate(self, mock_constraint, mock_acs_config):
        """Test getting downlink rate."""
        comms = CommunicationsSystem(
            band_capabilities=[
                BandCapability(
                    band="X", uplink_rate_mbps=10.0, downlink_rate_mbps=100.0
                )
            ]
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=480.0,
        )

        assert p.get_data_rate("X", "downlink") == 100.0
        assert p.get_data_rate("S", "downlink") == 0.0  # Band not supported

    def test_get_uplink_rate(self, mock_constraint, mock_acs_config):
        """Test getting uplink rate."""
        comms = CommunicationsSystem(
            band_capabilities=[
                BandCapability(
                    band="X", uplink_rate_mbps=10.0, downlink_rate_mbps=100.0
                )
            ]
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=480.0,
        )

        assert p.get_data_rate("X", "uplink") == 10.0
        assert p.get_data_rate("S", "uplink") == 0.0  # Band not supported

    def test_get_data_rate_invalid_direction(self, mock_constraint, mock_acs_config):
        """Test data rate with invalid direction."""
        comms = CommunicationsSystem(
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=5.0)]
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=480.0,
        )

        assert p.get_data_rate("S", "invalid") == 0.0


class TestPassDataVolume:
    """Test Pass.calculate_data_volume method."""

    def test_calculate_data_volume_no_comms(self, basic_pass):
        """Test data volume calculation with no comms config."""
        assert basic_pass.calculate_data_volume("S") == 0.0

    def test_calculate_data_volume_no_length(self, mock_constraint, mock_acs_config):
        """Test data volume calculation when pass has no length."""
        comms = CommunicationsSystem(
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=5.0)]
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=None,
        )

        assert p.calculate_data_volume("S") == 0.0

    def test_calculate_downlink_volume(self, mock_constraint, mock_acs_config):
        """Test calculating downlink data volume."""
        comms = CommunicationsSystem(
            band_capabilities=[
                BandCapability(band="X", downlink_rate_mbps=100.0)  # 100 Mbps
            ]
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=600.0,  # 10 minutes = 600 seconds
        )

        # 100 Mbps * 600 seconds = 60,000 Megabits
        volume = p.calculate_data_volume("X", "downlink")
        assert volume == 60000.0

    def test_calculate_uplink_volume(self, mock_constraint, mock_acs_config):
        """Test calculating uplink data volume."""
        comms = CommunicationsSystem(
            band_capabilities=[BandCapability(band="S", uplink_rate_mbps=2.0)]  # 2 Mbps
        )
        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=300.0,  # 5 minutes = 300 seconds
        )

        # 2 Mbps * 300 seconds = 600 Megabits
        volume = p.calculate_data_volume("S", "uplink")
        assert volume == 600.0


class TestPassCommunicationsIntegration:
    """Integration tests for Pass with CommunicationsSystem."""

    def test_complete_pass_with_multi_band_comms(
        self, mock_constraint, mock_acs_config
    ):
        """Test Pass with multi-band communications system."""
        comms = CommunicationsSystem(
            name="Dual S/X",
            band_capabilities=[
                BandCapability(band="S", uplink_rate_mbps=1.0, downlink_rate_mbps=10.0),
                BandCapability(
                    band="X", uplink_rate_mbps=0.0, downlink_rate_mbps=150.0
                ),
            ],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED, gimbal_range_deg=45.0
            ),
            polarization=Polarization.DUAL,
            pointing_accuracy_deg=5.0,
        )

        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="SGS",
            begin=1514764800.0,
            length=600.0,  # 10 minute pass
        )

        # Verify configuration
        assert p.comms_config.name == "Dual S/X"

        # Check data rates
        assert p.get_data_rate("S", "downlink") == 10.0
        assert p.get_data_rate("X", "downlink") == 150.0
        assert p.get_data_rate("S", "uplink") == 1.0
        assert p.get_data_rate("X", "uplink") == 0.0

        # Calculate data volumes
        s_volume = p.calculate_data_volume("S", "downlink")
        x_volume = p.calculate_data_volume("X", "downlink")

        assert s_volume == 6000.0  # 10 Mbps * 600s
        assert x_volume == 90000.0  # 150 Mbps * 600s

    def test_pass_with_high_rate_ka_band(self, mock_constraint, mock_acs_config):
        """Test Pass with high-rate Ka-band system."""
        comms = CommunicationsSystem(
            name="Ka-band HGA",
            band_capabilities=[
                BandCapability(
                    band="Ka", uplink_rate_mbps=5.0, downlink_rate_mbps=800.0
                )
            ],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED, gimbal_range_deg=90.0
            ),
            polarization=Polarization.CIRCULAR_RIGHT,
            pointing_accuracy_deg=0.5,  # Very tight for Ka-band
        )

        p = Pass(
            constraint=mock_constraint,
            acs_config=mock_acs_config,
            comms_config=comms,
            station="DSN",
            begin=1514764800.0,
            length=1800.0,  # 30 minute pass
        )

        # High data rate
        assert p.get_data_rate("Ka", "downlink") == 800.0

        # Large data volume
        volume = p.calculate_data_volume("Ka", "downlink")
        assert volume == 1440000.0  # 800 Mbps * 1800s = 1.44 million Megabits
