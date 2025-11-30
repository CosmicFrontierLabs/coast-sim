"""Tests for communications system configuration."""

import pytest

from conops.config import (
    AntennaPointing,
    AntennaType,
    BandCapability,
    CommunicationsSystem,
    Polarization,
)


class TestBandCapability:
    """Test BandCapability model."""

    def test_band_capability_creation(self):
        """Test creating a band capability."""
        band = BandCapability(band="X", uplink_rate_mbps=5.0, downlink_rate_mbps=50.0)
        assert band.band == "X"
        assert band.uplink_rate_mbps == 5.0
        assert band.downlink_rate_mbps == 50.0

    def test_band_capability_defaults(self):
        """Test default values for band capability."""
        band = BandCapability(band="S")
        assert band.band == "S"
        # Defaults now auto-populate for known bands
        assert band.uplink_rate_mbps == 2.0
        assert band.downlink_rate_mbps == 10.0

    def test_band_capability_validation(self):
        """Test validation of band capability values."""
        # Should not allow negative rates
        with pytest.raises(Exception):  # pydantic validation error
            BandCapability(band="X", uplink_rate_mbps=-1.0)

        with pytest.raises(Exception):  # pydantic validation error
            BandCapability(band="X", downlink_rate_mbps=-1.0)

    def test_supported_bands(self):
        """Test all supported band types."""
        for band_type in ["S", "X", "Ka", "Ku", "L", "C", "K"]:
            band = BandCapability(band=band_type, downlink_rate_mbps=10.0)
            assert band.band == band_type


class TestAntennaPointing:
    """Test AntennaPointing model."""

    def test_omni_antenna(self):
        """Test omnidirectional antenna configuration."""
        pointing = AntennaPointing(antenna_type=AntennaType.OMNI)
        assert pointing.antenna_type == AntennaType.OMNI
        assert not pointing.is_nadir_pointing()

    def test_fixed_nadir_pointing(self):
        """Test fixed nadir-pointing antenna (default)."""
        pointing = AntennaPointing(
            antenna_type=AntennaType.FIXED,
            fixed_azimuth_deg=0.0,
            fixed_elevation_deg=0.0,
        )
        assert pointing.antenna_type == AntennaType.FIXED
        assert pointing.is_nadir_pointing()

    def test_fixed_non_nadir_pointing(self):
        """Test fixed antenna not pointing nadir."""
        pointing = AntennaPointing(
            antenna_type=AntennaType.FIXED,
            fixed_azimuth_deg=180.0,
            fixed_elevation_deg=0.0,
        )
        assert pointing.antenna_type == AntennaType.FIXED
        assert not pointing.is_nadir_pointing()  # Zenith pointing

    def test_gimbaled_antenna(self):
        """Test gimbaled antenna configuration."""
        pointing = AntennaPointing(
            antenna_type=AntennaType.GIMBALED, gimbal_range_deg=30.0
        )
        assert pointing.antenna_type == AntennaType.GIMBALED
        assert pointing.gimbal_range_deg == 30.0
        assert not pointing.is_nadir_pointing()

    def test_antenna_pointing_validation(self):
        """Test validation of antenna pointing angles."""
        # Azimuth must be 0-360
        with pytest.raises(Exception):
            AntennaPointing(antenna_type=AntennaType.FIXED, fixed_azimuth_deg=400.0)

        # Elevation must be -90 to 90
        with pytest.raises(Exception):
            AntennaPointing(antenna_type=AntennaType.FIXED, fixed_elevation_deg=100.0)

        # Gimbal range must be 0-180
        with pytest.raises(Exception):
            AntennaPointing(antenna_type=AntennaType.GIMBALED, gimbal_range_deg=200.0)


class TestPolarization:
    """Test Polarization enum."""

    def test_all_polarizations(self):
        """Test all polarization types are accessible."""
        assert Polarization.LINEAR_HORIZONTAL == "linear_horizontal"
        assert Polarization.LINEAR_VERTICAL == "linear_vertical"
        assert Polarization.CIRCULAR_RIGHT == "circular_right"
        assert Polarization.CIRCULAR_LEFT == "circular_left"
        assert Polarization.DUAL == "dual"


class TestCommunicationsSystem:
    """Test CommunicationsSystem model."""

    def test_default_communications_system(self):
        """Test default communications system."""
        comms = CommunicationsSystem()
        assert comms.name == "Default Comms"
        assert len(comms.band_capabilities) == 1
        assert comms.band_capabilities[0].band == "S"
        assert comms.antenna_pointing.antenna_type == AntennaType.FIXED
        assert comms.polarization == Polarization.CIRCULAR_RIGHT
        assert comms.pointing_accuracy_deg == 5.0

    def test_custom_communications_system(self):
        """Test creating a custom communications system."""
        comms = CommunicationsSystem(
            name="X-band Comms",
            band_capabilities=[
                BandCapability(
                    band="X", uplink_rate_mbps=10.0, downlink_rate_mbps=100.0
                )
            ],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED, gimbal_range_deg=45.0
            ),
            polarization=Polarization.CIRCULAR_LEFT,
            pointing_accuracy_deg=2.0,
        )
        assert comms.name == "X-band Comms"
        assert len(comms.band_capabilities) == 1
        assert comms.band_capabilities[0].band == "X"
        assert comms.antenna_pointing.antenna_type == AntennaType.GIMBALED
        assert comms.antenna_pointing.gimbal_range_deg == 45.0
        assert comms.polarization == Polarization.CIRCULAR_LEFT
        assert comms.pointing_accuracy_deg == 2.0

    def test_multi_band_system(self):
        """Test communications system with multiple bands."""
        comms = CommunicationsSystem(
            name="Multi-band Comms",
            band_capabilities=[
                BandCapability(band="S", uplink_rate_mbps=2.0, downlink_rate_mbps=10.0),
                BandCapability(
                    band="X", uplink_rate_mbps=10.0, downlink_rate_mbps=100.0
                ),
                BandCapability(
                    band="Ka", uplink_rate_mbps=20.0, downlink_rate_mbps=500.0
                ),
            ],
        )
        assert len(comms.band_capabilities) == 3
        assert comms.band_capabilities[0].band == "S"
        assert comms.band_capabilities[1].band == "X"
        assert comms.band_capabilities[2].band == "Ka"

    def test_get_band(self):
        """Test retrieving a specific band capability."""
        comms = CommunicationsSystem(
            band_capabilities=[
                BandCapability(band="S", downlink_rate_mbps=10.0),
                BandCapability(band="X", downlink_rate_mbps=100.0),
            ]
        )
        s_band = comms.get_band("S")
        assert s_band is not None
        assert s_band.band == "S"
        assert s_band.downlink_rate_mbps == 10.0

        x_band = comms.get_band("X")
        assert x_band is not None
        assert x_band.band == "X"
        assert x_band.downlink_rate_mbps == 100.0

        # Non-existent band
        ka_band = comms.get_band("Ka")
        assert ka_band is None

    def test_get_downlink_rate(self):
        """Test getting downlink rate for a band."""
        comms = CommunicationsSystem(
            band_capabilities=[
                BandCapability(band="X", downlink_rate_mbps=150.0),
            ]
        )
        assert comms.get_downlink_rate("X") == 150.0
        assert comms.get_downlink_rate("S") == 0.0  # Not supported

    def test_get_uplink_rate(self):
        """Test getting uplink rate for a band."""
        comms = CommunicationsSystem(
            band_capabilities=[
                BandCapability(band="X", uplink_rate_mbps=15.0),
            ]
        )
        assert comms.get_uplink_rate("X") == 15.0
        assert comms.get_uplink_rate("Ka") == 0.0  # Not supported

    def test_can_communicate_omni(self):
        """Test communication check with omnidirectional antenna."""
        comms = CommunicationsSystem(
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
            pointing_accuracy_deg=5.0,
        )
        # Omni antenna should always be able to communicate
        assert comms.can_communicate(0.0)
        assert comms.can_communicate(10.0)
        assert comms.can_communicate(90.0)
        assert comms.can_communicate(180.0)

    def test_can_communicate_fixed(self):
        """Test communication check with fixed antenna."""
        comms = CommunicationsSystem(
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.FIXED),
            pointing_accuracy_deg=5.0,
        )
        # Within pointing accuracy
        assert comms.can_communicate(0.0)
        assert comms.can_communicate(3.0)
        assert comms.can_communicate(5.0)

        # Outside pointing accuracy
        assert not comms.can_communicate(6.0)
        assert not comms.can_communicate(10.0)
        assert not comms.can_communicate(90.0)

    def test_can_communicate_gimbaled(self):
        """Test communication check with gimbaled antenna."""
        comms = CommunicationsSystem(
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED, gimbal_range_deg=30.0
            ),
            pointing_accuracy_deg=10.0,
        )
        # Within pointing accuracy
        assert comms.can_communicate(5.0)
        assert comms.can_communicate(10.0)

        # Outside pointing accuracy
        assert not comms.can_communicate(11.0)
        assert not comms.can_communicate(20.0)

    def test_pointing_accuracy_validation(self):
        """Test pointing accuracy validation."""
        # Valid range
        comms = CommunicationsSystem(pointing_accuracy_deg=1.0)
        assert comms.pointing_accuracy_deg == 1.0

        # Should not allow negative
        with pytest.raises(Exception):
            CommunicationsSystem(pointing_accuracy_deg=-1.0)

        # Should not allow > 180
        with pytest.raises(Exception):
            CommunicationsSystem(pointing_accuracy_deg=200.0)


class TestCommunicationsIntegration:
    """Integration tests for communications system."""

    def test_complete_x_band_system(self):
        """Test a complete X-band communications system configuration."""
        comms = CommunicationsSystem(
            name="High-Rate X-band",
            band_capabilities=[
                BandCapability(
                    band="X", uplink_rate_mbps=20.0, downlink_rate_mbps=300.0
                )
            ],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED, gimbal_range_deg=60.0
            ),
            polarization=Polarization.CIRCULAR_RIGHT,
            pointing_accuracy_deg=3.0,
        )

        # Verify configuration
        assert comms.name == "High-Rate X-band"
        assert comms.get_downlink_rate("X") == 300.0
        assert comms.get_uplink_rate("X") == 20.0
        assert comms.antenna_pointing.antenna_type == AntennaType.GIMBALED
        assert comms.can_communicate(2.0)
        assert not comms.can_communicate(5.0)

    def test_nadir_pointing_s_band(self):
        """Test a nadir-pointing S-band system (typical for LEO)."""
        comms = CommunicationsSystem(
            name="LEO S-band",
            band_capabilities=[
                BandCapability(band="S", uplink_rate_mbps=1.0, downlink_rate_mbps=5.0)
            ],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.FIXED,
                fixed_azimuth_deg=0.0,
                fixed_elevation_deg=0.0,
            ),
            polarization=Polarization.CIRCULAR_RIGHT,
            pointing_accuracy_deg=10.0,
        )

        assert comms.antenna_pointing.is_nadir_pointing()
        assert comms.get_downlink_rate("S") == 5.0
        assert comms.can_communicate(9.0)
        assert not comms.can_communicate(11.0)

    def test_dual_band_system(self):
        """Test a dual-band S/X communications system."""
        comms = CommunicationsSystem(
            name="S/X Dual Band",
            band_capabilities=[
                BandCapability(band="S", uplink_rate_mbps=1.0, downlink_rate_mbps=10.0),
                BandCapability(
                    band="X", uplink_rate_mbps=10.0, downlink_rate_mbps=150.0
                ),
            ],
            antenna_pointing=AntennaPointing(
                antenna_type=AntennaType.GIMBALED, gimbal_range_deg=45.0
            ),
            polarization=Polarization.DUAL,
            pointing_accuracy_deg=5.0,
        )

        # Both bands should be available
        assert comms.get_downlink_rate("S") == 10.0
        assert comms.get_downlink_rate("X") == 150.0
        assert comms.get_uplink_rate("S") == 1.0
        assert comms.get_uplink_rate("X") == 10.0
        assert comms.polarization == Polarization.DUAL
