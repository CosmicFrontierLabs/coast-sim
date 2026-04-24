"""Tests for TelescopeConfig, TelescopeType, and Telescope."""

import pytest

from conops.config import Instrument, Payload, Telescope, TelescopeConfig, TelescopeType


class TestTelescopeType:
    def test_all_members_are_strings(self) -> None:
        for member in TelescopeType:
            assert isinstance(member.value, str)

    def test_ritchey_chretien_value(self) -> None:
        assert TelescopeType.RITCHEY_CHRETIEN.value == "Ritchey-Chrétien"

    def test_tma_value(self) -> None:
        assert TelescopeType.TMA.value == "Three Mirror Anastigmat"

    def test_other_is_default_telescope_type(self) -> None:
        tc = TelescopeConfig()
        assert tc.telescope_type == TelescopeType.OTHER


class TestTelescopeConfigDefaults:
    def test_all_optional_fields_default_to_none(self) -> None:
        tc = TelescopeConfig()
        assert tc.aperture_m is None
        assert tc.focal_length_m is None
        assert tc.f_number is None
        assert tc.tube_length_m is None

    def test_plate_scale_none_without_focal_length(self) -> None:
        assert TelescopeConfig().plate_scale_arcsec_per_um is None


class TestTelescopeConfigFnumber:
    def test_f_number_derived_from_aperture_and_focal_length(self) -> None:
        tc = TelescopeConfig(aperture_m=0.5, focal_length_m=5.0)
        assert tc.f_number == pytest.approx(10.0)

    def test_explicit_consistent_f_number_accepted(self) -> None:
        tc = TelescopeConfig(aperture_m=0.5, focal_length_m=5.0, f_number=10.0)
        assert tc.f_number == pytest.approx(10.0)

    def test_inconsistent_f_number_raises(self) -> None:
        with pytest.raises(ValueError, match="inconsistent"):
            TelescopeConfig(aperture_m=0.5, focal_length_m=5.0, f_number=8.0)

    def test_f_number_alone_accepted(self) -> None:
        tc = TelescopeConfig(f_number=8.0)
        assert tc.f_number == pytest.approx(8.0)

    def test_aperture_alone_leaves_f_number_none(self) -> None:
        tc = TelescopeConfig(aperture_m=0.3)
        assert tc.f_number is None

    def test_focal_length_alone_leaves_f_number_none(self) -> None:
        tc = TelescopeConfig(focal_length_m=3.0)
        assert tc.f_number is None


class TestTelescopeConfigValidation:
    def test_aperture_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TelescopeConfig(aperture_m=0.0)

    def test_focal_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TelescopeConfig(focal_length_m=-1.0)

    def test_f_number_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TelescopeConfig(f_number=0.0)

    def test_tube_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            TelescopeConfig(tube_length_m=0.0)


class TestTelescopeConfigPlateScale:
    def test_plate_scale_known_value(self) -> None:
        # 1 m focal length → 206265 arcsec / 1e6 µm = 0.206265 arcsec/µm
        tc = TelescopeConfig(focal_length_m=1.0)
        assert tc.plate_scale_arcsec_per_um == pytest.approx(206265e-6, rel=1e-5)

    def test_plate_scale_longer_focal_length_is_smaller(self) -> None:
        tc_short = TelescopeConfig(focal_length_m=1.0)
        tc_long = TelescopeConfig(focal_length_m=10.0)
        assert tc_long.plate_scale_arcsec_per_um < tc_short.plate_scale_arcsec_per_um


class TestTelescopeConfigRoundTrip:
    def test_full_config_round_trip(self) -> None:
        tc = TelescopeConfig(
            aperture_m=0.3,
            focal_length_m=3.0,
            telescope_type=TelescopeType.RITCHEY_CHRETIEN,
            tube_length_m=0.8,
        )
        assert tc.aperture_m == pytest.approx(0.3)
        assert tc.focal_length_m == pytest.approx(3.0)
        assert tc.f_number == pytest.approx(10.0)
        assert tc.telescope_type == TelescopeType.RITCHEY_CHRETIEN
        assert tc.tube_length_m == pytest.approx(0.8)

    def test_json_serialise_deserialise(self) -> None:
        tc = TelescopeConfig(
            aperture_m=0.5,
            focal_length_m=4.0,
            telescope_type=TelescopeType.CASSEGRAIN,
            tube_length_m=0.6,
        )
        restored = TelescopeConfig.model_validate_json(tc.model_dump_json())
        assert restored.aperture_m == pytest.approx(tc.aperture_m)
        assert restored.focal_length_m == pytest.approx(tc.focal_length_m)
        assert restored.f_number == pytest.approx(tc.f_number)
        assert restored.telescope_type == tc.telescope_type
        assert restored.tube_length_m == pytest.approx(tc.tube_length_m)


class TestTelescope:
    def test_telescope_is_instrument(self) -> None:
        assert issubclass(Telescope, Instrument)

    def test_default_name(self) -> None:
        t = Telescope()
        assert t.name == "Telescope"

    def test_default_boresight_is_plus_x(self) -> None:
        t = Telescope()
        assert t.boresight == (1.0, 0.0, 0.0)

    def test_custom_boresight_accepted(self) -> None:
        t = Telescope(boresight=(0.0, 1.0, 0.0))
        assert t.boresight == (0.0, 1.0, 0.0)

    def test_non_unit_boresight_raises(self) -> None:
        with pytest.raises(ValueError, match="unit vector"):
            Telescope(boresight=(1.0, 1.0, 0.0))

    def test_zero_boresight_raises(self) -> None:
        with pytest.raises(ValueError, match="unit vector"):
            Telescope(boresight=(0.0, 0.0, 0.0))

    def test_off_axis_boresight(self) -> None:
        import math

        v = (0.0, math.sin(math.radians(45)), math.cos(math.radians(45)))
        t = Telescope(boresight=v)
        assert t.boresight == pytest.approx(v, abs=1e-10)

    def test_default_optics_is_empty_config(self) -> None:
        t = Telescope()
        assert isinstance(t.optics, TelescopeConfig)
        assert t.optics.aperture_m is None

    def test_optics_f_number_derived(self) -> None:
        t = Telescope(
            name="Primary",
            optics=TelescopeConfig(aperture_m=0.6, focal_length_m=6.0),
        )
        assert t.optics.f_number == pytest.approx(10.0)

    def test_telescope_type_accessible(self) -> None:
        t = Telescope(
            optics=TelescopeConfig(telescope_type=TelescopeType.TMA),
        )
        assert t.optics.telescope_type == TelescopeType.TMA

    def test_telescope_inherits_power_method(self) -> None:
        t = Telescope()
        assert isinstance(t.power(), float)

    def test_json_round_trip(self) -> None:
        t = Telescope(
            name="Main Scope",
            boresight=(0.0, 0.0, 1.0),
            optics=TelescopeConfig(
                aperture_m=0.3,
                focal_length_m=3.0,
                telescope_type=TelescopeType.RITCHEY_CHRETIEN,
                tube_length_m=0.8,
            ),
        )
        restored = Telescope.model_validate_json(t.model_dump_json())
        assert restored.name == "Main Scope"
        assert restored.boresight == pytest.approx((0.0, 0.0, 1.0))
        assert restored.optics.aperture_m == pytest.approx(0.3)
        assert restored.optics.f_number == pytest.approx(10.0)
        assert restored.optics.telescope_type == TelescopeType.RITCHEY_CHRETIEN

    def test_telescope_in_payload(self) -> None:
        scope = Telescope(
            name="Primary Telescope",
            optics=TelescopeConfig(aperture_m=0.6, focal_length_m=6.0),
        )
        payload = Payload(instruments=[scope])
        assert len(payload.instruments) == 1
        assert isinstance(payload.instruments[0], Telescope)

    def test_payload_with_mixed_instruments(self) -> None:
        scope = Telescope(
            name="Primary Telescope",
            optics=TelescopeConfig(aperture_m=0.6, focal_length_m=6.0),
        )
        cam = Instrument(name="Fine Guidance Sensor")
        payload = Payload(instruments=[scope, cam])
        assert len(payload.instruments) == 2
        assert isinstance(payload.instruments[0], Telescope)
        assert type(payload.instruments[1]) is Instrument
