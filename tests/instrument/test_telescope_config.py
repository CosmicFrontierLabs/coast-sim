"""Tests for TelescopeConfig, TelescopeType, and Telescope."""

import pathlib

import pytest
import rust_ephem

from conops.config import (
    Instrument,
    MissionConfig,
    Payload,
    Telescope,
    TelescopeConfig,
    TelescopeType,
)
from conops.config.constraint import Constraint


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

    def test_payload_json_round_trip_preserves_telescope(self) -> None:
        scope = Telescope(
            name="Primary Telescope",
            boresight=(0.0, 0.0, 1.0),
            optics=TelescopeConfig(
                aperture_m=0.6,
                focal_length_m=6.0,
                telescope_type=TelescopeType.RITCHEY_CHRETIEN,
            ),
        )
        cam = Instrument(name="Fine Guidance Sensor")
        payload = Payload(instruments=[scope, cam])
        restored = Payload.model_validate_json(payload.model_dump_json())
        assert len(restored.instruments) == 2
        assert isinstance(restored.instruments[0], Telescope)
        assert type(restored.instruments[1]) is Instrument
        t = restored.instruments[0]
        assert isinstance(t, Telescope)
        assert t.name == "Primary Telescope"
        assert t.boresight == pytest.approx((0.0, 0.0, 1.0))
        assert t.optics.aperture_m == pytest.approx(0.6)
        assert t.optics.f_number == pytest.approx(10.0)
        assert t.optics.telescope_type == TelescopeType.RITCHEY_CHRETIEN

    def test_instrument_type_tag_on_instrument(self) -> None:
        assert Instrument().instrument_type == "Instrument"

    def test_instrument_type_tag_on_telescope(self) -> None:
        assert Telescope().instrument_type == "Telescope"

    def test_payload_json_round_trip_plain_instrument_no_tag_needed(self) -> None:
        # Legacy JSON without instrument_type falls back to Instrument
        import json

        raw = json.dumps({"instruments": [{"name": "Sensor"}]})
        payload = Payload.model_validate_json(raw)
        assert type(payload.instruments[0]) is Instrument

    def test_payload_json_round_trip_legacy_telescope_detected_by_keys(self) -> None:
        # Legacy JSON with boresight/optics but no instrument_type → Telescope
        import json

        raw = json.dumps(
            {
                "instruments": [
                    {
                        "name": "Old Scope",
                        "boresight": [1.0, 0.0, 0.0],
                        "optics": {},
                    }
                ]
            }
        )
        payload = Payload.model_validate_json(raw)
        assert isinstance(payload.instruments[0], Telescope)

    def test_config_alias_accepted_for_optics(self) -> None:
        tc = TelescopeConfig(
            aperture_m=0.5,
            f_number=12,
            tube_length_m=1.0,
            telescope_type=TelescopeType.TMA,
        )
        t = Telescope(name="Optical Imager", config=tc)
        assert t.optics.aperture_m == pytest.approx(0.5)
        assert t.optics.f_number == pytest.approx(12.0)
        assert t.optics.tube_length_m == pytest.approx(1.0)
        assert t.optics.telescope_type == TelescopeType.TMA

    def test_config_alias_serializes_correctly_to_yaml(
        self, tmp_path: pathlib.Path
    ) -> None:

        tc = TelescopeConfig(
            aperture_m=0.5,
            f_number=12,
            tube_length_m=1.0,
            telescope_type=TelescopeType.TMA,
        )
        scope = Telescope(name="Optical Imager", config=tc)
        payload = Payload(instruments=[scope])
        config = MissionConfig(payload=payload)
        yaml_path = tmp_path / "test.yaml"
        config.to_yaml_file(str(yaml_path))
        content = yaml_path.read_text()
        assert "aperture_m: 0.5" in content
        assert "f_number: 12.0" in content
        assert "tube_length_m: 1.0" in content
        assert "Three Mirror Anastigmat" in content

    def test_payload_legacy_config_key_detected_as_telescope(self) -> None:
        import json

        raw = json.dumps({"instruments": [{"name": "Old Scope", "config": {}}]})
        payload = Payload.model_validate_json(raw)
        assert isinstance(payload.instruments[0], Telescope)


class TestTelescopeConstraint:
    def _sun_constraint(self) -> Constraint:
        return Constraint(sun_constraint=rust_ephem.SunConstraint(min_angle=45.0))

    def test_no_constraint_by_default(self) -> None:
        t = Telescope()
        assert t.constraint is None

    def test_spacecraft_constraint_none_when_no_constraint(self) -> None:
        t = Telescope()
        assert t.spacecraft_constraint is None

    def test_spacecraft_constraint_none_when_constraint_has_no_active_components(
        self,
    ) -> None:
        t = Telescope(constraint=Constraint())
        assert t.spacecraft_constraint is None

    def test_spacecraft_constraint_returns_constraint_config(self) -> None:
        t = Telescope(constraint=self._sun_constraint())
        assert t.spacecraft_constraint is not None

    def test_spacecraft_constraint_no_offset_for_boresight_plus_x(self) -> None:
        # Boresight aligned with spacecraft forward (+x) — skip boresight_offset entirely
        c = self._sun_constraint()
        t = Telescope(boresight=(1.0, 0.0, 0.0), constraint=c)
        assert t.spacecraft_constraint is c.constraint

    def test_spacecraft_constraint_offset_for_off_axis_boresight(self) -> None:
        import math

        boresight = (0.0, math.sin(math.radians(45)), math.cos(math.radians(45)))
        c = self._sun_constraint()
        t = Telescope(boresight=boresight, constraint=c)
        t_default = Telescope(boresight=(1.0, 0.0, 0.0), constraint=c)
        # +X boresight returns the base constraint object (identity short-circuit)
        assert t_default.spacecraft_constraint is c.constraint
        # Off-axis boresight returns a new, distinct boresight-offset object
        assert t.spacecraft_constraint is not None
        assert t.spacecraft_constraint is not c.constraint

    def test_combined_telescope_spacecraft_constraint_no_telescopes(self) -> None:
        payload = Payload(instruments=[Instrument()])
        assert payload.combined_telescope_spacecraft_constraint() is None

    def test_combined_telescope_spacecraft_constraint_no_constraints(self) -> None:
        scope = Telescope()
        payload = Payload(instruments=[scope])
        assert payload.combined_telescope_spacecraft_constraint() is None

    def test_combined_telescope_spacecraft_constraint_single_telescope(self) -> None:
        scope = Telescope(constraint=self._sun_constraint())
        payload = Payload(instruments=[scope])
        result = payload.combined_telescope_spacecraft_constraint()
        assert result is not None

    def test_combined_telescope_spacecraft_constraint_multiple_telescopes(self) -> None:
        c = self._sun_constraint()
        scope1 = Telescope(name="T1", constraint=c)
        scope2 = Telescope(name="T2", boresight=(0.0, 0.0, 1.0), constraint=c)
        payload = Payload(instruments=[scope1, scope2])
        result = payload.combined_telescope_spacecraft_constraint()
        assert result is not None

    def test_combined_skips_non_telescope_instruments(self) -> None:
        cam = Instrument(name="Camera")
        scope = Telescope(constraint=self._sun_constraint())
        payload = Payload(instruments=[cam, scope])
        result = payload.combined_telescope_spacecraft_constraint()
        assert result is not None

    def test_invalidate_spacecraft_constraint_cache(self) -> None:
        c = self._sun_constraint()
        t = Telescope(boresight=(0.0, 0.0, 1.0), constraint=c)
        first = t.spacecraft_constraint
        assert first is not None
        # Cache is populated; invalidate and verify a fresh object is returned
        t.invalidate_spacecraft_constraint_cache()
        second = t.spacecraft_constraint
        assert second is not None
        # After invalidation a new ConstraintConfig is computed (different object)
        assert first is not second


class TestTelescopeConstraintMissionConfig:
    """Integration tests: Telescope.constraint propagates into MissionConfig."""

    def test_telescope_constraint_propagates_to_mission_constraint(self) -> None:
        from conops.config.constraint import Constraint

        scope = Telescope(
            constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
            )
        )
        payload = Payload(instruments=[scope])
        config = MissionConfig(payload=payload)
        assert config.constraint.telescope_hard_constraint is not None

    def test_no_telescope_constraint_leaves_field_none(self) -> None:

        scope = Telescope()  # no constraint
        payload = Payload(instruments=[scope])
        config = MissionConfig(payload=payload)
        assert config.constraint.telescope_hard_constraint is None

    def test_plain_instrument_payload_leaves_telescope_field_none(self) -> None:

        payload = Payload(instruments=[Instrument()])
        config = MissionConfig(payload=payload)
        assert config.constraint.telescope_hard_constraint is None

    def test_off_axis_telescope_constraint_appears_in_roll_dependent_constraint(
        self,
    ) -> None:
        # Off-axis boresight forces boresight_offset() → BoresightOffsetConstraint leaf.
        # roll_dependent_constraint should tree-walk telescope_hard_constraint and find it.

        scope = Telescope(
            boresight=(0.0, 0.0, 1.0),
            constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
            ),
        )
        payload = Payload(instruments=[scope])
        config = MissionConfig(payload=payload)
        assert config.constraint.telescope_hard_constraint is not None
        assert config.constraint.roll_dependent_constraint is not None

    def test_on_axis_telescope_constraint_not_in_roll_dependent_constraint(
        self,
    ) -> None:
        # Default boresight (1,0,0) — spacecraft_constraint returns the base SunConstraint
        # directly, with no boresight_offset wrapper.  The tree walk finds no
        # BoresightOffsetConstraint leaves, so roll_dependent_constraint is None
        # (assuming no star-tracker or radiator constraints are configured).
        from conops.config.spacecraft_bus import SpacecraftBus
        from conops.config.star_tracker import StarTrackerConfiguration

        scope = Telescope(
            boresight=(1.0, 0.0, 0.0),
            constraint=Constraint(
                sun_constraint=rust_ephem.SunConstraint(min_angle=45.0)
            ),
        )
        payload = Payload(instruments=[scope])
        # Use a bus with no star trackers so the only potential source is the telescope
        bus = SpacecraftBus(star_trackers=StarTrackerConfiguration(star_trackers=[]))
        config = MissionConfig(payload=payload, spacecraft_bus=bus)
        assert config.constraint.telescope_hard_constraint is not None
        assert config.constraint.roll_dependent_constraint is None
