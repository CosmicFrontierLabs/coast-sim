"""Tests for reaction wheel and MOI configuration parsing."""

from pathlib import Path

from conops import MissionConfig


def test_example_config_contains_wheels_and_moi():
    # Determine repository root by walking up from tests directory
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "examples" / "example_config.json"
    cfg = MissionConfig.from_json_file(str(example))

    acs = cfg.spacecraft_bus.attitude_control

    # Wheel flag enabled in example
    assert acs.wheel_enabled is True

    # Wheels list present and has three entries
    assert isinstance(acs.wheels, list)
    assert len(acs.wheels) == 3

    # Each wheel should have expected keys
    for w in acs.wheels:
        assert "name" in w
        assert "orientation" in w
        assert "max_torque" in w
        assert "max_momentum" in w

    # spacecraft_moi parsed as sequence of three values
    moi = acs.spacecraft_moi
    assert len(moi) == 3
    assert all(isinstance(x, (int, float)) for x in moi)


def test_legacy_single_wheel_fields_present_when_used():
    # Create minimal config with legacy fields
    data = {
        "spacecraft_bus": {
            "attitude_control": {
                "wheel_enabled": True,
                "wheel_max_torque": 0.2,
                "wheel_max_momentum": 2.5,
                "spacecraft_moi": [1.0, 1.0, 1.0],
            }
        },
        "constraint": {},
    }
    cfg = MissionConfig.model_validate(data)
    acs = cfg.spacecraft_bus.attitude_control
    assert acs.wheel_enabled is True
    assert acs.wheel_max_torque == 0.2
    assert acs.wheel_max_momentum == 2.5
