"""Tests for reaction wheel and MOI configuration parsing."""

from pathlib import Path

from conops import MissionConfig


def test_example_config_contains_wheels_and_moi():
    # Determine repository root by walking up from tests directory
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "examples" / "example_config.json"
    cfg = MissionConfig.from_json_file(str(example))

    acs = cfg.spacecraft_bus.attitude_control

    # Legacy wheel_enabled should be False when using wheels array
    # (wheel_enabled=True creates a legacy single wheel with wheel_max_torque/momentum)
    assert acs.wheel_enabled is False

    # Wheels list present and has four entries
    assert isinstance(acs.wheels, list)
    assert len(acs.wheels) == 4

    # Each wheel should have expected attributes (WheelSpec model)
    for w in acs.wheels:
        assert hasattr(w, "name")
        assert hasattr(w, "orientation")
        assert hasattr(w, "max_torque")
        assert hasattr(w, "max_momentum")

    # spacecraft_moi parsed as sequence of three values
    moi = acs.spacecraft_moi
    assert len(moi) == 3
    assert all(isinstance(x, (int, float)) for x in moi)


def test_legacy_wheel_fields_parsed_but_ignored():
    """Legacy wheel_enabled/wheel_max_torque/wheel_max_momentum are parsed but ignored.

    Wheels are only created from the explicit 'wheels' array.
    """
    data = {
        "spacecraft_bus": {
            "attitude_control": {
                "wheel_enabled": True,  # Ignored - no legacy wheel created
                "wheel_max_torque": 0.2,
                "wheel_max_momentum": 2.5,
                "spacecraft_moi": [1.0, 1.0, 1.0],
            }
        },
        "constraint": {},
    }
    cfg = MissionConfig.model_validate(data)
    acs = cfg.spacecraft_bus.attitude_control
    # Fields are still parsed (for backwards compatibility with config files)
    assert acs.wheel_enabled is True
    assert acs.wheel_max_torque == 0.2
    assert acs.wheel_max_momentum == 2.5
    # But no wheels array means no wheels created
    assert acs.wheels == []
