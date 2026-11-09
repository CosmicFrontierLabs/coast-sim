"""Momentum forecasts use each sequence step's physical body frame."""

import pytest

from conops import QueueDITL
from conops.common.vector import attitude_to_quat, quaternion_to_rotation_matrix
from conops.config import Instrument, Payload, SpacecraftFrame, Telescope
from conops.ditl.queue_ditl import _SequenceRun
from conops.simulation.slew import Slew
from conops.targets import Pointing
from conops.targets.sequence import ObservationSequence, SequenceStep

from .test_momentum_dumps import BEGIN, END, _config


@pytest.mark.parametrize("changed_frame", [False, True])
@pytest.mark.parametrize("instrument", ["Camera", "Scope"])
@pytest.mark.parametrize("switch_instrument", [False, True])
def test_sequence_momentum_forecast_uses_each_steps_physical_body_attitude(
    changed_frame: bool, instrument: str, switch_instrument: bool
) -> None:
    config = _config()
    if changed_frame:
        config.spacecraft_bus.frame = SpacecraftFrame(
            boresight=(0, 0, 1), sun_side=(1, 0, 0)
        )
    config.payload = Payload(
        instruments=[
            Instrument(name="Camera"),
            Telescope(name="Scope", boresight=(0, 1, 0)),
        ]
    )
    parent = Pointing(
        config=config,
        ra=80,
        dec=-20,
        obsid=20,
        instrument_name=instrument,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="a", duration_seconds=600, roll_offset_deg=5),
                SequenceStep(
                    name="b",
                    duration_seconds=600,
                    ra=85,
                    dec=-22,
                    roll_offset_deg=20,
                    instrument_name=("Scope" if instrument == "Camera" else "Camera")
                    if switch_instrument
                    else None,
                ),
            ]
        ),
        visits=1,
    )
    sim = QueueDITL(config=config, begin=BEGIN, end=END)
    sim._sequence = _SequenceRun(parent=parent, index=0, begun=BEGIN.timestamp())
    first = parent.step_pointing(0)
    first.collection_begin = BEGIN.timestamp() + 100
    instrument_roll = 37.0
    body = first.target_body_attitude(instrument_roll)
    slew = Slew.model_construct(
        config=config,
        slewstart=BEGIN.timestamp(),
        slewtime=100,
        startra=body[0],
        startdec=body[1],
        startroll=body[2],
        endra=body[0],
        enddec=body[1],
        endroll=body[2],
        instrument_roll=instrument_roll,
    )
    end, forecast = sim._visit_forecast(first, slew)
    starts = parent.sequence.step_starts(
        config.spacecraft_bus.attitude_control,
        sim._timing_for(parent),
        *parent.visit_base_pointing,
    )
    for index, start in enumerate(starts):
        roll = instrument_roll + (0 if index == 0 else 15)
        expected = parent.step_pointing(index).target_body_attitude(roll)
        observed = forecast(first.collection_begin + start + 1)
        assert quaternion_to_rotation_matrix(
            attitude_to_quat(*observed)
        ) == pytest.approx(
            quaternion_to_rotation_matrix(attitude_to_quat(*expected)), abs=1e-12
        )
    assert end >= first.collection_begin + starts[-1] + 600
