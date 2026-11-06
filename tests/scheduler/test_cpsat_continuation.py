"""Chunk resources reserve the decoded incumbent, including a dispatch seed."""

from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("ortools")

from conops.common import ACSMode  # noqa: E402
from conops.common.enums import AntennaType  # noqa: E402
from conops.config import (  # noqa: E402
    AntennaPointing,
    BandCapability,
    BudgetStep,
    CommunicationsSystem,
    DataGeneration,
    Instrument,
    InstrumentMode,
    ObservationTiming,
    ScienceData,
)
from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.schedulers.cpsat_scheduler import _kb_up  # noqa: E402
from conops.schedulers.local_search_scheduler import _Decoded, _Snapshot  # noqa: E402
from conops.targets import ObservationSequence, Pointing, SequenceStep  # noqa: E402
from conops.targets.plan_entry import extra_data_gb  # noqa: E402

from .planning_scenario import (  # noqa: E402
    BEGIN,
    HOUR,
    MIN,
    T0,
    make_config,
    make_target,
)
from .test_independent_recorders import storage  # noqa: E402


def _scenario(independent=False, timed=False):
    c = make_config(4, stations=timed)
    c.constraint.sun_constraint = None
    c.constraint.earth_constraint = None
    if timed:
        c.spacecraft_bus.communications = CommunicationsSystem(
            antenna_pointing=AntennaPointing(antenna_type=AntennaType.OMNI),
            band_capabilities=[BandCapability(band="S", downlink_rate_mbps=10)],
        )
    c.payload.instruments = [
        Instrument(
            name="camera",
            modes=[
                InstrumentMode(name="IDLE", power_w=10, observing=False),
                InstrumentMode(name="SCIENCE", power_w=210),
            ],
            data_generation=DataGeneration(
                science=ScienceData(rate_gbps=0.001, per_observation_gb=0.2)
            ),
        )
    ]
    c.payload.observation_timing = ObservationTiming(
        setup_steps=[BudgetStep(name="setup frames", duration_seconds=60, data_gb=0.3)],
        cleanup_steps=[
            BudgetStep(name="cleanup frames", duration_seconds=60, data_gb=0.4)
        ],
    )
    if independent:
        c.data_storage = storage()
    plain = make_target(c, 100, 105, 10, minutes=90, snapshot=20, ss_min=2)
    sequence = Pointing(
        config=c,
        obsid=101,
        ra=115,
        dec=10,
        visits=1,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(
                    name="calibration", duration_seconds=3 * MIN, science=False
                ),
                SequenceStep(name="science", duration_seconds=7 * MIN),
            ]
        ),
    )
    scheduler = CpSatScheduler(
        c,
        [plain, sequence],
        BEGIN,
        BEGIN + timedelta(hours=4),
        include_passes=False,
        workers=1,
        seed=1,
    )
    scheduler._start_from_priority_plan()
    prefix = scheduler._decode(
        [_Snapshot(obsid=100, seconds=600, release=T0 + 10 * MIN)]
    )
    # This is the old placement in the chunk being optimized, which must not
    # be charged as fixed production/energy in the replacement chunk model.
    old_chunk = scheduler._step(
        prefix.final, _Snapshot(obsid=100, seconds=1200, release=T0 + 30 * MIN)
    )
    assert old_chunk is not prefix.final
    return scheduler, prefix.final, old_chunk


def _tail(scheduler, before, *, shortened=False):
    snapshots = [
        _Snapshot(
            obsid=100,
            seconds=90 * MIN,
            release=T0 + (225 * MIN + 5 if shortened else HOUR + 125),
        ),
    ]
    if not shortened:
        snapshots.append(_Snapshot(obsid=101, seconds=7 * MIN, release=T0 + 100 * MIN))
    # A snapshot outside the horizon is dropped by exact decoding.
    snapshots.append(_Snapshot(obsid=100, seconds=600, release=T0 + 5 * HOUR))
    states = scheduler._decoded_states(before, snapshots)
    assert states[-1] is states[-2]
    return _Decoded.model_construct(sequence=snapshots, states=states)


def _blocks(decoded):
    return [
        after.timeline[after.cursor - 1]
        for before, after in zip(decoded.states, decoded.states[1:])
        if after.cursor > before.cursor
    ]


@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("shortened", [False, True])
def test_recorder_counts_actual_suffix_steps_without_the_old_chunk(
    independent, shortened, timed
):
    scheduler, state, old_chunk = _scenario(independent, timed)
    tail = _tail(scheduler, old_chunk, shortened=shortened)
    blocks = _blocks(tail)
    assert blocks
    # The snapshot release may be rounded forward on decoding; no entry
    # at that exact time need exist in either the greedy or decoded plan.
    assert float(blocks[0].entry.begin) > tail.sequence[0].release
    assert (
        sum(
            e.collection_seconds_between(e.begin, e.end)
            for e in (blocks[0].steps or [blocks[0].entry])
        )
        < tail.sequence[0].seconds
    )
    if not shortened:
        assert blocks[-1].is_visit and blocks[-1].steps[0].calibration
        assert extra_data_gb(blocks[-1].steps[0], scheduler.config) > 0
    # A lookup by obsid/release in the unrelated greedy plan cannot answer
    # this forecast. Emptying it must not drop the continuation's resources.
    scheduler._greedy_timeline = []
    entries = [e for b in [*state.timeline, *blocks] for e in (b.steps or [b.entry])]
    origin, _ = scheduler._chunk_origin(state)
    for _, ledger, offset in scheduler._ledger.members:
        assert ledger._timed_science == timed
        forecast = scheduler._chunk_recorder(state, tail, origin, ledger, offset, [])
        expected = ledger.initial_volumes(entries)
        _, fixed, _ = ledger.model_vectors(expected, origin)
        assert forecast.fixed == [_kb_up(value) for value in fixed]
        assert forecast.entering == _kb_up(ledger.level_at(expected, origin))


def test_battery_reserves_the_actual_suffix_and_ignores_dropped_snapshots():
    scheduler, state, old_chunk = _scenario()
    tail = _tail(scheduler, old_chunk)
    scheduler._greedy_timeline = []
    origin, _ = scheduler._chunk_origin(state)
    forecast = scheduler._chunk_battery(state, tail, origin, [])
    empty = scheduler._chunk_battery(state, None, origin, [])
    assert not any(empty.fixed)
    assert sum(forecast.fixed) > 0
    # Independent resource oracle: the continuation's full block energy,
    # with the chunk model's idle baseline removed at each actual sample.
    expected = [0] * len(forecast.fixed)
    for block in _blocks(tail):
        entries = block.steps or [block.entry]
        for entry in entries:
            begin, end = float(entry.begin), float(entry.end)
            for at in scheduler._battery.steps(begin, end):
                mode = (
                    ACSMode.SLEWING
                    if at < float(entry.begin) + entry.slewtime
                    else ACSMode.SCIENCE
                )
                eclipse = scheduler._battery.in_eclipse(float(at))
                watts = scheduler.config.spacecraft_bus.power_draw.power(
                    mode, in_eclipse=eclipse
                )
                # The test payload has exactly one observing camera, with
                # 10 W in SLEWING and 210 W in SCIENCE.
                watts += 10 if mode == ACSMode.SLEWING else 210
                run = sum(1 for edge in forecast.edges if origin + edge <= at)
                expected[run] += int(np.ceil(watts)) * scheduler.ctx.step_size - (
                    forecast.idle[run] * scheduler.ctx.step_size
                )
    assert forecast.fixed == expected


def test_a_changed_prefix_redecodes_the_reserved_continuation():
    scheduler, state, old_chunk = _scenario()
    original = _tail(scheduler, old_chunk)
    changed = scheduler._step(
        old_chunk, _Snapshot(obsid=100, seconds=1200, release=T0 + 65 * MIN)
    )
    assert changed is not old_chunk
    updated = _tail(scheduler, changed)
    before, after = _blocks(original)[0], _blocks(updated)[0]
    assert float(after.entry.collection_begin) > float(before.entry.collection_begin)
    origin, _ = scheduler._chunk_origin(state)
    for _, ledger, offset in scheduler._ledger.members:
        forecast = scheduler._chunk_recorder(state, updated, origin, ledger, offset, [])
        # Recorder production uses the shifted, rounded interval, not the
        # snapshot's old release or the unmodified incumbent's arrival.
        actual = [
            e
            for b in [*state.timeline, *_blocks(updated)]
            for e in (b.steps or [b.entry])
        ]
        expected = ledger.initial_volumes(actual)
        _, fixed, _ = ledger.model_vectors(expected, origin)
        assert forecast.fixed == [_kb_up(value) for value in fixed]
