"""A moving target in dispatch and in the planners: pointed where its slew
ends, tracked while observed, placed by the planner at its position."""

from datetime import timedelta

import pytest

from conops import DITL, create_ditl
from conops.common import ACSMode
from conops.config import MissionConfig
from conops.ditl.queue_ditl import QueueDITL
from conops.schedulers import PriorityScheduler, SequentialScheduler
from conops.targets import (
    MovingPointing,
    ObservationSequence,
    Plan,
    PlanEntry,
    Pointing,
    SequenceStep,
    track_from_table,
)

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
END = BEGIN + timedelta(hours=HOURS)
STEP = 60
RATE_DEG_PER_H = 1.0


def _mover(
    config: MissionConfig, minutes: int = 40, merit: float = 95.0
) -> MovingPointing:
    track = track_from_table(
        [T0 + k * HOUR for k in range(HOURS + 1)],
        [PATCH[0][0] + k * RATE_DEG_PER_H for k in range(HOURS + 1)],
        [PATCH[0][1]] * (HOURS + 1),
    )
    return MovingPointing(
        config=config,
        obsid=700,
        name="mover",
        merit=merit,
        exptime=minutes * MIN,
        ss_min=minutes * MIN,
        ss_max=minutes * MIN,
        track=track,
    )


def _fillers(config: MissionConfig) -> list[Pointing]:
    return [
        _target(config, 1 + k, *PATCH[1 + k], merit=50.0, minutes=180, snapshot=60)
        for k in range(2)
    ]


class TestDispatch:
    def test_pointed_where_the_slew_ends_and_tracked(self) -> None:
        config = _config(HOURS)
        mover = _mover(config)
        ditl = create_ditl(config, [mover, *_fillers(config)], BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        [entry] = [e for e in ditl.plan if e.obsid == 700]
        assert entry.moves and entry.slewtime > 0
        # The entry points where the slew ended, not where the body was at the start.
        arrival = float(entry.begin) + entry.slewtime
        assert (entry.ra, entry.dec) == pytest.approx(mover.position_at(arrival))
        assert entry.ra > PATCH[0][0]
        # Executed, the attitude follows the track and the mode is SCIENCE.
        hk = ditl.telemetry.housekeeping
        stamps = hk.timestamp
        samples = [
            (stamp.timestamp(), float(hk.ra[i]))
            for i, stamp in enumerate(stamps)
            if stamp is not None
            and entry.collection_begin <= stamp.timestamp() <= entry.end
            and str(hk.acs_mode[i]).endswith("SCIENCE")
        ]
        assert len(samples) >= 30
        tolerance = config.spacecraft_bus.attitude_control.slew_accuracy
        for utime, ra in samples:
            assert abs(ra - mover.position_at(utime)[0]) <= tolerance
        assert samples[-1][1] - samples[0][1] > 0.4
        assert ditl.validate_plan_matches_execution() == []
        assert any(
            "Tracking moving target 700" in e.description for e in ditl.log.events
        )

    def test_tracking_stops_when_the_observation_is_interrupted(self) -> None:
        """A Target of Opportunity cuts the tracked observation; the ACS brakes
        and the slew to the ToO starts, rather than waiting for the track's end."""
        config = _config(HOURS)
        mover = _mover(config, minutes=180)
        ditl = create_ditl(config, [mover, *_fillers(config)], BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP

        def arrive(d: QueueDITL) -> None:
            d.submit_too(
                obsid=9001,
                ra=PATCH[3][0],
                dec=PATCH[3][1],
                merit=1000.0,
                exptime=20 * MIN,
                name="ToO",
                submit_time=T0 + 30 * MIN,
            )

        arrive(ditl)
        assert ditl.calc()
        too = [e for e in ditl.plan if e.obsid == 9001]
        assert too and float(too[0].begin) < T0 + 45 * MIN
        modes = [str(m) for m in ditl.telemetry.housekeeping.acs_mode]
        assert any(m.endswith(str(ACSMode.SLEWING).split(".")[-1]) for m in modes)
        assert ditl.validate_plan_matches_execution() == []


class TestPlanner:
    @pytest.mark.parametrize("rate", [0.0, 30.0])
    def test_next_slew_leaves_the_tracking_endpoint(self, rate: float) -> None:
        config = _config(2)
        end = BEGIN + timedelta(hours=2)
        mover = _mover(config, minutes=30)
        mover.track = track_from_table(
            [T0, T0 + 2 * HOUR], [105.0, 105.0 + 2 * rate], [10.0, 10.0]
        )
        fixed = _target(config, 701, 105.0, 10.0, merit=40.0, minutes=20)
        scheduler = PriorityScheduler(config, [mover, fixed], BEGIN, end, step_size=2)
        plan = scheduler.schedule()
        assert [e.obsid for e in plan] == [700, 701]
        moving, successor = scheduler.timeline
        finish_ra = mover.position_at(float(moving.entry.end))[0]
        assert moving.attitude_out[0] == pytest.approx(finish_ra, abs=0.001)
        assert moving.end >= float(moving.entry.end)
        if rate:
            assert moving.end > float(moving.entry.end)
            assert moving.attitude_out[0] > moving.attitude_in[0] + 14.9
        assert successor.slew is not None
        assert successor.slew.startra == pytest.approx(moving.attitude_out[0])
        assert float(successor.entry.begin) >= scheduler.ctx.ceil_step(moving.end)
        replay = DITL(config=config, plan=plan, begin=BEGIN, end=end)
        replay.step_size = 2
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("duration", [0.5, 40.0, 60.0, 61.0])
    def test_short_observation_tracks(self, duration: float) -> None:
        config = _config(1)
        end = BEGIN + timedelta(hours=1)
        mover = _mover(config)
        mover.exptime = mover.ss_min = mover.ss_max = duration
        mover.track = track_from_table([T0, T0 + HOUR], [105.0, 135.0], [10.0, 10.0])
        plan = PriorityScheduler(config, [mover], BEGIN, end, step_size=2).schedule()
        assert len(plan) == 1
        replay = DITL(config=config, plan=plan, begin=BEGIN, end=end)
        replay.step_size = 2
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []
        assert any(
            "Tracking moving target 700" in e.description for e in replay.log.events
        )

    @pytest.mark.parametrize("reference_first", [True, False])
    @pytest.mark.parametrize("backend", ["priority", "queue"])
    def test_sequence_tracks_only_implicit_target_steps(
        self, reference_first: bool, backend: str
    ) -> None:
        config = _config(1)
        end = BEGIN + timedelta(hours=1)
        reference = SequenceStep(name="reference", duration_seconds=120, ra=110, dec=10)
        science = SequenceStep(name="target", duration_seconds=120)
        steps = [reference, science] if reference_first else [science, reference]
        mover = MovingPointing(
            config=config,
            obsid=700,
            name="sequence",
            track=track_from_table([T0, T0 + HOUR], [105, 135], [10, 10]),
            sequence=ObservationSequence(steps=steps),
        )
        # Dispatch and offline planning derive exactly the same step requests.
        assert [mover.step_pointing(i).moves for i in range(2)] == [
            step.name == "target" for step in steps
        ]
        if backend == "priority":
            plan = PriorityScheduler(
                config, [mover], BEGIN, end, step_size=2
            ).schedule()
        else:
            dispatch = create_ditl(config, [mover], BEGIN, end)
            dispatch.step_size = 2
            assert dispatch.calc()
            assert dispatch.validate_plan_matches_execution() == []
            plan = dispatch.plan
        assert len(plan) == 2
        assert [entry.moves for entry in plan] == [
            step.name == "target" for step in steps
        ]
        ref = next(entry for entry in plan if entry.step == "reference")
        assert (ref.ra, ref.dec) == (110, 10)
        replay = DITL(
            config=config,
            plan=Plan.model_validate_json(plan.model_dump_json()),
            begin=BEGIN,
            end=end,
        )
        replay.step_size = 2
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []

    def test_pointed_where_the_slew_ends_and_executes_exactly(self) -> None:
        """A planned entry of a moving target carries the target's position
        where its slew ends, as dispatch points it, not where the target
        was when it was requested; the executor then takes up the track
        from the slew's end with nothing to close."""
        config = _config(HOURS)
        mover = _mover(config, merit=40.0)
        track = track_from_table(
            [T0 + k * HOUR for k in range(HOURS + 1)],
            [PATCH[0][0] + k * 2.0 for k in range(HOURS + 1)],
            [PATCH[0][1]] * (HOURS + 1),
        )
        mover.track = track
        plan = PriorityScheduler(
            config, [*_fillers(config), mover], BEGIN, END, step_size=STEP
        ).schedule()
        [entry] = [e for e in plan if e.obsid == 700]
        assert entry.moves
        arrival = float(entry.begin) + float(entry.slewtime)
        assert (entry.ra, entry.dec) == pytest.approx(mover.position_at(arrival))
        assert arrival > T0 + 20 * MIN
        assert (entry.ra, entry.dec) != pytest.approx((track[0].ra, track[0].dec))
        from conops import DITL

        ditl = DITL(config=config, plan=plan, begin=BEGIN, end=END)
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []


class TestVisibilityOncePerRequest:
    def test_a_moving_target_s_windows_are_evaluated_once_across_plans(self) -> None:
        """A moving target's visibility follows its track over the whole
        ephemeris, which is costly to evaluate; the first plan evaluates
        it on the target, and every candidate of it in that plan and in
        later plans takes the target's windows."""
        from unittest.mock import patch

        from conops.targets import PlanEntry

        config = _config(HOURS)
        # Where the roll does not enter the windows, as with a mounted
        # instrument or field-of-regard scheduling.
        config.constraint.ignore_roll = True
        mover = _mover(config)
        targets = [mover, *_fillers(config)]
        evaluated: list[int] = []
        original = PlanEntry.visibility

        def counting(self: PlanEntry, *args: object, **kwargs: object) -> int:
            evaluated.append(int(self.obsid))
            return original(self, *args, **kwargs)

        with patch.object(PlanEntry, "visibility", counting):
            first = PriorityScheduler(
                config, targets, BEGIN, END, step_size=STEP
            ).schedule()
            assert evaluated.count(700) == 1
            evaluated.clear()
            second = PriorityScheduler(
                config, targets, BEGIN, END, step_size=STEP
            ).schedule()
            assert evaluated.count(700) == 0
        assert [e.obsid for e in first if e.obsid == 700]
        assert [(e.obsid, float(e.begin)) for e in first] == [
            (e.obsid, float(e.begin)) for e in second
        ]
