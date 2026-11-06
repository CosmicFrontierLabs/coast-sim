"""The science program as one file: typed requests without their run state,
the Targets of Opportunity with their triggers, the activities and the
groups, saved, loaded, bound and run."""

from datetime import timedelta
from pathlib import Path

import pytest

from conops import create_ditl
from conops.config import MissionConfig, ObservationGroup, ObservationGroups
from conops.config.observation_timing import BudgetStep, ObservationTiming
from conops.config.scheduler import SchedulerMode
from conops.ditl.queue_ditl import QueueDITL
from conops.targets import (
    MovingPointing,
    ObservationSequence,
    Plan,
    PlanEntry,
    Pointing,
    Program,
    ScheduledActivity,
    SequenceStep,
    TOORequest,
    track_from_table,
    visit_windows,
)

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config

HOURS = 8
END = BEGIN + timedelta(hours=HOURS)
STEP = 60


def _requests(config: MissionConfig) -> list[Pointing]:
    windowed = Pointing(
        config=config,
        obsid=101,
        name="windowed",
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=90.0,
        ss_min=10 * MIN,
        ss_max=10 * MIN,
        visit_windows=visit_windows([T0 + HOUR, T0 + 4 * HOUR], 30 * MIN),
        roll_offset_deg=5.0,
        protect_opening=True,
    )
    sequenced = Pointing(
        config=config,
        obsid=102,
        name="sequenced",
        ra=PATCH[1][0],
        dec=PATCH[1][1],
        merit=70.0,
        ss_min=5 * MIN,
        ss_max=HOUR,
        earliest_start=T0 + HOUR,
        deadline=T0 + 6 * HOUR,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="a", duration_seconds=600),
                SequenceStep(name="b", duration_seconds=300, science=False),
            ]
        ),
        visits=2,
        own_timing=ObservationTiming(
            setup_steps=[BudgetStep(name="warm", duration_seconds=60)]
        ),
    )
    moving = MovingPointing(
        config=config,
        obsid=103,
        name="mover",
        merit=95.0,
        ss_min=20 * MIN,
        ss_max=20 * MIN,
        track=track_from_table(
            [T0 + k * HOUR for k in range(HOURS + 1)],
            [PATCH[2][0] + 0.5 * k for k in range(HOURS + 1)],
            [PATCH[2][1]] * (HOURS + 1),
        ),
    )
    moving.exptime = 20 * MIN
    by_body = MovingPointing(
        config=config,
        obsid=104,
        name="body",
        merit=60.0,
        body="Jupiter",
        use_horizons=False,
        spice_kernel="k.bsp",
    )
    by_body.exptime = 15 * MIN
    return [windowed, sequenced, moving, by_body]


def _program(config: MissionConfig) -> Program:
    requests = _requests(config)
    too_request = Pointing(
        config=config,
        obsid=9001,
        name="burst",
        ra=PATCH[3][0],
        dec=PATCH[3][1],
        merit=1000.0,
        ss_min=10 * MIN,
        ss_max=20 * MIN,
    )
    too_request.exptime = 20 * MIN
    return Program(
        name="test program",
        description="a program for the round trip",
        begin=T0,
        end=T0 + HOURS * HOUR,
        requests=requests,
        toos=[
            TOORequest(
                request=too_request,
                submit_time=T0 + 2 * HOUR,
                deadline=T0 + 5 * HOUR,
                interrupt=False,
            )
        ],
        activities=[
            ScheduledActivity(
                name="calibration",
                begin=T0 + 3 * HOUR,
                duration_seconds=600,
                slide_seconds=HOUR,
            )
        ],
        observation_groups=ObservationGroups(
            groups=[
                ObservationGroup(
                    name="pair", members=[101, 102], prefer_contiguous=True
                )
            ]
        ),
    )


REQUEST_FIELDS = (
    "obsid",
    "name",
    "ra",
    "dec",
    "merit",
    "exptime",
    "ss_min",
    "ss_max",
    "earliest_start",
    "deadline",
    "visit_windows",
    "visits",
    "protect_opening",
    "sequence",
    "instrument_name",
    "instrument_modes",
    "science_data",
    "fixed_data_gb",
    "own_timing",
    "roll_deg",
    "roll_offset_deg",
    "desaturation",
    "track",
)


class TestRoundTrip:
    def test_saved_loaded_and_bound_equals_the_program_built(
        self, tmp_path: Path
    ) -> None:
        config = _config(HOURS)
        program = _program(config)
        path = program.save(tmp_path / "program.json")
        loaded = Program.load(path)
        assert (
            loaded.name == "test program"
            and loaded.begin == T0
            and loaded.end == T0 + HOURS * HOUR
        )
        assert [type(r) for r in loaded.requests] == [
            Pointing,
            Pointing,
            MovingPointing,
            MovingPointing,
        ]
        for built, read in zip(program.requests, loaded.requests, strict=True):
            for field in REQUEST_FIELDS:
                assert getattr(read, field) == getattr(built, field), field
        body = loaded.requests[3]
        assert isinstance(body, MovingPointing)
        assert (body.body, body.use_horizons, body.spice_kernel) == (
            "Jupiter",
            False,
            "k.bsp",
        )
        [too] = loaded.toos
        assert isinstance(too.request, Pointing)
        assert (too.obsid, too.submit_time, too.deadline, too.interrupt) == (
            9001,
            T0 + 2 * HOUR,
            T0 + 5 * HOUR,
            False,
        )
        assert too.request.exptime == 20 * MIN and not too.executed
        assert loaded.activities == program.activities
        assert loaded.observation_groups == program.observation_groups
        loaded.bind(config)
        assert all(
            r.config is config and r.ephem is config.constraint.ephem
            for r in loaded.requests
        )
        assert config.observation_groups.groups[0].name == "pair"

    def test_execution_fields_are_not_written(self, tmp_path: Path) -> None:
        config = _config(HOURS)
        program = _program(config)
        program.requests[0].begin = T0 + 100
        program.requests[0].end = T0 + 700
        program.requests[0].slewtime = 60
        program.requests[0].done = True
        text = (program.save(tmp_path / "p.json")).read_text()
        for field in (
            "slewtime",
            "slewdist",
            "great_circle_deg",
            "settle_seconds",
            "target_attitude",
            '"done"',
            "exporig",
        ):
            assert field not in text, field
        assert '"executed"' not in text
        loaded = Program.load(tmp_path / "p.json")
        first = loaded.requests[0]
        assert (first.begin, first.end, first.slewtime, first.done) == (0, 0, 0, False)

    def test_window_must_be_in_order(self) -> None:
        with pytest.raises(ValueError, match="window"):
            Program(begin=T0 + HOUR, end=T0)

    def test_legacy_too_fields_build_a_pointing(self) -> None:
        too = TOORequest(
            obsid=5, ra=10.0, dec=20.0, merit=50.0, exptime=600, name="old"
        )
        assert isinstance(too.request, Pointing)
        assert (too.obsid, too.ra, too.dec, too.merit, too.exptime, too.name) == (
            5,
            10.0,
            20.0,
            50.0,
            600,
            "old",
        )
        # As the dispatch queue builds one from the same fields.
        assert (too.request.ss_min, too.request.ss_max) == (300, 86400)
        read = TOORequest.model_validate(too.model_dump(mode="json"))
        assert read.request.obsid == 5 and isinstance(read.request, Pointing)


class TestTypedPlanEntries:
    def test_a_pointing_in_a_plan_reloads_as_a_pointing(self, tmp_path: Path) -> None:
        config = _config(HOURS)
        [windowed, *_] = _requests(config)
        windowed.begin, windowed.end = T0 + HOUR, T0 + HOUR + 1200
        plan = Plan(
            entries=[
                windowed,
                PlanEntry(name="plain", begin=T0 + 2 * HOUR, end=T0 + 2 * HOUR + 600),
            ]
        )
        path = plan.save(tmp_path / "plan.json")
        loaded = Plan.load(path)
        first, second = loaded.entries
        assert isinstance(first, Pointing) and type(second) is PlanEntry
        assert first.visit_windows == windowed.visit_windows
        assert first.protect_opening and first.roll_offset_deg == 5.0

    def test_a_file_without_kinds_holds_plain_entries(self, tmp_path: Path) -> None:
        plan = Plan(entries=[PlanEntry(name="p", begin=T0, end=T0 + 600)])
        path = plan.save(tmp_path / "old.json")
        text = path.read_text().replace('"kind": "entry",', "")
        path.write_text(text)
        loaded = Plan.load(path)
        assert type(loaded.entries[0]) is PlanEntry

    def test_unscheduled_entries_do_not_set_the_span(self) -> None:
        config = _config(HOURS)
        requests = _requests(config)
        plan = Plan(entries=requests)
        assert plan.start.startswith("1970")
        scheduled = PlanEntry(name="s", begin=T0 + HOUR, end=T0 + 2 * HOUR)
        plan = Plan(entries=[*requests, scheduled])
        assert plan._start_ts == T0 + HOUR and plan._end_ts == T0 + 2 * HOUR


class TestRuns:
    def test_dispatch_from_a_loaded_program_matches_the_built_one(
        self, tmp_path: Path
    ) -> None:
        config = _config(HOURS)
        config.scheduler.mode = SchedulerMode.DISPATCH
        built = _program(config)
        built.requests = [
            r for r in built.requests if r.obsid != 104
        ]  # no body lookup offline
        path = built.save(tmp_path / "program.json")

        def run(program: Program) -> list[tuple[int, float, float]]:
            cfg = _config(HOURS)
            cfg.scheduler.mode = SchedulerMode.DISPATCH
            ditl = create_ditl(cfg, program=program, begin=BEGIN, end=END)
            assert isinstance(ditl, QueueDITL)
            ditl.step_size = STEP
            assert ditl.calc()
            assert len(ditl.too_register) == 1
            return [(int(e.obsid), float(e.begin), float(e.end)) for e in ditl.plan]

        from_built = run(built)
        from_loaded = run(Program.load(path))
        assert from_built == from_loaded
        assert any(obsid == 9001 for obsid, _, _ in from_loaded)
        assert any(obsid == 103 for obsid, _, _ in from_loaded)

    def test_planned_run_from_a_program(self, tmp_path: Path) -> None:
        config = _config(HOURS)
        program = _program(config)
        program.requests = [r for r in program.requests if r.obsid != 104]
        program.toos = []
        path = program.save(tmp_path / "program.json")
        cfg = _config(HOURS)
        cfg.scheduler.mode = SchedulerMode.PLANNED
        ditl = create_ditl(cfg, program=Program.load(path), begin=BEGIN, end=END)
        assert ditl.calc()
        assert any(int(e.obsid) == 101 for e in ditl.plan)
        assert any(e.name == "calibration" for e in ditl.plan)

    def test_a_planned_run_refuses_toos(self) -> None:
        config = _config(HOURS)
        config.scheduler.mode = SchedulerMode.PLANNED
        program = _program(config)
        program.requests = [r for r in program.requests if r.obsid != 104]
        with pytest.raises(ValueError, match="Targets of Opportunity"):
            create_ditl(config, program=program, begin=BEGIN, end=END)

    def test_program_and_targets_are_exclusive(self) -> None:
        config = _config(HOURS)
        program = _program(config)
        with pytest.raises(ValueError, match="not both"):
            create_ditl(
                config, targets=_requests(config), program=program, begin=BEGIN, end=END
            )


class TestFollowUps:
    def test_contact_and_track_fields_are_not_written(self, tmp_path: Path) -> None:
        config = _config(HOURS)
        program = _program(config)
        text = program.save(tmp_path / "p.json").read_text()
        for field in (
            "station",
            "station_lat_deg",
            "contact_begin",
            "contact_end",
            "track_start_ra",
            "track_end_roll",
        ):
            assert f'"{field}"' not in text, field

    def test_original_exposure_survives_the_round_trip(self, tmp_path: Path) -> None:
        config = _config(HOURS)
        program = _program(config)
        loaded = Program.load(program.save(tmp_path / "p.json"))
        for built, read in zip(program.requests, loaded.requests, strict=True):
            assert (read.exptime, read.exporig) == (built.exptime, built.exporig)
            assert read.exporig is not None

    def test_too_target_is_bound_for_visibility(self) -> None:
        config = _config(HOURS)
        [too] = _program(config).toos
        target = too.target_for(config)
        assert target.constraint is config.constraint
        assert target.ephem is config.constraint.ephem
        assert target.acs_config is config.spacecraft_bus.attitude_control
        target.visibility()
        assert target.windows

    def test_a_sequence_too_is_observed_step_by_step(self) -> None:
        """A ToO holding a sequence request: dispatch takes it up as a visit
        of its steps, as any sequence target."""
        config = _config(HOURS)
        config.scheduler.mode = SchedulerMode.DISPATCH
        follow_up = Pointing(
            config=config,
            obsid=9002,
            name="follow-up",
            ra=PATCH[3][0],
            dec=PATCH[3][1],
            merit=1000.0,
            ss_min=5 * MIN,
            ss_max=HOUR,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="first", duration_seconds=600),
                    SequenceStep(name="second", duration_seconds=600),
                ]
            ),
        )
        program = Program(
            begin=T0,
            end=T0 + HOURS * HOUR,
            requests=[r for r in _requests(config) if r.obsid in (101, 102)],
            toos=[TOORequest(request=follow_up, submit_time=T0 + HOUR)],
        )
        ditl = create_ditl(config, program=program, begin=BEGIN, end=END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        steps = [e for e in ditl.plan if int(e.obsid) == 9002]
        assert [e.step for e in steps] == ["first", "second"]
        assert all(float(e.begin) >= T0 + HOUR for e in steps)
        assert ditl.too_register[0].executed


class TestSequenceToOWhileObserving:
    def test_admitted_through_the_preview_while_an_observation_runs(self) -> None:
        """With cleanup budgets, dispatch previews the admission of a ToO before
        it ends the observation under way; a ToO holding a sequence is matched
        back to its candidate through the visit's parent there too."""
        config = _config(HOURS)
        config.scheduler.mode = SchedulerMode.DISPATCH
        config.payload.observation_timing = ObservationTiming(
            cleanup_steps=[BudgetStep(name="readout", duration_seconds=120)]
        )
        follow_up = Pointing(
            config=config,
            obsid=9003,
            name="follow-up",
            ra=PATCH[3][0],
            dec=PATCH[3][1],
            merit=1000.0,
            ss_min=5 * MIN,
            ss_max=HOUR,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="first", duration_seconds=600),
                    SequenceStep(name="second", duration_seconds=600),
                ]
            ),
        )
        long_one = Pointing(
            config=config,
            obsid=105,
            name="long",
            ra=PATCH[1][0],
            dec=PATCH[1][1],
            merit=50.0,
            ss_min=10 * MIN,
            ss_max=4 * HOUR,
        )
        long_one.exptime = 4 * HOUR
        program = Program(
            begin=T0,
            end=T0 + HOURS * HOUR,
            requests=[long_one],
            toos=[TOORequest(request=follow_up, submit_time=T0 + HOUR)],
        )
        ditl = create_ditl(config, program=program, begin=BEGIN, end=END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        assert any("TOO interrupt" in e.description for e in ditl.log.events)
        steps = [e for e in ditl.plan if int(e.obsid) == 9003]
        assert [e.step for e in steps] == ["first", "second"]
        assert ditl.too_register[0].executed
        assert ditl.validate_plan_matches_execution() == []

    def test_flat_fields_build_a_bound_request(self) -> None:
        config = _config(HOURS)
        too = TOORequest(
            config=config,
            obsid=7,
            ra=10.0,
            dec=20.0,
            merit=50.0,
            exptime=600,
            name="flat",
        )
        assert too.request.config is config
        bound = too.target_for(config)
        assert bound.target_attitude == too.request.target_attitude
        assert too.request.model_dump() == bound.model_dump()


class TestVisitUnderWayAndLowerRequests:
    def test_a_visit_under_way_is_not_cut_for_a_lower_ranked_request(self) -> None:
        """A sequence visit that cannot wait for a lower-ranked request is
        admitted whole; its first step's pointing carries no deadline of its
        own, and the deadline dispatch computed for it then yielded to the
        request and cut the step minutes in, voiding the visit. A visit under
        way is not cut for a request ranked below it."""
        config = _config(HOURS)
        config.scheduler.mode = SchedulerMode.DISPATCH
        visit = Pointing(
            config=config,
            obsid=201,
            name="visit",
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            merit=95.0,
            ss_min=5 * MIN,
            ss_max=2 * HOUR,
            earliest_start=T0,
            deadline=T0 + 10 * MIN,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="first", duration_seconds=2400),
                    SequenceStep(name="second", duration_seconds=2400),
                ]
            ),
        )
        # Ranked below the visit, due to start inside it: the visit cannot
        # wait for it, so it is admitted whole and the request is missed.
        later = Pointing(
            config=config,
            obsid=202,
            name="later",
            ra=PATCH[2][0],
            dec=PATCH[2][1],
            merit=50.0,
            ss_min=10 * MIN,
            ss_max=10 * MIN,
            deadline=T0 + 30 * MIN,
        )
        later.exptime = 10 * MIN
        program = Program(begin=T0, end=T0 + HOURS * HOUR, requests=[visit, later])
        ditl = create_ditl(config, program=program, begin=BEGIN, end=END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        steps = [e for e in ditl.plan if int(e.obsid) == 201]
        assert [e.step for e in steps] == ["first", "second"]
        for entry in steps:
            assert float(entry.collection_end) - float(entry.collection_begin) == (
                pytest.approx(2400, abs=STEP)
            )
        assert not any("cut short" in e.description for e in ditl.log.events)
