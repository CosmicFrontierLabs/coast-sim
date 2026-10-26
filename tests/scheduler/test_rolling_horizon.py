"""RollingHorizonDITL: follow a plan rebuilt as the simulation runs."""

from datetime import timedelta
from pathlib import Path

import pytest

from conops.config import ScienceData, TargetConfig
from conops.config.observation_categories import ObservationCategory
from conops.ditl import DITLLogStore, ReplanReason, RollingHorizonDITL
from conops.simulation.passes import PassTimes
from conops.targets import Pointing

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target


def _run(
    hours: float,
    targets: list[Pointing],
    *,
    toos: tuple[dict[str, float | int | str | None], ...] = (),
    **options: bool | timedelta,
) -> RollingHorizonDITL:
    config = targets[0].config if targets else _config(int(hours) + 1)
    assert config is not None
    ditl = RollingHorizonDITL(
        config,
        targets,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
        **options,  # type: ignore[arg-type]
    )
    ditl.step_size = 60
    for too in toos:
        ditl.submit_too(**too)  # type: ignore[arg-type]
    assert ditl.calc()
    return ditl


def _patch_targets(
    hours: int, minutes: float = 60, snapshot: float = 20
) -> list[Pointing]:
    config = _config(hours)
    return [
        _target(
            config, 100 + i, ra, dec, merit=90 - i, minutes=minutes, snapshot=snapshot
        )
        for i, (ra, dec) in enumerate(PATCH)
    ]


def _executed_seconds(ditl: RollingHorizonDITL) -> float:
    return sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)


class TestScheduledReplanning:
    def test_replans_on_schedule_and_executes_as_planned(self) -> None:
        targets = _patch_targets(3)

        ditl = _run(
            3, targets, horizon=timedelta(hours=2), replan_interval=timedelta(hours=1)
        )

        assert [r.reason for r in ditl.replans] == [
            ReplanReason.INITIAL,
            ReplanReason.SCHEDULED,
            ReplanReason.SCHEDULED,
        ]
        assert [r.utime - T0 for r in ditl.replans] == [0, HOUR, 2 * HOUR]
        assert ditl.validate_plan_matches_execution() == []

    def test_collection_is_credited_to_targets(self) -> None:
        targets = _patch_targets(3)

        ditl = _run(3, targets, replan_interval=timedelta(hours=1))

        credited = sum(t.collected_seconds for t in targets)
        assert credited == pytest.approx(_executed_seconds(ditl))
        assert credited > 0
        assert all(
            t.exptime == pytest.approx(60 * MIN - t.collected_seconds) for t in targets
        )

    def test_commit_lead_time_sets_the_cutoff(self) -> None:
        targets = _patch_targets(3)

        ditl = _run(
            3,
            targets,
            replan_interval=timedelta(hours=1),
            commit_lead_time=timedelta(minutes=30),
        )

        first, *later = ditl.replans
        # The first plan is built before the run, so it takes effect at once.
        assert first.cutoff == first.utime
        assert all(r.cutoff == r.utime + 30 * MIN for r in later)
        assert ditl.validate_plan_matches_execution() == []

    def test_replanning_does_not_reobserve_finished_exposure(self) -> None:
        targets = _patch_targets(3, minutes=20)

        ditl = _run(3, targets, replan_interval=timedelta(minutes=30))

        assert all(t.collected_seconds <= 20 * MIN for t in targets)
        assert ditl.validate_plan_matches_execution() == []

    def test_replan_during_a_contact_starts_from_its_braking_rest(self) -> None:
        """A contact ends at rest after its last tracking sample, not on it."""
        config = _config(10, stations=True)
        # Slow braking, so the rest is well past the last sample.
        config.spacecraft_bus.attitude_control.slew_acceleration = 0.01
        targets = [_target(config, 1, *PATCH[0], minutes=600, snapshot=10)]

        # The second plan is built at 8 h, during the first contact.
        ditl = _run(9, targets, replan_interval=timedelta(hours=8))

        replan = ditl.replans[1].utime
        assert any(
            e.obstype.name == "GSP" and float(e.begin) < replan < float(e.end)
            for e in ditl.plan
        )
        planned = sum(
            e.collection_seconds_between(e.begin, e.end)
            for e in ditl.plan
            if e.obstype.name == "AT"
        )
        assert planned == pytest.approx(_executed_seconds(ditl))
        assert ditl.validate_plan_matches_execution() == []

    def test_finished_targets_still_count_towards_program_shares(self) -> None:
        """Program A has its whole share once its first target is done."""
        config = _config(
            2,
            weights=TargetConfig(completion_deficit_weight=100.0),
            categories=[
                ObservationCategory(
                    name="A", obsid_min=1, obsid_max=99, time_share=1.0
                ),
                ObservationCategory(name="B", obsid_min=100, obsid_max=199),
            ],
        )
        targets = [
            _target(config, 1, *PATCH[0], merit=500, minutes=10),
            _target(config, 2, *PATCH[1], merit=10, minutes=60, snapshot=10),
            _target(config, 100, *PATCH[2], merit=100, minutes=60, snapshot=10),
        ]

        ditl = _run(
            1,
            targets,
            horizon=timedelta(minutes=15),
            replan_interval=timedelta(minutes=15),
        )

        assert targets[0].done
        science = [int(e.obsid) for e in ditl.plan if e.obstype.name == "AT"]
        # Without target 1's science, A would look behind and win at 10 + 100.
        assert science[:2] == [1, 100]


GRB = {
    "obsid": 1_000_001,
    "name": "GRB",
    "ra": 115.0,
    "dec": 10.0,
    "merit": 500.0,
    "exptime": 600,
    "submit_time": T0 + 20 * MIN,
}


class TestPassPrediction:
    def test_the_runs_passes_are_predicted_once_for_every_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each plan used to predict the whole simulation's passes for itself,
        at several seconds a day; the run predicts them once and every plan
        reserves the same passes from them."""
        config = _config(14, stations=True)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH)
        ]
        predictions: list[bool] = []
        original = PassTimes.get

        def counted(self: PassTimes, *args: object, **kwargs: object) -> None:
            predictions.append(bool(kwargs.get("apply_scheduling", True)))
            original(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(PassTimes, "get", counted)
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=14),
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=3),
        )
        ditl.step_size = 60
        assert ditl.calc()

        assert len(ditl.replans) >= 4
        # One prediction for the plans, which DITL also commands from, so it
        # does not reconstruct the passes for itself.
        assert predictions == [True]
        assert any(e.obstype.name == "GSP" for e in ditl.plan)
        assert ditl.validate_plan_matches_execution() == []


def _long_observation() -> list[Pointing]:
    config = _config(3)
    return [
        _target(config, 1, 105.0, 10.0, merit=10, minutes=60, snapshot=60, ss_min=10)
    ]


class TestTargetsOfOpportunity:
    def test_a_too_beats_the_value_an_observation_was_selected_at(self) -> None:
        """After program shares change, an entry keeps its value at selection,
        not its value at the start of the horizon, so a ToO worth more than
        that interrupts it."""
        programs = [
            ObservationCategory(name="A", obsid_min=1, obsid_max=2, time_share=0.5),
            ObservationCategory(name="B", obsid_min=2, obsid_max=3, time_share=0.5),
        ]
        config = _config(
            3,
            weights=TargetConfig(completion_deficit_weight=100),
            categories=programs,
        )
        # A is worth 150 at first, B 110; A's third snapshot is selected once
        # each has 1200 s, at 100.
        a = _target(config, 1, 105.0, 10.0, merit=100, minutes=60, snapshot=20)
        b = _target(config, 2, 110.0, 10.0, merit=60, minutes=60, snapshot=20)

        ditl = _run(
            3,
            [a, b],
            toos=(
                {
                    **GRB,
                    "merit": 120.0,
                    "submit_time": T0 + 50 * MIN,
                    "deadline": T0 + 55 * MIN,
                },
            ),
            replan_interval=timedelta(hours=4),
        )

        assert [int(e.obsid) for e in ditl.plan if e.obstype.name == "AT"][:3] == [
            1,
            2,
            1,
        ]
        third = [e for e in ditl.plan if e.obstype.name == "AT"][2]
        assert float(third.merit) == pytest.approx(100.0)
        assert any(r.interrupted_obsid == 1 for r in ditl.replans)
        assert ditl.too_response_times()[1_000_001] is not None
        assert ditl.validate_plan_matches_execution() == []

    def test_close_deadline_triggers_rapid_replan_and_interrupt(self) -> None:
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
        )

        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert len(rapid) == 1
        assert rapid[0].trigger_obsid == GRB["obsid"]
        assert rapid[0].interrupted_obsid == 1
        response = ditl.too_response_times()[1_000_001]
        assert response is not None and response <= 10 * MIN
        assert ditl.too_register[0].executed
        assert ditl.validate_plan_matches_execution() == []

    def test_interrupted_observation_resumes_after_the_too(self) -> None:
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
        )

        science = [int(e.obsid) for e in ditl.plan if e.obstype.name == "AT"]
        assert science == [1, 1_000_001, 1]

    def test_a_too_that_may_not_interrupt_starts_after_the_observation(
        self,
    ) -> None:
        """A rapid replan, but the observation in progress runs to its end."""
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 90 * MIN, "interrupt": False},),
            replan_interval=timedelta(hours=4),
        )

        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert len(rapid) == 1
        assert rapid[0].interrupted_obsid is None
        science = [int(e.obsid) for e in ditl.plan if e.obstype.name == "AT"]
        assert science == [1, 1_000_001]
        assert ditl.too_register[0].executed
        assert ditl.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("deferred_first", [True, False])
    def test_a_waiting_too_does_not_take_the_interrupting_toos_place(
        self, deferred_first: bool
    ) -> None:
        """A higher-merit ToO submitted with ``interrupt=False`` alongside an
        interrupting one must not fill the cut, which would push the
        interrupting ToO past its deadline and undo the interrupt. It follows
        the interrupting ToO instead, and the observation resumes after both."""
        deferred = {
            **GRB,
            "obsid": 1_000_002,
            "name": "Deferred",
            "merit": 500.0,
            "exptime": 1800,
            "deadline": T0 + 90 * MIN,
            "interrupt": False,
        }
        allowed = {**GRB, "merit": 400.0, "deadline": T0 + 30 * MIN}
        ditl = _run(
            3,
            _long_observation(),
            toos=(deferred, allowed) if deferred_first else (allowed, deferred),
            replan_interval=timedelta(hours=4),
        )

        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert [(r.trigger_obsid, r.interrupted_obsid) for r in rapid] == [
            (GRB["obsid"], 1)
        ]
        science = [int(e.obsid) for e in ditl.plan if e.obstype.name == "AT"]
        assert science == [1, 1_000_001, 1_000_002, 1]
        assert all(too.executed for too in ditl.too_register)
        assert ditl.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("allowed_deadline", [80, 30])
    def test_the_interrupting_too_keeps_its_slot_when_a_waiting_too_is_planned(
        self, allowed_deadline: int
    ) -> None:
        """The interrupting ToO's entry is pinned, with its slew and setup,
        while a waiting ToO is planned in after it. Bounding only the waiting
        ToO's collection start by the entry's end let its slew and setup take
        the slot, and the interrupting ToO follow it. The deadline that the
        pushed-back interrupting ToO would miss changes nothing: both fit in
        order, and neither waits for the next plan."""
        targets = _long_observation()
        assert targets[0].config is not None
        targets[0].config.payload.observation_timing.setup_seconds = 300
        deferred = {
            **GRB,
            "obsid": 1_000_002,
            "name": "Deferred",
            "merit": 500.0,
            "exptime": 1800,
            "deadline": T0 + 90 * MIN,
            "interrupt": False,
        }
        allowed = {**GRB, "merit": 400.0, "deadline": T0 + allowed_deadline * MIN}
        ditl = _run(
            3,
            targets,
            toos=(deferred, allowed),
            replan_interval=timedelta(hours=4),
        )

        assert [r.interrupted_obsid for r in ditl.replans][-1] == 1
        science = [e for e in ditl.plan if e.obstype.name == "AT"]
        assert [int(e.obsid) for e in science] == [1, 1_000_001, 1_000_002, 1]
        _, allowed_entry, deferred_entry, _ = science
        # Pinned where the first pass placed it, straight after the cut.
        assert float(allowed_entry.begin) == T0 + 20 * MIN
        assert float(deferred_entry.begin) >= float(allowed_entry.end)
        assert all(too.executed for too in ditl.too_register)
        assert ditl.validate_plan_matches_execution() == []

    def test_a_held_toos_collected_science_still_counts_towards_its_share(
        self,
    ) -> None:
        """A ``interrupt=False`` ToO already observing is held out of the
        interrupt attempt's placement, but its collected science must still
        count towards its program's share: three ToOs in program A, which has
        delivered all the science, so the interrupting one is worth its merit
        less the deficit weight times half, 150, not 250, and the later ToO
        worth 175 can interrupt it in turn before its deadline."""
        programs = [
            ObservationCategory(
                name="A", obsid_min=1_000_000, obsid_max=1_000_010, time_share=0.5
            ),
            ObservationCategory(name="B", obsid_min=1, obsid_max=10, time_share=0.5),
        ]
        config = _config(
            3,
            weights=TargetConfig(completion_deficit_weight=100),
            categories=programs,
        )
        initial = {
            **GRB,
            "name": "Initial",
            "ra": 105.0,
            "merit": 10.0,
            "exptime": 3600,
            "submit_time": T0,
            "interrupt": False,
        }
        first = {
            **GRB,
            "obsid": 1_000_002,
            "name": "First",
            "merit": 200.0,
            "exptime": 3600,
            "deadline": T0 + 30 * MIN,
        }
        second = {
            **GRB,
            "obsid": 1_000_003,
            "name": "Second",
            "ra": 110.0,
            "merit": 225.0,
            "exptime": 600,
            "submit_time": T0 + 40 * MIN,
            "deadline": T0 + 50 * MIN,
        }
        ditl = RollingHorizonDITL(
            config,
            [],
            begin=BEGIN,
            end=BEGIN + timedelta(hours=3),
            replan_interval=timedelta(hours=4),
        )
        ditl.step_size = 60
        for too in (initial, first, second):
            ditl.submit_too(**too)  # type: ignore[arg-type]
        assert ditl.calc()

        science = [e for e in ditl.plan if e.obstype.name == "AT"]
        first_entry = next(e for e in science if int(e.obsid) == 1_000_002)
        assert float(first_entry.merit) == pytest.approx(150.0)
        interrupted = [
            r.interrupted_obsid for r in ditl.replans if r.interrupted_obsid is not None
        ]
        assert interrupted == [1_000_001, 1_000_002]
        response = ditl.too_response_times()[1_000_003]
        assert response is not None and response <= 10 * MIN
        assert all(too.executed for too in ditl.too_register)
        assert ditl.validate_plan_matches_execution() == []

    def test_a_non_interruptible_observation_is_not_cut(self) -> None:
        """The ToO that interrupts this observation in its default category
        cannot when the category is not interruptible, and misses its
        deadline."""
        locked = ObservationCategory(
            name="Locked", obsid_min=1, obsid_max=2, interruptible=False
        )
        config = _config(3, categories=[locked])
        target = _target(
            config, 1, 105.0, 10.0, merit=10, minutes=60, snapshot=60, ss_min=10
        )

        ditl = _run(
            3,
            [target],
            toos=({**GRB, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
        )

        assert all(r.interrupted_obsid is None for r in ditl.replans)
        assert ditl.too_response_times()[1_000_001] is None
        assert ditl.validate_plan_matches_execution() == []

    def test_without_interrupts_the_too_misses_its_deadline(self) -> None:
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
            allow_interrupts=False,
        )

        assert ditl.replans[-1].interrupted_obsid is None
        assert ditl.too_response_times()[1_000_001] is None
        assert not ditl.too_register[0].executed

    def test_distant_deadline_waits_for_the_scheduled_replan(self) -> None:
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 3 * HOUR},),
            replan_interval=timedelta(hours=1),
        )

        assert ReplanReason.RAPID not in [r.reason for r in ditl.replans]
        response = ditl.too_response_times()[1_000_001]
        assert response is not None and response >= 40 * MIN - MIN

    def test_deadline_when_next_plan_takes_effect_needs_a_rapid_replan(
        self,
    ) -> None:
        """The next plan takes effect at its cutoff but still has to slew."""
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + HOUR + 30 * MIN},),
            replan_interval=timedelta(hours=1),
            commit_lead_time=timedelta(minutes=30),
        )

        assert ReplanReason.RAPID in [r.reason for r in ditl.replans]
        assert ditl.too_response_times()[1_000_001] is not None

    def test_rapid_replans_can_be_disabled(self) -> None:
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
            rapid_replans=False,
        )

        assert ReplanReason.RAPID not in [r.reason for r in ditl.replans]

    def test_interrupt_keeps_science_already_collected(self) -> None:
        """With no lead time, cleanup follows the cut instead of preceding it."""
        targets = _long_observation()
        assert targets[0].config is not None
        targets[0].config.payload.observation_timing.cleanup_seconds = 120

        ditl = _run(
            3,
            targets,
            toos=({**GRB, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
        )

        assert ditl.replans[-1].interrupted_obsid == 1
        planned = sum(
            e.collection_seconds_between(e.begin, e.end)
            for e in ditl.plan
            if e.obstype.name == "AT"
        )
        assert planned == pytest.approx(_executed_seconds(ditl))
        assert ditl.validate_plan_matches_execution() == []

    def test_simultaneous_urgent_toos_interrupt_for_the_highest_ranked(
        self,
    ) -> None:
        """A low-merit ToO submitted alongside must not block the interrupt."""
        low = {
            **GRB,
            "obsid": 1_000_002,
            "name": "Low",
            "merit": 1.0,
            "deadline": T0 + 30 * MIN,
        }
        ditl = _run(
            3,
            _long_observation(),
            toos=(low, {**GRB, "deadline": T0 + 30 * MIN}),
            replan_interval=timedelta(hours=4),
        )

        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert [r.trigger_obsid for r in rapid] == [GRB["obsid"]]
        assert rapid[0].interrupted_obsid == 1
        assert ditl.too_response_times()[1_000_001] is not None

    def test_response_time_runs_to_the_start_of_collection(self) -> None:
        """Collection starting partway through a step counts from that point."""
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
        )

        entry = next(e for e in ditl.plan if e.obsid == GRB["obsid"])
        assert entry.collection_begin is not None
        assert (float(entry.collection_begin) - T0) % 60
        assert ditl.too_response_times()[1_000_001] == pytest.approx(
            float(entry.collection_begin) - float(GRB["submit_time"])  # type: ignore[arg-type]
        )

    def test_urgent_too_on_a_scheduled_boundary_interrupts(self) -> None:
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "deadline": T0 + 22 * MIN},),
            replan_interval=timedelta(minutes=20),
        )

        boundary = next(r for r in ditl.replans if r.utime == GRB["submit_time"])
        assert boundary.reason is ReplanReason.SCHEDULED
        assert boundary.trigger_obsid == GRB["obsid"]
        assert boundary.interrupted_obsid == 1
        assert ditl.too_register[0].executed
        assert ditl.validate_plan_matches_execution() == []

    def test_committed_observation_blocking_the_deadline_triggers_rapid_replan(
        self,
    ) -> None:
        config = _config(3)
        targets = [
            _target(
                config, 1, 105.0, 10.0, merit=10, minutes=120, snapshot=120, ss_min=10
            )
        ]

        ditl = _run(
            3,
            targets,
            toos=({**GRB, "deadline": T0 + 90 * MIN},),
            replan_interval=timedelta(hours=1),
        )

        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert [r.interrupted_obsid for r in rapid] == [1]
        assert ditl.too_register[0].executed

    def test_infeasible_urgent_too_does_not_interrupt(self) -> None:
        """A ToO inside the Sun keepout cannot replace the observation."""
        ditl = _run(
            3,
            _long_observation(),
            toos=({**GRB, "ra": 215.0, "dec": -15.0, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
        )

        assert all(r.interrupted_obsid is None for r in ditl.replans)
        assert not ditl.too_register[0].executed
        science = [int(e.obsid) for e in ditl.plan if e.obstype.name == "AT"]
        assert science[:1] == [1] and science.count(1) == 1
        assert ditl.validate_plan_matches_execution() == []

    def test_infeasible_top_ranked_too_falls_back_to_the_next(self) -> None:
        """A Sun-blocked higher-ranked ToO must not cost a feasible one its slot."""
        blocked = {
            **GRB,
            "obsid": 1_000_002,
            "name": "Blocked",
            "ra": 215.0,
            "dec": -15.0,
            "merit": 600.0,
            "deadline": T0 + 30 * MIN,
        }
        ditl = _run(
            3,
            _long_observation(),
            toos=(blocked, {**GRB, "deadline": T0 + 30 * MIN}),
            replan_interval=timedelta(hours=4),
        )

        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert [(r.trigger_obsid, r.interrupted_obsid) for r in rapid] == [
            (GRB["obsid"], 1)
        ]
        assert ditl.too_response_times()[1_000_001] is not None
        assert ditl.too_response_times()[1_000_002] is None
        assert ditl.validate_plan_matches_execution() == []

    def test_rejected_interrupt_leaves_no_events_in_the_store(
        self, tmp_path: Path
    ) -> None:
        config = _long_observation()[0].config
        assert config is not None
        targets = [
            _target(
                config, 1, 105.0, 10.0, merit=10, minutes=60, snapshot=60, ss_min=10
            )
        ]
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=3),
            replan_interval=timedelta(hours=4),
        )
        ditl.step_size = 60
        ditl.submit_too(**{**GRB, "ra": 215.0, "dec": -15.0, "deadline": T0 + 30 * MIN})  # type: ignore[arg-type]
        with DITLLogStore(tmp_path / "log.sqlite") as store:
            ditl.log.store = store
            ditl.log.run_id = "run"
            assert ditl.calc()
            stored = [e.description for e in store.fetch_events("run")]

        assert all(r.interrupted_obsid is None for r in ditl.replans)
        assert not any("interrupts observation" in d for d in stored)
        assert sorted(stored) == sorted(e.description for e in ditl.log.events)

    def test_interrupt_ranking_uses_delivered_program_shares(self) -> None:
        """Program B is ahead, so its ToO does not outrank program A's work."""
        config = _config(
            3,
            weights=TargetConfig(completion_deficit_weight=100.0),
            categories=[
                ObservationCategory(
                    name="A", obsid_min=1, obsid_max=99, time_share=0.5
                ),
                ObservationCategory(
                    name="B", obsid_min=100, obsid_max=2_000_000, time_share=0.5
                ),
            ],
        )
        delivered = _target(config, 100, *PATCH[2], minutes=1000)
        delivered.record_collection(T0, 60_000)
        targets = [
            _target(
                config, 1, 105.0, 10.0, merit=50, minutes=60, snapshot=60, ss_min=10
            ),
            delivered,
        ]

        ditl = _run(
            3,
            targets,
            toos=({**GRB, "merit": 70.0, "deadline": T0 + 30 * MIN},),
            replan_interval=timedelta(hours=4),
        )

        rapid = [r for r in ditl.replans if r.reason is ReplanReason.RAPID]
        assert rapid and all(r.interrupted_obsid is None for r in rapid)


class TestConsecutiveIntervals:
    def test_completed_too_is_not_observed_again_in_the_next_interval(
        self,
    ) -> None:
        config = _config(3)
        ditl = RollingHorizonDITL(
            config,
            [],
            begin=BEGIN,
            end=BEGIN + timedelta(hours=1),
            replan_interval=timedelta(hours=4),
        )
        ditl.step_size = 60
        too = ditl.submit_too(
            obsid=1_000_001, ra=115.0, dec=10.0, merit=500.0, exptime=600, name="GRB"
        )
        assert ditl.calc()
        assert too.executed
        response = ditl.too_response_times()[1_000_001]
        assert response is not None

        ditl.begin += timedelta(hours=1)
        ditl.end += timedelta(hours=1)
        assert ditl.calc()

        assert [int(t.obsid) for t in ditl.targets] == [1_000_001]
        assert ditl.targets[0].collected_seconds == pytest.approx(600)
        assert not any(int(e.obsid) == 1_000_001 for e in ditl.plan)
        assert too.executed
        assert ditl.too_response_times()[1_000_001] == response

    def test_the_next_interval_predicts_its_own_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The passes are cached for the plans of one ``calc()``. A later
        interval would otherwise filter the first one's passes to nothing
        and silently plan no contacts at all."""
        config = _config(20, stations=True)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=120, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH)
        ]
        predictions = 0
        original = PassTimes.get

        def counted(self: PassTimes, *args: object, **kwargs: object) -> None:
            nonlocal predictions
            predictions += 1
            original(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(PassTimes, "get", counted)
        # The scenario's first contacts come about 8 h in; both intervals
        # have several.
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN + timedelta(hours=6),
            end=BEGIN + timedelta(hours=12),
            horizon=timedelta(hours=6),
            replan_interval=timedelta(hours=3),
        )
        ditl.step_size = 60
        assert ditl.calc()
        first = [e for e in ditl.plan if e.obstype.name == "GSP"]
        assert first
        assert predictions == 1

        ditl.begin += timedelta(hours=6)
        ditl.end += timedelta(hours=6)
        assert ditl.calc()

        second = [e for e in ditl.plan if e.obstype.name == "GSP"]
        assert second
        assert all(float(e.begin) >= ditl.ustart for e in second)
        # The second interval predicts for itself, once, and executes it all.
        assert predictions == 2
        assert ditl.validate_plan_matches_execution() == []


class TestScienceData:
    def test_too_science_data_reaches_its_plan_entries(self) -> None:
        config = _config(2)
        ditl = RollingHorizonDITL(
            config, [], begin=BEGIN, end=BEGIN + timedelta(hours=1)
        )
        ditl.step_size = 60
        science = {"Default Instrument": ScienceData(rate_gbps=0.001)}
        ditl.submit_too(
            obsid=1_000_001,
            ra=115.0,
            dec=10.0,
            merit=500.0,
            exptime=600,
            name="GRB",
            science_data=science,
        )
        assert ditl.calc()

        entries = [e for e in ditl.plan if int(e.obsid) == 1_000_001]
        assert entries
        assert all(e.science_data == science for e in entries)


class TestArguments:
    @pytest.mark.parametrize(
        "options",
        [
            {"horizon": timedelta(0)},
            {"replan_interval": timedelta(minutes=-1)},
            {"commit_lead_time": timedelta(minutes=-1)},
        ],
    )
    def test_rejects_invalid_timing(self, options: dict[str, timedelta]) -> None:
        config = _config(1)

        with pytest.raises(ValueError):
            RollingHorizonDITL(
                config, [], begin=BEGIN, end=BEGIN + timedelta(hours=1), **options
            )
