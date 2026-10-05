"""RollingHorizonDITL: follow a plan rebuilt as the simulation runs."""

from datetime import timedelta

import pytest

from conops.ditl import ReplanReason, RollingHorizonDITL
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


GRB = {
    "obsid": 1_000_001,
    "name": "GRB",
    "ra": 115.0,
    "dec": 10.0,
    "merit": 500.0,
    "exptime": 600,
    "submit_time": T0 + 20 * MIN,
}


def _long_observation() -> list[Pointing]:
    config = _config(3)
    return [
        _target(config, 1, 105.0, 10.0, merit=10, minutes=60, snapshot=60, ss_min=10)
    ]


class TestTargetsOfOpportunity:
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
