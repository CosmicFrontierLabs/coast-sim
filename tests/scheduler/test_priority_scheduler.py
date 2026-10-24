"""PriorityScheduler: plans must execute in DITL exactly as planned."""

from datetime import timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest
from rust_ephem import SunConstraint

from conops import DITL
from conops.common import ObsType
from conops.config import (
    Constraint,
    MissionConfig,
    OnboardRecorder,
    Payload,
    ScienceData,
    TargetConfig,
    Telescope,
)
from conops.config.observation_categories import ObservationCategory
from conops.schedulers import PriorityScheduler, SchedulingContext
from conops.schedulers.priority_scheduler import _Move
from conops.targets import Plan, PlanEntry, Pointing

from .planning_scenario import (
    BEGIN,
    CONSTRAINED,
    HOUR,
    MIN,
    PATCH,
    T0,
)
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target


def _execute(config: MissionConfig, plan: Plan, hours: int, step: int) -> DITL:
    """Execute a serialized copy of the plan in DITL."""
    replay = Plan.model_validate_json(plan.model_dump_json())
    ditl = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=replay,
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
    )
    ditl.step_size = step
    assert ditl.calc()
    return ditl


def _science(plan: Plan) -> list[PlanEntry]:
    return [entry for entry in plan if entry.obstype != ObsType.GSP]


def _planned_collection(plan: Plan) -> float:
    return sum(
        float(e.collection_end) - float(e.collection_begin)
        for e in _science(plan)
        if e.collection_begin is not None and e.collection_end is not None
    )


class TestExecution:
    def test_plan_executes_exactly_as_planned(self) -> None:
        hours = 4
        config = _config(hours)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=40, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours)
        )

        plan = scheduler.schedule()
        ditl = _execute(config, plan, hours, scheduler.ctx.step_size)

        assert len(_science(plan)) >= 5
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_plan_with_ground_passes_executes_exactly_as_planned(self) -> None:
        hours = 12
        config = _config(hours, stations=True)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours)
        )

        plan = scheduler.schedule()
        ditl = _execute(config, plan, hours, scheduler.ctx.step_size)

        assert any(entry.obstype == ObsType.GSP for entry in plan)
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_plan_entries_start_on_simulation_steps(self) -> None:
        config = _config(2)
        targets = [
            _target(config, 100 + i, ra, dec) for i, (ra, dec) in enumerate(PATCH)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=2)
        )

        plan = scheduler.schedule()

        assert all((entry.begin - T0) % scheduler.ctx.step_size == 0 for entry in plan)


class TestOrdering:
    def test_higher_merit_is_placed_first(self) -> None:
        config = _config(2)
        low = _target(config, 1, 105.0, 10.0, merit=10)
        high = _target(config, 2, 110.0, 10.0, merit=90)

        plan = PriorityScheduler(
            config, [low, high], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        assert [e.merit for e in _science(plan)] == [90.0, 10.0]

    def test_merit_orders_requests_without_the_legacy_alias(self) -> None:
        """Ordering follows the canonical merit field, not the legacy ``fom``."""
        config = _config(2)
        targets = []
        # Obsids whose tie-break alone would order the low-merit request first.
        for obsid, (ra, dec), merit in (
            (2, (105.0, 10.0), 10.0),
            (3, (110.0, 10.0), 90.0),
        ):
            target = Pointing(
                config=config,
                obsid=obsid,
                name=f"T{obsid}",
                ra=ra,
                dec=dec,
                merit=merit,
                ss_min=20 * MIN,
                ss_max=20 * MIN,
            )
            target.exptime = 20 * MIN
            targets.append(target)

        plan = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [3, 2]

    def test_delivered_program_shares_order_the_first_pass(self) -> None:
        """An over-served program loses its completion-deficit credit."""
        categories = [
            ObservationCategory(
                name="A", obsid_min=1, obsid_max=2, program="A", time_share=0.5
            ),
            ObservationCategory(
                name="B", obsid_min=2, obsid_max=3, program="B", time_share=0.5
            ),
        ]
        config = _config(
            2,
            weights=TargetConfig(completion_deficit_weight=100),
            categories=categories,
        )
        over_served = _target(config, 1, 105.0, 10.0, merit=60)
        under_served = _target(config, 2, 110.0, 10.0, merit=50)
        for target, delivered in ((over_served, 900.0), (under_served, 100.0)):
            target.record_collection(T0 - HOUR, delivered)
            target.exptime = 20 * MIN

        plan = PriorityScheduler(
            config, [over_served, under_served], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        # Each entry keeps the value it was selected at. By the time A is
        # placed, B's 1200 s snapshot has moved A to 900 of 2200 seconds.
        a_value = 60 + 100 * (0.5 - 900 / 2200)
        assert [e.merit for e in _science(plan)] == [90.0, pytest.approx(a_value)]


class TestScienceData:
    def test_entries_carry_the_targets_science_data(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0)
        target.science_data = {"Default Instrument": ScienceData(rate_gbps=0.001)}

        plan = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert _science(plan)
        assert all(e.science_data == target.science_data for e in _science(plan))


class TestSnapshots:
    def test_exposure_is_split_into_snapshots(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)

        plan = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        snapshots = _science(plan)
        assert len(snapshots) == 2
        assert all(entry.exposure == 20 * MIN for entry in snapshots)

    def test_remainder_below_minimum_snapshot_follows_a_full_one(self) -> None:
        """Twenty-five minutes in snapshots of twenty with a minimum of ten:
        the five-minute remainder is placed too, as dispatch finishes a
        target that has collected anything; a first snapshot is never
        shorter than the minimum."""
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=25, snapshot=20, ss_min=10)

        scheduler = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        )
        plan = scheduler.schedule()

        assert [entry.exposure for entry in _science(plan)] == [20 * MIN, 5 * MIN]
        assert scheduler.unplaced == []
        short = _target(config, 2, 105.0, 10.0, minutes=5, snapshot=20, ss_min=10)
        scheduler = PriorityScheduler(
            config, [short], BEGIN, BEGIN + timedelta(hours=2)
        )
        assert _science(scheduler.schedule()) == []

    def test_zero_minimum_snapshot_stops_when_the_request_is_met(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=1, ss_min=0)

        scheduler = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        )
        plan = scheduler.schedule()

        assert [entry.exposure for entry in _science(plan)] == [MIN]
        assert scheduler.unplaced == []

    def test_inputs_are_not_modified(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)

        PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert target.exptime == 40 * MIN
        assert not target.done


class TestDeadlinesAndConstraints:
    def test_collection_starts_by_the_deadline(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, deadline=T0 + 30 * MIN)

        plan = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        (entry,) = _science(plan)
        assert entry.collection_begin is not None
        assert entry.collection_begin <= T0 + 30 * MIN

    def test_collection_starts_no_earlier_than_the_earliest_start(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, earliest_start=T0 + 30 * MIN)

        plan = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        (entry,) = _science(plan)
        assert entry.collection_begin is not None
        assert T0 + 30 * MIN <= entry.collection_begin < T0 + 35 * MIN

    def test_other_requests_use_the_time_before_the_earliest_start(self) -> None:
        config = _config(2)
        later = _target(config, 1, 105.0, 10.0, merit=90, earliest_start=T0 + 40 * MIN)
        now = _target(config, 2, 110.0, 10.0, merit=10)

        plan = PriorityScheduler(
            config, [later, now], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [int(e.obsid) for e in _science(plan)] == [2, 1]

    def test_executes_as_planned_with_an_earliest_start(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, earliest_start=T0 + 30 * MIN)
        scheduler = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        )
        plan = scheduler.schedule()

        ditl = _execute(config, plan, 2, scheduler.ctx.step_size)

        assert ditl.validate_plan_matches_execution() == []

    def test_earliest_start_and_deadline_met_when_the_slew_lengthens(
        self,
    ) -> None:
        """Only the start at 9600 s collects between 9809 s and 9819 s.

        From the horizon start the slew takes 208 s, which puts the estimate
        for an earliest start of 9809 s at 9660 s. The slew takes 209 s from
        there, so collection there would begin at 9869 s, after the deadline.
        """
        config = _config(6)
        target = _target(
            config,
            33,
            271.24954384622714,
            -10.971713550878292,
            minutes=10,
            snapshot=10,
            ss_min=5,
            earliest_start=T0 + 9809,
            deadline=T0 + 9819,
        )
        scheduler = PriorityScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=6),
            step_size=60,
            include_passes=False,
        )
        plan = scheduler.schedule()

        (entry,) = _science(plan)
        assert entry.collection_begin is not None
        assert T0 + 9809 <= entry.collection_begin <= T0 + 9819
        ditl = _execute(config, plan, 6, scheduler.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []

    def test_request_opening_after_the_horizon_does_not_abort_the_plan(
        self,
    ) -> None:
        """A request whose earliest start is past the horizon goes unplaced
        without stopping others from being placed."""
        config = _config(2)
        later = _target(config, 1, *PATCH[0], merit=100, earliest_start=T0 + 3 * HOUR)
        now = _target(config, 2, *PATCH[1], merit=50)
        scheduler = PriorityScheduler(
            config,
            [later, now],
            BEGIN,
            BEGIN + timedelta(hours=2),
            include_passes=False,
        )
        plan = scheduler.schedule()

        assert [entry.obsid for entry in _science(plan)] == [2]
        assert [target.obsid for target in scheduler.unplaced] == [1]

    def test_priority_order_can_cost_a_short_window_request(self) -> None:
        """Placed first, a flexible high-merit request takes the slot B needed."""
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        scheduler = PriorityScheduler(
            config, [flexible, urgent], BEGIN, BEGIN + timedelta(hours=3)
        )
        plan = scheduler.schedule()

        assert [e.obsid for e in _science(plan)] == [1]
        assert scheduler.unplaced == [urgent]

    def test_urgency_orders_the_short_window_request_first(self) -> None:
        config = _config(3, weights=TargetConfig(urgency_weight=50))
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        scheduler = PriorityScheduler(
            config, [flexible, urgent], BEGIN, BEGIN + timedelta(hours=3)
        )
        plan = scheduler.schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        assert scheduler.unplaced == []

    def test_target_never_clear_of_the_sun_is_unplaced(self) -> None:
        config = _config(2)
        near_sun = _target(config, 1, 216.0, -14.5)

        scheduler = PriorityScheduler(
            config, [near_sun], BEGIN, BEGIN + timedelta(hours=2)
        )
        plan = scheduler.schedule()

        assert _science(plan) == []
        assert scheduler.unplaced == [near_sun]


class TestLockedEntries:
    def test_locked_collection_window_is_kept_and_executes(self) -> None:
        hours = 3
        config = _config(hours)
        locked = PlanEntry(
            config=config,
            name="Locked",
            ra=115.0,
            dec=10.0,
            roll=0.0,
            obsid=900,
            obstype=ObsType.AT,
            begin=T0 + HOUR,
            end=T0 + HOUR + 20 * MIN,
            collection_begin=T0 + HOUR,
            collection_end=T0 + HOUR + 20 * MIN,
            ss_min=60,
        )
        others = [
            _target(config, 100 + i, ra, dec, minutes=40)
            for i, (ra, dec) in enumerate(PATCH)
        ]

        scheduler = PriorityScheduler(
            config, others, BEGIN, BEGIN + timedelta(hours=hours), locked=[locked]
        )
        plan = scheduler.schedule()

        kept = next(entry for entry in plan if entry.obsid == 900)
        assert (kept.collection_begin, kept.collection_end) == (
            T0 + HOUR,
            T0 + HOUR + 20 * MIN,
        )
        ditl = _execute(config, plan, hours, scheduler.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []

    def test_locked_entry_outside_the_horizon_is_left_out(self) -> None:
        config = _config(6)
        start = T0 + 4 * HOUR
        locked = PlanEntry(
            config=config,
            name="Locked",
            ra=115.0,
            dec=10.0,
            roll=0.0,
            obsid=900,
            obstype=ObsType.AT,
            begin=start,
            end=start + 20 * MIN,
            collection_begin=start,
            collection_end=start + 20 * MIN,
            ss_min=60,
        )

        plan = PriorityScheduler(
            config, [], BEGIN, BEGIN + timedelta(hours=2), locked=[locked]
        ).schedule()

        assert all(entry.obsid != 900 for entry in plan)

    def test_locked_entry_needs_a_collection_window(self) -> None:
        config = _config(1)
        locked = PlanEntry(config=config, obsid=900, begin=T0, end=T0 + 600)

        with pytest.raises(ValueError, match="collection window"):
            PriorityScheduler(
                config, [], BEGIN, BEGIN + timedelta(hours=1), locked=[locked]
            ).schedule()

    def test_locked_entry_that_breaks_its_constraints_is_left_out(self) -> None:
        """A lock reachable at ingress but not holdable for its whole window."""
        hours = 6
        config = _config(hours)
        probe = _target(config, 900, *CONSTRAINED[0])
        probe.visibility()
        window = probe.windows[0]
        locked = PlanEntry(
            config=config,
            name="Locked",
            ra=CONSTRAINED[0][0],
            dec=CONSTRAINED[0][1],
            roll=0.0,
            obsid=900,
            obstype=ObsType.AT,
            begin=window[0] + 300,
            end=window[1] + 600,
            collection_begin=window[0] + 300,
            collection_end=window[1] + 600,
            ss_min=60,
        )

        scheduler = PriorityScheduler(
            config, [], BEGIN, BEGIN + timedelta(hours=hours), locked=[locked]
        )
        plan = scheduler.schedule()

        assert all(entry.obsid != 900 for entry in plan)
        ditl = _execute(config, plan, hours, scheduler.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []

    def test_locked_snapshot_counts_against_its_request(self) -> None:
        hours = 3
        config = _config(hours)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)
        first = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=hours)
        ).schedule()
        locked = _science(first)[0]

        replan = PriorityScheduler(
            config,
            [target],
            BEGIN,
            BEGIN + timedelta(hours=hours),
            locked=[locked],
        ).schedule()

        assert len(_science(replan)) == 2
        assert _planned_collection(replan) == pytest.approx(40 * MIN)


class TestMountedTelescope:
    def test_science_windows_follow_the_mounted_line_of_sight(self) -> None:
        """A body keepout must not discard windows at the science coordinates."""
        hours = 2
        config = _config(hours)
        config.constraint = Constraint(
            safety_constraint=SunConstraint(min_angle=10),
            ephem=config.constraint.ephem,
        )
        telescope = Telescope(name="Rear", boresight=(-1.0, 0.0, 0.0))
        config.payload = Payload(instruments=[telescope])
        target = _target(config, 1, 216.0, -14.5, minutes=20)
        target.instrument_name = telescope.name

        scheduler = PriorityScheduler(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        )
        plan = scheduler.schedule()

        assert len(_science(plan)) == 1
        ditl = _execute(config, plan, hours, scheduler.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []


class TestGroundContacts:
    def test_activity_after_a_contact_starts_from_the_braking_rest_attitude(
        self,
    ) -> None:
        """Tracking brakes past the profile, so a contact does not end on it."""
        hours = 12
        config = _config(hours, stations=True)
        contacts = PriorityScheduler(
            config, [], BEGIN, BEGIN + timedelta(hours=hours)
        ).schedule()
        contact = next(entry for entry in contacts if entry.obstype == ObsType.GSP)
        locked = PlanEntry(
            config=config,
            name="Locked",
            ra=contact.track_end_ra,
            dec=contact.track_end_dec,
            roll=contact.track_end_roll,
            obsid=900,
            obstype=ObsType.AT,
            begin=contact.end,
            end=contact.end + 600,
            collection_begin=contact.end,
            collection_end=contact.end + 600,
            ss_min=60,
        )

        scheduler = PriorityScheduler(
            config, [], BEGIN, BEGIN + timedelta(hours=hours), locked=[locked]
        )
        plan = scheduler.schedule()

        ditl = _execute(config, plan, hours, scheduler.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_contact_starting_between_steps_is_acquired_late(self) -> None:
        """A pass beginning off the step grid is picked up on the next step."""
        hours = 12
        step = 180
        config = _config(hours, stations=True)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours), step_size=step
        )
        plan = scheduler.schedule()

        predicted = {
            gspass.station: gspass for gspass in scheduler.ctx.predict_passes()
        }
        contacts = [entry for entry in plan if entry.obstype == ObsType.GSP]
        assert {entry.station for entry in contacts} == set(predicted)
        off_grid = [
            entry
            for entry in contacts
            if (predicted[entry.station].begin - T0) % step  # type: ignore[index]
        ]
        assert off_grid
        for entry in contacts:
            gspass = predicted[entry.station]  # type: ignore[index]
            acquire = scheduler.ctx.ceil_step(gspass.begin)
            assert entry.contact_begin == acquire
            assert entry.contact_end == gspass.end
            assert entry.exptime == round(gspass.end - acquire)

        ditl = _execute(config, plan, hours, step)
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_contact_blocked_at_its_first_step_is_acquired_at_a_later_one(
        self,
    ) -> None:
        """A lock leaves no time to reach the first step, but the next one fits."""
        hours = 12
        step = 180
        config = _config(hours, stations=True)
        ctx = SchedulingContext(
            config, BEGIN, BEGIN + timedelta(hours=hours), step_size=step
        )
        mal = next(p for p in ctx.predict_passes() if p.station == "MAL")
        first, second = ctx.steps(mal.begin, mal.end)[:2]
        locked = PlanEntry(
            config=config,
            name="Locked",
            ra=105.0,
            dec=10.0,
            roll=0.0,
            obsid=900,
            obstype=ObsType.AT,
            begin=mal.begin - 600,
            end=mal.begin + 60,
            collection_begin=mal.begin - 600,
            collection_end=mal.begin + 60,
            ss_min=60,
        )
        assert first <= locked.end < second

        scheduler = PriorityScheduler(
            config,
            [],
            BEGIN,
            BEGIN + timedelta(hours=hours),
            step_size=step,
            locked=[locked],
        )
        plan = scheduler.schedule()

        contact = next(e for e in plan if e.station == "MAL")
        assert contact.contact_begin == second
        assert contact.contact_end == mal.end
        assert contact.exptime == round(mal.end - second)
        ditl = _execute(config, plan, hours, step)
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(660)

    def test_downlink_is_credited_only_for_the_usable_contact(self) -> None:
        """Contacts acquired late and ending mid-step downlink only while in contact."""
        hours = 12
        step = 180
        config = _config(hours, stations=True)
        config.recorder = OnboardRecorder(capacity_gb=1000, current_volume_gb=100)
        scheduler = PriorityScheduler(
            config, [], BEGIN, BEGIN + timedelta(hours=hours), step_size=step
        )
        plan = scheduler.schedule()
        contacts = [entry for entry in plan if entry.obstype == ObsType.GSP]
        assert contacts
        # Some contact ends partway through a step.
        assert any((float(e.contact_end) - T0) % step for e in contacts)  # type: ignore[arg-type]

        ditl = _execute(config, plan, hours, step)

        assert ditl.validate_plan_matches_execution() == []
        expected = 0.0
        for entry in contacts:
            assert entry.contact_begin is not None and entry.contact_end is not None
            station = config.ground_stations.get(entry.station)  # type: ignore[arg-type]
            rate_mbps = ditl._get_effective_data_rate(station)
            assert rate_mbps
            seconds = float(entry.contact_end) - float(entry.contact_begin)
            expected += rate_mbps * seconds / 1000.0
        assert ditl.data_downlinked_gb[-1] == pytest.approx(expected)

    def test_contact_with_no_step_inside_it_is_not_reserved(self) -> None:
        """Steps too coarse to land inside a pass leave it unreserved."""
        hours = 12
        step = 900
        config = _config(hours, stations=True)
        scheduler = PriorityScheduler(
            config, [], BEGIN, BEGIN + timedelta(hours=hours), step_size=step
        )
        plan = scheduler.schedule()

        unreachable = {
            gspass.station
            for gspass in scheduler.ctx.predict_passes()
            if scheduler.ctx.ceil_step(gspass.begin) >= gspass.utime[-1]
        }
        assert unreachable
        stations = {entry.station for entry in plan if entry.obstype == ObsType.GSP}
        assert stations and not stations & unreachable
        ditl = _execute(config, plan, hours, step)
        assert ditl.validate_plan_matches_execution() == []


class TestSuccessorRetries:
    """A limit on later starts tried in a gap whose successor is unreachable."""

    HOURS = 12

    def _plan(
        self, retries: int | None
    ) -> tuple[MissionConfig, PriorityScheduler, int]:
        config = _config(self.HOURS, stations=True)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - 3 * i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        scheduler = PriorityScheduler(
            config,
            targets,
            BEGIN,
            BEGIN + timedelta(hours=self.HOURS),
            successor_retries=retries,
        )
        with patch.object(scheduler, "_connect", wraps=scheduler._connect) as connect:
            scheduler.schedule()
        return config, scheduler, connect.call_count

    def test_a_limit_searches_less_and_still_executes_exactly(self) -> None:
        _, exhaustive, exhaustive_calls = self._plan(None)
        config, limited, limited_calls = self._plan(0)

        assert limited_calls < exhaustive_calls
        assert len(limited.plan) > 0
        ditl = _execute(config, limited.plan, self.HOURS, 60)
        assert ditl.validate_plan_matches_execution() == []

    def test_rejects_a_negative_limit(self) -> None:
        config = _config(1)

        with pytest.raises(ValueError, match="successor_retries"):
            PriorityScheduler(
                config, [], BEGIN, BEGIN + timedelta(hours=1), successor_retries=-1
            )


class TestDelayingSuccessors:
    """A snapshot that only the next one blocks is fitted by moving that one later."""

    HOURS = 12

    def _plan(self) -> tuple[MissionConfig, PriorityScheduler]:
        config = _config(self.HOURS)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - 7 * i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=self.HOURS)
        )
        scheduler.schedule()
        return config, scheduler

    def test_moves_a_snapshot_later_and_still_executes_exactly(self) -> None:
        config, scheduler = self._plan()

        moves = [e for e in scheduler.log.events if e.description.startswith("Moved")]
        assert moves
        ditl = _execute(config, scheduler.plan, self.HOURS, 60)
        assert ditl.validate_plan_matches_execution() == []

    def test_plans_more_science_than_leaving_the_gap(self) -> None:
        _, delaying = self._plan()
        with patch.object(
            PriorityScheduler, "_fit_delaying_successors", return_value=None
        ):
            _, fixed = self._plan()

        assert _planned_collection(delaying.plan) > _planned_collection(fixed.plan)


class TestSlewBeforeVisible:
    """With slew_before_visible, slews may start before their target rises."""

    HOURS = 12

    def _plan(self, allowed: bool) -> tuple[MissionConfig, PriorityScheduler]:
        config = _config(self.HOURS)
        config.spacecraft_bus.attitude_control.slew_before_visible = allowed
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - 7 * i, minutes=60, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=self.HOURS)
        )
        scheduler.schedule()
        return config, scheduler

    @staticmethod
    def _early_slews(scheduler: PriorityScheduler) -> int:
        """Science slews that start while their target is not visible."""
        return sum(
            1
            for entry in _science(scheduler.plan)
            if entry.slewtime and not entry.visible(entry.begin, entry.begin)
        )

    def test_slews_start_before_the_target_rises_and_execute_exactly(self) -> None:
        config, scheduler = self._plan(allowed=True)

        assert self._early_slews(scheduler) > 0
        ditl = _execute(config, scheduler.plan, self.HOURS, 60)
        assert ditl.validate_plan_matches_execution() == []

    def test_without_it_every_slew_starts_while_the_target_is_visible(self) -> None:
        _, scheduler = self._plan(allowed=False)

        assert self._early_slews(scheduler) == 0


class TestFillingTimeATargetIsHidden:
    """Time a target is hidden between two of its snapshots is filled.

    The later snapshot starts as its target rises, so by default only staying
    on the hidden target reaches it on time. Fitting another snapshot first
    makes it start later, which is allowed.
    """

    HOURS = 12

    def _plan(
        self, config: MissionConfig | None = None
    ) -> tuple[MissionConfig, PriorityScheduler]:
        config = config if config is not None else _config(self.HOURS)
        rng = np.random.default_rng(5)
        targets = [
            _target(
                config,
                100 + i,
                rng.uniform(0, 360),
                rng.uniform(-90, 90),
                merit=rng.uniform(10, 100),
                minutes=2000 / 60,
                snapshot=1000 / 60,
                ss_min=5,
            )
            for i in range(40)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=self.HOURS)
        )
        scheduler.schedule()
        return config, scheduler

    @staticmethod
    def _long_gaps(scheduler: PriorityScheduler) -> list[float]:
        timeline = scheduler.timeline
        gaps = [
            float(after.entry.begin) - float(before.end)
            for before, after in zip(timeline, timeline[1:])
        ]
        return [gap for gap in gaps if gap > 20 * MIN]

    def test_leaves_no_long_gaps_and_executes_exactly(self) -> None:
        config, scheduler = self._plan()

        assert self._long_gaps(scheduler) == []
        ditl = _execute(config, scheduler.plan, self.HOURS, 60)
        assert ditl.validate_plan_matches_execution() == []

    def test_some_moves_cut_a_snapshot_short(self) -> None:
        _, scheduler = self._plan()

        moves = [e for e in scheduler.log.events if e.description.startswith("Moved")]
        assert any("shorter" in e.description for e in moves)

    def test_a_request_cut_short_is_placed_again(self) -> None:
        """Exposure a move cuts from a finished request is planned elsewhere."""
        config, scheduler = self._plan()

        events = scheduler.log.events
        cut = {
            k: int(e.obsid)
            for k, e in enumerate(events)
            if e.description.startswith("Moved") and "shorter" in e.description
        }
        assert cut
        placed_again = [
            obsid
            for k, obsid in cut.items()
            if any(
                e.description.startswith(f"Placed {obsid} ") for e in events[k + 1 :]
            )
        ]
        assert placed_again
        ditl = _execute(config, scheduler.plan, self.HOURS, 60)
        assert ditl.validate_plan_matches_execution() == []

    def test_program_shares_count_only_what_the_plan_delivers(self) -> None:
        """A move that cuts a snapshot short takes its science off its program."""
        config = _config(
            self.HOURS,
            weights=TargetConfig(completion_deficit_weight=50),
            categories=[
                ObservationCategory(
                    name="A", obsid_min=100, obsid_max=120, time_share=0.5
                ),
                ObservationCategory(
                    name="B", obsid_min=120, obsid_max=140, time_share=0.5
                ),
            ],
        )
        _, scheduler = self._plan(config)

        assert any("shorter" in e.description for e in scheduler.log.events)
        planned = {"A": 0.0, "B": 0.0}
        for entry in _science(scheduler.plan):
            program = "A" if int(entry.obsid) < 120 else "B"
            planned[program] += entry.collection_seconds_between(entry.begin, entry.end)
        assert scheduler._program_seconds == pytest.approx(planned)

    def test_without_moving_snapshots_the_gaps_stay(self) -> None:
        """Guards the test above: the scenario must have such gaps to fill."""
        with patch.object(
            PriorityScheduler, "_fit_delaying_successors", return_value=None
        ):
            _, scheduler = self._plan()

        assert self._long_gaps(scheduler)


class TestMovingCadenceVisits:
    """A cadence target's latest visit may move later to fill time before it.

    Each visit waits a cadence after the last, so visits often start as their
    target rises, with nothing placed before them yet.
    """

    HOURS = 12
    CADENCE = HOUR

    def _plan(self) -> tuple[MissionConfig, PriorityScheduler]:
        config = _config(
            self.HOURS,
            weights=TargetConfig(cadence_weight=40.0),
            categories=[
                ObservationCategory(
                    name="Monitor",
                    obsid_min=100,
                    obsid_max=200,
                    cadence_seconds=self.CADENCE,
                )
            ],
        )
        rng = np.random.default_rng(5)
        targets = [
            _target(
                config,
                100 + i,
                rng.uniform(0, 360),
                rng.uniform(-90, 90),
                merit=rng.uniform(10, 100),
                minutes=2000 / 60,
                snapshot=1000 / 60,
                ss_min=5,
            )
            for i in range(40)
        ]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, BEGIN + timedelta(hours=self.HOURS)
        )
        scheduler.schedule()
        return config, scheduler

    @staticmethod
    def _never_moving_cadence_visits(
        scheduler: PriorityScheduler, block: object
    ) -> bool:
        request = scheduler._movable.get(id(block))
        return request is not None and request.cadence is None

    def test_moves_visits_and_executes_exactly(self) -> None:
        config, scheduler = self._plan()

        assert any(e.description.startswith("Moved") for e in scheduler.log.events)
        assert TestFillingTimeATargetIsHidden._long_gaps(scheduler) == []
        ditl = _execute(config, scheduler.plan, self.HOURS, 60)
        assert ditl.validate_plan_matches_execution() == []

    def test_visits_stay_a_cadence_apart(self) -> None:
        """A moved visit is fitted again after the new snapshot, which can be
        earlier than it was: it must not move closer to the visit before."""
        _, scheduler = self._plan()

        visits: dict[int, list[PlanEntry]] = {}
        for entry in _science(scheduler.plan):
            visits.setdefault(int(entry.obsid), []).append(entry)
        spacings = [
            float(later.collection_begin) - float(earlier.collection_end)
            for entries in visits.values()
            for earlier, later in zip(entries, entries[1:])
        ]
        assert spacings
        assert min(spacings) >= self.CADENCE

    def test_without_moving_them_time_is_lost(self) -> None:
        """Guards the tests above: the scenario must need cadence visits moved."""
        _, moving = self._plan()
        with patch.object(
            PriorityScheduler, "_may_move", self._never_moving_cadence_visits
        ):
            _, fixed = self._plan()

        assert TestFillingTimeATargetIsHidden._long_gaps(fixed)
        assert _planned_collection(moving.plan) > _planned_collection(fixed.plan)


class TestWorthCutting:
    """When a move may cut a snapshot short to fit a new one before it."""

    @staticmethod
    def _request(value: float) -> Mock:
        request = Mock()
        request.steady_value = value
        return request

    def _move(self, lost: float, value: float = 50.0) -> _Move:
        return _Move(Mock(), self._request(value), Mock(), Mock(), lost)

    def test_moves_that_cut_nothing_are_always_worth_it(self) -> None:
        new = self._request(1.0)

        assert PriorityScheduler._worth_cutting(new, 10.0, [self._move(0.0)])

    def test_worth_it_when_the_new_snapshot_is_worth_more(self) -> None:
        new = self._request(20.0)

        assert PriorityScheduler._worth_cutting(new, 900.0, [self._move(300.0)])
        assert not PriorityScheduler._worth_cutting(new, 300.0, [self._move(300.0)])
