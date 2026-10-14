"""PriorityPlanner: plans must execute in DITL exactly as planned."""

from datetime import datetime, timedelta, timezone

import pytest
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import DITL
from conops.common import ObsType
from conops.config import (
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    TargetConfig,
    Telescope,
)
from conops.config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)
from conops.schedulers import PriorityPlanner
from conops.targets import Plan, PlanEntry, Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
T0 = BEGIN.timestamp()
MIN = 60
HOUR = 3600


def _config(
    hours: int,
    *,
    stations: bool = False,
    weights: TargetConfig | None = None,
    categories: list[ObservationCategory] | None = None,
) -> MissionConfig:
    ephem = TLEEphemeris(
        tle="examples/example.tle",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
        step_size=60,
    )
    config = MissionConfig(
        constraint=Constraint(
            sun_constraint=SunConstraint(min_angle=45),
            earth_constraint=EarthLimbConstraint(min_angle=10),
        ),
        ground_stations=GroundStationRegistry.default()
        if stations
        else GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1.0, slew_acceleration=0.5, settle_time=30.0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
        targets=weights or TargetConfig(),
        observation_categories=ObservationCategories(categories=categories or []),
    )
    config.random_seed = 1234
    config.constraint.ephem = ephem
    return config


def _target(
    config: MissionConfig,
    obsid: int,
    ra: float,
    dec: float,
    *,
    merit: float = 50.0,
    minutes: float = 20,
    snapshot: float | None = None,
    ss_min: float | None = None,
    deadline: float | None = None,
) -> Pointing:
    snapshot = minutes if snapshot is None else snapshot
    target = Pointing(
        config=config,
        obsid=obsid,
        name=f"T{obsid}",
        ra=ra,
        dec=dec,
        merit=merit,
        fom=merit,
        ss_min=(snapshot if ss_min is None else ss_min) * MIN,
        ss_max=snapshot * MIN,
        deadline=deadline,
    )
    target.exptime = minutes * MIN
    return target


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


# A patch of sky this orbit keeps visible throughout, and some targets that are
# constrained for part of every orbit.
PATCH = [(105.0, 10.0), (110.0, 10.0), (115.0, 10.0), (105.0, 0.0), (110.0, 20.0)]
CONSTRAINED = [(30.0, 40.0), (300.0, -50.0), (170.0, 60.0), (250.0, 10.0)]


class TestExecution:
    def test_plan_executes_exactly_as_planned(self) -> None:
        hours = 4
        config = _config(hours)
        targets = [
            _target(config, 100 + i, ra, dec, merit=90 - i, minutes=40, snapshot=20)
            for i, (ra, dec) in enumerate(PATCH + CONSTRAINED)
        ]
        planner = PriorityPlanner(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours)
        )

        plan = planner.schedule()
        ditl = _execute(config, plan, hours, planner.ctx.step_size)

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
        planner = PriorityPlanner(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours)
        )

        plan = planner.schedule()
        ditl = _execute(config, plan, hours, planner.ctx.step_size)

        assert any(entry.obstype == ObsType.GSP for entry in plan)
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_plan_entries_start_on_simulation_steps(self) -> None:
        config = _config(2)
        targets = [
            _target(config, 100 + i, ra, dec) for i, (ra, dec) in enumerate(PATCH)
        ]
        planner = PriorityPlanner(config, targets, BEGIN, BEGIN + timedelta(hours=2))

        plan = planner.schedule()

        assert all((entry.begin - T0) % planner.ctx.step_size == 0 for entry in plan)


class TestOrdering:
    def test_higher_merit_is_placed_first(self) -> None:
        config = _config(2)
        low = _target(config, 1, 105.0, 10.0, merit=10)
        high = _target(config, 2, 110.0, 10.0, merit=90)

        plan = PriorityPlanner(
            config, [low, high], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        assert [e.merit for e in _science(plan)] == [90.0, 10.0]

    def test_tier_outranks_merit(self) -> None:
        categories = [ObservationCategory(name="Cal", obsid_min=1, obsid_max=2, tier=1)]
        config = _config(2, categories=categories)
        calibration = _target(config, 1, 105.0, 10.0, merit=1)
        science = _target(config, 2, 110.0, 10.0, merit=90)

        plan = PriorityPlanner(
            config, [science, calibration], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [1, 2]

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

        plan = PriorityPlanner(
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

        plan = PriorityPlanner(
            config, [over_served, under_served], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        assert [e.merit for e in _science(plan)] == [90.0, 20.0]


class TestSnapshots:
    def test_exposure_is_split_into_snapshots(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)

        plan = PriorityPlanner(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        snapshots = _science(plan)
        assert len(snapshots) == 2
        assert all(entry.exposure == 20 * MIN for entry in snapshots)

    def test_remainder_below_minimum_snapshot_is_dropped(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=25, snapshot=20, ss_min=10)

        planner = PriorityPlanner(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        plan = planner.schedule()

        assert [entry.exposure for entry in _science(plan)] == [20 * MIN]
        assert planner.unplaced == []

    def test_zero_minimum_snapshot_stops_when_the_request_is_met(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=1, ss_min=0)

        planner = PriorityPlanner(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        plan = planner.schedule()

        assert [entry.exposure for entry in _science(plan)] == [MIN]
        assert planner.unplaced == []

    def test_inputs_are_not_modified(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)

        PriorityPlanner(config, [target], BEGIN, BEGIN + timedelta(hours=2)).schedule()

        assert target.exptime == 40 * MIN
        assert not target.done


class TestDeadlinesAndConstraints:
    def test_collection_starts_by_the_deadline(self) -> None:
        config = _config(2)
        target = _target(config, 1, 105.0, 10.0, deadline=T0 + 30 * MIN)

        plan = PriorityPlanner(
            config, [target], BEGIN, BEGIN + timedelta(hours=2)
        ).schedule()

        (entry,) = _science(plan)
        assert entry.collection_begin is not None
        assert entry.collection_begin <= T0 + 30 * MIN

    def test_priority_order_can_cost_a_short_window_request(self) -> None:
        """Placed first, a flexible high-merit request takes the slot B needed."""
        config = _config(3)
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        planner = PriorityPlanner(
            config, [flexible, urgent], BEGIN, BEGIN + timedelta(hours=3)
        )
        plan = planner.schedule()

        assert [e.obsid for e in _science(plan)] == [1]
        assert planner.unplaced == [urgent]

    def test_urgency_orders_the_short_window_request_first(self) -> None:
        config = _config(3, weights=TargetConfig(urgency_weight=50))
        flexible = _target(config, 1, 105.0, 10.0, merit=100, minutes=60)
        urgent = _target(
            config, 2, 110.0, 10.0, merit=70, minutes=30, deadline=T0 + 30 * MIN
        )

        planner = PriorityPlanner(
            config, [flexible, urgent], BEGIN, BEGIN + timedelta(hours=3)
        )
        plan = planner.schedule()

        assert [e.obsid for e in _science(plan)] == [2, 1]
        assert planner.unplaced == []

    def test_target_never_clear_of_the_sun_is_unplaced(self) -> None:
        config = _config(2)
        near_sun = _target(config, 1, 216.0, -14.5)

        planner = PriorityPlanner(config, [near_sun], BEGIN, BEGIN + timedelta(hours=2))
        plan = planner.schedule()

        assert _science(plan) == []
        assert planner.unplaced == [near_sun]


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

        planner = PriorityPlanner(
            config, others, BEGIN, BEGIN + timedelta(hours=hours), locked=[locked]
        )
        plan = planner.schedule()

        kept = next(entry for entry in plan if entry.obsid == 900)
        assert (kept.collection_begin, kept.collection_end) == (
            T0 + HOUR,
            T0 + HOUR + 20 * MIN,
        )
        ditl = _execute(config, plan, hours, planner.ctx.step_size)
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

        plan = PriorityPlanner(
            config, [], BEGIN, BEGIN + timedelta(hours=2), locked=[locked]
        ).schedule()

        assert all(entry.obsid != 900 for entry in plan)

    def test_locked_entry_needs_a_collection_window(self) -> None:
        config = _config(1)
        locked = PlanEntry(config=config, obsid=900, begin=T0, end=T0 + 600)

        with pytest.raises(ValueError, match="collection window"):
            PriorityPlanner(
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

        planner = PriorityPlanner(
            config, [], BEGIN, BEGIN + timedelta(hours=hours), locked=[locked]
        )
        plan = planner.schedule()

        assert all(entry.obsid != 900 for entry in plan)
        ditl = _execute(config, plan, hours, planner.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []

    def test_locked_snapshot_counts_against_its_request(self) -> None:
        hours = 3
        config = _config(hours)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)
        first = PriorityPlanner(
            config, [target], BEGIN, BEGIN + timedelta(hours=hours)
        ).schedule()
        locked = _science(first)[0]

        replan = PriorityPlanner(
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

        planner = PriorityPlanner(config, [target], BEGIN, BEGIN + timedelta(hours=2))
        plan = planner.schedule()

        assert len(_science(plan)) == 1
        ditl = _execute(config, plan, hours, planner.ctx.step_size)
        assert ditl.validate_plan_matches_execution() == []


class TestGroundContacts:
    def test_activity_after_a_contact_starts_from_the_braking_rest_attitude(
        self,
    ) -> None:
        """Tracking brakes past the profile, so a contact does not end on it."""
        hours = 12
        config = _config(hours, stations=True)
        contacts = PriorityPlanner(
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

        planner = PriorityPlanner(
            config, [], BEGIN, BEGIN + timedelta(hours=hours), locked=[locked]
        )
        plan = planner.schedule()

        ditl = _execute(config, plan, hours, planner.ctx.step_size)
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
        planner = PriorityPlanner(
            config, targets, BEGIN, BEGIN + timedelta(hours=hours), step_size=step
        )
        plan = planner.schedule()

        predicted = {gspass.station: gspass for gspass in planner.ctx.predict_passes()}
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
            acquire = planner.ctx.ceil_step(gspass.begin)
            assert entry.contact_begin == acquire
            assert entry.contact_end == gspass.end
            assert entry.exptime == round(gspass.end - acquire)

        ditl = _execute(config, plan, hours, step)
        assert ditl.validate_plan_matches_execution() == []
        executed = sum(hk.collection_seconds for hk in ditl.telemetry.housekeeping)
        assert executed == pytest.approx(_planned_collection(plan))

    def test_contact_with_no_step_inside_it_is_not_reserved(self) -> None:
        """Steps too coarse to land inside a pass leave it unreserved."""
        hours = 12
        step = 900
        config = _config(hours, stations=True)
        planner = PriorityPlanner(
            config, [], BEGIN, BEGIN + timedelta(hours=hours), step_size=step
        )
        plan = planner.schedule()

        unreachable = {
            gspass.station
            for gspass in planner.ctx.predict_passes()
            if planner.ctx.ceil_step(gspass.begin) >= gspass.utime[-1]
        }
        assert unreachable
        stations = {entry.station for entry in plan if entry.obstype == ObsType.GSP}
        assert stations and not stations & unreachable
        ditl = _execute(config, plan, hours, step)
        assert ditl.validate_plan_matches_execution() == []
