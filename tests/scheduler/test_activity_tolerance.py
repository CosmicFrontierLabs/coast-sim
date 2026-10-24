"""Activities that may slide within a tolerance window, periodic activities,
and eclipse intervals given to a run."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, create_ditl
from conops.common import ObsType
from conops.config import Constraint, SolarPanel, SolarPanelSet
from conops.schedulers import PriorityScheduler
from conops.targets import Plan, Pointing, ScheduledActivity
from conops.targets.activity import periodic_activities
from conops.visualization.mpl.ditl_timeline import _extract_eclipses

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
BLOCK_START = T0 + 2 * HOUR
BLOCK = 20 * MIN


def _long_targets(config) -> list[Pointing]:  # type: ignore[no-untyped-def]
    """Targets observed in long snapshots, so one is always under way."""
    return [
        _target(config, 100 + i, ra, dec, merit=90 - i, minutes=300, snapshot=60)
        for i, (ra, dec) in enumerate(PATCH[:3])
    ]


def _activities(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype == ObsType.ACTIVITY]


def _science(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype not in (ObsType.GSP, ObsType.ACTIVITY)]


def _dispatch(activities: list[ScheduledActivity]) -> QueueDITL:
    config = _config(HOURS)
    ditl = create_ditl(config, _long_targets(config), BEGIN, END, activities=activities)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl


class TestModel:
    def test_a_slide_gives_a_latest_start_and_not_with_fixed(self) -> None:
        block = ScheduledActivity(
            name="Cal",
            begin=BLOCK_START,
            duration_seconds=BLOCK,
            slide_seconds=30 * MIN,
        )
        assert block.slides and block.latest_start == BLOCK_START + 30 * MIN
        plain = ScheduledActivity(name="Cal", begin=BLOCK_START, duration_seconds=BLOCK)
        assert not plain.slides and plain.latest_start == plain.begin
        with pytest.raises(ValidationError, match="cannot slide"):
            ScheduledActivity(
                name="Cal",
                begin=BLOCK_START,
                duration_seconds=BLOCK,
                fixed=True,
                slide_seconds=60.0,
            )

    def test_periodic_activities(self) -> None:
        twice = periodic_activities(
            "Lamp",
            BEGIN,
            12 * HOUR,
            BLOCK,
            count=4,
            instrument_modes={},
            data_gb=0.5,
            slide_seconds=10 * MIN,
        )
        assert [a.begin for a in twice] == [T0 + k * 12 * HOUR for k in range(4)]
        assert all(
            (a.name, a.duration_seconds, a.data_gb, a.slide_seconds)
            == ("Lamp", BLOCK, 0.5, 10 * MIN)
            for a in twice
        )
        until = periodic_activities(
            "Lamp", T0, 12 * HOUR, BLOCK, until=T0 + 2 * 24 * HOUR
        )
        assert len(until) == 5
        with pytest.raises(ValueError, match="count or until"):
            periodic_activities("Lamp", T0, 12 * HOUR, BLOCK)
        with pytest.raises(ValueError, match="positive"):
            periodic_activities("Lamp", T0, 0.0, BLOCK, count=1)


class TestDispatch:
    def test_a_sliding_activity_waits_for_the_observation_under_way(self) -> None:
        """A block due while an observation collects waits for it, within
        its window, then runs its full length; without a window it cuts
        the observation at its start."""
        sliding = ScheduledActivity(
            name="Cal",
            begin=BLOCK_START,
            duration_seconds=BLOCK,
            slide_seconds=45 * MIN,
        )
        ditl = _dispatch([sliding])
        (entry,) = _activities(ditl.plan)
        under_way = [
            e
            for e in _science(ditl.plan)
            if float(e.begin) < BLOCK_START < float(e.end)
        ]
        assert len(under_way) == 1
        assert float(under_way[0].end) <= float(entry.begin) <= sliding.latest_start
        assert float(entry.begin) > BLOCK_START
        assert float(entry.end) - float(entry.begin) == pytest.approx(BLOCK)
        assert not under_way[0].interruptions
        assert any(
            "waits for observation" in e.description
            and "at the latest" in e.description
            for e in ditl.log.events
        )
        # Nothing starts while the block waits.
        assert not any(
            BLOCK_START <= float(e.begin) < float(entry.begin)
            for e in _science(ditl.plan)
        )
        # Without a window the block's start is the deadline: the science
        # before it ends by the start and the block begins on time.
        plain = ScheduledActivity(name="Cal", begin=BLOCK_START, duration_seconds=BLOCK)
        on_time = _dispatch([plain])
        (plain_entry,) = _activities(on_time.plan)
        assert float(plain_entry.begin) == BLOCK_START
        assert all(
            float(e.end) <= BLOCK_START or float(e.begin) >= plain.end
            for e in _science(on_time.plan)
        )

    def test_the_window_running_out_cuts_the_observation(self) -> None:
        short = ScheduledActivity(
            name="Cal", begin=BLOCK_START, duration_seconds=BLOCK, slide_seconds=5 * MIN
        )
        ditl = _dispatch([short])
        (entry,) = _activities(ditl.plan)
        assert BLOCK_START < float(entry.begin) <= short.latest_start
        under_way = [
            e
            for e in _science(ditl.plan)
            if float(e.begin) < BLOCK_START < float(e.end)
        ]
        assert len(under_way) == 1 and float(under_way[0].end) <= float(entry.begin)
        assert any("waits for observation" in e.description for e in ditl.log.events)

    def test_a_target_of_opportunity_fits_before_a_sliding_block(self) -> None:
        """A target of opportunity whose slew and collection would cross the
        block's start is observed before the block, which slides after it;
        with no window the block's start is a deadline the response cannot
        meet before it."""
        responses = {}
        for slide in (0.0, 45 * MIN):
            block = ScheduledActivity(
                name="Cal",
                begin=BLOCK_START,
                duration_seconds=BLOCK,
                slide_seconds=slide,
            )
            config = _config(HOURS)
            ditl = create_ditl(
                config, _long_targets(config), BEGIN, END, activities=[block]
            )
            assert isinstance(ditl, QueueDITL)
            ditl.step_size = STEP
            ditl.submit_too(
                obsid=30001,
                name="Urgent",
                ra=PATCH[4][0],
                dec=PATCH[4][1],
                merit=1000.0,
                exptime=15 * MIN,
                submit_time=BLOCK_START - 5 * MIN,
                deadline=BLOCK_START + 10 * MIN,
            )
            assert ditl.calc()
            too = [e for e in ditl.plan if int(e.obsid) == 30001]
            responses[slide] = too[0] if too else None
        assert responses[45 * MIN] is not None
        assert float(responses[45 * MIN].collection_begin) < BLOCK_START + 10 * MIN
        assert (
            responses[0.0] is None or float(responses[0.0].begin) >= BLOCK_START + BLOCK
        )


class TestPlanner:
    def test_science_runs_into_the_window_and_the_block_follows(self) -> None:
        """The planner lets a snapshot run into the block's window and begins
        the block after it, within the window, keeping its length; the plan
        replays without a mismatch."""
        sliding = ScheduledActivity(
            name="Cal",
            begin=BLOCK_START,
            duration_seconds=BLOCK,
            slide_seconds=45 * MIN,
        )
        config = _config(HOURS)
        scheduler = PriorityScheduler(
            config,
            _long_targets(config),
            BEGIN,
            END,
            step_size=STEP,
            activities=[sliding],
        )
        plan = scheduler.schedule()
        (entry,) = _activities(plan)
        # The entry begins where the hold before the block takes over, and
        # ends the block's length after the hold begins, within the window.
        held_from = float(entry.end) - BLOCK
        assert sliding.begin <= held_from <= sliding.latest_start
        before = [e for e in _science(plan) if float(e.end) <= held_from]
        assert before and float(before[-1].end) > BLOCK_START
        for e in _science(plan):
            assert float(e.end) <= held_from or float(e.begin) >= float(entry.end)
        replay = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(plan.model_dump_json()),
            begin=BEGIN,
            end=END,
        )
        replay.step_size = STEP
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []


class TestActivityAfterActivity:
    """A block due while another block is under way begins when that one
    ends and keeps its length, as one delayed by a slew or an observation
    does, in dispatch and in the plans."""

    WINDOW = ScheduledActivity(
        name="Maintenance", begin=BLOCK_START, duration_seconds=HOUR, fixed=True
    )

    def _due_inside_the_window(self, **fields: float) -> ScheduledActivity:
        return ScheduledActivity(
            name="Cal", begin=BLOCK_START + 30 * MIN, duration_seconds=BLOCK, **fields
        )

    def _check(self, plan: Plan | list, second: ScheduledActivity) -> None:
        window, cal = _activities(plan)
        assert float(window.begin) == BLOCK_START
        assert float(window.end) == self.WINDOW.end
        assert float(cal.begin) == self.WINDOW.end
        assert float(cal.end) - float(cal.begin) == pytest.approx(BLOCK)
        assert float(cal.end) > second.end

    def test_dispatch_begins_the_block_when_the_one_under_way_ends(self) -> None:
        cal = self._due_inside_the_window()
        ditl = _dispatch([self.WINDOW, cal])
        self._check(ditl.plan, cal)
        assert any(
            e.description.startswith("Scheduled activity Cal begins, ")
            and e.description.endswith("s after its start")
            for e in ditl.log.events
        )
        assert ditl.validate_plan_matches_execution() == []

    def test_dispatch_keeps_the_length_of_a_sliding_block_too(self) -> None:
        """The window ends after the block's latest start: the block still
        begins when the window ends, for its length."""
        cal = self._due_inside_the_window(slide_seconds=10 * MIN)
        ditl = _dispatch([self.WINDOW, cal])
        self._check(ditl.plan, cal)

    def test_dispatch_runs_a_fixed_block_after_the_one_under_way(self) -> None:
        """A fixed block due during another block cannot begin at its start;
        it begins when the block under way ends, for its length."""
        short = ScheduledActivity(name="Cal", begin=BLOCK_START, duration_seconds=BLOCK)
        window = ScheduledActivity(
            name="Maintenance",
            begin=BLOCK_START + 10 * MIN,
            duration_seconds=HOUR,
            fixed=True,
        )
        ditl = _dispatch([short, window])
        cal, maintenance = _activities(ditl.plan)
        assert float(cal.begin) == BLOCK_START and float(cal.end) == short.end
        assert float(maintenance.begin) == short.end
        assert float(maintenance.end) - float(maintenance.begin) == pytest.approx(HOUR)

    def test_the_planner_reserves_the_block_after_the_one_under_way(self) -> None:
        """The plan puts the block after the window, for its length, and
        says so; the plan replays without a mismatch."""
        cal = self._due_inside_the_window(slide_seconds=10 * MIN)
        config = _config(HOURS)
        scheduler = PriorityScheduler(
            config,
            _long_targets(config),
            BEGIN,
            END,
            step_size=STEP,
            activities=[self.WINDOW, cal],
        )
        plan = scheduler.schedule()
        self._check(plan, cal)
        assert any(
            "Cal is due during another activity" in e.description
            for e in scheduler.log.events
        )
        for e in _science(plan):
            assert (
                float(e.end) <= BLOCK_START or float(e.begin) >= self.WINDOW.end + BLOCK
            )
        replay = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(plan.model_dump_json()),
            begin=BEGIN,
            end=END,
        )
        replay.step_size = STEP
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []

    def test_the_planner_settles_the_block_back_to_its_start(self) -> None:
        """Where the block it waited for slides and ends before the block's
        start, the block begins at its start."""
        sliding = ScheduledActivity(
            name="Cal",
            begin=BLOCK_START,
            duration_seconds=BLOCK,
            slide_seconds=45 * MIN,
        )
        after = ScheduledActivity(
            name="Lamp", begin=BLOCK_START + 30 * MIN, duration_seconds=BLOCK
        )
        config = _config(HOURS)
        scheduler = PriorityScheduler(
            config,
            _long_targets(config),
            BEGIN,
            END,
            step_size=STEP,
            activities=[sliding, after],
        )
        plan = scheduler.schedule()
        cal, lamp = _activities(plan)
        assert float(cal.end) - float(cal.begin) == pytest.approx(BLOCK)
        assert float(lamp.end) - float(lamp.begin) == pytest.approx(BLOCK)
        assert float(lamp.begin) == max(after.begin, float(cal.end))
        assert float(lamp.begin) <= sliding.latest_start + BLOCK


class TestEclipseIntervals:
    def test_an_interval_is_eclipse_everywhere(self) -> None:
        """An interval given as eclipse is one for the constraint, the
        telemetry, the panels, the planners' ledger and the timeline plot,
        whatever the ephemeris says."""
        config = _config(HOURS)
        config.solar_panel = SolarPanelSet(panels=[SolarPanel(normal=(0.0, 0.0, -1.0))])
        start, end = T0 + 1 * HOUR, T0 + HOUR + 20 * MIN
        config.constraint.eclipse_intervals = [
            (
                BEGIN + timedelta(hours=1),
                (BEGIN + timedelta(hours=1, minutes=20)).isoformat(),
            )
        ]
        assert config.constraint.eclipse_intervals_unix() == [(start, end)]
        assert config.constraint.in_eclipse(0.0, 0.0, start + 60.0)
        assert config.constraint.in_eclipse_interval(start) and not (
            config.constraint.in_eclipse_interval(end)
        )
        ditl = create_ditl(config, _long_targets(config), BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        records = ditl.telemetry.housekeeping
        inside = [hk for hk in records if start <= hk.timestamp.timestamp() < end]
        assert inside and all(hk.in_eclipse for hk in inside)
        times = [hk.timestamp.timestamp() for hk in records]
        panel = [ditl.panel[i] for i, t in enumerate(times) if start <= t < end]
        assert panel and max(panel) == 0.0
        assert any(
            ditl.panel[i] > 0.0 for i, t in enumerate(times) if t < start - 10 * MIN
        ) or any(ditl.panel[i] > 0.0 for i, t in enumerate(times) if t > end + 10 * MIN)
        scheduler = PriorityScheduler(
            config, _long_targets(config), BEGIN, END, step_size=STEP
        )
        scheduler.schedule()
        assert scheduler._battery.in_eclipse(start + 60.0)
        segments = _extract_eclipses(ditl, T0, 0.0)
        assert any(
            s <= 1.0 + 1e-6 and s + d >= 1.0 + 20 / 60 - 1e-6 for s, d in segments
        )

    def test_intervals_are_validated(self) -> None:
        with pytest.raises(ValidationError, match="end after it starts"):
            Constraint(eclipse_intervals=[(T0 + HOUR, T0 + HOUR)])
        with pytest.raises(ValidationError, match="pair"):
            Constraint(eclipse_intervals=[(T0,)])  # type: ignore[list-item]
        given = Constraint(eclipse_intervals=[(BEGIN, BEGIN + timedelta(minutes=5))])
        assert given.eclipse_intervals == [(T0, T0 + 5 * MIN)]
