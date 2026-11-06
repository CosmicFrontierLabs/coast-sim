"""Reporting: science by category, the visit table, and the timeline's
activities, repoints and sequence steps."""

from datetime import datetime, timedelta, timezone

import matplotlib
import numpy as np
import pytest
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import QueueDITL, create_ditl
from conops.config import (
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    ObservationCategories,
    ObservationCategory,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.ditl.ditl_stats import CategorySummary, VisitRecord
from conops.targets import Pointing
from conops.targets.sequence import ObservationSequence, SequenceStep
from conops.visualization import plot_ditl_timeline
from conops.visualization.plotly.ditl_timeline import plot_ditl_timeline_plotly

from .planning_scenario import BEGIN, MIN, PATCH
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target
from .test_visit_windows import STARTS, VISIT, WINDOWS, _fillers, _timed_target

matplotlib.use("Agg")

HOURS = 12
STEP = 60
END = BEGIN + timedelta(hours=HOURS)


def _cut_visit_run(hours: int = HOURS) -> tuple[QueueDITL, Pointing]:
    """A windowed target whose first visit a target of opportunity cuts."""
    config = _config(hours)
    target = _timed_target(config)
    ditl = create_ditl(
        config, [target, *_fillers(config)], BEGIN, BEGIN + timedelta(hours=hours)
    )
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    cut_at = STARTS[0] + 6 * MIN
    ditl.submit_too(
        obsid=30001,
        ra=PATCH[0][0] + 2.0,
        dec=PATCH[0][1],
        merit=1000.0,
        exptime=5 * MIN,
        deadline=cut_at + 30 * MIN,
        name="Urgent",
        submit_time=cut_at,
    )
    assert ditl.calc()
    return ditl, target


def _legend(fig_ax) -> list[str]:  # type: ignore[no-untyped-def]
    _, ax = fig_ax
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


def _traces(fig) -> set[str]:  # type: ignore[no-untyped-def]
    return {trace.name for trace in fig.data if trace.showlegend}


class TestCategorySummary:
    def test_science_by_category_with_shares(self) -> None:
        """Categories split the run's science: targets, visits, science and
        slew hours per category, shares summing to one, and the allocated
        fraction where a category sets one. Every configured category has a
        row, in configured order, a category that collected nothing too."""
        ditl, _ = _cut_visit_run()
        ditl.config.observation_categories = ObservationCategories(
            categories=[
                ObservationCategory(
                    name="Monitoring",
                    obsid_min=900,
                    obsid_max=1000,
                    color="tab:blue",
                    program="monitoring",
                    time_share=0.1,
                ),
                ObservationCategory(
                    name="Survey",
                    obsid_min=1,
                    obsid_max=100,
                    color="tab:green",
                    program="survey",
                ),
                ObservationCategory(
                    name="Idle",
                    obsid_min=5000,
                    obsid_max=6000,
                    color="tab:grey",
                    time_share=0.2,
                ),
            ]
        )
        rows = ditl.summary_by_category()
        assert [type(r) for r in rows] == [CategorySummary] * len(rows)
        assert [r.category for r in rows][:3] == ["Monitoring", "Survey", "Idle"]
        by_name = {r.category: r for r in rows}
        idle = by_name["Idle"]
        assert (idle.targets, idle.visits, idle.science_hours, idle.share) == (
            0,
            0,
            0.0,
            0.0,
        )
        assert idle.time_share == 0.2
        monitoring = by_name["Monitoring"]
        assert monitoring.program == "monitoring"
        assert monitoring.targets == 1
        assert monitoring.visits == 4
        assert monitoring.science_hours == pytest.approx((3 * VISIT * MIN + 198) / 3600)
        assert monitoring.time_share == 0.1
        assert by_name["Survey"].time_share is None
        assert by_name["Survey"].targets == 3
        assert sum(r.share for r in rows) == pytest.approx(1.0)
        total = sum(r.science_hours for r in rows)
        assert total == pytest.approx(
            sum(float(e.exposure) for e in ditl.plan if e.obstype.name != "ACTIVITY")
            / 3600
        )
        assert all(r.slew_hours > 0 and r.slew_degrees > 0 for r in rows if r.visits)

    def test_printed_statistics_carry_the_tables(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl, _ = _cut_visit_run()
        ditl.print_statistics()
        out = capsys.readouterr().out
        assert "SCIENCE BY CATEGORY" in out
        assert "VISITS IN WINDOWS" in out
        assert "complete" in out and "Tries" in out
        verdict = next(line for line in out.splitlines() if line.startswith("V900"))
        assert verdict.split()[-2:] == ["complete", "2"]


class TestVisitTable:
    def test_visits_against_their_windows(self) -> None:
        """Each visit of a windowed target: its window, when it collected,
        its science, cut where a target of opportunity cut it, complete
        where whole, and a window with no visit missed."""
        ditl, target = _cut_visit_run(hours=6)
        rows = [r for r in ditl.visit_table() if r.obsid == 900]
        assert [type(r) for r in rows] == [VisitRecord] * len(rows)
        assert [r.visit for r in rows] == [0, 0, 1, 2]
        assert [r.status for r in rows] == ["cut", "complete", "complete", "missed"]
        first, again, second, third = rows
        assert first.window_start == WINDOWS[0].start
        assert first.window_deadline == WINDOWS[0].deadline
        assert first.exposure_seconds == 198.0
        assert again.exposure_seconds == VISIT * MIN
        assert again.collection_begin is not None
        assert WINDOWS[0].start <= again.collection_begin <= WINDOWS[0].deadline
        assert second.category == "Observation"
        assert second.slew_degrees > 0 and second.roll_deg is not None
        assert third.begin is None and third.exposure_seconds == 0.0
        assert third.window_start == WINDOWS[2].start
        # The same table from the requests given explicitly.
        given = [r for r in ditl.visit_table([target]) if r.obsid == 900]
        assert [r.status for r in given] == [r.status for r in rows]
        # Per window, the verdict: the whole visit taken in the first window
        # after two attempts, the second whole in one, the third missed.
        verdicts = [r for r in ditl.visit_table(per_window=True) if r.obsid == 900]
        assert [(r.visit, r.status, r.attempts) for r in verdicts] == [
            (0, "complete", 2),
            (1, "complete", 1),
            (2, "missed", 0),
        ]
        assert verdicts[0].exposure_seconds == VISIT * MIN
        # Targets without windows have rows too, one per snapshot, complete.
        fillers = [r for r in ditl.visit_table() if r.obsid in (1, 2, 3)]
        assert fillers and all(r.window_start is None for r in fillers)
        assert all(r.status == "complete" for r in fillers)


class TestTimeline:
    def test_dumps_and_sequence_steps_and_repoints_are_drawn(self) -> None:
        """Both timelines show momentum dumps on their own row, sequence
        steps hatched over the observation, and antenna repoints marked."""
        ephem = TLEEphemeris(
            tle="examples/example.tle",
            begin=datetime(2025, 11, 1, tzinfo=timezone.utc),
            end=datetime(2025, 11, 1, 4, tzinfo=timezone.utc),
            step_size=10,
        )
        config = MissionConfig(
            constraint=Constraint(
                sun_constraint=SunConstraint(min_angle=45),
                earth_constraint=EarthLimbConstraint(min_angle=10),
            ),
            ground_stations=GroundStationRegistry(stations=[]),
            solar_panel=SolarPanelSet(panels=[]),
            battery=Battery(watthour=100_000.0),
            spacecraft_bus=SpacecraftBus(
                inertia_tensor_body_kg_m2=np.diag([1000.0, 1200.0, 800.0]),
                attitude_control=AttitudeControlSystem(
                    max_slew_rate=0.5,
                    slew_acceleration=0.25,
                    settle_time=30.0,
                    stored_momentum=StoredMomentumConfig(
                        gravity_gradient_enabled=True,
                        capacity_n_m_s=1.0,
                        planning_level=0.6,
                        thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
                    ),
                ),
                star_trackers=StarTrackerConfiguration(
                    star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
                ),
                radiators=RadiatorConfiguration(radiators=[]),
            ),
        )
        config.constraint.ephem = ephem
        sequence = Pointing(
            config=config,
            obsid=2,
            name="Visit",
            ra=80.0,
            dec=-20.0,
            merit=100.0,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="first", duration_seconds=20 * MIN),
                    SequenceStep(name="second", duration_seconds=20 * MIN),
                ]
            ),
            visits=4,
        )
        ditl = create_ditl(config, [sequence], ephem.begin, ephem.end)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 10
        assert ditl.calc()
        assert any(e.desaturation for e in ditl.plan)
        assert any(e.step is not None for e in ditl.plan)
        # A sequence visit's steps, begun a simulation step apart, are one
        # visit: as many as the log says began.
        begun = sum(
            "Sequence visit of Visit begins" in e.description for e in ditl.log.events
        )
        (row,) = (r for r in ditl.summary_by_category() if r.visits)
        assert begun >= 2 and row.visits == begun
        assert len([r for r in ditl.visit_table() if r.obsid == 2]) == begun
        ditl.repoints_in_observations = [
            (ephem.begin.timestamp() + 1800.0, 2, "reported")
        ]
        labels = _legend(plot_ditl_timeline(ditl))
        assert {"Momentum Dump", "Sequence Step", "Antenna Repoint"} <= set(labels)
        assert "Activity" not in labels
        traces = _traces(plot_ditl_timeline_plotly(ditl))
        assert {"Momentum Dump", "Sequence Step", "Antenna Repoint"} <= traces

    def test_a_run_without_them_draws_as_before(self) -> None:
        config = _config(6)
        ditl = create_ditl(
            config,
            [_target(config, 1, *PATCH[0], minutes=60, snapshot=20)],
            BEGIN,
            BEGIN + timedelta(hours=6),
        )
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        labels = _legend(plot_ditl_timeline(ditl))
        assert not {
            "Momentum Dump",
            "Activity",
            "Sequence Step",
            "Antenna Repoint",
        } & set(labels)
        assert not {
            "Momentum Dump",
            "Activity",
            "Sequence Step",
            "Antenna Repoint",
        } & _traces(plot_ditl_timeline_plotly(ditl))
