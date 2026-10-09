from datetime import timedelta

import pytest

from conops import DITL
from conops.common import ACSMode
from conops.config import (
    AttitudeControlSystem,
    DataGeneration,
    GroundStationRegistry,
    Instrument,
    MissionConfig,
    ObservationTiming,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.targets import Plan, PlanEntry
from scripts import check_default_plan_output as scenario
from scripts.check_default_plan_output import (
    DEFAULT_BASELINE,
    DEFAULT_NUMERIC_ABS_TOL,
    build_default_plan_payload,
    compare_to_baseline,
)


def test_default_plan_output_matches_baseline() -> None:
    actual = build_default_plan_payload()
    diffs = compare_to_baseline(
        actual, DEFAULT_BASELINE, abs_tol=DEFAULT_NUMERIC_ABS_TOL
    )
    assert not diffs, "\n".join(diffs[:25])


@pytest.mark.parametrize("budgets", [(0, 0, 0), (8.25, 1.5, 2.5)])
def test_budgeted_plan_matches_fractional_collection_and_data(monkeypatch, budgets):
    constructor = scenario.QueueDITL
    simulations = []

    def timed_simulation(**kwargs):
        payload = kwargs["config"].payload
        payload.observation_timing = ObservationTiming(
            setup_seconds=budgets[0],
            cleanup_seconds=budgets[1],
            handoff_seconds=budgets[2],
        )
        payload.instruments = [
            Instrument(data_generation=DataGeneration(rate_gbps=0.001))
        ]
        ditl = constructor(**kwargs)
        simulations.append(ditl)
        return ditl

    monkeypatch.setattr(scenario, "QueueDITL", timed_simulation)
    scenario.build_default_plan_payload()
    ditl = simulations[0]
    science = [entry for entry in ditl.plan if entry.collection_begin is not None]
    planned_collection = sum(
        entry.collection_end - entry.collection_begin for entry in science
    )
    assert science
    assert not ditl.validate_plan_matches_execution()
    assert all(
        entry.collection_begin == entry.begin + entry.slewtime + budgets[0]
        for entry in science
    )
    assert all(
        entry.collection_end == entry.end - sum(budgets[1:]) for entry in science
    )
    assert sum(
        hk.collection_seconds for hk in ditl.telemetry.housekeeping
    ) == pytest.approx(planned_collection)
    assert ditl.data_generated_gb[-1] == pytest.approx(planned_collection * 0.001)


def test_replay_with_real_acs_does_not_collect_during_a_longer_than_planned_slew():
    config = MissionConfig(
        constraint=scenario.DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=2, slew_acceleration=0.125, settle_time=36
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.payload.instruments = [
        Instrument(data_generation=DataGeneration(rate_gbps=0.01))
    ]
    begin = scenario.SCENARIO_BEGIN
    end = begin + timedelta(seconds=120)
    ephem = scenario.DeterministicEphemeris(begin, end, step_size_seconds=60)
    config.constraint.ephem = ephem
    entry = PlanEntry(
        begin=begin.timestamp(),
        end=end.timestamp(),
        slewtime=50,
        ra=180,
        dec=0,
        roll=0,
        obsid=7,
    )
    ditl = DITL(
        config=config, ephem=ephem, plan=Plan(entries=[entry]), begin=begin, end=end
    )

    assert ditl.calc()
    assert 120 < ditl.acs.last_slew.slewtime <= 143
    assert all(h.acs_mode == ACSMode.SLEWING for h in ditl.telemetry.housekeeping)
    assert all(h.collection_seconds == 0 for h in ditl.telemetry.housekeeping)
    assert ditl.data_generated_gb[-1] == 0


def test_replay_with_real_acs_cancels_a_slew_deferred_to_the_entry_end(monkeypatch):
    """A slew deferred to exactly its entry's end must not start."""
    config = MissionConfig(
        constraint=scenario.DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        spacecraft_bus=SpacecraftBus(
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    begin = scenario.SCENARIO_BEGIN
    entry_end = begin + timedelta(seconds=60)
    end = begin + timedelta(seconds=180)
    ephem = scenario.DeterministicEphemeris(begin, end, step_size_seconds=60)
    config.constraint.ephem = ephem
    entry = PlanEntry(
        begin=begin.timestamp(),
        end=entry_end.timestamp(),
        ra=180,
        dec=0,
        roll=0,
        obsid=7,
    )
    # The target first becomes visible as the entry ends, so ACS defers the
    # entry's slew to the step at which the entry expires.
    monkeypatch.setattr(PlanEntry, "next_vis", lambda self, utime: entry.end)
    ditl = DITL(
        config=config, ephem=ephem, plan=Plan(entries=[entry]), begin=begin, end=end
    )

    assert ditl.calc()
    assert ditl.acs.command_queue == []
    assert all(c.slew is None or c.slew.obsid != 7 for c in ditl.acs.executed_commands)
    assert all(h.acs_mode != ACSMode.SLEWING for h in ditl.telemetry.housekeeping)


def test_queue_plan_replays_identically_through_plan_driven_ditl(monkeypatch):
    """A serialized QueueDITL plan, executed by DITL, must fly the same timeline."""
    constructor = scenario.QueueDITL
    simulations = []

    def recorded_simulation(**kwargs):
        ditl = constructor(**kwargs)
        simulations.append(ditl)
        return ditl

    monkeypatch.setattr(scenario, "QueueDITL", recorded_simulation)
    scenario.build_default_plan_payload()
    queue = simulations[0]
    plan = Plan.model_validate_json(queue.plan.model_dump_json())
    assert len(plan) > 1

    replay = DITL(
        config=queue.config,
        ephem=queue.ephem,
        plan=plan,
        begin=queue.begin,
        end=queue.end,
    )
    replay.step_size = queue.step_size
    monkeypatch.setattr(
        scenario.rust_ephem,
        "get_eop_provenance",
        lambda: {"ut1": {"available": False}, "polar_motion": {"available": False}},
    )
    assert replay.calc()

    assert replay.utime == queue.utime
    assert [int(m) for m in replay.mode] == [int(m) for m in queue.mode]
    assert [int(o) for o in replay.obsid] == [int(o) for o in queue.obsid]
    assert replay.ra == pytest.approx(queue.ra)
    assert replay.dec == pytest.approx(queue.dec)
    assert replay.roll == pytest.approx(queue.roll)
    assert [hk.collection_seconds for hk in replay.telemetry.housekeeping] == [
        hk.collection_seconds for hk in queue.telemetry.housekeeping
    ]
    assert replay.validate_plan_matches_execution() == []
