"""End-to-end dispatch behaviour of the dynamic merit model in QueueDITL."""

from collections.abc import Sequence
from datetime import timedelta
from unittest.mock import patch

import pytest
import rust_ephem

from conops import QueueDITL
from conops.common import ObsType
from conops.config import (
    AttitudeControlSystem,
    Battery,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    TargetConfig,
)
from conops.config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)
from scripts.check_default_plan_output import (
    SCENARIO_BEGIN,
    DeterministicConstraint,
    DeterministicEphemeris,
)

BEGIN = SCENARIO_BEGIN.timestamp()
HOUR = 3600


def _run(
    targets: Sequence[dict[str, float | int | str | None]],
    *,
    weights: TargetConfig | None = None,
    categories: Sequence[ObservationCategory] = (),
    toos: Sequence[dict[str, float | int | str | None]] = (),
    hours: int = 4,
) -> QueueDITL:
    """Run QueueDITL over an unconstrained sky with the given requests."""
    end = SCENARIO_BEGIN + timedelta(hours=hours)
    ephem = DeterministicEphemeris(SCENARIO_BEGIN, end)
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1.0, slew_acceleration=0.5, settle_time=10.0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
        targets=weights or TargetConfig(),
        observation_categories=ObservationCategories(categories=list(categories)),
    )
    config.constraint.ephem = ephem
    ditl = QueueDITL(config=config, ephem=ephem, begin=SCENARIO_BEGIN, end=end)
    for target in targets:
        ditl.queue.add(**target)  # type: ignore[arg-type]
    for too in toos:
        ditl.submit_too(**too)  # type: ignore[arg-type]
    provenance = {"ut1": {"available": False}, "polar_motion": {"available": False}}
    with patch.object(rust_ephem, "get_eop_provenance", lambda: provenance):
        assert ditl.calc()
    return ditl


def _science_order(ditl: QueueDITL) -> list[int]:
    return [int(e.obsid) for e in ditl.plan if e.obstype == ObsType.AT]


# The trade study's Figure 1: a flexible high-merit request (A) and a lower-merit
# request that must start within half an hour (B).
FLEXIBLE = {
    "obsid": 10,
    "name": "A",
    "ra": 20.0,
    "dec": 0.0,
    "merit": 100.0,
    "exptime": HOUR,
    "ss_min": HOUR,
    "ss_max": HOUR,
}
SHORT_WINDOW = {
    "obsid": 11,
    "name": "B",
    "ra": 40.0,
    "dec": 0.0,
    "merit": 70.0,
    "exptime": HOUR // 2,
    "ss_min": HOUR // 2,
    "ss_max": HOUR // 2,
    "deadline": BEGIN + HOUR // 2,
}


class TestUrgency:
    def test_priority_order_loses_the_short_window_request(self) -> None:
        ditl = _run([FLEXIBLE, SHORT_WINDOW])

        assert _science_order(ditl) == [10]

    def test_urgency_observes_both(self) -> None:
        ditl = _run([FLEXIBLE, SHORT_WINDOW], weights=TargetConfig(urgency_weight=50.0))

        assert _science_order(ditl) == [11, 10]

    def test_plan_entry_carries_the_frozen_value(self) -> None:
        ditl = _run([FLEXIBLE, SHORT_WINDOW], weights=TargetConfig(urgency_weight=50.0))

        first = ditl.plan[0]
        assert first.obsid == 11
        assert first.merit == pytest.approx(120.0)
        assert any(
            event.description.startswith("Selected 11:")
            and "urgency=+50.000" in event.description
            for event in ditl.log.events
        )

    def test_target_is_not_selected_after_its_deadline(self) -> None:
        late = {**SHORT_WINDOW, "deadline": BEGIN - 1}

        ditl = _run([late])

        assert _science_order(ditl) == []


class TestTier:
    def test_higher_tier_wins_regardless_of_merit(self) -> None:
        categories = [
            ObservationCategory(name="Low", obsid_min=10, obsid_max=11, tier=0),
            ObservationCategory(name="High", obsid_min=11, obsid_max=12, tier=1),
        ]
        no_deadline = {**SHORT_WINDOW, "deadline": None}

        ditl = _run([FLEXIBLE, no_deadline], categories=categories)

        assert _science_order(ditl) == [11, 10]


class TestTargetOfOpportunity:
    def test_too_interrupts_and_is_observed_before_its_deadline(self) -> None:
        submit = BEGIN + HOUR // 4
        resumable = {**FLEXIBLE, "ss_min": 300}
        ditl = _run(
            [resumable],
            toos=[
                {
                    "obsid": 30001,
                    "ra": 60.0,
                    "dec": 10.0,
                    "merit": 1000.0,
                    "exptime": 600,
                    "name": "GRB",
                    "submit_time": submit,
                    "deadline": submit + HOUR // 2,
                }
            ],
        )

        too = ditl.too_register[0]
        too_entry = next(e for e in ditl.plan if e.obsid == 30001)
        assert too.executed
        assert submit <= too_entry.collection_begin <= submit + HOUR // 2
        assert _science_order(ditl) == [10, 30001, 10]
        assert any("TOO interrupt: GRB" in e.description for e in ditl.log.events)

    def test_lower_value_too_waits_for_the_current_observation(self) -> None:
        ditl = _run(
            [FLEXIBLE],
            toos=[
                {
                    "obsid": 30001,
                    "ra": 60.0,
                    "dec": 10.0,
                    "merit": 50.0,
                    "exptime": 600,
                    "name": "Low",
                    "submit_time": BEGIN + HOUR // 4,
                }
            ],
        )

        assert _science_order(ditl) == [10, 30001]
        assert not any("TOO interrupt" in e.description for e in ditl.log.events)

    def test_unschedulable_too_does_not_interrupt(self) -> None:
        resumable = {**FLEXIBLE, "ss_min": 300}
        ditl = _run(
            [resumable],
            toos=[
                {
                    "obsid": 30001,
                    "ra": 60.0,
                    "dec": 10.0,
                    "merit": 1000.0,
                    "exptime": 120,  # below the 300 s minimum snapshot
                    "name": "Short",
                    "submit_time": BEGIN + HOUR // 4,
                }
            ],
        )

        assert _science_order(ditl) == [10]
        assert not any("TOO interrupt" in e.description for e in ditl.log.events)

    @pytest.mark.parametrize("higher_first", [True, False])
    def test_simultaneous_toos_pick_the_higher_value(self, higher_first: bool) -> None:
        submit = BEGIN + HOUR // 4
        lower = {
            "obsid": 30001,
            "ra": 60.0,
            "dec": 10.0,
            "merit": 500.0,
            "exptime": 600,
            "name": "Lower",
            "submit_time": submit,
        }
        higher = {
            "obsid": 30002,
            "ra": 80.0,
            "dec": -10.0,
            "merit": 1000.0,
            "exptime": 600,
            "name": "Higher",
            "submit_time": submit,
            "deadline": submit + 200,
        }
        resumable = {**FLEXIBLE, "ss_min": 300}

        ditl = _run(
            [resumable],
            toos=[higher, lower] if higher_first else [lower, higher],
        )

        higher_entry = next(e for e in ditl.plan if e.obsid == 30002)
        assert higher_entry.collection_begin <= submit + 200
        assert _science_order(ditl)[:2] == [10, 30002]
