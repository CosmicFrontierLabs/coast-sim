"""End-to-end dispatch behaviour of the dynamic merit model in QueueDITL."""

from collections.abc import Callable, Sequence
from datetime import timedelta
from unittest.mock import patch

import numpy as np
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
from conops.simulation import SAA
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
    prepare: Callable[[QueueDITL], None] | None = None,
    step_size: int = 300,
    settle_time: float = 10.0,
) -> QueueDITL:
    """Run QueueDITL over an unconstrained sky with the given requests."""
    end = SCENARIO_BEGIN + timedelta(hours=hours)
    ephem = DeterministicEphemeris(SCENARIO_BEGIN, end, step_size_seconds=step_size)
    config = MissionConfig(
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1.0, slew_acceleration=0.5, settle_time=settle_time
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
    if prepare is not None:
        prepare(ditl)
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

    def test_target_is_not_selected_before_its_earliest_start(self) -> None:
        later = {**FLEXIBLE, "earliest_start": BEGIN + HOUR}

        ditl = _run([later])

        (entry,) = (e for e in ditl.plan if e.obstype == ObsType.AT)
        assert entry.collection_begin >= BEGIN + HOUR


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


class TestCadenceDispatch:
    def test_cadence_credits_the_end_of_the_collected_interval(self) -> None:
        """A visit ending mid-step must not look a full step older than it is."""
        visit = {
            "ra": 20.0,
            "dec": 0.0,
            "exptime": 900,
            "ss_min": 300,
            "ss_max": 300,
        }
        ditl = _run(
            [
                {**visit, "obsid": 10, "name": "A", "merit": 100.0},
                {**visit, "obsid": 11, "name": "B", "merit": 99.0},
            ],
            weights=TargetConfig(cadence_weight=50.0),
            categories=[
                ObservationCategory(
                    name="Monitor", obsid_min=10, obsid_max=12, cadence_seconds=60
                )
            ],
            hours=1,
            step_size=60,
        )

        assert _science_order(ditl)[:6] == [10, 11, 10, 11, 10, 11]


def _saa_from_start(saa_end: float) -> Callable[[QueueDITL], None]:
    """Attach an SAA interval that runs from the start of the run to ``saa_end``."""

    def attach(ditl: QueueDITL) -> None:
        saa = SAA()
        saa.ephem = ditl.ephem
        saa.saatimes = np.array([[BEGIN, BEGIN + saa_end]])
        saa.calculated = True
        ditl.acs.saa = saa

    return attach


DEADLINE_TARGET = {
    "obsid": 10,
    "name": "A",
    "ra": 20.0,
    "dec": 0.0,
    "merit": 100.0,
    "exptime": 600,
    "ss_min": 100,
    "ss_max": 600,
}


class TestDeadlineScienceAvailability:
    def test_target_is_skipped_when_saa_delays_science_past_its_deadline(
        self,
    ) -> None:
        """The slew fits the deadline, but science only resumes after SAA exit.

        A 25 s settle ends the slew exactly on the +120 s tick, where the
        spacecraft is at rest inside the SAA and collects nothing until +600 s.
        """
        ditl = _run(
            [{**DEADLINE_TARGET, "deadline": BEGIN + 180}],
            hours=1,
            prepare=_saa_from_start(599),
            step_size=60,
            settle_time=25.0,
        )

        assert _science_order(ditl) == []
        assert any(
            "collection would start" in event.description for event in ditl.log.events
        )

    @pytest.mark.parametrize("saa_end", [60, 90])
    def test_saa_exit_before_mid_step_slew_completion_keeps_the_target(
        self, saa_end: int
    ) -> None:
        """The tick before slew completion is SLEWING, not SAA, so science
        starts mid-step when the slew ends (+105 s), inside the +110 s deadline."""
        ditl = _run(
            [{**DEADLINE_TARGET, "deadline": BEGIN + 110}],
            hours=1,
            prepare=_saa_from_start(saa_end),
            step_size=60,
        )

        assert _science_order(ditl) == [10]
        entry = ditl.plan[0]
        assert entry.collection_begin == pytest.approx(BEGIN + 105)
        first = next(
            hk for hk in ditl.telemetry.housekeeping if hk.collection_seconds > 0
        )
        assert first.timestamp.timestamp() == BEGIN + 60
        assert first.collection_seconds == pytest.approx(15)
        assert ditl.validate_plan_matches_execution() == []


# A long, lower-merit request (C) and a higher-merit request (D) whose window
# opens an hour in and allows ten minutes to start.
LONG = {
    "obsid": 12,
    "name": "C",
    "ra": 20.0,
    "dec": 0.0,
    "merit": 60.0,
    "exptime": 3 * HOUR,
    "ss_min": 600,
    "ss_max": 3 * HOUR,
}
TIMED = {
    "obsid": 13,
    "name": "D",
    "ra": 40.0,
    "dec": 0.0,
    "merit": 90.0,
    "exptime": HOUR // 2,
    "ss_min": HOUR // 2,
    "ss_max": HOUR // 2,
    "earliest_start": BEGIN + HOUR,
    "deadline": BEGIN + HOUR + 600,
}


class TestLookAhead:
    def test_a_long_observation_ends_when_a_higher_merit_window_opens(self) -> None:
        """The timed request's ten minutes of slack are spent on the long
        observation, less a step, a slew and the setup; the request then
        starts within its window."""
        ditl = _run([LONG, TIMED], hours=5)

        assert _science_order(ditl) == [12, 13, 12]
        first, timed = ditl.plan[0], next(e for e in ditl.plan if e.obsid == 13)
        assert first.obsid == 12
        allowance = ditl.queue._slew_allowance()
        assert allowance is not None
        setup = ditl.queue.observation_timing.setup_seconds
        assert BEGIN + HOUR < first.end <= BEGIN + HOUR + 600 - allowance - setup
        assert BEGIN + HOUR <= timed.collection_begin <= BEGIN + HOUR + 600
        assert any(
            e.description.startswith("Snapshot of 12 ends at")
            and e.description.endswith("when request 13 must start")
            for e in ditl.log.events
        )

    def test_a_request_with_slack_lets_the_observation_run_to_its_latest_start(
        self,
    ) -> None:
        """With two hours of slack, the long observation runs on past the
        opening and ends at the request's latest start, less a slew and the
        setup; the request then starts by its deadline."""
        slack = {**TIMED, "deadline": BEGIN + HOUR + 2 * HOUR}

        ditl = _run([LONG, slack], hours=6)

        assert _science_order(ditl) == [12, 13, 12]
        first, timed = ditl.plan[0], next(e for e in ditl.plan if e.obsid == 13)
        assert first.obsid == 12
        allowance = ditl.queue._slew_allowance()
        assert allowance is not None
        setup = ditl.queue.observation_timing.setup_seconds
        latest = BEGIN + 3 * HOUR - allowance - setup
        assert first.end > BEGIN + 2 * HOUR
        assert first.end <= latest
        assert timed.collection_begin <= BEGIN + 3 * HOUR
        # Asked to keep its opening, the request starts there as before.
        kept = {**slack, "protect_opening": True}
        ditl = _run([LONG, kept], hours=6)
        assert _science_order(ditl) == [12, 13, 12]
        assert ditl.plan[0].end <= BEGIN + HOUR
        timed = next(e for e in ditl.plan if e.obsid == 13)
        assert timed.collection_begin <= BEGIN + HOUR + 600

    def test_a_request_without_a_deadline_does_not_shorten_the_observation(
        self,
    ) -> None:
        """Unbounded slack: the request waits for the long observation."""
        open_ended = {**TIMED, "deadline": None}

        ditl = _run([LONG, open_ended], hours=5)

        assert _science_order(ditl) == [12, 13]
        assert ditl.plan[0].end >= BEGIN + 3 * HOUR

    def test_a_lower_merit_window_does_not_shorten_the_observation(self) -> None:
        """Ranked below the long request, the timed one waits for it and
        misses its window."""
        lower = {**TIMED, "merit": 30.0}

        ditl = _run([LONG, lower], hours=5)

        assert _science_order(ditl) == [12]
        assert ditl.plan[0].end >= BEGIN + 3 * HOUR

    def test_a_shorter_candidate_is_preferred_when_the_minimum_does_not_fit(
        self,
    ) -> None:
        """The long request needs an hour at least, which does not fit before
        the window opens half an hour in; a ten-minute request does."""
        hour_min = {**LONG, "ss_min": HOUR}
        soon = {
            **TIMED,
            "earliest_start": BEGIN + HOUR // 2,
            "deadline": BEGIN + HOUR // 2 + 600,
        }
        short = {
            "obsid": 14,
            "name": "E",
            "ra": 30.0,
            "dec": 0.0,
            "merit": 40.0,
            "exptime": 600,
            "ss_min": 600,
            "ss_max": 600,
        }

        ditl = _run([hour_min, soon, short], hours=5)

        assert _science_order(ditl) == [14, 13, 12]
        assert any(
            "passed over: their minimum snapshot does not fit before request 13 must start"
            in e.description
            for e in ditl.log.events
        )
