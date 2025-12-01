"""Tests for DITLLogStore persistence and retrieval."""

import tempfile
from pathlib import Path

from conops.common import ACSMode
from conops.ditl import DITLEvent, DITLLogStore


def test_log_store_add_and_fetch_events():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "logs.sqlite"
        store = DITLLogStore(db)
        run_id = "run-1"
        # Add a couple of events
        ev1 = DITLEvent.from_utime(
            utime=1000.0,
            event_type="PASS",
            description="Starting pass",
            obsid=123,
            acs_mode=ACSMode.PASS,
        )
        ev2 = DITLEvent.from_utime(
            utime=1010.0,
            event_type="OBSERVATION",
            description="Begin observation",
            obsid=456,
            acs_mode=ACSMode.SCIENCE,
        )
        store.add_event(run_id, ev1)
        store.add_event(run_id, ev2)

        events = store.fetch_events(run_id)
        assert len(events) == 2
        assert events[0].event_type == "PASS"
        assert events[1].event_type == "OBSERVATION"
        # filter by type
        obs = store.fetch_events(run_id, event_type="OBSERVATION")
        assert len(obs) == 1
        assert obs[0].description == "Begin observation"
        # filter by time
        later = store.fetch_events(run_id, start_time=1005.0)
        assert len(later) == 1
        assert later[0].time == 1010.0
        # runs listing
        runs = store.fetch_runs()
        assert run_id in runs
        store.close()
