"""Additional tests to increase coverage for DITLLogStore."""

import tempfile
from pathlib import Path

from conops.common import ACSMode
from conops.ditl import DITLEvent, DITLLogStore


def test_add_events_batch_and_filters():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "batch.sqlite"
        store = DITLLogStore(db)
        run_id = "batch-run"
        events = [
            DITLEvent.from_utime(990.0, "INFO", "pre", None, None),
            DITLEvent.from_utime(1000.0, "PASS", "start", 1, ACSMode.PASS),
            DITLEvent.from_utime(1020.0, "OBSERVATION", "obs", 2, ACSMode.SCIENCE),
        ]
        store.add_events(run_id, events)

        # end_time filter
        early = store.fetch_events(run_id, end_time=1005.0)
        assert len(early) == 2
        assert [e.event_type for e in early] == ["INFO", "PASS"]

        # Specific type filter
        only_pass = store.fetch_events(run_id, event_type="PASS")
        assert len(only_pass) == 1
        assert only_pass[0].description == "start"

        # Empty runs listing after different run added
        store.add_event("other-run", DITLEvent.from_utime(1000.0, "INFO", "x"))
        runs = set(store.fetch_runs())
        assert "batch-run" in runs and "other-run" in runs

        # Close is safe to call multiple times
        store.close()
        store.close()
