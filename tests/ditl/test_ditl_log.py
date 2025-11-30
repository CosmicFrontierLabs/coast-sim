"""Tests for DITLLog container behavior and optional persistence."""

from conops.common import ACSMode
from conops.ditl import DITLLog, DITLLogStore


def test_ditl_log_basic_methods():
    log = DITLLog()
    assert len(log) == 0
    log.log_event(utime=1000.0, event_type="INFO", description="Hello")
    assert len(log) == 1
    e = log[0]
    assert e.description == "Hello"
    # iter works
    assert [ev.event_type for ev in log] == ["INFO"]
    # clear
    log.clear()
    assert len(log) == 0


def test_ditl_log_with_store_persists_events(tmp_path):
    # set up store
    store = DITLLogStore(tmp_path / "logs.sqlite")
    log = DITLLog(run_id="run-x", store=store)
    # add few events
    log.log_event(1000.0, "PASS", "Start pass", obsid=1, acs_mode=ACSMode.PASS)
    log.log_event(1010.0, "SLEW", "Slewing", obsid=None, acs_mode=ACSMode.SLEWING)
    # fetch from store
    evs = store.fetch_events("run-x")
    assert len(evs) == 2
    assert evs[0].event_type == "PASS"
    assert evs[1].event_type == "SLEW"
    # flush also works and is idempotent-ish (may add duplicates)
    log.flush_to_store()
    store.close()
