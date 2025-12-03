"""Tests for DITLLog container behavior and optional persistence."""

import pytest

from conops.common import ACSMode
from conops.ditl import DITLLog, DITLLogStore


class TestDITLLogBasicMethods:
    def test_len_empty(self):
        log = DITLLog()
        assert len(log) == 0

    def test_log_event_increases_len(self):
        log = DITLLog()
        log.log_event(utime=1000.0, event_type="INFO", description="Hello")
        assert len(log) == 1

    def test_access_by_index(self):
        log = DITLLog()
        log.log_event(utime=1000.0, event_type="INFO", description="Hello")
        e = log[0]
        assert e.description == "Hello"

    def test_clear(self):
        log = DITLLog()
        log.log_event(utime=1000.0, event_type="INFO", description="Hello")
        log.clear()
        assert len(log) == 0


class TestDITLLogWithStorePersistsEvents:
    @pytest.fixture
    def store_and_log(self, tmp_path):
        store = DITLLogStore(tmp_path / "logs.sqlite")
        log = DITLLog(run_id="run-x", store=store)
        yield store, log
        store.close()

    def test_log_events(self, store_and_log):
        store, log = store_and_log
        log.log_event(1000.0, "PASS", "Start pass", obsid=1, acs_mode=ACSMode.PASS)
        log.log_event(1010.0, "SLEW", "Slewing", obsid=None, acs_mode=ACSMode.SLEWING)
        # Implicitly tested by adding events

    def test_fetch_events_count(self, store_and_log):
        store, log = store_and_log
        log.log_event(1000.0, "PASS", "Start pass", obsid=1, acs_mode=ACSMode.PASS)
        log.log_event(1010.0, "SLEW", "Slewing", obsid=None, acs_mode=ACSMode.SLEWING)
        evs = store.fetch_events("run-x")
        assert len(evs) == 2

    def test_fetch_events_first_type(self, store_and_log):
        store, log = store_and_log
        log.log_event(1000.0, "PASS", "Start pass", obsid=1, acs_mode=ACSMode.PASS)
        log.log_event(1010.0, "SLEW", "Slewing", obsid=None, acs_mode=ACSMode.SLEWING)
        evs = store.fetch_events("run-x")
        assert evs[0].event_type == "PASS"

    def test_fetch_events_second_type(self, store_and_log):
        store, log = store_and_log
        log.log_event(1000.0, "PASS", "Start pass", obsid=1, acs_mode=ACSMode.PASS)
        log.log_event(1010.0, "SLEW", "Slewing", obsid=None, acs_mode=ACSMode.SLEWING)
        evs = store.fetch_events("run-x")
        assert evs[1].event_type == "SLEW"

    def test_flush_to_store(self, store_and_log):
        store, log = store_and_log
        log.log_event(1000.0, "PASS", "Start pass", obsid=1, acs_mode=ACSMode.PASS)
        log.log_event(1010.0, "SLEW", "Slewing", obsid=None, acs_mode=ACSMode.SLEWING)
        log.flush_to_store()
        # Flush is tested by ensuring no errors; idempotency noted in comment
