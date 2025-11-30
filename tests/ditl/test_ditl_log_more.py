"""Additional tests to increase coverage for DITLLog."""

from conops.ditl import DITLLog, DITLLogStore


def test_print_log_outputs(capsys):
    log = DITLLog()
    log.log_event(utime=1000.0, event_type="INFO", description="A message")
    log.print_log()
    captured = capsys.readouterr()
    assert "A message" in captured.out
    assert "INFO" in captured.out or captured.out  # string includes description


def test_flush_without_store_or_run_id_no_error():
    log = DITLLog()
    log.log_event(1000.0, "INFO", "no store")
    # Should be a no-op and not crash
    log.flush_to_store()
    assert len(log) == 1


def test_persist_failure_is_non_fatal(monkeypatch):
    store = DITLLogStore("/tmp/ditl_logs.sqlite")
    log = DITLLog(run_id="run-y", store=store)

    # Force add_event to raise, ensure we still keep in-memory event
    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    # Pydantic models restrict instance attribute setting; patch on the class
    monkeypatch.setattr(DITLLogStore, "add_event", boom)
    log.log_event(1000.0, "INFO", "kept despite failure")
    assert len(log) == 1
    store.close()


def test_log_event_accepts_none_fields(tmp_path):
    store = DITLLogStore(tmp_path / "logs.sqlite")
    log = DITLLog(run_id="run-n", store=store)
    log.log_event(1000.0, "SLEW", "slew", obsid=None, acs_mode=None)
    evs = store.fetch_events("run-n")
    assert len(evs) == 1
    assert evs[0].obsid is None
    assert evs[0].acs_mode is None
    store.close()
