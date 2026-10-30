import pytest
from reproduce import bisection_probe, booking_probe, outage_probe


def test_separate_contact_spans_preserve_outage_losses():
    simulated, chronological = outage_probe()
    assert simulated == pytest.approx(chronological)


def test_scheduler_finds_a_proven_valid_variable_rate_placement():
    count, valid = bisection_probe()
    assert valid, "Independent candidate scan must establish a feasible placement"
    assert count > 0


def test_replayed_contacts_do_not_run_past_their_booking():
    assert booking_probe() == pytest.approx(0.0)
