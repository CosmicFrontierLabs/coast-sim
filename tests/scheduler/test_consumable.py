"""A consumable's level over a horizon: accrual, removals and the limit."""

import pytest

from conops.schedulers.consumable import ConsumableLedger, Removal


def _ledger(removals, *, limit=80.0, start=0.0, rate=0.001):
    """A horizon of 1000 s accruing ``rate`` per second outside removals."""
    return ConsumableLedger(
        0.0, 1000.0, removals, limit=limit, start=start, accrual=lambda s: rate * s
    )


class TestSegments:
    def test_removals_cut_the_horizon_into_segments(self) -> None:
        ledger = _ledger([Removal(600.0, 700.0, 30.0), Removal(200.0, 300.0, 20.0)])
        assert ledger.segments == 3
        assert ledger.bounds == [0.0, 200.0, 600.0, 1000.0]
        assert ledger.removal == [0.0, 20.0, 30.0]
        times = (0, 199, 200, 599, 600, 1000)
        assert [ledger.segment(t) for t in times] == [0, 0, 1, 1, 2, 2]

    def test_accrual_excludes_the_removals(self) -> None:
        ledger = _ledger([Removal(200.0, 300.0, 20.0, accrual=0.3)])
        assert ledger.accrual == pytest.approx([0.2, 0.7])
        assert ledger.removal_accrual == pytest.approx([0.0, 0.3])

    def test_removals_outside_the_horizon_are_ignored(self) -> None:
        ledger = ConsumableLedger(
            100.0,
            900.0,
            [Removal(950.0, 1000.0, 1.0), Removal(0.0, 100.0, 1.0)],
            limit=None,
            start=0.0,
            accrual=lambda s: 0.0,
        )
        assert ledger.segments == 1

    def test_a_removal_straddling_the_horizon_is_scaled_to_the_part_inside(
        self,
    ) -> None:
        ledger = ConsumableLedger(
            250.0,
            900.0,
            [Removal(200.0, 300.0, 20.0, accrual=0.3)],
            limit=None,
            start=0.0,
            accrual=lambda s: 0.0,
        )
        assert ledger.bounds == [250.0, 250.0, 900.0]
        assert ledger.removal == pytest.approx([0.0, 10.0])
        assert ledger.removal_accrual == pytest.approx([0.0, 0.15])


class TestLevels:
    def test_what_a_removal_cannot_take_off_carries_over(self) -> None:
        ledger = _ledger(
            [Removal(200.0, 300.0, 20.0, 0.3), Removal(600.0, 700.0, 30.0, 0.3)],
            start=5.0,
        )
        # Segment 0: 5 + 0.2 + 40 = 45.2; the removal leaves 45.2 + 0.3 - 20.
        # Segment 1: 25.5 + 0.3 + 10 = 35.8; the removal leaves 6.1.
        # Segment 2: 6.1 + 0.3 = 6.4.
        assert ledger.levels((40.0, 10.0, 0.0)) == pytest.approx([45.2, 35.8, 6.4])

    def test_a_removal_takes_no_more_than_is_there(self) -> None:
        ledger = _ledger([Removal(200.0, 300.0, 50.0)])
        assert ledger.levels((1.0, 0.0)) == pytest.approx([1.2, 0.7])

    def test_fits_checks_every_later_level(self) -> None:
        ledger = _ledger([Removal(200.0, 300.0, 20.0), Removal(600.0, 700.0, 1.0)])
        planned = (0.0, 0.0, 70.0)
        # 30 in segment 0 leaves 10.2 after the first removal, 9.5 after the
        # second, and the last segment at 79.8 of 80; 31 puts it over, though
        # both are well within the limit where they are added.
        assert ledger.fits(planned, 0, 30.0)
        assert not ledger.fits(planned, 0, 31.0)
        assert not ledger.fits(planned, 0, 85.0)
        assert ledger.add(planned, 0, 10.0) == (10.0, 0.0, 70.0)

    def test_fits_keeps_the_level_below_the_limit_not_at_it(self) -> None:
        """The limit is where an alert fires, so reaching it is not within it."""
        ledger = _ledger([], rate=0.0)
        assert ledger.fits((0.0,), 0, 79.999)
        assert not ledger.fits((0.0,), 0, 80.0)

    def test_a_removals_lead_in_accrues_at_its_own_rate(self) -> None:
        """The 50 s before the removal accrue 1.0 instead of the plain rate,
        in the segment before it; clipped to the horizon like the removal, and
        counted even when the removal itself is past the horizon's end."""
        lead = Removal(200.0, 300.0, 20.0, lead=50.0, lead_accrual=1.0)
        ledger = _ledger([lead])
        assert ledger.bounds == [0.0, 200.0, 1000.0]
        assert ledger.accrual == pytest.approx([0.15 + 1.0, 0.7])
        clipped = ConsumableLedger(
            180.0, 1000.0, [lead], limit=80.0, start=0.0, accrual=lambda s: 0.001 * s
        )
        assert clipped.accrual == pytest.approx([0.4, 0.7])
        after = ConsumableLedger(
            0.0,
            1000.0,
            [Removal(1020.0, 1100.0, 20.0, lead=50.0, lead_accrual=1.0)],
            limit=80.0,
            start=0.0,
            accrual=lambda s: 0.001 * s,
        )
        assert after.segments == 1
        assert after.accrual == pytest.approx([0.97 + 0.6])

    def test_no_limit_fits_anything(self) -> None:
        ledger = _ledger([], limit=None)
        assert not ledger.active
        assert ledger.fits((0.0,), 0, 1e9)
