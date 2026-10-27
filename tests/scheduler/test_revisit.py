"""A target that has collected anything may finish with a snapshot shorter
than ``ss_min``: an interrupted target is revisited for its remainder, and a
remainder below the minimum is not lost."""

from datetime import timedelta

from conops import QueueDITL, create_ditl
from conops.targets.plan_entry import snapshot_min_seconds

from .planning_scenario import BEGIN, MIN, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 3


class TestSnapshotMinimum:
    def test_relaxes_to_the_remainder_once_anything_is_collected(self) -> None:
        config = _config(HOURS)
        target = _target(config, 1, 105.0, 10.0, minutes=50, snapshot=20)
        assert snapshot_min_seconds(target) == 1200.0
        target.record_collection(T0 + 1200, 1200.0)
        assert snapshot_min_seconds(target) == 1200.0, "a full snapshot remains"
        target.record_collection(T0 + 2400, 1200.0)
        assert snapshot_min_seconds(target) == 600.0, "only the remainder is left"
        assert target.snapshot_min_seconds == 600.0

    def test_a_remainder_shorter_than_the_step_is_rounding(self) -> None:
        """An exposure a second over two snapshots: the second's record ends
        the target at the simulation step, and had it not, the remainder
        would get no snapshot of its own."""
        config = _config(HOURS)
        target = _target(config, 1, 105.0, 10.0, minutes=40, snapshot=20)
        target.exptime = 2401.0
        target.record_collection(T0 + 1200, 1200.0, step=60.0)
        assert target.exptime == 1201.0
        assert not target.done
        target.record_collection(T0 + 2400, 1200.0, step=60.0)
        assert target.exptime == 0
        assert target.done
        leftover = _target(config, 2, 105.0, 10.0, minutes=40, snapshot=20)
        leftover.exptime = 2401.0
        leftover.record_collection(T0 + 2400, 2400.0)
        assert leftover.exptime == 1.0
        assert snapshot_min_seconds(leftover) == 1.0
        assert snapshot_min_seconds(leftover, 60.0) == 1200.0

    def test_an_open_ended_target_keeps_its_minimum(self) -> None:
        config = _config(HOURS)
        target = _target(config, 1, 105.0, 10.0, minutes=50, snapshot=20)
        target.exptime = None  # type: ignore[assignment]
        target.record_collection(T0 + 1200, 1200.0)
        assert snapshot_min_seconds(target) == 1200.0


class TestDispatch:
    def test_a_remainder_below_the_minimum_is_collected(self) -> None:
        """Fifty minutes in twenty-minute snapshots: the last ten minutes used
        to be left uncollected, since no snapshot could be shorter than twenty."""
        config = _config(HOURS)
        target = _target(config, 1, 105.0, 10.0, minutes=50, snapshot=20)
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=HOURS))
        ditl.step_size = 60
        assert isinstance(ditl, QueueDITL)
        assert ditl.calc()

        queued = ditl.queue.targets[0]
        assert queued.collected_seconds == 3000.0
        assert queued.done
        snapshots = [e for e in ditl.plan if e.obstype.name == "AT"]
        assert len(snapshots) == 3

    def test_a_remainder_shorter_than_the_step_ends_the_target(self) -> None:
        """A twenty-minute snapshot of an exposure a second longer: the
        second is rounding, so the target is done after one snapshot and
        is not revisited for it."""
        config = _config(HOURS)
        target = _target(config, 1, 105.0, 10.0, minutes=20, snapshot=20)
        target.exptime = 20 * MIN + 1.0
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=HOURS))
        ditl.step_size = 60
        assert isinstance(ditl, QueueDITL)
        assert ditl.calc()

        queued = ditl.queue.targets[0]
        assert queued.collected_seconds == 1200.0
        assert queued.done
        snapshots = [e for e in ditl.plan if e.obstype.name == "AT"]
        assert len(snapshots) == 1

    def test_an_interrupted_target_is_revisited_for_its_remainder(self) -> None:
        """A single thirty-minute snapshot cut short by a ToO after some
        minutes: the target is revisited for what remains, below its minimum,
        and records the interruption."""
        config = _config(HOURS)
        target = _target(config, 1, 105.0, 10.0, merit=10, minutes=30, snapshot=30)
        ditl = create_ditl(config, [target], BEGIN, BEGIN + timedelta(hours=HOURS))
        ditl.step_size = 60
        assert isinstance(ditl, QueueDITL)
        ditl.submit_too(
            obsid=1_000_001,
            name="GRB",
            ra=115.0,
            dec=10.0,
            merit=500.0,
            exptime=600,
            submit_time=T0 + 15 * MIN,
            deadline=T0 + 40 * MIN,
        )
        assert ditl.calc()

        queued = next(t for t in ditl.queue.targets if int(t.obsid) == 1)
        assert len(queued.interruptions) == 1
        assert T0 < queued.interruptions[0] < T0 + 30 * MIN
        assert queued.collected_seconds == 1800.0
        assert queued.done
        visits = [e for e in ditl.plan if e.obstype.name == "AT" and e.obsid == 1]
        assert len(visits) == 2
        assert ditl.too_register[0].executed
