"""Back-compatible name for :class:`~conops.schedulers.SequentialPlanner`."""

from collections.abc import Iterable
from datetime import timedelta
from typing import TYPE_CHECKING

from ..common import dtutcfromtimestamp
from ..config import MissionConfig
from ..targets import PlanEntry, TargetList
from .sequential_planner import SequentialPlanner

if TYPE_CHECKING:
    from ..ditl.ditl_log import DITLLog


class DumbScheduler(SequentialPlanner):
    """Deprecated name for :class:`~conops.schedulers.SequentialPlanner`.

    The horizon starts at the beginning of the ephemeris and lasts ``days``.
    Targets are added to ``targlist`` after construction, ``mintime`` is the
    old name for ``min_window``, and an observation may run past the horizon
    to the end of the ephemeris. Repeated :meth:`schedule` calls extend the
    same plan. Without a ``log``, events are printed.
    """

    def __init__(
        self, config: MissionConfig, days: int = 1, log: "DITLLog | None" = None
    ) -> None:
        if config is None:
            raise ValueError("Config must be provided to DumbScheduler")
        ephem = self._validated_ephem(config)
        begin = dtutcfromtimestamp(ephem.timestamp[0].timestamp())
        super().__init__(config, (), begin, begin + timedelta(days=days), log=log)
        self.days = days
        self._print_events = log is None
        self.issurvey = False
        self.gimbled = False  # Default: not gimbled
        self.sidemount = False  # Default: not side-mounted

    @property
    def targets(self) -> list[PlanEntry]:
        return list(self.targlist)

    @targets.setter
    def targets(self, value: Iterable[PlanEntry]) -> None:
        self.targlist = TargetList(targets=list(value))

    @property
    def mintime(self) -> float:
        return self.min_window

    @mintime.setter
    def mintime(self, value: float) -> None:
        self.min_window = value

    def _start_run(self) -> None:
        # Runs accumulate: earlier entries stay planned and are not repeated.
        pass

    def _horizon(self) -> tuple[float, float]:
        self.end = self.begin + timedelta(days=self.days)
        return super()._horizon()

    def _latest_end(self, uend: float, ephem_end: float) -> float:
        return ephem_end

    def _log(self, utime: float, description: str) -> None:
        if self._print_events:
            print(description)
        else:
            super()._log(utime, description)
