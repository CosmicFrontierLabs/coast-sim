import numpy as np

from ..common import unixtime2date
from ..config import AttitudeControlSystem, Constraint
from .plan_entry import PlanEntry


class Pointing(PlanEntry):
    """Define the basic parameters of an observing target with visibility checking."""

    ra: float
    dec: float
    obsid: int
    name: str
    merit: float

    def __init__(
        self,
        constraint: Constraint | None = None,
        acs_config: AttitudeControlSystem | None = None,
        ra: float = 0.0,
        dec: float = 0.0,
        obsid: int = 0,
        name: str = "FakeTarget",
        merit: float = 100.0,
        exptime: int | None = None,
        ss_min: int = 300,
        ss_max: int = 86400,
    ):
        PlanEntry.__init__(self, constraint=constraint, acs_config=acs_config)
        assert self.constraint == constraint, "Constraint not properly set in Pointing"
        self.done = False
        self.obstype = "AT"
        self.isat = False
        self.ra = ra
        self.dec = dec
        self.targetid = obsid
        self.obsid = obsid
        self.name = name
        # ``fom`` is maintained as a legacy alias for ``merit`` for
        # backwards compatibility (e.g. tests and older code). The
        # canonical field we use internally is ``merit`` which can be
        # recomputed each scheduling iteration by ``Queue.meritsort``.
        self.fom = merit
        self.merit = merit
        self._exptime: int | None = exptime
        self._exporig: int | None = exptime
        self._done = False
        # Snapshot min/max size
        self.ss_min = ss_min  # seconds
        self.ss_max = ss_max  # seconds

    def is_visible(self, utime):
        """Is a target visible at this time"""
        return not self.constraint.inoccult(self.ra, self.dec, utime)

    def in_sun(self, utime):
        """Is this target in Sun constraint?"""
        return self.constraint.in_sun(self.ra, self.dec, utime)

    def in_earth(self, utime):
        """Is this target in Earth constraint?"""
        return self.constraint.in_earth(self.ra, self.dec, utime)

    def in_moon(self, utime):
        """Is this target in Moon constraint?"""
        return self.constraint.in_moon(self.ra, self.dec, utime)

    def in_panel(self, utime):
        """Is this target in Panel constraint?"""
        return self.constraint.in_panel(self.ra, self.dec, utime)

    def next_vis(self, utime):
        """When is this target visible next?"""
        # Are we currently in a visibility window, if yes, return back the current time
        if self.visible(utime, utime):
            return utime

        # Are there no visibility windows? Then just return False
        if len(self.windows) == 0:
            return False
        try:
            visstarts = np.array(self.windows).transpose()[0]
            windex = np.where(visstarts - utime > 0)[0][0]
            return visstarts[windex]
        except Exception:
            return False

    def __str__(self):
        return f"{unixtime2date(self.begin)} {self.name} ({self.targetid}) RA={self.ra:.4f}, Dec={self.dec:4f}, Roll={self.roll:.1f}, Merit={self.merit}"

    @property
    def exptime(self) -> int | None:
        return self._exptime

    @exptime.setter
    def exptime(self, t: int):
        if self._exptime is None:
            self._exporig = t
        self._exptime = t

    @property
    def done(self):
        if self.exptime is not None and self.exptime <= 0:
            self._done = True
        return self._done

    @done.setter
    def done(self, v):
        self._done = v

    def reset(self):
        if self._exporig is not None:
            self._exptime = self._exporig
        self.done = False
        self.begin = 0
        self.end = 0
        self.slewtime = 0
