from typing import Literal

import numpy as np
from pydantic import Field, PrivateAttr, computed_field

from ..common import unixtime2date
from ..common.enums import ObsType
from .plan_entry import PlanEntry


class Pointing(PlanEntry):
    """Define the basic parameters of an observing target with visibility checking."""

    obsid: int = 0
    name: str = "FakeTarget"
    merit: float = 100.0
    isat: bool = False
    # ``fom`` is maintained as a legacy alias for ``merit`` for
    # backwards compatibility (e.g. tests and older code). The
    # canonical field we use internally is ``merit`` which can be
    # recomputed each scheduling iteration by ``Queue.meritsort``.
    fom: float = Field(default=100.0, exclude=True)
    obstype: ObsType = ObsType.AT
    roll: float = 0.0
    _done: bool = PrivateAttr(default=False)

    def in_sun(self, utime: float) -> bool:
        """Is this target in Sun constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_sun(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_earth(self, utime: float) -> bool:
        """Is this target in Earth constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_earth(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_moon(self, utime: float) -> bool:
        """Is this target in Moon constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_moon(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_panel(self, utime: float) -> bool:
        """Is this target in Panel constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_panel(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_orbit(self, utime: float) -> bool:
        """Is this target in Orbit constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_orbit(
            self.ra, self.dec, utime, target_roll=self.roll
        )

    def in_star_tracker_hard(self, utime: float, acs_mode: int | None = None) -> bool:
        """Is this target in star tracker hard constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_star_tracker_hard(
            self.ra, self.dec, utime, target_roll=self.roll, acs_mode=acs_mode
        )

    def in_star_tracker_soft(self, utime: float, acs_mode: int | None = None) -> bool:
        """Is this target in star tracker soft constraint?"""
        assert self.config is not None, "Config must be set to evaluate constraints"
        return self.config.constraint.in_star_tracker_soft(
            self.ra, self.dec, utime, target_roll=self.roll, acs_mode=acs_mode
        )

    def next_vis(self, utime: float) -> float | Literal[False]:
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
            return float(visstarts[windex])
        except Exception:
            return False

    def __str__(self) -> str:
        return f"{unixtime2date(self.begin)} {self.name} ({self.obsid}) RA={self.ra:.4f}, Dec={self.dec:4f}, Roll={self.roll:.1f}, Merit={self.merit}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def done(self) -> bool:
        if self.exptime is not None and self.exptime <= 0:
            self._done = True
        return self._done

    @done.setter
    def done(self, v: bool) -> None:
        self._done = v

    def reset(self) -> None:
        if self._exporig is not None:
            self._exptime = self._exporig
        self.done = False
        self.begin = 0
        self.end = 0
        self.slewtime = 0
