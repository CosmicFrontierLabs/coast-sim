from typing import Literal

import numpy as np

from ..common import unixtime2date
from .plan_entry import PlanEntry


class Pointing(PlanEntry):
    """Define the basic parameters of an observing target with visibility checking."""

    model_config = PlanEntry.model_config

    # Additional fields for Pointing
    isat: bool = False
    obstype: str = "AT"
    done: bool = False

    def in_sun(self, utime: float) -> bool:
        """Is this target in Sun constraint?"""
        in_sun = self.config.constraint.in_sun(self.ra, self.dec, utime)
        return in_sun

    def in_earth(self, utime: float) -> bool:
        """Is this target in Earth constraint?"""
        return self.config.constraint.in_earth(self.ra, self.dec, utime)

    def in_moon(self, utime: float) -> bool:
        """Is this target in Moon constraint?"""
        return self.config.constraint.in_moon(self.ra, self.dec, utime)

    def in_panel(self, utime: float) -> bool:
        """Is this target in Panel constraint?"""
        return self.config.constraint.in_panel(self.ra, self.dec, utime)

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

    def reset(self) -> None:
        if self.exporig is not None:
            self.exptime = self.exporig
        self.done = False
        self.begin = 0
        self.end = 0
        self.slewtime = 0
