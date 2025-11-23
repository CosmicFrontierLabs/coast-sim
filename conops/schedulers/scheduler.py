from typing import cast

import numpy as np
import rust_ephem

from ..common import dtutcfromtimestamp
from ..config import Config, Constraint
from ..simulation.saa import SAA
from ..targets import Plan, PlanEntry, TargetList


class DumbScheduler:
    """A simple (dumb) scheduler for spacecraft observations.

    The scheduler iterates through times in the Ephemeris and finds targets that
    satisfy sun/anti-sun constraints and exposure time windows.
    """

    def __init__(self, constraint: Constraint, days: int = 1) -> None:
        if constraint is None:
            raise ValueError("Constraint must be provided to DumbScheduler")

        self.mintime = 5 * 60  # seconds (5 minutes)
        # self.maxtime = 240*60 # example maximum (commented out in original)
        self.constraint = constraint
        self.ephem: rust_ephem.TLEEphemeris = self.constraint.ephem  # type: ignore[assignment]
        if self.ephem is None:
            raise ValueError("Constraint.ephem must be set")

        self.ppst: Plan = Plan()
        self.scheduled: list[int] = []
        self.days = days
        self.saa: SAA | None = None  # will be created lazily
        self.targlist: TargetList = TargetList()
        self.gimbled = False
        self.sidemount = False
        # Get sun/anti-sun constraints from Constraint class
        self.suncons = cast(
            rust_ephem.SunConstraint, self.constraint.sun_constraint
        ).min_angle
        self.antisuncons = cast(
            rust_ephem.SunConstraint, self.constraint.anti_sun_constraint
        ).max_angle
        self.step_size = 60
        self.issurvey = False
        self.config: Config | None = None  # optional: can be set externally

    def _init_saa(self) -> None:
        if self.saa is None:
            self.saa = SAA()
            self.saa.ephem = self.ephem
            self.saa.calc()

    def _get_default_slew(self) -> int:
        # Default slew time for the first observation in a plan (seconds)
        return 180

    def schedule(self) -> None:
        """Main scheduling loop."""
        # Ensure SAA object and its passages are computed
        self._init_saa()

        # starting time index
        i = 0
        ephem_utime = [dt.timestamp() for dt in self.ephem.timestamp]
        end_limit = ephem_utime[0] + 86400 * self.days

        while ephem_utime[i] < end_limit:
            found = False
            selected_target: PlanEntry | None = None
            selected_obslen = 0
            selected_slewtime = 0

            # Candidate targets (exptime > 0)
            candidates = [t for t in self.targlist if t.exptime > 0]

            # iterate until we find one suitable candidate at this time index
            for task in candidates:
                self.current_time = ephem_utime[i]

                # Determine slew time based on prior plan entry (if any)
                if self.ppst and len(self.ppst) > 0:
                    try:
                        last_entry = self.ppst[-1]
                        task.calc_slewtime(last_entry.ra, last_entry.dec)
                        slewtime = task.slewtime
                    except Exception:
                        # fallback to default if last entry isn't usable
                        slewtime = self._get_default_slew()
                else:
                    slewtime = self._get_default_slew()

                # Check constraints for the observation window
                obs_start = self.current_time
                obs_end = self.current_time + task.exptime + slewtime

                # Get ephemeris time indices for observation window
                begin_idx = self.ephem.index(dtutcfromtimestamp(obs_start))
                end_idx = self.ephem.index(dtutcfromtimestamp(obs_end)) + 1

                # Evaluate constraints at each timestep in the observation window
                utime_window = self.ephem.timestamp[begin_idx:end_idx]
                in_occult = self.constraint.inoccult(
                    ra=task.ra,
                    dec=task.dec,
                    utime=utime_window,
                )

                # goodtime = 1 where constraints are satisfied (NOT in occult)
                goodtime = np.bitwise_not(in_occult).astype(int).tolist()

                # compute contiguous available observation length from start
                obslen = -self.step_size
                for k in range(len(goodtime)):
                    if goodtime[k] == 1:
                        obslen += self.step_size
                    else:
                        break

                # Check observation length relative to min time + slewtime and requested exposure
                if (
                    obslen >= (self.mintime + slewtime)
                    and obslen <= (task.exptime + slewtime)
                    and task.targetid not in self.scheduled
                ):
                    found = True
                    selected_target = task
                    selected_obslen = obslen
                    selected_slewtime = slewtime
                    break

            if not found:
                print(
                    "WARNING: No target found at time index %s (utime=%s); stopping scheduling",
                    i,
                    ephem_utime[i],
                )
                break

            # Create and populate the plan entry
            assert selected_target is not None
            ppt = PlanEntry(
                constraint=self.constraint,
                acs_config=(
                    self.config.spacecraft_bus.attitude_control
                    if hasattr(self, "config") and self.config is not None
                    else None
                ),
            )

            ppt.suncons = self.suncons
            ppt.ephem = self.ephem
            # keep entry angles in units that other code expects (note: original used target.ra)
            ppt.ra = selected_target.ra
            ppt.dec = selected_target.dec
            ppt.begin = ephem_utime[i]  # numeric start time
            ppt.slewtime = selected_slewtime

            # End time as begin + selected available window length (obslen includes slewtime)
            endtime = ppt.begin + selected_obslen
            if endtime > ephem_utime[-1]:
                endtime = ephem_utime[-1]

            ppt.end = endtime

            # Compute actual exposure assigned to this plan entry (seconds)
            # exposure is time after slew and before end
            exposure_time = max(0, endtime - ppt.begin - ppt.slewtime)
            # Do not exceed the target's remaining requested exposure
            exposure_time = min(exposure_time, selected_target.exptime)  # type: ignore[attr-defined]
            # assign to the PlanEntry if the attribute exists; fallback to attribute assignment
            try:
                ppt.exposure = exposure_time
            except Exception:
                setattr(ppt, "exposure", exposure_time)

            # Update the target's remaining requested exposure
            selected_target.exptime -= exposure_time  # type: ignore[attr-defined]

            ppt.obsid = selected_target.targetid
            assert self.saa is not None
            ppt.saa = self.saa
            ppt.sidemount = self.sidemount
            ppt.gimbled = self.gimbled
            ppt.merit = selected_target.merit
            ppt.name = selected_target.name

            self.scheduled.append(selected_target.targetid)
            self.ppst.extend([ppt])

            # Move to next index for scheduling after this observation
            i = self.ephem.index(dtutcfromtimestamp(ppt.end))

        print("Scheduled %d targets", len(self.ppst))
