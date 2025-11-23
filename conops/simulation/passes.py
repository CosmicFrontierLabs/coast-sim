import time
from typing import Any

import numpy as np
import rust_ephem  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from ..common import ics_date_conv, roll_over_angle, unixtime2date
from ..config import Config, Constraint, GroundStationRegistry
from .slew import Slew


class Pass(BaseModel):
    """A groundstation pass consisting of a pre-pass slew (composition) and the dwell phase.

    Composition replaces inheritance from `Slew` for clearer semantics: a pass *contains* a
    slew rather than *is* a slew. The helper `pre_slew` computes the maneuver required before
    the contact begins.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Core dependencies
    ephem: rust_ephem.TLEEphemeris | None = None
    constraint: Constraint | None = None
    acs_config: Any | None = None  # AttitudeControlSystem, avoiding circular import
    pre_slew: Slew | None = None

    # Pass metadata
    station: str
    begin: float
    length: float | None = None

    # What type of observation is this, a Ground Station Pass (GSP)
    obstype: str = "GSP"

    # Ground station pointing vectors (start/end of contact)
    gsstartra: float = 0.0
    gsstartdec: float = 0.0
    gsendra: float = 0.0
    gsenddec: float = 0.0

    # Recorded pointing profile during the pass dwell
    utime: list[float] = Field(default_factory=list)
    ra: list[float] = Field(default_factory=list)
    dec: list[float] = Field(default_factory=list)

    # Scheduling / status
    slewrequired: float = 0.0
    slewlate: float = 0.0
    possible: bool = True
    obsid: int = 0xFFFF

    # Internal caching for slew prediction
    _cached_slew_inputs: tuple[float, float, float, float] | None = None
    _slew_grace: float | None = None  # computed from ephemeris step size

    def model_post_init(self, __context) -> None:
        """Initialize dependent objects after Pydantic validation."""
        if self.constraint is not None and self.pre_slew is None:
            # Create the pre-pass slew with provided acs_config
            self.pre_slew = Slew(self.constraint, acs_config=self.acs_config)
        if self.pre_slew is not None and self.ephem is None:
            self.ephem = self.pre_slew.ephem

    # Expose selected slew attributes via properties for external code expecting them
    @property
    def startra(self) -> float | None:  # current pointing at slew start
        if self.pre_slew is None:
            return None
        return self.pre_slew.startra

    @startra.setter
    def startra(self, value: float):
        if self.pre_slew is None:
            return
        self.pre_slew.startra = value

    @property
    def startdec(self):
        return self.pre_slew.startdec

    @startdec.setter
    def startdec(self, value: float):
        if self.pre_slew is None:
            return
        self.pre_slew.startdec = value

    @property
    def endra(self) -> float:  # end of slew == start of pass
        return self.gsstartra

    @property
    def enddec(self) -> float:
        return self.gsstartdec

    @property
    def slewtime(self) -> float | None:
        if self.pre_slew is None:
            return None
        return self.pre_slew.slewtime

    @property
    def slewstart(self) -> float | None:
        if self.pre_slew is None:
            return None
        return self.pre_slew.slewstart

    @slewstart.setter
    def slewstart(self, value: float):
        if self.pre_slew is None:
            return
        self.pre_slew.slewstart = value

    @property
    def slewdist(self) -> float:
        if self.pre_slew is None:
            return 0.0
        return self.pre_slew.slewdist

    @property
    def slewpath(self) -> np.ndarray | None:
        if self.pre_slew is None:
            return None
        return self.pre_slew.slewpath

    @property
    def slewsecs(self) -> np.ndarray:
        return getattr(self.pre_slew, "slewsecs", np.array([]))

    @property
    def end(self) -> float:
        assert self.length is not None, "Pass length must be set"
        return self.begin + self.length

    def __str__(self):
        """Return string of details on the pass"""
        return f"{unixtime2date(self.begin):18s}  {self.station:3s}  {self.length / 60.0:4.1f} mins"  #  {self.time_to_pass():12s}"

    @property
    def slewend(self) -> float | None:
        """End time of the slew maneuver preceding the pass."""
        if self.slewstart is None or self.slewtime is None:
            return None
        return self.slewstart + self.slewtime

    def is_slewing(self, utime: float) -> bool:
        """Return True if we are in the pre-pass slew phase at utime."""
        if self.slewstart is None or self.slewend is None:
            return False
        return self.slewstart <= utime < self.slewend

    def in_pass(self, utime: float) -> bool:
        if utime >= self.begin and utime <= self.end and self.possible:
            return True
        else:
            return False

    def time_to_pass(self) -> str:
        """Return a string for how long it is until the next pass"""
        now = time.time()
        timetopass = (self.begin - now) / 60.0

        if timetopass < 60.0:
            timetopassstring = "%.0f mins " % timetopass
        else:
            timetopassstring = "%.1f hours" % (timetopass / 60.0)

        return timetopassstring

    def ra_dec(self, utime: float) -> tuple[float | None, float | None]:
        if utime < self.begin:
            if self.pre_slew is None:
                return None, None
            return self.pre_slew.slew_ra_dec(utime)
        else:
            return self.pass_ra_dec(utime)

    def pass_ra_dec(self, utime: float) -> tuple[float | None, float | None]:
        """Return the RA/Dec of the Spacecraft during a groundstation pass"""
        if self.in_pass(utime):
            ras = roll_over_angle(self.ra)  # Deal with RA roll-overs, 360 -> 0
            ra = (
                np.interp(utime, self.utime, ras) % 360
            )  # so "interp" doesn't do some crazy thing
            dec = np.interp(utime, self.utime, self.dec)
        else:
            # print("PASS OVER _ GET OVER IT")
            ra = np.interp(utime, self.utime, self.ra)
            dec = np.interp(utime, self.utime, self.dec)
        return ra, dec

    def time_to_slew(self, utime: float) -> bool:
        """Determine whether to begin slewing for this pass.

        Returns True when slew should start now, False otherwise. Applies a small grace
        window (ephemeris step size) allowing a slightly late start without abandoning
        the pass. Abandons only if lateness exceeds the grace.
        """
        if not self.possible:
            return False

        # Ensure pointing inputs are sensible (avoid computing slew from zeros unless intentional)
        if (
            (self.startra == 0 or self.startra is None)
            and (self.startdec == 0 or self.startdec is None)
            and (self.endra != 0 or self.enddec != 0)
        ):
            # Current pointing unknown -> cannot decide yet
            return False

        # Update end target of slew (start-of-pass pointing)
        assert self.pre_slew is not None, "Pre-slew must be initialized"
        self.pre_slew.endra = self.endra
        self.pre_slew.enddec = self.enddec

        # asserts for mypy mostly
        assert self.startra is not None, "Start ra must be set"
        assert self.startdec is not None, "Start dec must be set"
        assert self.begin is not None, "Pass begin time must be set"
        assert self.endra is not None, "End ra must be set"
        assert self.enddec is not None, "End dec must be set"
        assert self.slewtime is not None, "Slew time must be set"

        # Cache & predict only when inputs change
        inputs = (self.startra, self.startdec, self.endra, self.enddec)
        if inputs != self._cached_slew_inputs:
            self.pre_slew.calc_slewtime()
            self._cached_slew_inputs = inputs
            # Recompute grace window (one ephemeris time step)
            if self.ephem is not None:
                # Compute step_size from timestamp difference
                self._slew_grace = float(
                    self.ephem.timestamp[1].timestamp()
                    - self.ephem.timestamp[0].timestamp()
                )
            else:
                # Fallback to 0 to preserve previous strict behavior
                self._slew_grace = 0.0

        self.slewrequired = self.begin - self.slewtime
        now_delta = utime - self.slewrequired

        # Start when within one step before required time
        if now_delta < -(self._slew_grace or 0):
            # Too early
            return False

        # Lateness evaluation
        if now_delta > 0:
            self.slewlate = now_delta
            if self._slew_grace is not None and self.slewlate > self._slew_grace:
                # Excessive lateness: abort pass
                self.possible = False
                print(
                    "%s: Slew start late by %.1fs (> %.1fs grace). Abandoning pass %s.",
                    unixtime2date(utime),
                    self.slewlate,
                    self._slew_grace,
                    self.station,
                )
                return False
            else:
                print(
                    "%s: Slew starting late by %.1fs (grace %.1fs).",
                    unixtime2date(utime),
                    self.slewlate,
                    self._slew_grace,
                )
        else:
            # Early or exactly on time
            self.slewlate = now_delta  # negative or zero
            if now_delta < 0:
                print(
                    "%s: Slew starting early by %.1fs.",
                    unixtime2date(utime),
                    -now_delta,
                )
            else:
                print("%s: Slew starting exactly on time.", unixtime2date(utime))

        # Record slew start
        self.slewstart = utime
        return True


# Spacecraft Groundstations
# Ghana (5.74 deg N, 0.30 deg W) and
# the backup is
# Mwulire, Rwanda (1.96 deg S, 30.39 deg E).


class PassTimes:
    """Like the swift PassTimes class, except we calculate our passes based on epheris and groundstation location"""

    passes: list[Pass]
    constraint: Constraint
    ephem: rust_ephem.TLEEphemeris
    ground_stations: GroundStationRegistry
    config: Config

    _current_pass: Pass | None

    def __init__(
        self,
        constraint: Constraint,
        config: Config,
    ):
        self.constraint = constraint
        assert self.constraint.ephem is not None, "Ephemeris must be set for Pass class"
        self.ephem = self.constraint.ephem

        self.config = config
        self.passes = []
        self.length = 1

        # Ground stations registry from config
        if config.ground_stations is None:
            self.ground_stations = GroundStationRegistry.default()
        else:
            self.ground_stations = config.ground_stations

        # What makes a good pass
        self.minelev = 10.0
        self.minlen = 8 * 60  # 10 mins
        self.schedule_chance = 1.0  # base chance of getting a pass

    def __getitem__(self, number: int) -> Pass:
        return self.passes[number]

    def __len__(self) -> int:
        return len(self.passes)

    def next_pass(self, utime: float) -> Pass | None:
        for gspass in self.passes:
            if utime < gspass.begin:
                return gspass
        return None

    def request_passes(self, req_gsnum: int, gsprob: float = 0.9) -> list[Pass]:
        """Request passes at a particular rate, including random probability of scheduling"""
        mean_between = 86400 / req_gsnum
        sched = list()
        last = 0.0
        for gspass in self.passes:
            if gspass.begin - last > mean_between and np.random.random() < gsprob:
                sched.extend([gspass])
                last = sched[-1].begin
        return sched

    def get(self, year: int, day: int, length: int = 1) -> None:
        """Calculate the passes using rust_ephem GroundEphemeris for vectorized operations."""
        ustart = ics_date_conv(f"{year}-{day:03d}-00:00:00")
        assert self.ephem is not None, "Ephemeris is not set"

        # Use binary search instead of np.where for finding start index
        # Prefer adapter datetimes if available, otherwise use Time.unix
        timestamp_unix = np.array([dt.timestamp() for dt in self.ephem.timestamp])
        startindex = np.searchsorted(timestamp_unix, ustart)

        # Calculate end index
        num_steps = int(86400 * length / self.ephem.step_size)
        endindex = min(startindex + num_steps, len(timestamp_unix))

        # Get spacecraft positions for the time range (vectorized)
        sc_pos_all = self.ephem.gcrs_pv.position[startindex:endindex]  # Shape: (N, 3)

        # Get timestamps as datetime objects for GroundEphemeris
        timestamps = self.ephem.timestamp[startindex:endindex]
        begin_time = timestamps[0]
        end_time = timestamps[-1]

        # Process each ground station
        for station in self.ground_stations:
            # Create GroundEphemeris for this station (vectorized ground station ephemeris)

            gs_ephem = rust_ephem.GroundEphemeris(
                latitude=station.latitude_deg,
                longitude=station.longitude_deg,
                height=station.elevation_m,
                begin=begin_time,
                end=end_time,
                step_size=self.ephem.step_size,
            )

            # Get ground station positions in GCRS (vectorized)
            gs_pos_all = gs_ephem.gcrs_pv.position  # Shape: (N, 3)

            # Compute vectors from GS to SC in GCRS (vectorized)
            gs_to_sc_gcrs = sc_pos_all - gs_pos_all  # Shape: (N, 3)

            # For accurate elevation angles, we need to work in ITRS (Earth-fixed) frame
            # and convert to topocentric East-North-Up coordinates
            sc_itrs_all = self.ephem.itrs_pv.position[startindex:endindex]
            gs_itrs_all = gs_ephem.itrs_pv.position

            # Vector from GS to SC in ITRS frame
            gs_to_sc_itrs = sc_itrs_all - gs_itrs_all  # Shape: (N, 3)

            # Rotation matrix from ITRS to topocentric ENU (East-North-Up) frame
            lat_rad = np.radians(station.latitude_deg)
            lon_rad = np.radians(station.longitude_deg)

            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            sin_lon = np.sin(lon_rad)
            cos_lon = np.cos(lon_rad)

            # ENU rotation matrix (constant for this ground station)
            r_enu = np.array(
                [
                    [-sin_lon, cos_lon, 0],
                    [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                    [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
                ]
            )

            # Transform all GS-to-SC vectors to ENU frame (vectorized matrix multiplication)
            gs_to_sc_enu = (r_enu @ gs_to_sc_itrs.T).T  # Shape: (N, 3)

            # In ENU frame: East=x, North=y, Up=z
            # Elevation = arctan(Up / sqrt(East^2 + North^2))
            horizontal_dist = np.sqrt(gs_to_sc_enu[:, 0] ** 2 + gs_to_sc_enu[:, 1] ** 2)
            elevations = np.degrees(np.arctan2(gs_to_sc_enu[:, 2], horizontal_dist))

            # Compute RA/Dec for SC-to-GS pointing (vectorized)
            # We want the vector from SC to GS (opposite direction) in GCRS
            sc_to_gs = -gs_to_sc_gcrs
            ra_all = np.arctan2(sc_to_gs[:, 1], sc_to_gs[:, 0])
            dec_all = np.arcsin(sc_to_gs[:, 2] / np.linalg.norm(sc_to_gs, axis=1))
            ra_all_deg = np.degrees(ra_all)
            dec_all_deg = np.degrees(dec_all)

            # Find passes using elevation threshold
            min_elev = (
                station.min_elevation_deg
                if hasattr(station, "min_elevation_deg")
                else self.minelev
            )

            # Detect pass boundaries
            above_threshold = elevations > min_elev
            pass_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
            pass_ends = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1

            # Handle edge cases
            if above_threshold[0]:
                pass_starts = np.concatenate([[0], pass_starts])
            if above_threshold[-1]:
                pass_ends = np.concatenate([pass_ends, [len(above_threshold)]])

            # Create Pass objects
            for start_idx, end_idx in zip(pass_starts, pass_ends):
                global_start_idx = startindex + start_idx
                global_end_idx = (
                    startindex + end_idx
                )  # end_idx is already the first point below threshold

                passstart = timestamp_unix[global_start_idx]
                passend = timestamp_unix[global_end_idx]
                passlen = passend - passstart

                if passlen >= self.minlen:
                    # Combine global and station-specific schedule probabilities
                    combined_prob = self.schedule_chance * getattr(
                        station, "schedule_probability", 1.0
                    )
                    rand_val = np.random.random()
                    if rand_val <= combined_prob:
                        gspass = Pass(
                            constraint=self.constraint,
                            ephem=self.ephem,
                            acs_config=self.config.spacecraft_bus.attitude_control,
                            station=station.code,
                            begin=passstart,
                            gsstartra=ra_all_deg[start_idx],
                            gsstartdec=dec_all_deg[start_idx],
                            gsendra=ra_all_deg[end_idx - 1],
                            gsenddec=dec_all_deg[end_idx - 1],
                            length=passlen,
                        )

                        # Record the path during the pass
                        for i in range(start_idx, end_idx):
                            gspass.utime.append(timestamp_unix[startindex + i])
                            gspass.ra.append(ra_all_deg[i])
                            gspass.dec.append(dec_all_deg[i])

                        self.passes.append(gspass)

        # Order the passes by time
        self.passes.sort(key=lambda x: x.begin, reverse=False)

    def get_current_pass(self) -> Pass | None:
        """Get the current active pass being tracked."""
        # Handle legacy PassTimes objects that don't have _current_pass
        if not hasattr(self, "_current_pass"):
            self._current_pass = None
        return self._current_pass

    def check_pass_timing(
        self, utime: float, current_ra: float, current_dec: float, step_size: float
    ) -> dict[str, Any]:
        """Check pass timing and return actions needed.

        Returns dict with:
            - 'start_pass': Pass object if a pass should start now
            - 'end_pass': True if current pass has ended
            - 'updated_pass': Pass object with updated timing
        """
        # Ensure _current_pass exists (for legacy objects)
        if not hasattr(self, "_current_pass"):
            self._current_pass = None

        result: dict[str, Any] = {
            "start_pass": None,
            "end_pass": False,
            "updated_pass": None,
        }

        # Check if current pass has ended
        if self._current_pass is not None and utime > self._current_pass.end:
            result["end_pass"] = True
            self._current_pass = None
            return result

        # Look for next pass if none is scheduled
        if self._current_pass is None:
            next_pass = self.next_pass(utime)
            if next_pass is not None:
                self._current_pass = next_pass

        # Update pass timing and check if it should start
        if self._current_pass is not None:
            # Update pass timing with current spacecraft position
            if (
                current_ra != 0 or current_dec != 0
            ) and self._current_pass.slewtime is not None:
                self._current_pass.startra = current_ra
                self._current_pass.startdec = current_dec

                self._current_pass.slewrequired = (
                    self._current_pass.begin - self._current_pass.slewtime - step_size
                )

            result["updated_pass"] = self._current_pass

            # Check if it's time to start the pass
            time_to_pass = self._current_pass.slewrequired - utime
            if 0 < time_to_pass <= step_size:
                result["start_pass"] = self._current_pass

        return result
