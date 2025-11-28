import time
from typing import Any

import numpy as np
import rust_ephem
from pydantic import BaseModel, Field

from conops.common.vector import vec2radec

from ..common import ics_date_conv, unixtime2date
from ..config import Config, Constraint, GroundStationRegistry


class Pass(BaseModel):
    """A groundstation pass consisting of the dwell phase.

    The pass contains pointing profile information for ground station contacts.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Core dependencies
    ephem: rust_ephem.TLEEphemeris | None = None
    constraint: Constraint | None = None
    acs_config: Any | None = None  # AttitudeControlSystem, avoiding circular import

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
    obsid: int = 0xFFFF

    @property
    def end(self) -> float:
        assert self.length is not None, "Pass length must be set"
        return self.begin + self.length

    def __str__(self):
        """Return string of details on the pass"""
        return f"{unixtime2date(self.begin):18s}  {self.station:3s}  {self.length / 60.0:4.1f} mins"  #  {self.time_to_pass():12s}"

    def in_pass(self, utime: float) -> bool:
        if utime >= self.begin and utime <= self.end:
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
        """Return the RA/Dec of the Spacecraft during a groundstation pass.
        Note: If utime is outside the pass, returns the earliest or latest
        RA/Dec in the pass."""

        # Find the closest time index
        idx = np.searchsorted(self.utime, utime)
        return self.ra[idx - 1], self.dec[idx - 1]

    def time_to_slew(self, utime: float, ra: float, dec: float) -> bool:
        """Determine whether to begin slewing for this pass.

        Calculates the slew time between current RA/Dec and the appropriate pointing of the pass.
        If on time, slews to pass start. If late, slews to where the pass currently is.
        Returns True when the pass time minus slew time is less than 60 seconds away.
        """
        assert self.ephem is not None, "Ephemeris must be set for Pass class"

        # Determine target pointing: if we're late, target where pass currently is
        if utime >= self.begin:
            # We're late - target current pass position
            target_ra, target_dec = self.ra_dec(utime)
            if target_ra is None or target_dec is None:
                return False
        else:
            # On time - target pass start
            if not self.ra or not self.dec:
                return False
            target_ra, target_dec = self.ra[0], self.dec[0]

        # Using ACS config, calculate slew time
        if self.acs_config is not None:
            slewdist, _ = self.acs_config.predict_slew(
                startra=ra,
                startdec=dec,
                endra=target_ra,
                enddec=target_dec,
            )
            slewtime = self.acs_config.slew_time(slewdist)
        else:
            raise ValueError("ACS config must be set to calculate slew time")
        # Determine if we need to start slewing now
        time_until_slew = (self.begin - slewtime) - utime

        # If we are within 2 ephem steps of needing to slew, return True
        # FIXME: Is this buffer necessary?
        if time_until_slew <= self.ephem.step_size * 2:
            return True
        else:
            return False


class PassTimes:
    """Like the swift PassTimes class, except we calculate our passes based on epheris and groundstation location"""

    passes: list[Pass]
    constraint: Constraint
    ephem: rust_ephem.TLEEphemeris
    ground_stations: GroundStationRegistry
    config: Config

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

    def current_pass(self, utime: float) -> Pass | None:
        """Get the current active pass being tracked."""
        for gspass in self.passes:
            if gspass.in_pass(utime):
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

            # Calculate satellite RA/Dec as seen from ground station
            sat_ra, sat_dec = np.degrees(
                vec2radec(
                    (
                        self.ephem.gcrs_pv.position[startindex:endindex]
                        - gs_ephem.gcrs_pv.position
                    ).T
                )
            )

            # Validate RA values are within [0, 360)
            if np.any(sat_ra < 0) or np.any(sat_ra >= 360):
                raise ValueError("Calculated satellite RA out of bounds [0, 360)")

            # Fast vectorized approach: compute Earth limb constraint directly
            # The Earth limb constraint checks if the angle from the observer to the target
            # passes through Earth (i.e., target is below the horizon + min_angle)

            # Get ground station position in GCRS
            gs_pos = gs_ephem.gcrs_pv.position  # Shape: (N, 3)

            # Vector from ground station to satellite (target direction)
            gs_to_sat = (
                self.ephem.gcrs_pv.position[startindex:endindex] - gs_pos
            )  # Shape: (N, 3)

            # Normalize to get unit vector toward satellite
            gs_to_sat_dist = np.linalg.norm(gs_to_sat, axis=1, keepdims=True)
            gs_to_sat_unit = gs_to_sat / gs_to_sat_dist

            # Vector from Earth center to ground station
            earth_to_gs = gs_pos  # GCRS origin is Earth center
            earth_to_gs_dist = np.linalg.norm(earth_to_gs, axis=1, keepdims=True)
            earth_to_gs_unit = earth_to_gs / earth_to_gs_dist

            # Angle between "up" (away from Earth center) and target direction
            # cos(angle) = dot(earth_to_gs_unit, gs_to_sat_unit)
            cos_angle = np.sum(earth_to_gs_unit * gs_to_sat_unit, axis=1)

            # Calculate elevation above local horizon
            elevation_angle = np.degrees(np.arcsin(cos_angle))

            # Find passes using elevation threshold
            min_elev = (
                station.min_elevation_deg
                if hasattr(station, "min_elevation_deg")
                else self.minelev
            )

            # Target is visible if elevation > min_elev
            is_visible = elevation_angle > min_elev

            # Find pass boundaries by detecting transitions
            transitions = np.diff(is_visible.astype(int))
            pass_starts = np.where(transitions == 1)[0] + 1
            pass_ends = np.where(transitions == -1)[0] + 1

            # Handle edge cases
            if is_visible[0]:
                pass_starts = np.concatenate([[0], pass_starts])
            if is_visible[-1]:
                pass_ends = np.concatenate([pass_ends, [len(is_visible)]])

            # Create Pass objects
            for start_idx, end_idx in zip(pass_starts, pass_ends):
                global_start_idx = startindex + start_idx
                global_end_idx = (
                    startindex + end_idx
                )  # end_idx is already the first point below threshold

                passstart = timestamp_unix[global_start_idx]
                passend = timestamp_unix[global_end_idx]
                passlen = passend - passstart

                # Only consider passes that meet minimum length
                if passlen >= self.minlen:
                    # Combine global and station-specific schedule probabilities
                    combined_prob = self.schedule_chance * getattr(
                        station, "schedule_probability", 1.0
                    )
                    rand_val = np.random.random()
                    if rand_val <= combined_prob:
                        # Schedule this pass if the dice roll allows it
                        gspass = Pass(
                            constraint=self.constraint,
                            ephem=self.ephem,
                            acs_config=self.config.spacecraft_bus.attitude_control,
                            station=station.code,
                            begin=passstart,
                            gsstartra=sat_ra[start_idx],
                            gsstartdec=sat_dec[start_idx],
                            gsendra=sat_ra[end_idx - 1],
                            gsenddec=sat_dec[end_idx - 1],
                            length=passlen,
                        )

                        # Record the path during the pass
                        gspass.utime = timestamp_unix[
                            startindex + start_idx : startindex + end_idx
                        ].tolist()
                        gspass.ra = sat_ra[start_idx:end_idx].tolist()
                        gspass.dec = sat_dec[start_idx:end_idx].tolist()

                        self.passes.append(gspass)

        # Order the passes by time
        self.passes.sort(key=lambda x: x.begin, reverse=False)
