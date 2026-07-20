import time
from numbers import Integral
from typing import Literal, Protocol

import numpy as np
import rust_ephem
from pydantic import BaseModel, ConfigDict, Field

from ..common import find_boundaries, ics_date_conv, unixtime2date
from ..common.enums import ACSMode, AntennaType, ObsType, SlewAlgorithm
from ..common.vector import (
    attitude_for_body_vector_tracking,
    body_vector_to_eci,
    quaternion_attitude_distance,
    radec2vec,
    separation,
    vec2radec,
)
from ..config import (
    Constraint,
    GroundStation,
    GroundStationRegistry,
    MissionConfig,
)
from ..config.constants import DTOR
from ..config.constraint import in_attitude_constraint_scopes
from .slew import Slew

# Legacy ground-pass roll for profiles without explicit roll samples.
GSP_TRACK_ROLL = 0.0
LEGACY_GSP_ANTENNA_BODY_VECTOR = (-1.0, 0.0, 0.0)


def pass_slew_trigger_buffer(step_size: float) -> float:
    """Return how early pass handling can trigger a slew, in seconds."""
    return max(0.0, 2.0 * float(step_size))


def _config_random_seed(config: MissionConfig) -> int | None:
    seed = config.random_seed
    if seed is None or isinstance(seed, bool) or not isinstance(seed, Integral):
        return None
    return int(seed)


def _unit_vector_or_none(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return None
    return vector / norm


def _perpendicular_reference(axis: np.ndarray) -> np.ndarray | None:
    for candidate in (
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ):
        projected = candidate - float(np.dot(candidate, axis)) * axis
        reference = _unit_vector_or_none(projected)
        if reference is not None:
            return reference
    return None


def _rotate_about_axis(
    vector: np.ndarray, axis: np.ndarray, angle_deg: float
) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    rotated = (
        vector * np.cos(angle_rad)
        + np.cross(axis, vector) * np.sin(angle_rad)
        + axis * float(np.dot(axis, vector)) * (1.0 - np.cos(angle_rad))
    )
    return np.asarray(rotated, dtype=float)


def _tracking_phase_reference_eci(
    target_vector: np.ndarray, phase_deg: float
) -> np.ndarray | None:
    target_axis = _unit_vector_or_none(np.asarray(target_vector, dtype=float))
    if target_axis is None:
        return None
    reference = _perpendicular_reference(target_axis)
    if reference is None:
        return None
    return _rotate_about_axis(reference, target_axis, phase_deg)


class RandomSource(Protocol):
    def random(self) -> float: ...


class Pass(BaseModel):
    """A groundstation pass consisting of the dwell phase.

    The pass contains pointing profile information for ground station contacts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core dependencies
    ephem: rust_ephem.Ephemeris | None = None
    config: MissionConfig | None = None

    # Pass metadata
    station: str
    begin: float
    length: float

    # What type of observation is this, a Ground Station Pass (GSP)
    obstype: ObsType = ObsType.GSP

    # Ground station pointing vectors (start/end of contact)
    antenna_boresight_body: tuple[float, float, float] = LEGACY_GSP_ANTENNA_BODY_VECTOR
    gsstartra: float = 0.0
    gsstartdec: float = 0.0
    gsstartroll: float = GSP_TRACK_ROLL
    gsendra: float = 0.0
    gsenddec: float = 0.0
    gsendroll: float = GSP_TRACK_ROLL

    # Recorded pointing profile during the pass dwell
    utime: list[float] = Field(default_factory=list)
    ra: list[float] = Field(default_factory=list)
    dec: list[float] = Field(default_factory=list)
    roll: list[float] = Field(default_factory=list)
    station_ra: list[float] = Field(default_factory=list)
    station_dec: list[float] = Field(default_factory=list)

    # Scheduling / status
    slewrequired: float = 0.0
    slewlate: float = 0.0
    obsid: int = 0xFFFF

    @property
    def end(self) -> float:
        assert self.length is not None, "Pass length must be set"
        return self.begin + self.length

    def __str__(self) -> str:
        """Return string of details on the pass"""
        return f"{unixtime2date(self.begin):18s}  {self.station:3s}  {self.length / 60.0:4.1f} mins"

    def in_pass(self, utime: float) -> bool:
        return self.begin <= utime <= self.end

    def time_to_pass(self) -> str:
        """Return a string for how long it is until the next pass"""
        now = time.time()
        timetopass = (self.begin - now) / 60.0

        if timetopass < 60.0:
            timetopassstring = "%.0f mins " % timetopass
        else:
            timetopassstring = "%.1f hours" % (timetopass / 60.0)

        return timetopassstring

    def _profile_index(self, utime: float) -> int | None:
        if not self.utime:
            return None

        idx = int(np.searchsorted(self.utime, utime))
        if idx >= len(self.utime):
            return len(self.utime) - 1
        return idx

    def ra_dec(self, utime: float) -> tuple[float | None, float | None]:
        """Return the RA/Dec of the spacecraft during a groundstation pass.
        Note: If utime is outside the pass, returns the earliest or latest
        RA/Dec in the pass. If no profile is available, returns None values."""

        idx = self._profile_index(utime)
        if idx is None:
            return None, None
        return self.ra[idx], self.dec[idx]

    def roll_at(self, utime: float) -> float:
        """Return the spacecraft roll during a groundstation pass."""

        idx = self._profile_index(utime)
        if idx is None or not self.roll:
            return self.gsstartroll
        return self.roll[idx]

    def attitude_at(self, utime: float) -> tuple[float | None, float | None, float]:
        """Return RA, Dec, and roll during a groundstation pass."""

        ra, dec = self.ra_dec(utime)
        return ra, dec, self.roll_at(utime)

    def station_ra_dec(self, utime: float) -> tuple[float | None, float | None]:
        """Return the spacecraft-to-ground-station line of sight."""

        idx = self._profile_index(utime)
        if idx is None or not self.station_ra or not self.station_dec:
            return None, None
        return self.station_ra[idx], self.station_dec[idx]

    def pointing_error(
        self,
        spacecraft_ra: float,
        spacecraft_dec: float,
        target_ra: float,
        target_dec: float,
    ) -> float:
        """Calculate pointing error between spacecraft pointing and target.

        Args:
            spacecraft_ra: Current spacecraft RA (degrees)
            spacecraft_dec: Current spacecraft Dec (degrees)
            target_ra: Target RA (degrees)
            target_dec: Target Dec (degrees)

        Returns:
            Angular separation in degrees
        """
        # Convert to radians

        return (
            separation(
                [spacecraft_ra * DTOR, spacecraft_dec * DTOR],
                [target_ra * DTOR, target_dec * DTOR],
            )
            / DTOR
        )

    def can_communicate(
        self,
        spacecraft_ra: float,
        spacecraft_dec: float,
        utime: float | None = None,
        spacecraft_roll: float = 0.0,
    ) -> bool:
        """Check if communication is possible given spacecraft pointing.

        Args:
            spacecraft_ra: Current spacecraft RA (degrees)
            spacecraft_dec: Current spacecraft Dec (degrees)
            utime: Time during pass (optional, uses begin if None)
            spacecraft_roll: Spacecraft roll angle in degrees

        Returns:
            True if communication is possible, False otherwise
        """
        if self.config is None or self.config.spacecraft_bus.communications is None:
            # No comms config, assume communication is always possible
            return True

        # Determine target pointing for this time
        if utime is None:
            utime = self.begin

        target_ra, target_dec = self.ra_dec(utime)
        if target_ra is None or target_dec is None:
            return False

        error = self.antenna_pointing_error(
            spacecraft_ra, spacecraft_dec, spacecraft_roll, utime
        )
        if error is None:
            error = self.pointing_error(
                spacecraft_ra, spacecraft_dec, target_ra, target_dec
            )

        # Check if within acceptable range
        return self.config.spacecraft_bus.communications.can_communicate(error)

    def antenna_pointing_error(
        self, spacecraft_ra: float, spacecraft_dec: float, roll: float, utime: float
    ) -> float | None:
        """Return fixed antenna pointing error to the station line of sight."""

        target_ra, target_dec = self.station_ra_dec(utime)
        if target_ra is None or target_dec is None:
            return None

        antenna_vec = body_vector_to_eci(
            spacecraft_ra, spacecraft_dec, roll, self.antenna_boresight_body
        )
        if antenna_vec is None:
            return None
        target_vec = radec2vec(target_ra * DTOR, target_dec * DTOR)
        dot = float(np.dot(antenna_vec, target_vec))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(np.rad2deg(np.arccos(dot)))

    def get_data_rate(
        self, band: str, direction: Literal["downlink", "uplink"] = "downlink"
    ) -> float:
        """Get data rate for this pass in the specified band.

        Args:
            band: Frequency band (e.g., "S", "X", "Ka")
            direction: "downlink" or "uplink"

        Returns:
            Data rate in Mbps, or 0.0 if band not supported
        """
        assert self.config is not None, "Config must be set for Pass class"
        if self.config.spacecraft_bus.communications is None:
            return 0.0

        if direction.lower() == "downlink":
            return self.config.spacecraft_bus.communications.get_downlink_rate(band)
        elif direction.lower() == "uplink":
            return self.config.spacecraft_bus.communications.get_uplink_rate(band)
        else:
            return 0.0

    def calculate_data_volume(
        self, band: str, direction: Literal["downlink", "uplink"] = "downlink"
    ) -> float:
        """Calculate total data volume for this pass.

        Args:
            band: Frequency band (e.g., "S", "X", "Ka")
            direction: "downlink" or "uplink"

        Returns:
            Data volume in Megabits
        """
        assert self.config is not None, "Config must be set for Pass class"
        if self.config.spacecraft_bus.communications is None:
            return 0.0

        rate_mbps = self.get_data_rate(band, direction)
        return rate_mbps * self.length  # Mbps * seconds = Megabits

    def _slew_time_to_target(
        self,
        utime: float,
        ra: float,
        dec: float,
        roll: float,
        target_ra: float,
        target_dec: float,
        target_roll: float,
    ) -> float:
        assert self.config is not None, "Config must be set for Pass class"
        if self.config.spacecraft_bus.attitude_control is None:
            raise ValueError("ACS config must be set to calculate slew time")

        acs_config = self.config.spacecraft_bus.attitude_control
        # This scalar shortcut is equivalent to Slew.calc_slewtime() only for
        # quaternion slews. Constraint-avoiding and future slew algorithms keep
        # the full path unless their scalar equivalence has been proven.
        if acs_config.slew_algorithm == SlewAlgorithm.QUATERNION:
            slewdist = quaternion_attitude_distance(
                ra,
                dec,
                roll,
                target_ra,
                target_dec,
                target_roll,
            )
            return round(acs_config.slew_time(slewdist))

        slew = Slew(config=self.config)
        slew.startra = ra
        slew.startdec = dec
        slew.startroll = roll
        slew.endra = target_ra
        slew.enddec = target_dec
        slew.endroll = target_roll
        slew.slewstart = utime
        return slew.calc_slewtime()

    def time_to_slew(
        self, utime: float, ra: float, dec: float, roll: float = 0.0
    ) -> bool:
        """Determine whether to begin slewing for this pass.

        Calculates the slew time between current RA/Dec and the appropriate pointing of the pass.
        If on time, slews to pass start. If late, slews to where the pass currently is.
        Returns True when the pass time minus slew time is less than 60 seconds away.
        """
        assert self.ephem is not None, "Ephemeris must be set for Pass class"
        assert self.config is not None, "Config must be set for Pass class"

        # Determine target pointing: if we're late, target where pass currently is
        if utime >= self.begin:
            # We're late - target current pass position
            target_ra, target_dec, target_roll = self.attitude_at(utime)
            if target_ra is None or target_dec is None:
                return False
        else:
            # On time - target pass start
            if not self.ra or not self.dec:
                return False
            target_ra, target_dec, target_roll = (
                self.ra[0],
                self.dec[0],
                self.roll[0] if self.roll else self.gsstartroll,
            )

        slewtime = self._slew_time_to_target(
            utime, ra, dec, roll, target_ra, target_dec, target_roll
        )
        # Determine if we need to start slewing now
        time_until_slew = (self.begin - slewtime) - utime

        if time_until_slew <= pass_slew_trigger_buffer(self.ephem.step_size):
            return True
        else:
            return False


class PassTimes:
    """PassTimes class for calculating passes based on ephemeris and ground station location"""

    passes: list[Pass]
    constraint: Constraint
    ephem: rust_ephem.Ephemeris
    ground_stations: GroundStationRegistry
    config: MissionConfig

    def __init__(
        self,
        config: MissionConfig,
        rng: RandomSource | None = None,
    ):
        self.constraint = config.constraint
        assert self.constraint.ephem is not None, (
            "Ephemeris must be set for PassTimes class"
        )
        self.ephem = self.constraint.ephem

        self.config = config
        self.rng = (
            rng
            if rng is not None
            else np.random.default_rng(_config_random_seed(config))
        )
        self.passes = []
        self.dropped_overlapping_passes: list[tuple[Pass, Pass]] = []
        self.dropped_constraint_passes: list[Pass] = []
        self.length = 1

        # Ground stations registry from config
        if config.ground_stations is None:
            self.ground_stations = GroundStationRegistry.default()
        else:
            self.ground_stations = config.ground_stations

        # What makes a good pass
        self.minlen = 8 * 60  # 10 mins
        self.schedule_chance = 1.0  # base chance of getting a pass

    def _random(self) -> float:
        return float(self.rng.random())

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
            if gspass.begin - last <= mean_between:
                continue
            if gsprob >= 1.0 or (gsprob > 0.0 and self._random() < gsprob):
                sched.extend([gspass])
                last = sched[-1].begin
        return sched

    def _station_downlink_rate_mbps(self, gspass: Pass) -> float:
        try:
            rate = self.ground_stations.get(gspass.station).get_overall_max_downlink()
        except KeyError:
            return 0.0
        return 0.0 if rate is None else rate

    def _pass_selection_key(self, gspass: Pass) -> tuple[float, float, float]:
        rate = self._station_downlink_rate_mbps(gspass)
        return (rate * gspass.length, gspass.length, -gspass.begin)

    def _gsp_antenna_boresight_body(self) -> tuple[float, float, float] | None:
        communications = self.config.spacecraft_bus.communications
        if communications is None:
            return LEGACY_GSP_ANTENNA_BODY_VECTOR
        antenna = communications.antenna_pointing
        if antenna.antenna_type != AntennaType.FIXED:
            return None
        boresight = antenna.fixed_boresight_body
        return (float(boresight[0]), float(boresight[1]), float(boresight[2]))

    def _pass_attitude_violates_scopes(
        self, ra: float, dec: float, roll: float, utime: float
    ) -> bool:
        return bool(
            in_attitude_constraint_scopes(
                self.constraint,
                self.config.attitude_constraint_scopes_for_mode(ACSMode.PASS),
                ra,
                dec,
                utime,
                target_roll=roll,
                acs_mode=ACSMode.PASS,
            )
        )

    def _pass_profile_violates_scopes(
        self,
        utimes: list[float],
        track_ra: list[float],
        track_dec: list[float],
        track_roll: list[float],
    ) -> bool:
        return any(
            self._pass_attitude_violates_scopes(ra, dec, roll, utime)
            for utime, ra, dec, roll in zip(
                utimes, track_ra, track_dec, track_roll, strict=True
            )
        )

    def _gsp_tracking_phase_candidates(self) -> list[float]:
        candidates = [0.0]
        phase_step_deg = (
            self.config.spacecraft_bus.attitude_control.gsp_tracking_phase_step_deg
        )
        offset = phase_step_deg
        while offset <= 180.0:
            candidates.append(float(offset))
            if offset < 180.0:
                candidates.append(float(360.0 - offset))
            offset += phase_step_deg
        return candidates

    def _tracking_attitude_profile_for_phase(
        self,
        antenna_boresight: tuple[float, float, float],
        target_vectors: np.ndarray,
        phase_deg: float,
    ) -> list[tuple[float, float, float]] | None:
        attitude_profile: list[tuple[float, float, float]] = []
        for target_vector in target_vectors:
            reference_eci = _tracking_phase_reference_eci(target_vector, phase_deg)
            if reference_eci is None:
                return None
            attitude = attitude_for_body_vector_tracking(
                antenna_boresight,
                target_vector,
                reference_eci=reference_eci,
            )
            if attitude is None:
                return None
            attitude_profile.append(attitude)
        return attitude_profile

    def _fixed_phase_tracking_attitude_profile(
        self,
        antenna_boresight: tuple[float, float, float],
        target_vectors: np.ndarray,
        track_utime: list[float],
    ) -> tuple[list[tuple[float, float, float]] | None, bool]:
        fallback_profile: list[tuple[float, float, float]] | None = None
        for phase_deg in self._gsp_tracking_phase_candidates():
            attitude_profile = self._tracking_attitude_profile_for_phase(
                antenna_boresight, target_vectors, phase_deg
            )
            if attitude_profile is None:
                continue
            if fallback_profile is None:
                fallback_profile = attitude_profile
            track_ra = [attitude[0] for attitude in attitude_profile]
            track_dec = [attitude[1] for attitude in attitude_profile]
            track_roll = [attitude[2] for attitude in attitude_profile]
            if not self._pass_profile_violates_scopes(
                track_utime, track_ra, track_dec, track_roll
            ):
                return attitude_profile, True

        return fallback_profile, False

    def _safe_tracking_attitudes_by_sample(
        self,
        antenna_boresight: tuple[float, float, float],
        target_vectors: np.ndarray,
        track_utime: list[float],
    ) -> list[dict[float, tuple[float, float, float]]] | None:
        safe_attitudes: list[dict[float, tuple[float, float, float]]] = []
        for target_vector, utime in zip(target_vectors, track_utime, strict=True):
            sample_attitudes: dict[float, tuple[float, float, float]] = {}
            for phase_deg in self._gsp_tracking_phase_candidates():
                attitude_profile = self._tracking_attitude_profile_for_phase(
                    antenna_boresight,
                    np.array([target_vector], dtype=float),
                    phase_deg,
                )
                if attitude_profile is None:
                    continue
                attitude = attitude_profile[0]
                if not self._pass_attitude_violates_scopes(*attitude, utime):
                    sample_attitudes[phase_deg] = attitude
            if not sample_attitudes:
                return None
            safe_attitudes.append(sample_attitudes)
        return safe_attitudes

    def _step_motion_feasible(
        self,
        previous_attitude: tuple[float, float, float],
        attitude: tuple[float, float, float],
        dt: float,
    ) -> bool:
        if dt <= 0.0:
            return False
        attitude_distance = quaternion_attitude_distance(
            *previous_attitude,
            *attitude,
        )
        acs = self.config.spacecraft_bus.attitude_control
        return bool(acs.motion_time(attitude_distance) <= dt)

    def _dynamic_phase_tracking_attitude_profile(
        self,
        safe_attitudes: list[dict[float, tuple[float, float, float]]],
        track_utime: list[float],
    ) -> list[tuple[float, float, float]] | None:
        phase_rank = {
            phase: index
            for index, phase in enumerate(self._gsp_tracking_phase_candidates())
        }
        paths: dict[float, tuple[float, float, list[float]]] = {
            phase: (0.0, 0.0, [phase]) for phase in safe_attitudes[0]
        }

        for sample_index in range(1, len(safe_attitudes)):
            dt = float(track_utime[sample_index] - track_utime[sample_index - 1])
            next_paths: dict[float, tuple[float, float, list[float]]] = {}
            for phase, attitude in safe_attitudes[sample_index].items():
                best: tuple[float, float, list[float]] | None = None
                for previous_phase, (
                    previous_max_step,
                    previous_total_step,
                    previous_path,
                ) in paths.items():
                    previous_attitude = safe_attitudes[sample_index - 1][previous_phase]
                    if not self._step_motion_feasible(previous_attitude, attitude, dt):
                        continue
                    step_distance = quaternion_attitude_distance(
                        *previous_attitude,
                        *attitude,
                    )
                    candidate = (
                        max(previous_max_step, step_distance),
                        previous_total_step + step_distance,
                        previous_path + [phase],
                    )
                    if best is None or (candidate[0], candidate[1]) < (
                        best[0],
                        best[1],
                    ):
                        best = candidate
                if best is not None:
                    next_paths[phase] = best
            paths = next_paths
            if not paths:
                return None

        _, (_, _, phase_path) = min(
            paths.items(),
            key=lambda item: (
                item[1][0],
                item[1][1],
                [phase_rank[phase] for phase in item[1][2]],
            ),
        )
        return [
            safe_attitudes[sample_index][phase]
            for sample_index, phase in enumerate(phase_path)
        ]

    def _constraint_safe_tracking_attitude_profile(
        self,
        antenna_boresight: tuple[float, float, float],
        target_vectors: np.ndarray,
        track_utime: list[float],
    ) -> tuple[list[tuple[float, float, float]], bool] | None:
        fallback_profile, fixed_phase_safe = (
            self._fixed_phase_tracking_attitude_profile(
                antenna_boresight, target_vectors, track_utime
            )
        )
        if fixed_phase_safe:
            assert fallback_profile is not None
            return fallback_profile, True

        safe_attitudes = self._safe_tracking_attitudes_by_sample(
            antenna_boresight, target_vectors, track_utime
        )
        if safe_attitudes is not None:
            dynamic_profile = self._dynamic_phase_tracking_attitude_profile(
                safe_attitudes, track_utime
            )
            if dynamic_profile is not None:
                return dynamic_profile, True

        if fallback_profile is None:
            return None
        return fallback_profile, False

    def _deconflict_overlapping_passes(self) -> None:
        selected: list[Pass] = []
        self.dropped_overlapping_passes = []

        for gspass in sorted(self.passes, key=lambda x: x.begin, reverse=False):
            if not selected or selected[-1].end <= gspass.begin:
                selected.append(gspass)
                continue

            incumbent = selected[-1]
            if self._pass_selection_key(gspass) > self._pass_selection_key(incumbent):
                selected[-1] = gspass
                self.dropped_overlapping_passes.append((incumbent, gspass))
            else:
                self.dropped_overlapping_passes.append((gspass, incumbent))

        self.passes = selected

    def _station_visibility(
        self,
        station: GroundStation,
        startindex: int,
        endindex: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute elevation angle and target-direction unit vectors for a station.

        Uses a fast vectorized Earth-limb approach: the angle from the ground
        station's local "up" direction to the satellite direction is compared
        against the station's minimum elevation.

        Returns (elevation_angle, gs_to_sat_unit) arrays over [startindex, endindex).
        """
        gs_ephem = rust_ephem.GroundEphemeris(
            latitude=station.latitude_deg,
            longitude=station.longitude_deg,
            height=station.elevation_m,
            begin=self.ephem.timestamp[startindex],
            end=self.ephem.timestamp[endindex - 1],
            step_size=self.ephem.step_size,
        )

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

        return elevation_angle, gs_to_sat_unit

    def _build_pass(
        self,
        station: GroundStation,
        startindex: int,
        start_idx: int,
        end_idx: int,
        timestamp_unix: np.ndarray,
        gs_to_sat_unit: np.ndarray,
    ) -> tuple[Pass, bool] | None:
        """Build a Pass for one visibility window.

        Returns None if the window is too short, loses the scheduling-probability
        dice roll, or no constraint-safe tracking attitude profile can be built.
        Otherwise returns (gspass, profile_safe).
        """
        global_start_idx = startindex + start_idx
        global_end_idx = (
            startindex + end_idx
        )  # end_idx is already the first point below threshold
        # Clamp to last valid index to avoid overflow on short ephemeris windows
        global_end_idx = min(global_end_idx, len(timestamp_unix) - 1)

        passstart = timestamp_unix[global_start_idx]
        passend = timestamp_unix[global_end_idx]
        passlen = passend - passstart

        # Only consider passes that meet minimum length
        if passlen < self.minlen:
            return None

        # Combine global and station-specific schedule probabilities
        combined_prob = self.schedule_chance * station.schedule_probability
        should_schedule = combined_prob >= 1.0 or (
            combined_prob > 0.0 and self._random() <= combined_prob
        )
        if not should_schedule:
            return None

        target_vectors = -gs_to_sat_unit[start_idx:end_idx]
        target_ra, target_dec = np.degrees(vec2radec(target_vectors.T))
        track_utime = timestamp_unix[
            startindex + start_idx : startindex + end_idx
        ].tolist()
        antenna_boresight = self._gsp_antenna_boresight_body()
        if antenna_boresight is None:
            return None
        profile_result = self._constraint_safe_tracking_attitude_profile(
            antenna_boresight, target_vectors, track_utime
        )
        if profile_result is None:
            return None
        attitude_profile, profile_safe = profile_result
        track_ra = [attitude[0] for attitude in attitude_profile]
        track_dec = [attitude[1] for attitude in attitude_profile]
        track_roll = [attitude[2] for attitude in attitude_profile]

        gspass = Pass(
            config=self.config,
            ephem=self.ephem,
            station=station.code,
            begin=passstart,
            antenna_boresight_body=antenna_boresight,
            gsstartra=track_ra[0],
            gsstartdec=track_dec[0],
            gsstartroll=track_roll[0],
            gsendra=track_ra[-1],
            gsenddec=track_dec[-1],
            gsendroll=track_roll[-1],
            length=passlen,
        )

        # Record the path during the pass
        gspass.utime = track_utime
        gspass.ra = track_ra
        gspass.dec = track_dec
        gspass.roll = track_roll
        gspass.station_ra = target_ra.tolist()
        gspass.station_dec = target_dec.tolist()

        return gspass, profile_safe

    def get(self, year: int, day: int, length: int = 1) -> None:
        """Calculate the passes using rust_ephem GroundEphemeris for vectorized operations."""
        ustart = ics_date_conv(f"{year}-{day:03d}-00:00:00")
        self.dropped_constraint_passes = []

        # Use binary search instead of np.where for finding start index
        # Prefer adapter datetimes if available, otherwise use Time.unix
        timestamp_unix = np.array([dt.timestamp() for dt in self.ephem.timestamp])
        startindex = int(np.searchsorted(timestamp_unix, ustart))

        # Calculate end index
        num_steps = int(86400 * length / self.ephem.step_size)
        endindex = min(startindex + num_steps, len(timestamp_unix))

        # Process stations that are allowed to drive spacecraft GSP tracking. RF
        # uplink/downlink capability is modeled separately on each station.
        for station in self.ground_stations.stations:
            if not station.schedule_for_tracking:
                continue

            elevation_angle, gs_to_sat_unit = self._station_visibility(
                station, startindex, endindex
            )
            is_visible = elevation_angle > station.min_elevation_deg
            pass_starts, pass_ends = find_boundaries(is_visible)

            for start_idx, end_idx in zip(pass_starts, pass_ends):
                result = self._build_pass(
                    station,
                    startindex,
                    start_idx,
                    end_idx,
                    timestamp_unix,
                    gs_to_sat_unit,
                )
                if result is None:
                    continue
                gspass, profile_safe = result
                if not profile_safe:
                    self.dropped_constraint_passes.append(gspass)
                    continue
                self.passes.append(gspass)

        # Order the passes by time
        self.passes.sort(key=lambda x: x.begin, reverse=False)
        self._deconflict_overlapping_passes()
