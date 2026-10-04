"""Check an executed DITL timeline against the plan it claims to have flown."""

from bisect import bisect_left
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..common import ACSMode, ObsType, angular_separation, unixtime2date
from ..common.vector import attitude_to_quat
from ..config import MissionConfig
from ..simulation.passes import Pass
from ..targets import Plan, PlanEntry

PLAN_SCIENCE_OBSTYPES = frozenset({ObsType.PPT, ObsType.AT, ObsType.TOO})
"""Entry types that describe a science observation of a fixed target."""

QUEUE_SCIENCE_OBSTYPES = frozenset({ObsType.AT, ObsType.TOO})
"""Science entry types a queue-built plan exports; its PPT entries are not checked."""

AttitudeViolation = tuple[str, str]
"""Violated constraint name and the scope label it was checked under."""


class PlanExecutionMismatchError(RuntimeError):
    """Raised when an exported plan entry does not match executed ACS telemetry."""


class PlanExecutionMismatch(BaseModel):
    """A single mismatch between an exported plan entry and ACS telemetry."""

    model_config = ConfigDict(frozen=True)

    utime: float
    message: str
    obsid: int | None = None

    def __str__(self) -> str:
        """Return the mismatch message as the string representation."""
        return self.message


def entry_obstype(entry: PlanEntry) -> ObsType | None:
    """Return the entry's obstype as an ObsType, or None if it cannot be coerced."""
    obstype = getattr(entry, "obstype", None)
    if isinstance(obstype, ObsType):
        return obstype
    try:
        return ObsType(obstype)
    except (TypeError, ValueError):
        return None


class PlanExecutionValidator:
    """Compare a plan with the per-step telemetry recorded while executing it.

    The validator reads only the plan, the telemetry arrays and the scheduled
    ground-station passes, so it applies equally to a plan a scheduler built
    while simulating (QueueDITL) and to a plan supplied for execution (DITL).

    Args:
        config: Mission configuration (slew accuracy sets the pointing tolerance).
        plan: Plan whose science and contact entries are checked.
        utime, ra, dec, roll, obsid, mode: Per-step executed telemetry.
        end_time: Simulation end, used to timestamp a telemetry-length mismatch.
        passes: Ground-station passes that GSP entries are matched against.
        attitude_violation_at: Returns the attitude constraint, if any, that
            telemetry sample ``index`` violates under ``mode``'s scopes.
        rate_mismatches: Attitude-rate violations already found in telemetry.
        dropped_science_windows: ``(begin, end, obsid)`` windows of science the
            scheduler deliberately dropped from the plan as under-collected.
        science_obstypes: Entry types checked as science observations.
    """

    def __init__(
        self,
        *,
        config: MissionConfig,
        plan: Plan,
        utime: Sequence[float],
        ra: Sequence[float],
        dec: Sequence[float],
        roll: Sequence[float],
        obsid: Sequence[int],
        mode: Sequence[ACSMode | int],
        end_time: float,
        passes: Sequence[Pass],
        attitude_violation_at: Callable[[int, ACSMode], AttitudeViolation | None],
        rate_mismatches: Sequence[PlanExecutionMismatch] = (),
        dropped_science_windows: Sequence[tuple[float, float, int]] = (),
        science_obstypes: frozenset[ObsType] = QUEUE_SCIENCE_OBSTYPES,
    ) -> None:
        self.config = config
        self.plan = plan
        self.utime = utime
        self.ra = ra
        self.dec = dec
        self.roll = roll
        self.obsid = obsid
        self.mode = mode
        self.end_time = end_time
        self.passes = passes
        self.attitude_violation_at = attitude_violation_at
        self.rate_mismatches = rate_mismatches
        self.dropped_science_windows = dropped_science_windows
        self.science_obstypes = science_obstypes

    def validate(self) -> list[PlanExecutionMismatch]:
        """Return plan/ACS mismatches over exported science and contact intervals."""
        mismatch = self._telemetry_length_mismatch()
        if mismatch is not None:
            return [mismatch]

        mismatches = list(self.rate_mismatches)
        tolerance_deg = self._tolerance_deg()
        structure_mismatches = self.plan_structure_mismatches()
        if structure_mismatches:
            return mismatches + structure_mismatches

        for entry in self.plan:
            obstype = entry_obstype(entry)
            if obstype in self.science_obstypes:
                mismatches.extend(self._science_entry_mismatches(entry, tolerance_deg))
            elif obstype == ObsType.GSP:
                mismatches.extend(self._gsp_entry_mismatches(entry, tolerance_deg))
        mismatches.extend(self.attitude_constraint_mismatches())
        mismatches.extend(self.unplanned_execution_mismatches())
        return mismatches

    def attitude_constraint_mismatches(self) -> list[PlanExecutionMismatch]:
        """Report samples whose attitude violated a constraint scoped to their mode.

        An empty scope list for a mode disables validation for that mode.
        """
        mismatches: list[PlanExecutionMismatch] = []
        for i in range(len(self.utime)):
            mode = self._mode_at_index(i)
            if mode is None:
                mismatches.append(self._unknown_mode_mismatch(i))
                continue

            violation = self.attitude_violation_at(i, mode)
            if violation is None:
                continue

            constraint_name, scope_label = violation
            mismatches.append(
                self._attitude_constraint_mismatch(i, constraint_name, scope_label)
            )
        return mismatches

    def plan_structure_mismatches(self) -> list[PlanExecutionMismatch]:
        """Check plan entries for invalid intervals and non-monotonic ordering."""
        mismatches: list[PlanExecutionMismatch] = []
        previous_begin: float | None = None
        for index, entry in enumerate(self.plan):
            obstype = entry_obstype(entry)
            if obstype is None or (
                obstype == ObsType.PPT and obstype not in self.science_obstypes
            ):
                continue
            begin = float(entry.begin)
            end = float(entry.end)
            obsid = int(entry.obsid) if entry.obsid is not None else None
            if end <= begin:
                mismatches.append(
                    self._mismatch(
                        begin,
                        "plan",
                        "invalid_interval",
                        (
                            f"entry {index} obsid {obsid} ends at or before it begins "
                            f"({end:.0f} <= {begin:.0f})"
                        ),
                        obsid=obsid,
                    )
                )
            if previous_begin is not None and begin < previous_begin:
                mismatches.append(
                    self._mismatch(
                        begin,
                        "plan",
                        "non_monotonic_begin",
                        (
                            f"entry {index} obsid {obsid} begins before previous "
                            f"entry ({begin:.0f} < {previous_begin:.0f})"
                        ),
                        obsid=obsid,
                    )
                )
            previous_begin = begin
        return mismatches

    def unplanned_execution_mismatches(self) -> list[PlanExecutionMismatch]:
        """Check every executed science/pass sample is covered by a plan entry."""
        mismatches: list[PlanExecutionMismatch] = []

        # Build obsid → entries lookups once to avoid O(N×M) linear plan scans.
        science_by_obsid: dict[int, list[PlanEntry]] = {}
        gsp_by_obsid: dict[int, list[PlanEntry]] = {}
        for entry in self.plan:
            obstype = entry_obstype(entry)
            if obstype in self.science_obstypes:
                science_by_obsid.setdefault(int(entry.obsid), []).append(entry)
            elif obstype == ObsType.GSP:
                gsp_by_obsid.setdefault(int(entry.obsid), []).append(entry)

        dropped_by_obsid: dict[int, list[tuple[float, float]]] = {}
        for start, end, dropped_obsid in self.dropped_science_windows:
            dropped_by_obsid.setdefault(dropped_obsid, []).append((start, end))

        for i, utime in enumerate(self.utime):
            mode = self._mode_at_index(i)
            if mode == ACSMode.SCIENCE:
                obsid = int(self.obsid[i])
                entries = science_by_obsid.get(obsid, [])
                covered = any(float(e.begin) <= utime < float(e.end) for e in entries)
                if not covered:
                    if any(s <= utime < e for s, e in dropped_by_obsid.get(obsid, [])):
                        continue
                    mismatches.append(
                        self._mismatch(
                            utime,
                            "execution",
                            "unplanned_science",
                            f"obsid {obsid} has no matching exported science entry",
                            obsid=obsid,
                        )
                    )
            elif mode == ACSMode.PASS:
                obsid = int(self.obsid[i])
                entries = gsp_by_obsid.get(obsid, [])
                if not any(float(e.begin) <= utime <= float(e.end) for e in entries):
                    mismatches.append(
                        self._mismatch(
                            utime,
                            "execution",
                            "unplanned_contact",
                            f"obsid {obsid} has no matching exported GSP entry",
                            obsid=obsid,
                        )
                    )
        return mismatches

    def _mode_at_index(self, index: int) -> ACSMode | None:
        """Return the telemetry mode at index as an ACSMode, or None if it cannot be coerced."""
        mode = self.mode[index]
        if isinstance(mode, ACSMode):
            return mode
        try:
            return ACSMode(mode)
        except (TypeError, ValueError):
            return None

    def _tolerance_deg(self) -> float:
        """Return the pointing error tolerance, in degrees."""
        tolerance = float(self.config.spacecraft_bus.attitude_control.slew_accuracy)
        return tolerance if tolerance > 0 else 0.01

    def _telemetry_length_mismatch(self) -> PlanExecutionMismatch | None:
        """Return a mismatch if the per-step telemetry lists are not all the same length."""
        lengths = {
            "utime": len(self.utime),
            "ra": len(self.ra),
            "dec": len(self.dec),
            "roll": len(self.roll),
            "obsid": len(self.obsid),
            "mode": len(self.mode),
        }
        if len(set(lengths.values())) == 1:
            return None
        return PlanExecutionMismatch(
            utime=self.end_time,
            message=f"telemetry_length_mismatch: {lengths}",
        )

    @staticmethod
    def _science_start_time(entry: PlanEntry) -> float:
        """Return the time science collection begins for an entry, after its slew."""
        if entry.collection_begin is not None:
            return entry.collection_begin
        return float(entry.begin) + max(0.0, float(entry.slewtime))

    def _window_indices(self, start: float, end: float) -> range:
        """Return the range of telemetry indices whose utime falls within [start, end)."""
        lo = max(0, bisect_left(self.utime, start))
        hi = min(len(self.utime), bisect_left(self.utime, end))
        return range(lo, hi)

    @staticmethod
    def _mismatch(
        utime: float,
        interval: str,
        mismatch_type: str,
        detail: str,
        obsid: int | None = None,
    ) -> PlanExecutionMismatch:
        """Build a PlanExecutionMismatch with a formatted, timestamped message."""
        return PlanExecutionMismatch(
            utime=utime,
            message=f"{unixtime2date(utime)} {interval} {mismatch_type}: {detail}",
            obsid=obsid,
        )

    def _mode_mismatch(
        self,
        entry: PlanEntry,
        utime: float,
        actual_mode: ACSMode | None,
        expected_mode: ACSMode,
        interval: str,
    ) -> PlanExecutionMismatch:
        """Build a mismatch for telemetry executing in an unexpected ACS mode."""
        return self._mismatch(
            utime,
            interval,
            "mode_mismatch",
            f"obsid {entry.obsid} expected {expected_mode} got {actual_mode}",
            obsid=int(entry.obsid),
        )

    def _obsid_mismatch(
        self, entry: PlanEntry, utime: float, actual_obsid: int, interval: str
    ) -> PlanExecutionMismatch:
        """Build a mismatch for telemetry reporting an unexpected obsid."""
        return self._mismatch(
            utime,
            interval,
            "obsid_mismatch",
            f"expected {int(entry.obsid)} got {int(actual_obsid)}",
            obsid=int(entry.obsid),
        )

    def _pointing_mismatch(
        self,
        entry: PlanEntry,
        utime: float,
        error_deg: float,
        interval: str,
    ) -> PlanExecutionMismatch:
        """Build a mismatch for pointing error exceeding tolerance."""
        return self._mismatch(
            utime,
            interval,
            "pointing_mismatch",
            f"obsid {int(entry.obsid)} error {error_deg:.3f} deg",
            obsid=int(entry.obsid),
        )

    def _attitude_constraint_mismatch(
        self, index: int, constraint_name: str, scope_label: str
    ) -> PlanExecutionMismatch:
        """Build a mismatch for an attitude constraint violation at a telemetry sample."""
        mode = self._mode_at_index(index)
        obsid = int(self.obsid[index])
        return self._mismatch(
            self.utime[index],
            "attitude",
            "constraint_violation",
            (
                f"mode {mode.name if mode is not None else self.mode[index]} "
                f"obsid {obsid} violates {constraint_name} ({scope_label}); "
                f"ra={float(self.ra[index]):.3f} "
                f"dec={float(self.dec[index]):.3f} "
                f"roll={float(self.roll[index]):.3f}"
            ),
            obsid=obsid,
        )

    def _unknown_mode_mismatch(self, index: int) -> PlanExecutionMismatch:
        """Build a mismatch for a telemetry sample whose mode could not be resolved."""
        return self._mismatch(
            self.utime[index],
            "attitude",
            "unknown_mode",
            f"cannot apply constraint scopes for mode {self.mode[index]}",
            obsid=int(self.obsid[index]),
        )

    def _science_entry_mismatches(
        self, entry: PlanEntry, tolerance_deg: float
    ) -> list[PlanExecutionMismatch]:
        """Check telemetry over a science entry's window against expected mode, obsid, and pointing."""
        start = self._science_start_time(entry)
        end = (
            min(entry.end, entry.collection_end)
            if entry.collection_end is not None
            else float(entry.end)
        )
        mismatches: list[PlanExecutionMismatch] = []
        if end <= start:
            return mismatches

        samples = 0
        science_samples = 0
        for i in self._window_indices(start, end):
            utime = self.utime[i]
            samples += 1
            mode = self._mode_at_index(i)
            if mode == ACSMode.SAA:
                continue
            if mode != ACSMode.SCIENCE:
                mismatches.append(
                    self._mode_mismatch(entry, utime, mode, ACSMode.SCIENCE, "science")
                )
                continue

            science_samples += 1
            if int(self.obsid[i]) != int(entry.obsid):
                mismatches.append(
                    self._obsid_mismatch(entry, utime, self.obsid[i], "science")
                )

            expected_attitude = entry.spacecraft_attitude or (
                entry.ra,
                entry.dec,
                entry.roll,
            )
            error_deg = angular_separation(
                float(self.ra[i]),
                float(self.dec[i]),
                float(expected_attitude[0]),
                float(expected_attitude[1]),
            )
            if error_deg > tolerance_deg:
                mismatches.append(
                    self._pointing_mismatch(entry, utime, error_deg, "science")
                )

        if samples > 0 and science_samples == 0:
            mismatches.append(
                self._mismatch(
                    start,
                    "science",
                    "no_science_execution",
                    f"obsid {int(entry.obsid)}",
                    obsid=int(entry.obsid),
                )
            )
        return mismatches

    def _gsp_entry_mismatches(
        self, entry: PlanEntry, tolerance_deg: float
    ) -> list[PlanExecutionMismatch]:
        """Check telemetry over a GSP entry's contact window against expected mode, obsid, and antenna pointing."""
        contact_begin = entry.contact_begin
        contact_end = entry.contact_end
        start = (
            float(contact_begin)
            if contact_begin is not None
            else self._science_start_time(entry)
        )
        end = float(contact_end) if contact_end is not None else float(entry.end)
        mismatches: list[PlanExecutionMismatch] = []
        if end <= start:
            return mismatches

        gspass = matching_pass_for_entry(entry, self.passes)
        missing_profile = gspass is None or not gspass.ra or not gspass.dec
        if missing_profile:
            mismatches.append(
                self._mismatch(
                    start,
                    "contact",
                    "pass_profile_missing",
                    f"obsid {int(entry.obsid)} station {entry.station}",
                    obsid=int(entry.obsid),
                )
            )

        for i in self._window_indices(start, end):
            utime = self.utime[i]
            mode = self._mode_at_index(i)
            if mode != ACSMode.PASS:
                mismatches.append(
                    self._mode_mismatch(entry, utime, mode, ACSMode.PASS, "contact")
                )

            if int(self.obsid[i]) != int(entry.obsid):
                mismatches.append(
                    self._obsid_mismatch(entry, utime, self.obsid[i], "contact")
                )

            if missing_profile:
                continue
            assert gspass is not None
            expected_ra, expected_dec, expected_roll = gspass.attitude_at(utime)
            if expected_ra is None or expected_dec is None:
                continue

            actual_quat = attitude_to_quat(
                float(self.ra[i]), float(self.dec[i]), float(self.roll[i])
            )
            expected_quat = attitude_to_quat(
                float(expected_ra), float(expected_dec), float(expected_roll)
            )
            dot = min(1.0, abs(float(np.dot(actual_quat, expected_quat))))
            error_deg = float(np.rad2deg(2.0 * np.arccos(dot)))
            antenna_error = gspass.antenna_pointing_error(
                float(self.ra[i]), float(self.dec[i]), float(self.roll[i]), utime
            )
            if antenna_error is not None:
                error_deg = max(error_deg, antenna_error)
            if error_deg > tolerance_deg:
                mismatches.append(
                    self._pointing_mismatch(entry, utime, error_deg, "contact")
                )
        return mismatches


def matching_pass_for_entry(entry: PlanEntry, passes: Sequence[Pass]) -> Pass | None:
    """Find the pass matching a GSP plan entry's station and contact window."""
    station = entry.station
    contact_begin = entry.contact_begin
    contact_end = entry.contact_end
    for gspass in passes:
        if station is not None and gspass.station != station:
            continue
        begin_matches = (
            contact_begin is None
            or abs(float(gspass.begin) - float(contact_begin)) <= 1e-6
        )
        end_matches = (
            contact_end is None or abs(float(gspass.end) - float(contact_end)) <= 1e-6
        )
        if begin_matches and end_matches:
            return gspass
    return None
