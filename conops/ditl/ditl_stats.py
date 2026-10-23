from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from ..common import ChargeState, normalize_acs_mode, unixtime2date
from ..config.config import MissionConfig
from ..config.observation_categories import ObservationCategories
from ..simulation.acs import ACS
from ..simulation.coverage import StationCoverage
from ..simulation.passes import PassTimes
from ..targets import Plan, PlanEntry, Pointing
from .plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from .telemetry import Telemetry


@dataclass(frozen=True)
class CategorySummary:
    """What one observation category got over a run."""

    category: str
    program: str
    targets: int
    """Distinct targets observed."""
    visits: int
    """Visits taken: a snapshot, or one visit of a sequence, each."""
    science_hours: float
    slew_hours: float
    """Hours slewing into the category's visits."""
    slew_degrees: float
    """Degrees slewed into the category's visits."""
    share: float
    """The category's fraction of the run's science time."""
    time_share: float | None
    """The fraction allocated to it, where the category sets one."""


@dataclass(frozen=True)
class VisitRecord:
    """One visit of a target as executed, against its window where it has one."""

    obsid: int
    name: str
    category: str
    visit: int | None
    """The visit window it was taken in, for a target with windows."""
    window_start: float | None
    window_deadline: float | None
    begin: float | None
    """When the entry began, with its slew; None for a visit not taken."""
    collection_begin: float | None
    end: float | None
    exposure_seconds: float
    slew_degrees: float
    roll_deg: float | None
    status: str
    """``complete``, ``cut`` (less science than planned) or ``missed`` (a
    window with no visit)."""
    attempts: int = 1
    """How many times the visit was begun, in a per-window record: a visit
    cut short and taken again within its window counts each attempt."""


@dataclass(frozen=True)
class _Visit:
    entries: list[PlanEntry]

    @property
    def first(self) -> PlanEntry:
        return self.entries[0]

    @property
    def exposure(self) -> float:
        return float(sum(e.exposure for e in self.entries if not e.calibration))

    @property
    def planned(self) -> float:
        return float(sum(float(e.ss_max) for e in self.entries if not e.calibration))


class DITLStats:
    """Mixin class providing statistics printing functionality for DITL simulations."""

    config: MissionConfig
    begin: datetime
    end: datetime
    step_size: int
    utime: list[float]
    mode: list[int]
    ra: list[float]
    dec: list[float]
    roll: list[float]
    obsid: list[int]
    panel: list[float]
    power: list[float]
    batterylevel: list[float]
    recorder_fill_fraction: list[float]
    data_generated_gb: list[float]
    data_downlinked_gb: list[float]
    executed_passes: PassTimes
    coverage: StationCoverage | None
    repoints_in_observations: list[tuple[float, int, str]]
    acs: ACS
    telemetry: Telemetry

    # ── Reporting ──────────────────────────────────────────────────────

    def _requests(self) -> list[Pointing]:
        """The targets the run was given, where it keeps them."""
        queue = getattr(self, "queue", None)
        targets = getattr(queue, "targets", None) or getattr(self, "targets", None)
        return [t for t in (targets or []) if isinstance(t, Pointing)]

    def _science_visits(self) -> list[_Visit]:
        """The run's science visits, in time order: each snapshot entry, or
        the step entries of one visit of a sequence, as one. Dispatch begins
        a visit's next step at the simulation step after the last ended, so
        steps up to a step apart belong to the same visit."""
        plan = getattr(self, "plan", None)
        if not isinstance(plan, Plan):
            return []
        step = float(getattr(self, "step_size", 0) or 0)
        entries = sorted(
            (
                e
                for e in plan
                if entry_obstype(e) in PLAN_SCIENCE_OBSTYPES and e.obsid is not None
            ),
            key=lambda e: (float(e.begin), float(e.end)),
        )
        visits: list[_Visit] = []
        for entry in entries:
            last = visits[-1] if visits else None
            if (
                last is not None
                and entry.step is not None
                and last.first.step is not None
                and int(last.first.obsid) == int(entry.obsid)
                and last.first.visit == entry.visit
                and float(entry.begin) <= float(last.entries[-1].end) + step + 1e-6
            ):
                last.entries.append(entry)
            else:
                visits.append(_Visit([entry]))
        return visits

    def summary_by_category(self) -> list[CategorySummary]:
        """What each observation category got over the run, from the plan
        as executed: targets, visits, science and slew hours, the degrees
        slewed into its visits, its share of the run's science time, and
        the fraction allocated to it where the category sets one. One row
        per configured category, in their order, a category that collected
        nothing included, then any category science fell to outside them."""
        categories = getattr(self.config, "observation_categories", None)
        # A test double's configuration has no categories.
        if not isinstance(categories, ObservationCategories):
            return []
        visits = self._science_visits()
        by_name: dict[str, list[_Visit]] = {
            category.name: [] for category in categories.categories
        }
        for visit in visits:
            name = categories.get_category(int(visit.first.obsid)).name
            by_name.setdefault(name, []).append(visit)
        total = sum(v.exposure for v in visits)
        rows = []
        for name, taken in by_name.items():
            category = next(
                (c for c in categories.categories if c.name == name),
                categories.get_category(int(taken[0].first.obsid)) if taken else None,
            )
            if category is None:
                continue
            science = sum(v.exposure for v in taken)
            rows.append(
                CategorySummary(
                    category=name,
                    program=category.program_name,
                    targets=len({int(v.first.obsid) for v in taken}),
                    visits=len(taken),
                    science_hours=science / 3600.0,
                    slew_hours=sum(float(v.first.slewtime) for v in taken) / 3600.0,
                    slew_degrees=sum(float(v.first.slewdist) for v in taken),
                    share=science / total if total > 0 else 0.0,
                    time_share=category.time_share,
                )
            )
        return rows

    def visit_table(
        self, targets: Sequence[Pointing] | None = None, *, per_window: bool = False
    ) -> list[VisitRecord]:
        """Each visit of the run against its plan: the window it was taken
        in and when, its science against what was planned, the slew into it
        and its roll. ``targets``, or the run's own requests, give the
        windows; a window of theirs with no visit is a ``missed`` row.
        A visit with less science than planned, less a step, is ``cut``.

        Every attempt is a row, so a visit cut short and taken again within
        its window is two. With ``per_window`` a window is one row, its
        verdict: the whole visit where one was taken, with the attempts it
        took, else the last attempt, cut; a target without windows keeps a
        row per snapshot."""
        categories = getattr(self.config, "observation_categories", None)
        if not isinstance(categories, ObservationCategories):
            return []
        requests = list(targets) if targets is not None else self._requests()
        by_obsid = {int(t.obsid): t for t in requests}
        step = float(getattr(self, "step_size", 0) or 0)
        rows: list[VisitRecord] = []
        taken: dict[tuple[int, int], bool] = {}
        for visit in self._science_visits():
            first = visit.first
            obsid = int(first.obsid)
            target = by_obsid.get(obsid)
            windows = getattr(target, "visit_windows", None) or getattr(
                first, "visit_windows", None
            )
            index = first.visit
            window = (
                windows[index]
                if windows and index is not None and 0 <= index < len(windows)
                else None
            )
            planned = visit.planned
            if target is not None and target.visit_windows:
                planned = float(target.per_visit_seconds)
            exposure = visit.exposure
            status = "cut" if exposure < planned - step - 1e-6 else "complete"
            if index is not None and status == "complete":
                taken[(obsid, index)] = True
            rows.append(
                VisitRecord(
                    obsid=obsid,
                    name=str(first.name),
                    category=categories.get_category(obsid).name,
                    visit=index,
                    window_start=window.start if window is not None else None,
                    window_deadline=window.deadline if window is not None else None,
                    begin=float(first.begin),
                    collection_begin=(
                        float(first.collection_begin)
                        if first.collection_begin is not None
                        else None
                    ),
                    end=float(visit.entries[-1].end),
                    exposure_seconds=exposure,
                    slew_degrees=float(first.slewdist),
                    roll_deg=float(first.roll) if first.roll != -1.0 else None,
                    status=status,
                )
            )
        for target in requests:
            for index, window in enumerate(target.visit_windows):
                obsid = int(target.obsid)
                if (obsid, index) in taken:
                    continue
                if any(r.obsid == obsid and r.visit == index for r in rows):
                    continue
                rows.append(
                    VisitRecord(
                        obsid=obsid,
                        name=str(target.name),
                        category=categories.get_category(obsid).name,
                        visit=index,
                        window_start=window.start,
                        window_deadline=window.deadline,
                        begin=None,
                        collection_begin=None,
                        end=None,
                        exposure_seconds=0.0,
                        slew_degrees=0.0,
                        roll_deg=None,
                        status="missed",
                    )
                )
        if per_window:
            rows = self._per_window(rows)
        return sorted(
            rows,
            key=lambda r: (
                r.begin if r.begin is not None else (r.window_start or 0.0),
                r.obsid,
            ),
        )

    @staticmethod
    def _per_window(rows: list[VisitRecord]) -> list[VisitRecord]:
        """One record per visit window: the whole visit taken in it, with
        the attempts it took, else the last attempt."""
        by_window: dict[tuple[int, int], list[VisitRecord]] = {}
        verdicts: list[VisitRecord] = []
        for record in rows:
            if record.visit is None or record.window_start is None:
                verdicts.append(record)
                continue
            by_window.setdefault((record.obsid, record.visit), []).append(record)
        for attempts in by_window.values():
            taken = [r for r in attempts if r.status == "complete"]
            verdict = taken[0] if taken else attempts[-1]
            begun = [r for r in attempts if r.status != "missed"]
            verdicts.append(replace(verdict, attempts=len(begun)))
        return verdicts

    def _print_reporting(self) -> None:
        """Print the category summary and, for targets with visit windows,
        the visit table."""
        rows = self.summary_by_category()
        if rows:
            print("\n" + "-" * 70)
            print("SCIENCE BY CATEGORY")
            print("-" * 70)
            print(
                f"{'Category':<18} {'Program':<12} {'Targets':>7} {'Visits':>6} "
                f"{'Science h':>9} {'Slew h':>6} {'Slew deg':>8} {'Share':>6} {'Alloc':>6}"
            )
            print("-" * 70)
            for row in rows:
                allocated = (
                    f"{row.time_share:>6.2f}"
                    if row.time_share is not None
                    else "     -"
                )
                print(
                    f"{row.category:<18} {row.program:<12} {row.targets:>7} "
                    f"{row.visits:>6} {row.science_hours:>9.2f} {row.slew_hours:>6.2f} "
                    f"{row.slew_degrees:>8.1f} {row.share:>6.2f} {allocated}"
                )
        visits = [
            r for r in self.visit_table(per_window=True) if r.window_start is not None
        ]
        if visits:
            print("\n" + "-" * 70)
            print("VISITS IN WINDOWS")
            print("-" * 70)
            print(
                f"{'Target':<16} {'Visit':>5} {'Window opens':<18} {'Collects':<18} "
                f"{'Science s':>9} {'Status':<8} {'Tries':>5}"
            )
            print("-" * 70)
            for record in visits:
                collects = (
                    unixtime2date(record.collection_begin)
                    if record.collection_begin is not None
                    else "-"
                )
                opens = (
                    unixtime2date(record.window_start) if record.window_start else "-"
                )
                print(
                    f"{record.name[:16]:<16} {(record.visit or 0) + 1:>5} {opens:<18} "
                    f"{collects:<18} {record.exposure_seconds:>9.0f} {record.status:<8} "
                    f"{record.attempts:>5}"
                )

    def instrument_mode_hours(self) -> dict[str, dict[str, float]]:
        """Hours each instrument with modes spent in each mode over the run,
        by instrument then mode, from the telemetry: each step counts whole
        in the mode the instrument is in at its start. Empty when no
        instrument defines modes."""
        hours: dict[str, dict[str, float]] = {}
        samples = getattr(getattr(self, "telemetry", None), "housekeeping", None)
        if not isinstance(samples, list):
            return hours
        step_hours = float(self.step_size) / 3600.0
        for sample in samples:
            modes = getattr(sample, "instrument_modes", None)
            if not isinstance(modes, dict):
                continue
            for instrument, mode in modes.items():
                by_mode = hours.setdefault(instrument, {})
                by_mode[mode] = by_mode.get(mode, 0.0) + step_hours
        return hours

    def print_statistics(self) -> None:
        """Print comprehensive statistics about the DITL simulation.

        Displays information about:
        - Simulation time period and duration
        - Mode distribution (time spent in each ACS mode)
        - Observation statistics (unique targets, total observations)
        - Power and battery statistics
        - Solar panel performance
        - Queue information (if available)
        - ACS commands (if available)
        - Ground station pass statistics (if available)
        """

        # Basic simulation info
        print("=" * 70)
        print("DITL SIMULATION STATISTICS")
        print("=" * 70)
        print(f"\nConfiguration: {self.config.name}")
        print(f"Start Time: {self.begin.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"End Time: {self.end.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        duration_hours = (self.end - self.begin).total_seconds() / 3600
        print(f"Duration: {duration_hours:.2f} hours ({duration_hours / 24:.2f} days)")
        print(f"Time Steps: {len(self.utime)}")
        print(f"Step Size: {self.step_size} seconds")

        # Mode statistics
        print("\n" + "-" * 70)
        print("MODE DISTRIBUTION")
        print("-" * 70)
        if self.mode:
            mode_counts = Counter(self.mode)
            total_steps = len(self.mode)
            print(f"{'Mode':<20} {'Count':<10} {'Percentage':<12} {'Time (hours)':<15}")
            print("-" * 70)
            for mode_val, count in sorted(mode_counts.items()):
                acs_mode = normalize_acs_mode(mode_val)
                mode_name = (
                    acs_mode.name if acs_mode is not None else f"UNKNOWN({mode_val})"
                )
                percentage = (count / total_steps) * 100
                time_hours = (count * self.step_size) / 3600
                print(
                    f"{mode_name:<20} {count:<10} {percentage:>6.2f}%      {time_hours:>10.2f}"
                )

        # Instrument modes
        mode_hours = self.instrument_mode_hours()
        if mode_hours:
            print("\n" + "-" * 70)
            print("INSTRUMENT MODE HOURS")
            print("-" * 70)
            print(f"{'Instrument':<24} {'Mode':<16} {'Time (hours)':<15}")
            print("-" * 70)
            for instrument, by_mode in mode_hours.items():
                for mode_name, time_hours in sorted(
                    by_mode.items(), key=lambda item: item[1], reverse=True
                ):
                    print(f"{instrument:<24} {mode_name:<16} {time_hours:>10.2f}")

        self._print_reporting()

        # Observation statistics
        print("\n" + "-" * 70)
        print("OBSERVATION STATISTICS")
        print("-" * 70)
        if self.obsid:
            unique_obsids = set(self.obsid)
            # Filter out special ObsIDs (like 0 or 999xxx for charging)
            science_obsids = [o for o in unique_obsids if o > 0 and o < 999000]
            print(f"Total Unique Observations: {len(science_obsids)}")
            print(
                f"Total Observation Steps: {sum(1 for o in self.obsid if 0 < o < 999000)}"
            )

            if science_obsids:
                # Count time per obsid
                obsid_counts = Counter([o for o in self.obsid if 0 < o < 999000])
                print("\nTop 10 Observations by Time:")
                print(f"{'ObsID':<10} {'Steps':<10} {'Time (hours)':<15}")
                print("-" * 35)
                for obsid, count in obsid_counts.most_common(10):
                    time_hours = (count * self.step_size) / 3600
                    print(f"{obsid:<10} {count:<10} {time_hours:>10.2f}")

        # Pointing statistics
        print("\n" + "-" * 70)
        print("POINTING STATISTICS")
        print("-" * 70)
        if self.ra and self.dec:
            print(f"Total Pointing Updates: {len(self.ra)}")
            print(f"RA Range: {min(self.ra):.2f}° to {max(self.ra):.2f}°")
            print(f"Dec Range: {min(self.dec):.2f}° to {max(self.dec):.2f}°")
            if self.roll:
                print(f"Roll Range: {min(self.roll):.2f}° to {max(self.roll):.2f}°")

        # Battery statistics
        print("\n" + "-" * 70)
        print("POWER AND BATTERY STATISTICS")
        print("-" * 70)
        if self.batterylevel:
            print(f"Battery Capacity: {self.config.battery.watthour:.2f} Wh")
            print(f"Initial Charge: {self.batterylevel[0] * 100:.1f}%")
            print(f"Final Charge: {self.batterylevel[-1] * 100:.1f}%")
            print(f"Min Charge: {min(self.batterylevel) * 100:.1f}%")
            print(f"Max Charge: {max(self.batterylevel) * 100:.1f}%")
            print(f"Avg Charge: {np.mean(self.batterylevel) * 100:.1f}%")
            max_dod = self.config.battery.max_depth_of_discharge
            print(f"Max Depth of Discharge: {max_dod * 100:.1f}%")
            violations = sum(1 for bl in self.batterylevel if bl < max_dod)
            if violations > 0:
                print(
                    f"⚠️  DoD Violations: {violations} steps ({violations / len(self.batterylevel) * 100:.2f}%)"
                )

        # Charge state statistics
        if hasattr(self, "charge_state") and self.charge_state:
            print("\nBattery Charging State Distribution:")
            charge_state_counts = Counter(self.charge_state)
            total_steps = len(self.charge_state)
            print(
                f"{'State':<20} {'Count':<10} {'Percentage':<12} {'Time (hours)':<15}"
            )
            print("-" * 70)
            for state_val, count in sorted(charge_state_counts.items()):
                try:
                    state_name = ChargeState(state_val).name
                except ValueError:
                    state_name = f"UNKNOWN({state_val})"
                percentage = (count / total_steps) * 100
                time_hours = (count * self.step_size) / 3600
                print(
                    f"{state_name:<20} {count:<10} {percentage:>6.2f}%      {time_hours:>10.2f}"
                )

        if hasattr(self, "power") and self.power:
            print("\nPower Consumption:")
            print(f"  Average: {np.mean(self.power):.2f} W")
            print(f"  Peak: {max(self.power):.2f} W")
            print(f"  Minimum: {min(self.power):.2f} W")

            # Subsystem power breakdown if available
            if (
                hasattr(self, "power_bus")
                and hasattr(self, "power_payload")
                and self.power_bus
                and self.power_payload
            ):
                print("\n  Subsystem Breakdown:")
                avg_bus = np.mean(self.power_bus)
                avg_payload = np.mean(self.power_payload)
                print(
                    f"    Bus Average: {avg_bus:.2f} W ({avg_bus / np.mean(self.power) * 100:.1f}%)"
                )
                print(
                    f"    Payload Average: {avg_payload:.2f} W ({avg_payload / np.mean(self.power) * 100:.1f}%)"
                )
                print(f"    Bus Peak: {max(self.power_bus):.2f} W")
                print(f"    Payload Peak: {max(self.power_payload):.2f} W")

        if hasattr(self, "panel_power") and self.panel_power:
            print("\nSolar Panel Generation:")
            print(f"  Average: {np.mean(self.panel_power):.2f} W")
            print(f"  Peak: {max(self.panel_power):.2f} W")
            total_generated = sum(self.panel_power) * self.step_size / 3600  # Wh
            total_consumed = sum(self.power) * self.step_size / 3600  # Wh
            print(f"  Total Generated: {total_generated:.2f} Wh")
            print(f"  Total Consumed: {total_consumed:.2f} Wh")
            print(f"  Net Energy: {total_generated - total_consumed:.2f} Wh")

        if hasattr(self, "panel") and self.panel:
            print("\nSolar Panel Illumination:")
            avg_illumination = np.mean(self.panel) * 100
            print(f"  Average: {avg_illumination:.1f}%")
            eclipse_steps = sum(1 for p in self.panel if p < 0.01)
            print(
                f"  Eclipse Time: {eclipse_steps * self.step_size / 3600:.2f} hours ({eclipse_steps / len(self.panel) * 100:.1f}%)"
            )

        # Data Management statistics
        if hasattr(self, "recorder_volume_gb") and self.recorder_volume_gb:
            print("\n" + "-" * 70)
            print("DATA MANAGEMENT STATISTICS")
            print("-" * 70)
            print(f"Recorder Capacity: {self.config.recorder.capacity_gb:.2f} Gb")
            print(f"Initial Volume: {self.recorder_volume_gb[0]:.2f} Gb")
            print(f"Final Volume: {self.recorder_volume_gb[-1]:.2f} Gb")
            print(f"Peak Volume: {max(self.recorder_volume_gb):.2f} Gb")

            if self.recorder_fill_fraction:
                print("\nFill Level:")
                print(f"  Initial: {self.recorder_fill_fraction[0] * 100:.1f}%")
                print(f"  Final: {self.recorder_fill_fraction[-1] * 100:.1f}%")
                print(f"  Peak: {max(self.recorder_fill_fraction) * 100:.1f}%")
                print(f"  Average: {np.mean(self.recorder_fill_fraction) * 100:.1f}%")

            total_generated = (
                self.data_generated_gb[-1] if self.data_generated_gb else 0.0
            )

            if self.data_generated_gb:
                print(f"\nData Generated: {total_generated:.2f} Gb")

                # Calculate generation rate
                duration_hours = (self.end - self.begin).total_seconds() / 3600
                if duration_hours > 0:
                    avg_rate = total_generated / duration_hours
                    print(
                        f"  Average Rate: {avg_rate:.3f} Gb/hour ({avg_rate * 1000:.2f} Mbps)"
                    )

            if self.data_downlinked_gb:
                total_downlinked = (
                    self.data_downlinked_gb[-1] if self.data_downlinked_gb else 0
                )
                print(f"\nData Downlinked: {total_downlinked:.2f} Gb")

                # Calculate downlink efficiency
                if total_generated > 0:
                    efficiency = (total_downlinked / total_generated) * 100
                    print(f"  Downlink Efficiency: {efficiency:.1f}%")

                # Calculate downlink rate
                duration_hours = (self.end - self.begin).total_seconds() / 3600
                if duration_hours > 0:
                    avg_rate = total_downlinked / duration_hours
                    print(
                        f"  Average Rate: {avg_rate:.3f} Gb/hour ({avg_rate * 1000:.2f} Mbps)"
                    )

            # Recorder alert statistics
            if hasattr(self, "recorder_alert") and self.recorder_alert:
                alert_counts = Counter(self.recorder_alert)
                print("\nRecorder Alerts:")
                print(
                    f"  Yellow Threshold: {self.config.recorder.yellow_threshold * 100:.0f}%"
                )
                print(
                    f"  Red Threshold: {self.config.recorder.red_threshold * 100:.0f}%"
                )

                yellow_count = alert_counts.get(1, 0)  # 1 = yellow alert
                red_count = alert_counts.get(2, 0)  # 2 = red alert
                total_steps = len(self.recorder_alert)

                if yellow_count > 0:
                    yellow_time = yellow_count * self.step_size / 3600
                    print(
                        f"  Yellow Alerts: {yellow_count} steps ({yellow_time:.2f} hours, {yellow_count / total_steps * 100:.1f}%)"
                    )
                else:
                    print("  Yellow Alerts: None")

                if red_count > 0:
                    red_time = red_count * self.step_size / 3600
                    print(
                        f"  Red Alerts: {red_count} steps ({red_time:.2f} hours, {red_count / total_steps * 100:.1f}%)"
                    )
                else:
                    print("  Red Alerts: None")

        # Queue statistics (if available)
        if hasattr(self, "queue"):
            print("\n" + "-" * 70)
            print("TARGET QUEUE STATISTICS")
            print("-" * 70)
            print(f"Total Targets in Queue: {len(self.queue.targets)}")
            completed = sum(1 for t in self.queue.targets if getattr(t, "done", False))
            print(f"Completed Targets: {completed}")
            print(f"Remaining Targets: {len(self.queue.targets) - completed}")

        # ACS Command statistics (if available)
        if self.acs.executed_commands:
            print("\n" + "-" * 70)
            print("ACS COMMAND STATISTICS")
            print("-" * 70)
            cmd_counts = Counter(
                [cmd.command_type for cmd in self.acs.executed_commands]
            )
            print(f"Total ACS Commands: {len(self.acs.executed_commands)}")
            print(f"\n{'Command Type':<25} {'Count':<10}")
            print("-" * 35)
            for cmd_type, count in sorted(
                cmd_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"{cmd_type.name:<25} {count:<10}")

        # Ground station pass statistics (if available)
        if len(self.executed_passes.passes) > 0:
            print("\n" + "-" * 70)
            print("GROUND STATION PASS STATISTICS")
            print("-" * 70)
            print(f"Total Passes Executed: {len(self.executed_passes.passes)}")
            total_pass_time = (
                sum((p.end - p.begin) for p in self.executed_passes.passes) / 3600
            )
            print(f"Total Pass Time: {total_pass_time:.2f} hours")

        # Station coverage for an antenna that needs no attitude
        if self.coverage is not None:
            print("\n" + "-" * 70)
            print("GROUND STATION COVERAGE")
            print("-" * 70)
            begin, end = self.begin.timestamp(), self.end.timestamp()
            spans = self.coverage.spans(begin, end)
            in_contact = sum(span.end - span.begin for span in spans) / 3600
            print(f"Contacts: {len(spans)}")
            print(f"Handovers: {self.coverage.handovers}")
            print(
                f"Time in Contact: {in_contact:.2f} hours "
                f"({100 * self.coverage.fraction(begin, end):.1f}% of the run)"
            )
            repoints = self.coverage.repoints_in(begin, end)
            kinds = Counter(repoint.kind for repoint in repoints)
            print(
                f"Antenna Repoints: {len(repoints)} "
                f"(acquisition {kinds['acquisition']}, handover {kinds['handover']}, "
                f"tracking {kinds['tracking']})"
            )
            turns = [r.maneuver_deg for r in repoints if r.maneuver_deg is not None]
            if turns:
                print(
                    f"Repoint Maneuver: mean {sum(turns) / len(turns):.1f} deg, "
                    f"max {max(turns):.1f} deg"
                )
            noted = getattr(self, "repoints_in_observations", [])
            forbidden = sum(1 for _, _, setting in noted if setting == "forbidden")
            print(
                f"Repoints During Sensitive Observations: {len(noted)} "
                f"({forbidden} in observations that forbid them)"
            )

        print("\n" + "=" * 70)
