from collections import Counter
from datetime import datetime

import numpy as np

from ..common import ChargeState, normalize_acs_mode
from ..config.config import MissionConfig
from ..config.data_storage import DataStorage
from ..simulation.acs import ACS
from ..simulation.coverage import StationCoverage
from ..simulation.passes import PassTimes
from .telemetry import Telemetry


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
            views = (
                self.config.recorder_views()
                if isinstance(self.config.data_storage, DataStorage)
                else {"default": self.config}
            )
            if isinstance(self.config.data_storage, DataStorage):
                for name, view in views.items():
                    print(
                        f"Recorder {name}: capacity {view.recorder.capacity_gb:.2f} Gb, "
                        f"final volume {view.recorder.current_volume_gb:.2f} Gb"
                    )
                print(
                    "Aggregate volumes below are totals; fill and alert are the worst store."
                )
            else:
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
                for name, view in views.items():
                    print(
                        f"  {name}: yellow {view.recorder.yellow_threshold:.0%}, red {view.recorder.red_threshold:.0%}"
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

        # Usable links during passes without spacecraft tracking
        if self.coverage is not None:
            print("\n" + "-" * 70)
            print("GROUND STATION PASS STATISTICS")
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
