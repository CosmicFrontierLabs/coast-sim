from collections import Counter
from datetime import datetime

import numpy as np

from ..config.config import MissionConfig


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
        from ..common import ACSMode

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
                mode_name = (
                    ACSMode(mode_val).name
                    if mode_val in [m.value for m in ACSMode]
                    else f"UNKNOWN({mode_val})"
                )
                percentage = (count / total_steps) * 100
                time_hours = (count * self.step_size) / 3600
                print(
                    f"{mode_name:<20} {count:<10} {percentage:>6.2f}%      {time_hours:>10.2f}"
                )

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
            if hasattr(self, "roll") and self.roll:
                print(f"Roll Range: {min(self.roll):.2f}° to {max(self.roll):.2f}°")

        # Battery statistics
        print("\n" + "-" * 70)
        print("POWER AND BATTERY STATISTICS")
        print("-" * 70)
        if self.batterylevel:
            battery_capacity = getattr(
                self.config.battery,
                "watthour",
                getattr(self.config.battery, "capacity", None),
            )
            if battery_capacity is not None:
                print(f"Battery Capacity: {battery_capacity:.2f} Wh")
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
            from ..common import ChargeState

            print("\nBattery Charging State Distribution:")
            charge_state_counts = Counter(self.charge_state)
            total_steps = len(self.charge_state)
            print(
                f"{'State':<20} {'Count':<10} {'Percentage':<12} {'Time (hours)':<15}"
            )
            print("-" * 70)
            for state_val, count in sorted(charge_state_counts.items()):
                state_name = (
                    ChargeState(state_val).name
                    if state_val in [s.value for s in ChargeState]
                    else f"UNKNOWN({state_val})"
                )
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
        if (
            hasattr(self, "recorder_volume_gb")
            and self.recorder_volume_gb
            and self.config.recorder is not None
        ):
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

            if self.data_generated_gb:
                total_generated = (
                    self.data_generated_gb[-1] if self.data_generated_gb else 0
                )
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
                if hasattr(self, "data_generated_gb") and self.data_generated_gb:
                    total_generated = self.data_generated_gb[-1]
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
        if hasattr(self, "acs") and hasattr(self.acs, "commands"):
            print("\n" + "-" * 70)
            print("ACS COMMAND STATISTICS")
            print("-" * 70)
            cmd_counts = Counter([cmd.command_type for cmd in self.acs.commands])
            print(f"Total ACS Commands: {len(self.acs.commands)}")
            print(f"\n{'Command Type':<25} {'Count':<10}")
            print("-" * 35)
            for cmd_type, count in sorted(
                cmd_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"{cmd_type.name:<25} {count:<10}")

        # Ground station pass statistics (if available)
        if hasattr(self, "executed_passes") and len(self.executed_passes.passes) > 0:
            print("\n" + "-" * 70)
            print("GROUND STATION PASS STATISTICS")
            print("-" * 70)
            print(f"Total Passes Executed: {len(self.executed_passes.passes)}")
            total_pass_time = (
                sum((p.end - p.begin) for p in self.executed_passes.passes) / 3600
            )
            print(f"Total Pass Time: {total_pass_time:.2f} hours")

        print("\n" + "=" * 70)
