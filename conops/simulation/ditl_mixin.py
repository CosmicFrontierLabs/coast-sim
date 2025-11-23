from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import rust_ephem

from ..config import Config
from .acs import ACS
from .passes import PassTimes
from .plan_entry import PlanEntry
from .ppst import Plan


class DITLMixin:
    ppt: PlanEntry | None
    ra: list[float]
    dec: list[float]
    roll: list[float]
    mode: list[int]
    panel: list[float]
    power: list[float]
    panel_power: list[float]
    batterylevel: list[float]
    charge_state: list[int]
    obsid: list[int]
    ppst: Plan
    utime: list
    ephem: rust_ephem.TLEEphemeris | None
    # Subsystem power tracking
    power_bus: list[float]
    power_payload: list[float]

    def __init__(self, config: Config) -> None:
        # Defining telemetry data points
        self.config = config
        self.ra = []
        self.dec = []
        self.utime = []
        self.mode = []
        self.obsid = []
        self.ephem = None
        # Defining when the model is run
        self.begin = datetime(
            2018, 11, 27, 0, 0, 0, tzinfo=timezone.utc
        )  # Default: Nov 27, 2018 (day 331)
        self.end = datetime(
            2018, 11, 28, 0, 0, 0, tzinfo=timezone.utc
        )  # Default: 1 day later
        self.step_size = 60  # seconds
        self.ustart = 0.0  # Calculate these
        self.uend = 0.0  # later
        self.ppst = Plan()
        self.saa = None
        self.passes = PassTimes(constraint=self.config.constraint, config=config)
        self.executed_passes = PassTimes(
            constraint=self.config.constraint, config=config
        )

        # Set up event based ACS
        assert self.config.constraint.ephem is not None, (
            "Ephemeris must be set in Config Constraint"
        )
        self.acs = ACS(constraint=self.config.constraint, config=self.config)

        # Current target
        self.ppt = None

    def plot(self) -> None:
        """Plot DITL timeline"""
        timehours = (np.array(self.utime) - self.utime[0]) / 3600

        _ = plt.figure(figsize=(10, 8))
        ax = plt.subplot(711)
        plt.plot(timehours, self.ra)
        ax.xaxis.set_visible(False)
        plt.ylabel("RA")
        ax.set_title(f"Timeline for DITL Simulation: {self.config.name}")

        ax = plt.subplot(712)
        ax.plot(timehours, self.dec)
        ax.xaxis.set_visible(False)

        plt.ylabel("Dec")
        ax = plt.subplot(713)
        ax.plot(timehours, self.mode)
        ax.xaxis.set_visible(False)

        plt.ylabel("Mode")
        ax = plt.subplot(714)
        ax.plot(timehours, self.batterylevel)
        ax.axhline(
            y=self.config.battery.max_depth_of_discharge, color="r", linestyle="--"
        )
        ax.xaxis.set_visible(False)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Batt. charge")

        ax = plt.subplot(715)
        ax.plot(timehours, self.panel)
        ax.xaxis.set_visible(False)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Panel Ill.")

        ax = plt.subplot(716)
        # Check if subsystem power data is available
        if (
            hasattr(self, "power_bus")
            and hasattr(self, "power_payload")
            and self.power_bus
            and self.power_payload
        ):
            # Line plot showing power breakdown
            ax.plot(timehours, self.power_bus, label="Bus", alpha=0.8)
            ax.plot(timehours, self.power_payload, label="Payload", alpha=0.8)
            ax.plot(timehours, self.power, label="Total", linewidth=2, alpha=0.9)
            ax.legend(loc="upper right", fontsize="small")
        else:
            # Fall back to total power only
            ax.plot(timehours, self.power, label="Total")
        ax.set_ylim(0, max(self.power) * 1.1)
        ax.set_ylabel("Power (W)")
        ax.xaxis.set_visible(False)

        ax = plt.subplot(717)
        ax.plot(timehours, self.obsid)
        ax.set_ylabel("ObsID")
        ax.set_xlabel("Time (hour of day)")

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
        from collections import Counter

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
            if self.roll:
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

        if self.power:
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

        if self.panel_power:
            print("\nSolar Panel Generation:")
            print(f"  Average: {np.mean(self.panel_power):.2f} W")
            print(f"  Peak: {max(self.panel_power):.2f} W")
            total_generated = sum(self.panel_power) * self.step_size / 3600  # Wh
            total_consumed = sum(self.power) * self.step_size / 3600  # Wh
            print(f"  Total Generated: {total_generated:.2f} Wh")
            print(f"  Total Consumed: {total_consumed:.2f} Wh")
            print(f"  Net Energy: {total_generated - total_consumed:.2f} Wh")

        if self.panel:
            print("\nSolar Panel Illumination:")
            avg_illumination = np.mean(self.panel) * 100
            print(f"  Average: {avg_illumination:.1f}%")
            eclipse_steps = sum(1 for p in self.panel if p < 0.01)
            print(
                f"  Eclipse Time: {eclipse_steps * self.step_size / 3600:.2f} hours ({eclipse_steps / len(self.panel) * 100:.1f}%)"
            )

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
