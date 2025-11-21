from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import rust_ephem

from .acs import ACS
from .config import Config
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
    obsid: list[int]
    ppst: Plan
    utime: list
    ephem: rust_ephem.TLEEphemeris | None

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
        ax.plot(timehours, self.power)
        ax.set_ylim(0, max(self.power) * 1.1)
        ax.set_ylabel("Power (W)")

        ax = plt.subplot(717)
        ax.plot(timehours, self.obsid)
        ax.set_ylabel("ObsID")
        ax.set_xlabel("Time (hour of day)")
