"""Fault Management System for Spacecraft Operations.

This module provides an extensible fault monitoring and response system that:
- Monitors multiple parameters against configurable yellow/red thresholds
- Tracks time spent in each fault state (nominal, yellow, red)
- Automatically triggers safe mode on RED conditions
- Supports both "below" and "above" threshold directions

Configuration Example (JSON):
    {
        "fault_management": {
            "thresholds": {
                "battery_level": {
                    "name": "battery_level",
                    "yellow": 0.5,
                    "red": 0.4,
                    "direction": "below"
                },
                "temperature": {
                    "name": "temperature",
                    "yellow": 50.0,
                    "red": 60.0,
                    "direction": "above"
                }
            },
            "states": {},
            "safe_mode_on_red": true
        }
    }

Usage Example (Python):
    from conops.fault_management import FaultManagement

    # Create fault management system
    fm = FaultManagement()

    # Add thresholds programmatically
    fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
    fm.add_threshold("temperature", yellow=50.0, red=60.0, direction="above")

    # Check parameters each simulation cycle
    classifications = fm.check(
        values={"battery_level": 0.45, "temperature": 55.0},
        utime=current_time,
        step_size=1.0,
        acs=spacecraft_acs
    )

    # Get accumulated statistics
    stats = fm.statistics()
    # Returns: {"battery_level": {"yellow_seconds": 120.0, "red_seconds": 0.0, "current": "yellow"}, ...}

Threshold Directions:
    - "below": Fault triggered when value <= threshold (e.g., battery_level)
    - "above": Fault triggered when value >= threshold (e.g., temperature, power_draw)

Safe Mode Behavior:
    When safe_mode_on_red=True (default), any parameter reaching RED state will:
    1. Enqueue an ACSCommand to enter safe mode
    2. Safe mode is irreversible once entered
    3. Spacecraft points solar panels at Sun for maximum power generation
    4. All queued commands are cleared
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class FaultState:
    current: str = "nominal"  # nominal | yellow | red
    yellow_seconds: float = 0.0
    red_seconds: float = 0.0


class FaultThreshold(BaseModel):
    """Threshold configuration for a single monitored parameter.

    Attributes:
        name: Parameter name (e.g. 'battery_level').
        yellow: Value at/ beyond which a YELLOW fault is flagged.
        red: Value at/ beyond which a RED fault is flagged.
        direction: 'below' or 'above' indicating fault when value passes *below* or *above* limit.
    """

    name: str
    yellow: float
    red: float
    direction: str = Field(default="below")  # 'below' or 'above'

    def classify(self, value: float) -> str:
        """Return nominal|yellow|red for the given value."""
        if self.direction == "below":
            if value <= self.red:
                return "red"
            if value <= self.yellow:
                return "yellow"
            return "nominal"
        else:  # direction == 'above'
            if value >= self.red:
                return "red"
            if value >= self.yellow:
                return "yellow"
            return "nominal"


class FaultManagement(BaseModel):
    """Extensible Fault Management system.

    Monitors configured parameters each simulation cycle, classifies them
    into nominal / yellow / red states, records time spent in each state,
    and triggers ACS safe mode entry on RED conditions (once) where configured.
    """

    thresholds: dict[str, FaultThreshold] = Field(default_factory=dict)
    states: dict[str, FaultState] = Field(default_factory=dict)
    safe_mode_on_red: bool = True  # Global policy: enter safe mode for any RED

    def ensure_state(self, name: str) -> FaultState:
        if name not in self.states:
            self.states[name] = FaultState()
        return self.states[name]

    def check(
        self,
        values: dict[str, float],
        utime: float,
        step_size: float,
        acs: ACS | None = None,
    ) -> dict[str, str]:
        """Evaluate all monitored parameters.

        Args:
            values: Mapping of parameter name -> current numeric value.
            utime: Current unix time of simulation.
            step_size: Simulation time step in seconds (used for duration accumulation).
            acs: ACS instance to trigger safe mode if needed.

        Returns:
            Dict mapping parameter name to classification string.
        """
        classifications: dict[str, str] = {}
        for name, val in values.items():
            thresh = self.thresholds.get(name)
            if thresh is None:
                continue  # Not monitored
            state = thresh.classify(val)
            classifications[name] = state
            st = self.ensure_state(name)
            # Accumulate time
            if state == "yellow":
                st.yellow_seconds += step_size
            elif state == "red":
                st.red_seconds += step_size
            st.current = state
            # Safe mode trigger
            if (
                state == "red"
                and self.safe_mode_on_red
                and acs is not None
                and not acs.in_safe_mode
            ):
                # Enqueue a safe mode command directly (avoid wrapper)
                from .acs import ACSCommand, ACSCommandType  # local import

                command = ACSCommand(
                    command_type=ACSCommandType.ENTER_SAFE_MODE,
                    execution_time=utime,
                )
                acs.enqueue_command(command)

        return classifications

    def statistics(self) -> dict[str, dict[str, float | str]]:
        """Return accumulated statistics for all parameters."""
        return {
            name: {
                "yellow_seconds": st.yellow_seconds,
                "red_seconds": st.red_seconds,
                "current": st.current,
            }
            for name, st in self.states.items()
        }

    def add_threshold(
        self, name: str, yellow: float, red: float, direction: str = "below"
    ) -> None:
        self.thresholds[name] = FaultThreshold(
            name=name, yellow=yellow, red=red, direction=direction
        )


# Lazy import for type checking to avoid circular import
try:  # pragma: no cover
    from .acs import ACS  # noqa: F401
except Exception:  # pragma: no cover
    pass
