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
    from datetime import datetime, timezone

    from conops.common import ACSMode
    from conops.config.fault_management import FaultManagement
    from conops.ditl.telemetry import Housekeeping

    # Create fault management system
    fm = FaultManagement()

    # Add thresholds programmatically
    fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
    fm.add_threshold(
        "temperature",
        yellow=50.0,
        red=60.0,
        direction="above",
        acs_modes=[ACSMode.SCIENCE],
    )

    # Check parameters each simulation cycle
    hk = Housekeeping(
        timestamp=datetime.now(tz=timezone.utc),
        battery_level=0.45,
        temperature=55.0,
        acs_mode=ACSMode.SCIENCE,
    )
    classifications = fm.check(housekeeping=hk, acs=spacecraft_acs)

    # Get accumulated statistics
    stats = fm.statistics()
    # Returns: {"battery_level": {"yellow_seconds": 120.0, "red_seconds": 0.0, "current": "yellow"}, ...}

Threshold Directions:
    - "below": Fault triggered when value <= threshold (e.g., battery_level)
    - "above": Fault triggered when value >= threshold (e.g., temperature, power_draw)

Safe Mode Behavior:
    When safe_mode_on_red=True (default), any parameter reaching RED state will:
    1. Set the safe_mode_requested flag to True
    2. The DITL loop checks this flag and enqueues the ENTER_SAFE_MODE command
    3. Safe mode is irreversible once entered
    4. Spacecraft points solar panels at Sun for maximum power generation
    5. All queued commands are cleared

ACS Mode Filtering:
    Thresholds can be restricted to specific ACS modes using the acs_modes parameter.
    This allows different fault policies for different operational modes.

    Examples:

    # Only check star tracker count during SCIENCE mode (when precision pointing matters)
    fm.add_threshold(
        "star_tracker_functional_count",
        yellow=2.0,
        red=1.0,
        direction="below",
        acs_modes=[ACSMode.SCIENCE],
    )

    # Check thermal limits in all modes except SAFE mode (thermal control always matters)
    fm.add_threshold(
        "temperature",
        yellow=50.0,
        red=60.0,
        direction="above",
        acs_modes=[ACSMode.SCIENCE, ACSMode.SLEW, ACSMode.SETTLE],
    )

    # Check battery level in all modes (None = no filtering)
    fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")

    # During fault checking, the current ACS mode is determined from:
    # 1. housekeeping.acs_mode (preferred)
    # 2. acs.acsmode (fallback)
    # Thresholds are only evaluated when the current mode is in acs_modes list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field
from rust_ephem.constraints import ConstraintConfig

from ..common import ACSMode, normalize_acs_mode
from ..common.common import dtutcfromtimestamp

if TYPE_CHECKING:
    from ..ditl.telemetry import Housekeeping
    from ..simulation import ACS


@dataclass
class FaultEvent:
    """Records a single fault management event.

    Attributes:
        utime: Unix timestamp when the event occurred
        event_type: Type of event (threshold_transition, constraint_violation, safe_mode_trigger)
        name: Name of the parameter or constraint that triggered the event
        cause: Human-readable description of what happened
        metadata: Optional additional data (e.g., current values, thresholds, durations)
    """

    utime: float
    event_type: str
    name: str
    cause: str
    metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        """Concise human-readable summary for printing a list of events."""
        try:
            time_dt = dtutcfromtimestamp(self.utime)
            time_str = time_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = f"utime={self.utime}"

        base = f"[{time_str}] {self.event_type.upper():<20}: {self.name} - {self.cause}"

        if not self.metadata:
            return base

        # Build a short preview of metadata: show up to 3 key=repr(value) pairs, truncating long values
        parts: list[str] = []
        for i, (k, v) in enumerate(self.metadata.items()):
            if i >= 3:
                parts.append("...")
                break
            v_repr = repr(v)
            if len(v_repr) > 80:
                v_repr = v_repr[:77] + "..."
            parts.append(f"{k}={v_repr}")

        return f"{base} | " + ", ".join(parts)


@dataclass
class FaultState:
    """Tracks constraint violations (both red limit and regular thresholds).

    Attributes:
        in_violation: Whether currently violating the constraint
        continuous_violation_seconds: Current continuous violation duration
        current: Current state for threshold-based faults (nominal/yellow/red)
        yellow_seconds: Time spent in yellow state (for threshold-based faults)
        red_seconds: Time spent in red state (for threshold-based faults)
    """

    in_violation: bool = False
    continuous_violation_seconds: float = 0.0
    current: str = "nominal"  # Fault status: nominal | yellow | red
    yellow_seconds: float = 0.0  # Number of seconds in yellow state
    red_seconds: float = 0.0  # Number of seconds in red state


class FaultConstraint(BaseModel):
    """Spacecraft-level red limit constraint for health and safety.

    These constraints define absolute limits that the spacecraft should never violate
    for extended periods. They are typically looser than instrument constraints which
    are optimized for data quality. Red limits exist purely for spacecraft health and safety.

    Attributes:
        name: Descriptive name for the constraint (e.g., 'spacecraft_sun_limit')
        constraint: rust_ephem ConstraintConfig defining the constraint
        time_threshold_seconds: Maximum continuous time allowed in violation before triggering safe mode.
                                If None, no automatic safe mode trigger occurs.
        description: Optional human-readable description of the constraint purpose

    Example:
        >>> # Spacecraft must not point within 30 degrees of Sun for more than 5 minutes
        >>> constraint = FaultConstraint(
        ...     name="spacecraft_sun_limit",
        ...     constraint=rust_ephem.SunConstraint(min_angle=30.0),
        ...     time_threshold_seconds=300.0,
        ...     description="Prevent thermal damage from prolonged sun exposure"
        ... )
    """

    name: str = Field(description="Constraint name identifier")
    constraint: ConstraintConfig = Field(description="Constraint configuration object")
    time_threshold_seconds: float | None = Field(
        default=None,
        description="Maximum continuous violation time before safe mode trigger",
    )
    description: str = Field(
        default="", description="Human-readable description of constraint purpose"
    )

    model_config = {"arbitrary_types_allowed": True}


class FaultThreshold(BaseModel):
    """Threshold configuration for a single monitored parameter.

    Attributes:
        name: Parameter name (e.g. 'battery_level').
        yellow: Value at or beyond which a YELLOW fault is flagged.
        red: Value at or beyond which a RED fault is flagged.
        direction: 'below' or 'above' indicating fault when value passes *below* or *above* limit.
        acs_modes: ACS modes where this parameter should be checked (None = all modes).

    Examples:
        >>> # Check battery level in all modes
        >>> battery_threshold = FaultThreshold(
        ...     name="battery_level",
        ...     yellow=0.5,
        ...     red=0.4,
        ...     direction="below"
        ... )

        >>> # Only check star tracker count during SCIENCE mode
        >>> star_tracker_threshold = FaultThreshold(
        ...     name="star_tracker_functional_count",
        ...     yellow=2.0,
        ...     red=1.0,
        ...     direction="below",
        ...     acs_modes=[ACSMode.SCIENCE]
        ... )

        >>> # Check thermal limits in multiple modes
        >>> thermal_threshold = FaultThreshold(
        ...     name="temperature",
        ...     yellow=50.0,
        ...     red=60.0,
        ...     direction="above",
        ...     acs_modes=[ACSMode.SCIENCE, ACSMode.SLEW, ACSMode.SETTLE]
        ... )
    """

    name: str = Field(description="Parameter name to monitor")
    yellow: float = Field(description="Yellow threshold value")
    red: float = Field(description="Red threshold value")
    direction: str = Field(
        default="below", description="Fault direction: 'below' or 'above'"
    )
    acs_modes: list[ACSMode] | None = Field(
        default=None,
        description="ACS modes where this parameter should be checked",
    )

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

    Also supports spacecraft-level red limit constraints for health and safety that
    can trigger safe mode after sustained violations beyond a time threshold.
    """

    thresholds: list[FaultThreshold] = Field(
        default_factory=list, description="List of parameter thresholds to monitor"
    )
    red_limit_constraints: list[FaultConstraint] = Field(
        default_factory=list,
        description="List of spacecraft-level red limit constraints",
    )
    states: dict[str, FaultState] = Field(
        default_factory=dict,
        description="Current fault states for monitored parameters",
    )
    safe_mode_on_red: bool = Field(
        default=True, description="Whether to trigger safe mode on any RED condition"
    )
    safe_mode_requested: bool = Field(
        default=False, description="Flag indicating safe mode has been requested"
    )
    events: list[FaultEvent] = Field(
        default_factory=list, description="Event log with timestamps and causes"
    )

    def ensure_state(self, name: str) -> FaultState:
        if name not in self.states:
            self.states[name] = FaultState()
        return self.states[name]

    def _trigger_safe_mode(
        self,
        *,
        utime: float,
        name: str,
        cause: str,
        metadata: dict[str, Any],
        acs: ACS,
    ) -> None:
        if not self.safe_mode_on_red:
            return
        if acs.in_safe_mode:
            return
        if self.safe_mode_requested:
            return
        self.safe_mode_requested = True
        self.events.append(
            FaultEvent(
                utime=utime,
                event_type="safe_mode_trigger",
                name=name,
                cause=cause,
                metadata=metadata,
            )
        )

    def check(
        self,
        housekeeping: Housekeeping,
        acs: ACS,
    ) -> dict[str, str]:
        """
        Evaluate all monitored parameters and red limit constraints.

        Args:
            housekeeping: Housekeeping telemetry packet containing all spacecraft state.
            acs: ACS instance to trigger safe mode

        Returns:
            Dict mapping parameter name to classification string (for thresholds only).
        """
        ephem = acs.ephem
        if ephem is None:
            raise ValueError("ACS ephemeris must be set for fault management checks")

        step_size = ephem.step_size
        utime = housekeeping.timestamp.timestamp()

        thresholds_by_name = {t.name: t for t in self.thresholds}
        values: dict[str, float] = {}
        for name in thresholds_by_name:
            val = getattr(housekeeping, name, None)
            if val is not None:
                values[name] = float(val)

        ra = housekeeping.ra
        dec = housekeeping.dec
        classifications: dict[str, str] = {}

        for name, val in values.items():
            threshold = thresholds_by_name[name]
            # Check if this fault has ACS mode restrictions and if so, whether
            # current mode is in the allowed list before classifying the fault.
            # If ACS mode info is not available, skip checking this threshold.
            if threshold.acs_modes is not None:
                current_mode = housekeeping.acs_mode
                if current_mode is None:
                    current_mode = acs.acsmode
                current_mode = normalize_acs_mode(current_mode)
                if current_mode is None:
                    continue
                if current_mode not in threshold.acs_modes:
                    continue
            state = threshold.classify(val)
            classifications[name] = state
            st = self.ensure_state(name)

            previous_state = st.current
            if previous_state != state:
                self.events.append(
                    FaultEvent(
                        utime=utime,
                        event_type="threshold_transition",
                        name=name,
                        cause=f"Transitioned from {previous_state} to {state}",
                        metadata={
                            "previous_state": previous_state,
                            "new_state": state,
                            "value": val,
                            "yellow_threshold": threshold.yellow,
                            "red_threshold": threshold.red,
                            "direction": threshold.direction,
                        },
                    )
                )

            if state == "yellow":
                st.yellow_seconds += step_size
            elif state == "red":
                st.red_seconds += step_size
            st.current = state

            if state == "red":
                self._trigger_safe_mode(
                    utime=utime,
                    name=name,
                    cause=f"RED threshold exceeded for {name}",
                    metadata={
                        "value": val,
                        "red_threshold": threshold.red,
                        "direction": threshold.direction,
                    },
                    acs=acs,
                )

        # Check spacecraft-level red limit constraints if ephemeris and pointing provided
        if (
            ephem is not None
            and ra is not None
            and dec is not None
            and self.red_limit_constraints
        ):
            dt = dtutcfromtimestamp(utime)

            for red_limit in self.red_limit_constraints:
                # Ensure state exists
                fault_state = self.ensure_state(red_limit.name)

                # Check if currently in constraint violation
                # Note: in_constraint returns True when constraint is VIOLATED
                in_violation = cast(
                    bool,
                    red_limit.constraint.in_constraint(
                        ephemeris=ephem, target_ra=ra, target_dec=dec, time=dt
                    ),
                )

                # Log constraint violation events
                previous_violation_state = fault_state.in_violation
                fault_state.in_violation = in_violation

                if in_violation:
                    # Log new violation
                    if not previous_violation_state:
                        self.events.append(
                            FaultEvent(
                                utime=utime,
                                event_type="constraint_violation",
                                name=red_limit.name,
                                cause=f"Entered constraint violation for {red_limit.name}",
                                metadata={
                                    "constraint_type": type(
                                        red_limit.constraint
                                    ).__name__,
                                    "ra": ra,
                                    "dec": dec,
                                    "description": red_limit.description,
                                },
                            )
                        )

                    # Accumulate violation time
                    fault_state.current = "red"
                    fault_state.red_seconds += step_size
                    fault_state.continuous_violation_seconds += step_size

                    # Check if we've exceeded the time threshold
                    if (
                        red_limit.time_threshold_seconds is not None
                        and fault_state.continuous_violation_seconds
                        >= red_limit.time_threshold_seconds
                    ):
                        self._trigger_safe_mode(
                            utime=utime,
                            name=red_limit.name,
                            cause="Constraint violation exceeded time threshold",
                            metadata={
                                "constraint_type": type(red_limit.constraint).__name__,
                                "continuous_violation_seconds": fault_state.continuous_violation_seconds,
                                "time_threshold_seconds": red_limit.time_threshold_seconds,
                                "ra": ra,
                                "dec": dec,
                            },
                            acs=acs,
                        )
                else:
                    # Log violation cleared
                    if previous_violation_state:
                        self.events.append(
                            FaultEvent(
                                utime=utime,
                                event_type="constraint_violation",
                                name=red_limit.name,
                                cause=f"Cleared constraint violation for {red_limit.name}",
                                metadata={
                                    "constraint_type": type(
                                        red_limit.constraint
                                    ).__name__,
                                    "total_violation_seconds": fault_state.continuous_violation_seconds,
                                    "ra": ra,
                                    "dec": dec,
                                },
                            )
                        )
                    # Reset continuous violation counter when constraint is satisfied
                    fault_state.continuous_violation_seconds = 0.0

        return classifications

    def statistics(self) -> dict[str, dict[str, float | str | bool]]:
        """Return accumulated statistics for all parameters and red limit constraints.

        Returns:
            Dict mapping parameter/constraint name to statistics. For threshold-based
            parameters, includes yellow_seconds, red_seconds, and current state. For
            red limit constraints, includes in_violation, red_seconds,
            and continuous_violation_seconds.
        """
        stats: dict[str, dict[str, float | str | bool]] = {}

        for name, st in self.states.items():
            # Check if this is a red limit constraint or special constraint
            if any(c.name == name for c in self.red_limit_constraints):
                # Red limit constraint stats
                stats[name] = {
                    "in_violation": st.in_violation,
                    "red_seconds": st.red_seconds,
                    "continuous_violation_seconds": st.continuous_violation_seconds,
                }
            else:
                # Threshold-based parameter stats
                stats[name] = {
                    "yellow_seconds": st.yellow_seconds,
                    "red_seconds": st.red_seconds,
                    "current": st.current,
                }

        return stats

    def add_threshold(
        self,
        name: str,
        yellow: float,
        red: float,
        direction: str = "below",
        acs_modes: list[ACSMode] | None = None,
    ) -> None:
        """Add a parameter threshold for fault monitoring.

        Args:
            name: Parameter name to monitor (must match Housekeeping attribute name)
            yellow: Value at or beyond which a YELLOW fault is flagged
            red: Value at or beyond which a RED fault is flagged
            direction: 'below' or 'above' indicating fault direction
            acs_modes: ACS modes where this threshold should be checked.
                      None (default) means check in all modes.

        Examples:
            >>> fm = FaultManagement()
            >>> # Check battery in all modes
            >>> fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
            >>> # Only check star trackers during science operations
            >>> fm.add_threshold("star_tracker_count", yellow=2.0, red=1.0,
            ...                  direction="below", acs_modes=[ACSMode.SCIENCE])
        """
        # Import here to avoid circular imports
        from ..ditl.telemetry import Housekeeping

        # Check if name is a valid Housekeeping attribute
        valid_housekeeping_fields = set(Housekeeping.model_fields.keys())
        if name not in valid_housekeeping_fields:
            raise ValueError(
                f"Threshold name '{name}' is not a valid Housekeeping attribute. "
                f"Valid predefined fields are: {sorted(valid_housekeeping_fields)}. "
            )

        self.thresholds.append(
            FaultThreshold(
                name=name,
                yellow=yellow,
                red=red,
                direction=direction,
                acs_modes=acs_modes,
            )
        )

    def add_red_limit_constraint(
        self,
        name: str,
        constraint: ConstraintConfig,
        time_threshold_seconds: float | None = None,
        description: str = "",
    ) -> None:
        """Add a spacecraft-level red limit constraint.

        Args:
            name: Unique identifier for the constraint
            constraint: rust_ephem ConstraintConfig defining the constraint
            time_threshold_seconds: Max continuous violation time before safe mode (None = no trigger)
            description: Human-readable description
        """
        self.red_limit_constraints.append(
            FaultConstraint(
                name=name,
                constraint=constraint,
                time_threshold_seconds=time_threshold_seconds,
                description=description,
            )
        )
