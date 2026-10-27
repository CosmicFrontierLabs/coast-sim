from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property

import yaml
from pydantic import Field, field_validator, model_validator
from rust_ephem.constraints import ConstraintConfig

from ..common.enums import ACSMode
from ._base import ConfigModel
from .battery import Battery
from .constraint import AttitudeConstraintScope, Constraint, DefaultConstraint
from .data_generator import ScienceData
from .data_storage import DataStorage, RecorderView
from .fault_management import FaultManagement
from .groundstation import GroundStationRegistry
from .instrument import Payload
from .observation_categories import ObservationCategories
from .observation_timing import ObservationTiming
from .recorder import OnboardRecorder
from .scheduler import SchedulerConfig
from .solar_panel import SolarPanelSet
from .spacecraft_bus import SpacecraftBus
from .targets import TargetConfig
from .visualization import VisualizationConfig


def _default_attitude_constraint_scopes() -> dict[str, list[AttitudeConstraintScope]]:
    return {
        ACSMode.SCIENCE.name: [
            AttitudeConstraintScope.HARDWARE_SAFETY,
            AttitudeConstraintScope.IMAGING_QUALITY,
            AttitudeConstraintScope.POWER_GENERATION,
        ],
        ACSMode.CHARGING.name: [
            AttitudeConstraintScope.HARDWARE_SAFETY,
            AttitudeConstraintScope.POWER_GENERATION,
        ],
        ACSMode.SLEWING.name: [AttitudeConstraintScope.HARDWARE_SAFETY],
        ACSMode.PASS.name: [
            AttitudeConstraintScope.HARDWARE_SAFETY,
            AttitudeConstraintScope.POWER_GENERATION,
            AttitudeConstraintScope.GROUND_CONTACT,
        ],
        ACSMode.SAA.name: [AttitudeConstraintScope.HARDWARE_SAFETY],
        ACSMode.SAFE.name: [
            AttitudeConstraintScope.HARDWARE_SAFETY,
            AttitudeConstraintScope.POWER_GENERATION,
        ],
        ACSMode.IDLE.name: [AttitudeConstraintScope.HARDWARE_SAFETY],
        ACSMode.DESAT.name: [AttitudeConstraintScope.HARDWARE_SAFETY],
    }


def _mode_name(mode: ACSMode | int | str) -> str:
    if isinstance(mode, ACSMode):
        return mode.name
    if isinstance(mode, int):
        return ACSMode(mode).name
    raw = mode.strip().upper()
    if raw in ACSMode.__members__:
        return raw
    return ACSMode(int(raw)).name


def _normalize_attitude_constraint_scopes(
    value: dict[
        ACSMode | int | str,
        list[AttitudeConstraintScope | str] | tuple[AttitudeConstraintScope | str, ...],
    ]
    | None,
) -> dict[str, list[AttitudeConstraintScope]]:
    scopes = _default_attitude_constraint_scopes()
    if value is None:
        return scopes
    for mode, configured_scopes in value.items():
        normalized: list[AttitudeConstraintScope] = []
        for scope in configured_scopes:
            enum_scope = AttitudeConstraintScope(scope)
            if enum_scope not in normalized:
                normalized.append(enum_scope)
        scopes[_mode_name(mode)] = normalized
    return scopes


WAITING_MODES: tuple[ACSMode | None, ...] = (
    None,
    ACSMode.IDLE,
    ACSMode.CHARGING,
    ACSMode.SAA,
)
"""The modes the spacecraft waits in between observations, the nominal rate
first: what a reserve of housekeeping over a wait has to cover."""


def housekeeping_while_waiting(config: object, duration_seconds: float) -> float:
    """The most housekeeping ``config`` generates over an interval spent
    waiting between observations: in whichever of the nominal, ``IDLE``,
    ``CHARGING`` and ``SAA`` rates is the highest, as dispatch can spend the
    interval in any of those modes and a reserve made at a lower rate would
    be overrun.
    ``config`` is anything with ``housekeeping_generated(seconds, mode=)``,
    a mission configuration or a test double's."""
    generated = getattr(config, "housekeeping_generated")
    return max(float(generated(duration_seconds, mode=mode)) for mode in WAITING_MODES)


class MissionConfig(ConfigModel):
    """
    Configuration class for the spacecraft and its subsystems.
    """

    name: str = Field(default="Default Config", description="Mission name/identifier")
    random_seed: int | None = Field(
        default=None,
        description="Optional seed for stochastic planning decisions.",
    )
    attitude_constraint_scopes: dict[str, list[AttitudeConstraintScope]] = Field(
        default_factory=_default_attitude_constraint_scopes,
        description=(
            "Constraint validation scopes by ACS mode for executed attitude samples. "
            "Values are lists containing 'hardware_safety', 'imaging_quality', "
            "'power_generation', and/or 'ground_contact'. Omitted modes use "
            "the default scopes."
        ),
    )
    spacecraft_bus: SpacecraftBus = Field(
        default_factory=SpacecraftBus,
        description="Spacecraft Bus Configuration. Defines the main spacecraft platform including power, attitude control, and communications",
    )
    solar_panel: SolarPanelSet = Field(
        default_factory=SolarPanelSet,
        description="Solar Panel Configuration. Defines solar array characteristics and power generation capability",
    )
    payload: Payload = Field(
        default_factory=Payload,
        description="Payload/Instrument Configuration. Defines science instruments with power draw and data generation rates",
    )
    battery: Battery = Field(
        default_factory=Battery,
        description="Battery Configuration. Defines battery capacity, voltage, and charging parameters",
    )
    constraint: Constraint = Field(
        default_factory=DefaultConstraint,
        description="Pointing Constraints. Defines geometric constraints for spacecraft pointing",
    )
    ground_stations: GroundStationRegistry = Field(
        default_factory=GroundStationRegistry,
        description="Ground Station Network. Defines ground station locations and communication capabilities",
    )
    recorder: OnboardRecorder = Field(
        default_factory=OnboardRecorder,
        # Omit default values so named stores round-trip without this field.
        # Validation still rejects explicitly supplying both storage models.
        exclude_if=lambda recorder: recorder == OnboardRecorder(),
        description="Onboard Data Recorder. Defines storage capacity and fill thresholds",
    )
    data_storage: DataStorage | None = Field(
        default=None, description="Independent stores and explicit data/downlink routes"
    )

    fault_management: FaultManagement = Field(
        default_factory=FaultManagement,
        description="Fault Management System. Defines thresholds and responses for system health monitoring",
    )
    observation_categories: ObservationCategories = Field(
        default_factory=ObservationCategories.default_categories,
        description="Observation Categories. Defines categories for different observation types with ID ranges",
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        exclude=True,
        description="Visualization Configuration",
    )
    targets: TargetConfig = Field(
        default_factory=TargetConfig,
        description="Target Configuration. Defines observational targets and scheduling parameters",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description=(
            "Scheduler. How simulations built with create_ditl decide what to "
            "observe: queue dispatch, one plan, or rolling replanning"
        ),
    )

    @model_validator(mode="after")
    def _one_storage_model(self) -> MissionConfig:
        self.__dict__.pop("_recorder_view_cache", None)
        if (
            isinstance(self.data_storage, DataStorage)
            and "recorder" in self.model_fields_set
        ):
            raise ValueError("configure either recorder or data_storage, not both")
        return self

    @cached_property
    def _recorder_view_cache(
        self,
    ) -> tuple[int, dict[str, MissionConfig | RecorderView]]:
        if self.data_storage is None:
            return id(self), {"default": self}
        self.data_storage.require_ready(self)
        return id(self), {
            name: RecorderView(self, name) for name in self.data_storage.stores
        }

    def recorder_views(
        self, *, refresh: bool = False
    ) -> dict[str, MissionConfig | RecorderView]:
        """Validated views, reused through a run's candidate and step loops.

        Constructors refresh these before each run. After editing nested
        configuration in place, other callers must request a refresh too.
        Recorder volumes themselves remain live through the cached views.
        """
        if refresh:
            self.__dict__.pop("_recorder_view_cache", None)
        # Pydantic copies cached properties; copied views must bind to this
        # configuration, even for a shallow model_copy(update=...).
        if self._recorder_view_cache[0] != id(self):
            self.__dict__.pop("_recorder_view_cache", None)
        return self._recorder_view_cache[1]

    @field_validator("attitude_constraint_scopes", mode="before")
    @classmethod
    def _validate_attitude_constraint_scopes(
        cls,
        value: dict[
            ACSMode | int | str,
            list[AttitudeConstraintScope | str]
            | tuple[AttitudeConstraintScope | str, ...],
        ]
        | None,
    ) -> dict[str, list[AttitudeConstraintScope]]:
        return _normalize_attitude_constraint_scopes(value)

    @model_validator(mode="after")
    def _panels_see_the_given_eclipses(self) -> MissionConfig:
        """The solar panels take the intervals the constraint gives as
        eclipse, read as the constraint's list stands."""
        constraint = getattr(self, "constraint", None)
        take = getattr(
            getattr(self, "solar_panel", None), "take_eclipse_intervals_from", None
        )
        # A test double's constraint or panels take nothing.
        if callable(take) and isinstance(constraint, Constraint):
            take(constraint.eclipse_intervals_unix)
        return self

    @model_validator(mode="after")
    def _budget_steps_name_known_modes(self) -> MissionConfig:
        """The calibration steps of every timing budget name instruments the
        payload has, in modes they define, allowed at once."""
        payload = getattr(self, "payload", None)
        check = getattr(payload, "check_request", None)
        # A test double's payload or categories check nothing.
        if not callable(check):
            return self
        timings = [getattr(payload, "observation_timing", None)]
        categories = getattr(self, "observation_categories", None)
        if isinstance(categories, ObservationCategories):
            timings.extend(
                category.observation_timing for category in categories.categories
            )
        for timing in timings:
            if not isinstance(timing, ObservationTiming):
                continue
            for step in (*timing.setup_steps, *timing.cleanup_steps):
                if step.instrument_modes:
                    check(step.instrument_modes)
        return self

    def observation_timing_for(self, obsid: int) -> ObservationTiming:
        """The setup, cleanup and handoff budgets of the observation ``obsid``:
        its category's where that sets them, else the payload's."""
        timing = self.observation_categories.get_category(int(obsid)).observation_timing
        return timing if timing is not None else self.payload.observation_timing

    def observation_timings(self) -> list[ObservationTiming]:
        """Every timing an observation can have: the payload's and each
        category's own."""
        return [
            self.payload.observation_timing,
            *(
                category.observation_timing
                for category in self.observation_categories.categories
                if category.observation_timing is not None
            ),
        ]

    def attitude_constraint_scopes_for_mode(
        self, mode: ACSMode | int
    ) -> list[AttitudeConstraintScope]:
        return self.attitude_constraint_scopes.get(_mode_name(mode), [])

    @model_validator(mode="after")
    def init_fault_management_defaults(self) -> MissionConfig:
        """Initialize default fault thresholds if none provided.

        Currently sets up a battery_level threshold using the battery
        max_depth_of_discharge as the minimum allowed charge level for YELLOW
        and (max_depth_of_discharge + 0.1) as RED for demonstration.
        Also sets up recorder_fill_fraction threshold using the recorder's
        yellow and red thresholds.
        Users can override via config serialization.
        """
        if self.fault_management is None:
            return self
        # Only add battery threshold if not already present
        if not any(t.name == "battery_level" for t in self.fault_management.thresholds):
            # Yellow alert at minimum allowed charge level (1.0 - max_depth_of_discharge)
            yellow = 1.0 - self.battery.max_depth_of_discharge
            # Red alert at 10% below the minimum allowed charge level
            red = max(yellow - 0.1, 0.0)  # Ensure non-negative
            self.fault_management.add_threshold(
                name="battery_level", yellow=yellow, red=red, direction="below"
            )
        # Add recorder threshold if not already present
        if not isinstance(self.data_storage, DataStorage) and not any(
            t.name == "recorder_fill_fraction" for t in self.fault_management.thresholds
        ):
            # Use recorder's built-in thresholds
            self.fault_management.add_threshold(
                name="recorder_fill_fraction",
                yellow=self.recorder.yellow_threshold,
                red=self.recorder.red_threshold,
                direction="above",
            )
        if isinstance(self.data_storage, DataStorage):
            # The legacy aggregate threshold is meaningless for independent stores.
            self.fault_management.thresholds = [
                t
                for t in self.fault_management.thresholds
                if t.name != "recorder_fill_fraction"
            ]
            for name, store in self.data_storage.stores.items():
                key = f"recorders.{name}.fill_fraction"
                if store.recorder is not None and not any(
                    t.name == key for t in self.fault_management.thresholds
                ):
                    self.fault_management.add_threshold(
                        name=key,
                        yellow=store.recorder.yellow_threshold,
                        red=store.recorder.red_threshold,
                        direction="above",
                    )
        # Add star tracker threshold if not already present and star trackers are configured
        star_trackers = None
        try:
            star_trackers = self.spacecraft_bus.star_trackers
            has_star_trackers = (
                hasattr(star_trackers, "num_trackers")
                and star_trackers.num_trackers() > 0
            )
        except AttributeError:
            has_star_trackers = False

        existing_st_threshold = next(
            (
                t
                for t in self.fault_management.thresholds
                if t.name == "star_tracker_functional_count"
            ),
            None,
        )
        if existing_st_threshold is not None:
            # Config-file-loaded thresholds default triggers_safe_mode=True; force
            # it False — a soft-constraint degradation should alert but not safehold
            # (reduced pointing quality is operationally significant but not an emergency).
            existing_st_threshold.triggers_safe_mode = False
        elif has_star_trackers and star_trackers is not None:
            # Monitors soft-constraint violations (degraded pointing quality).
            # functional_count = num_trackers - soft_violation_count, so dropping
            # below num_trackers means at least one tracker is in a soft keepout.
            num_trackers = star_trackers.num_trackers()
            yellow = num_trackers - 1  # any soft violation → YELLOW and RED
            red = num_trackers - 1  # any soft violation → RED
            self.fault_management.add_threshold(
                name="star_tracker_functional_count",
                yellow=yellow,
                red=red,
                direction="below",
                triggers_safe_mode=False,
            )

        # Add star_tracker_hard_violations threshold if not already present.
        # star_tracker_hard_violations counts trackers inside HARD_KEEPOUT zones;
        # any non-zero value is immediately RED. Hard violations are tracked
        # separately from functional_count (which only reflects soft violations).
        # direction="above" fires when value >= threshold, so yellow=red=0.5 fires
        # on the first integer violation (violations=1 >= 0.5). monitor-only:
        # the scheduler avoids hard keepouts; a violation is a scheduling bug,
        # not a spacecraft emergency requiring a safehold.
        if (
            not any(
                t.name == "star_tracker_hard_violations"
                for t in self.fault_management.thresholds
            )
            and has_star_trackers
        ):
            self.fault_management.add_threshold(
                name="star_tracker_hard_violations",
                yellow=0.5,
                red=0.5,
                direction="above",
                triggers_safe_mode=False,
            )

        # Add ppt_unavailable threshold if not already present.
        # bool True (=1.0) > 0.5 → YELLOW; False (=0.0) is nominal; red set
        # above 1.0 so only YELLOW fires (no RED for this alert).
        if not any(
            t.name == "ppt_unavailable" for t in self.fault_management.thresholds
        ):
            self.fault_management.add_threshold(
                name="ppt_unavailable",
                yellow=0.5,
                red=1.5,  # unreachable by bool — YELLOW-only alert
                direction="above",
            )

        # Propagate star tracker hard exclusions into mission-level planning constraint.
        if star_trackers is not None and has_star_trackers:
            self.constraint.star_tracker_hard_constraint = (
                star_trackers.startracker_hard_constraint
            )
            self.constraint.star_tracker_soft_constraint = (
                star_trackers.startracker_constraint
            )
            self.constraint.star_tracker_enforce_modes = (
                star_trackers.modes_require_lock
            )
        else:
            self.constraint.star_tracker_hard_constraint = None
            self.constraint.star_tracker_soft_constraint = None
            self.constraint.star_tracker_enforce_modes = None

        radiators = None
        try:
            radiators = self.spacecraft_bus.radiators
            num_radiators = radiators.num_radiators()
            has_radiators = isinstance(num_radiators, int) and num_radiators > 0
        except (AttributeError, TypeError):
            has_radiators = False

        if radiators is not None and has_radiators:
            self.constraint.radiator_hard_constraint = (
                radiators.radiator_hard_constraint
            )
        else:
            self.constraint.radiator_hard_constraint = None

        telescope_constraint = self.payload.combined_telescope_spacecraft_constraint()
        self.constraint.telescope_hard_constraint = (
            telescope_constraint
            if isinstance(telescope_constraint, ConstraintConfig)
            else None
        )

        # Add radiator_hard_violations threshold if not already present.
        # Same semantics as star_tracker_hard_violations: any non-zero count fires
        # RED immediately (violations=1 >= 0.5). monitor-only by default.
        if (
            not any(
                t.name == "radiator_hard_violations"
                for t in self.fault_management.thresholds
            )
            and has_radiators
        ):
            self.fault_management.add_threshold(
                name="radiator_hard_violations",
                yellow=0.5,
                red=0.5,
                direction="above",
                triggers_safe_mode=False,
            )

        # Add telescope_hard_violations threshold if not already present.
        # telescope_hard_violations is 0 or 1 (single constraint); any violation
        # fires RED immediately. monitor-only by default.
        has_telescope_hard = isinstance(telescope_constraint, ConstraintConfig)
        if (
            not any(
                t.name == "telescope_hard_violations"
                for t in self.fault_management.thresholds
            )
            and has_telescope_hard
        ):
            self.fault_management.add_threshold(
                name="telescope_hard_violations",
                yellow=0.5,
                red=0.5,
                direction="above",
                triggers_safe_mode=False,
            )

        self.constraint.invalidate_combined_constraint_cache()
        return self

    def housekeeping_generated(
        self, duration_seconds: float, *, mode: ACSMode | None = None
    ) -> float:
        """Housekeeping data generated by the bus and payload over an interval.

        Args:
            duration_seconds: Length of the interval in seconds.
            mode: ACS mode during the interval; None for nominal rates.

        Returns:
            float: Data generated in Gigabits.
        """
        return self.spacecraft_bus.data_generated(
            duration_seconds, mode=mode
        ) + self.payload.housekeeping_generated(duration_seconds, mode=mode)

    def housekeeping_while_waiting(self, duration_seconds: float) -> float:
        """The most housekeeping the bus and payload generate over an
        interval the spacecraft spends waiting between observations (see
        :func:`housekeeping_while_waiting`).

        Args:
            duration_seconds: Length of the interval in seconds.

        Returns:
            float: Data generated in Gigabits.
        """
        return housekeeping_while_waiting(self, duration_seconds)

    def science_generated(
        self,
        collection_seconds: float,
        science: Mapping[str, ScienceData] | None = None,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> float:
        """Science data an observation generates, having collected for some seconds.

        See :meth:`Payload.science_generated <conops.config.Payload.science_generated>`.

        Args:
            collection_seconds: Seconds of science the observation collected.
            science: The observed target's science data, keyed by instrument
                name.
            instrument_modes: The mode each instrument is in during the
                observation, by name; None: every instrument generates.

        Returns:
            float: Data generated in Gigabits.
        """
        return self.payload.science_generated(
            collection_seconds, science, instrument_modes
        )

    def data_generated(
        self,
        duration_seconds: float,
        *,
        mode: ACSMode | None = None,
        collection_seconds: float | None = None,
        science: Mapping[str, ScienceData] | None = None,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> float:
        """Data an observation generates over an interval.

        Housekeeping data from the bus and payload for the interval, plus the
        observation's science data for the seconds it collected.

        Args:
            duration_seconds: Length of the interval in seconds.
            mode: ACS mode during the interval, which sets housekeeping rates;
                None for nominal rates.
            collection_seconds: Seconds of science collected in the interval;
                None for all of it.
            science: The observed target's science data, keyed by instrument
                name.

        Returns:
            float: Data generated in Gigabits.
        """
        if collection_seconds is None:
            collection_seconds = duration_seconds
        return self.housekeeping_generated(
            duration_seconds, mode=mode
        ) + self.science_generated(collection_seconds, science, instrument_modes)

    @classmethod
    def from_json_file(cls, filepath: str) -> MissionConfig:
        """Load configuration from a JSON file."""
        with open(filepath) as f:
            return cls.model_validate_json(f.read())

    def to_json_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_yaml_file(cls, filepath: str) -> MissionConfig:
        """Load configuration from a YAML file.

        Args:
            filepath: Path to the YAML configuration file.

        Returns:
            MissionConfig: Validated configuration object.
        """
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)
            return cls.model_validate(config_dict)

    def to_yaml_file(self, filepath: str) -> None:
        """Save configuration to a YAML file with annotations.

        The YAML output includes helpful comments explaining:
        - Default units for physical quantities (e.g., power in Watts, time in seconds)
        - Purpose and meaning of configuration settings
        - Valid ranges or constraints where applicable

        Args:
            filepath: Path where the YAML file will be written.
        """
        config_dict = self.model_dump(mode="json", exclude_none=True)

        # Build annotated YAML with comments
        yaml_lines = []
        yaml_lines.append("# COAST-Sim Mission Configuration File")
        yaml_lines.append("# Generated YAML configuration with annotations")
        yaml_lines.append("#")
        yaml_lines.append(
            "# This file defines the complete spacecraft configuration including:"
        )
        yaml_lines.append("#   - Spacecraft bus and subsystems")
        yaml_lines.append("#   - Power generation and storage")
        yaml_lines.append("#   - Payload instruments")
        yaml_lines.append("#   - Ground stations")
        yaml_lines.append("#   - Operational constraints")
        yaml_lines.append("#")
        yaml_lines.append("# Units Legend:")
        yaml_lines.append("#   Power: Watts (W)")
        yaml_lines.append("#   Energy: Watt-hours (Wh)")
        yaml_lines.append("#   Time: seconds (s) unless otherwise specified")
        yaml_lines.append("#   Angles: degrees (deg)")
        yaml_lines.append(
            "#   Data rates: Gigabits per second (Gbps) or Megabits per second (Mbps)"
        )
        yaml_lines.append("#   Data volume: Gigabits (Gb)")
        yaml_lines.append("#   Slew rates: degrees per second (deg/s)")
        yaml_lines.append("#   Acceleration: degrees per second squared (deg/s²)")
        yaml_lines.append("")

        # Dynamically iterate through all top-level fields
        for field_name, field_info in self.__class__.model_fields.items():
            # Skip visualization as it's excluded
            if field_info.exclude:
                continue

            if field_name not in config_dict:
                continue

            # Add section header comment
            if field_info.description:
                yaml_lines.append(f"# {field_info.description}")

            field_value = config_dict[field_name]
            field_model = field_info.annotation

            # For scalar top-level fields, just output directly
            if not isinstance(field_value, (dict, list)):
                yaml_str = yaml.safe_dump(
                    {field_name: field_value}, default_flow_style=False, sort_keys=False
                )
                yaml_lines.append(yaml_str.rstrip())
            else:
                # For dict/list fields, use the key and then indent content
                yaml_lines.append(f"{field_name}:")
                self._add_annotated_yaml_content(
                    yaml_lines, field_value, field_model, indent=1
                )
            yaml_lines.append("")

        # Write to file
        with open(filepath, "w") as f:
            f.write("\n".join(yaml_lines))

    def _add_annotated_yaml_content(
        self,
        lines: list[str],
        content: object,
        model_class: object,
        indent: int = 1,
    ) -> None:
        """Add YAML content with annotations from Pydantic model field descriptions.

        Args:
            lines: List of YAML lines to append to
            content: The data content to serialize (dict, list, or scalar)
            model_class: The Pydantic model class for field descriptions (can be generic type)
            indent: Indentation level for nested content
        """
        ind = "  " * indent

        # Handle dict with potential nested model
        if isinstance(content, dict):
            if hasattr(model_class, "model_fields"):
                # Iterate through fields in the model
                for field_name, field_info in model_class.model_fields.items():
                    if field_name not in content:
                        continue

                    field_value = content[field_name]
                    field_annotation = self._resolve_annotation(field_info.annotation)

                    # Add field description as a comment
                    if field_info.description:
                        lines.append(f"{ind}# {field_name}: {field_info.description}")

                    # Handle lists
                    if isinstance(field_value, list):
                        lines.append(f"{ind}{field_name}:")
                        # Get the item type from list annotation
                        item_model = self._get_list_item_type(field_info.annotation)
                        if (
                            field_value
                            and isinstance(field_value[0], dict)
                            and item_model
                            and hasattr(item_model, "model_fields")
                        ):
                            # For lists of dicts with a model, add annotated items
                            for item in field_value:
                                item_fields = [
                                    (item_field_name, item_field_info)
                                    for (
                                        item_field_name,
                                        item_field_info,
                                    ) in item_model.model_fields.items()
                                    if item_field_name in item
                                ]
                                item_fields.sort(key=lambda field: field[0] != "name")
                                if not item_fields:
                                    lines.append(f"{ind}  - {{}}")
                                    continue

                                first_field = True
                                for item_field_name, item_field_info in item_fields:
                                    item_val = item[item_field_name]
                                    if (
                                        item_field_name != "name"
                                        and item_field_info.description
                                    ):
                                        comment_indent = (
                                            f"{ind}  " if first_field else f"{ind}    "
                                        )
                                        lines.append(
                                            f"{comment_indent}# {item_field_name}: "
                                            f"{item_field_info.description}"
                                        )
                                    yaml_str = yaml.safe_dump(
                                        {item_field_name: item_val},
                                        default_flow_style=False,
                                        sort_keys=False,
                                    )
                                    for line_index, line in enumerate(
                                        yaml_str.rstrip().split("\n")
                                    ):
                                        if line:
                                            if first_field and line_index == 0:
                                                lines.append(f"{ind}  - {line}")
                                            else:
                                                lines.append(f"{ind}    {line}")
                                    first_field = False
                        else:
                            # Default list handling
                            self._add_annotated_yaml_content(
                                lines,
                                field_value,
                                item_model if item_model else object,
                                indent + 1,
                            )
                    # Handle nested models
                    elif isinstance(field_value, dict) and hasattr(
                        field_annotation, "model_fields"
                    ):
                        expected_keys = set(field_annotation.model_fields.keys())
                        if set(field_value.keys()).issubset(expected_keys):
                            lines.append(f"{ind}{field_name}:")
                            self._add_annotated_yaml_content(
                                lines,
                                field_value,
                                field_annotation,
                                indent=indent + 1,
                            )
                        else:
                            yaml_str = yaml.safe_dump(
                                {field_name: field_value},
                                default_flow_style=False,
                                sort_keys=False,
                            )
                            for line in yaml_str.rstrip().split("\n"):
                                if line:
                                    lines.append(ind + line)
                    else:
                        # Scalar or non-pydantic value
                        yaml_str = yaml.safe_dump(
                            {field_name: field_value},
                            default_flow_style=False,
                            sort_keys=False,
                        )
                        for line in yaml_str.rstrip().split("\n"):
                            if line:
                                lines.append(ind + line)
            else:
                # No model class, just dump as-is
                yaml_str = yaml.safe_dump(
                    content, default_flow_style=False, sort_keys=False
                )
                for line in yaml_str.rstrip().split("\n"):
                    if line:
                        lines.append(ind + line)
                    else:
                        lines.append("")
        # Handle list
        elif isinstance(content, list):
            yaml_str = yaml.safe_dump(
                content, default_flow_style=False, sort_keys=False
            )
            for line in yaml_str.rstrip().split("\n"):
                if line:
                    lines.append(ind + line)
        else:
            # Scalar value
            yaml_str = yaml.safe_dump(
                content, default_flow_style=False, sort_keys=False
            )
            lines.append(ind + yaml_str.rstrip())

    def _resolve_annotation(self, annotation: object) -> object:
        """Unwrap Annotated and Union annotations to the underlying type."""
        import types
        import typing

        if hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
            origin = typing.get_origin(annotation)
            args = typing.get_args(annotation)

            if origin is typing.Annotated and args:
                return self._resolve_annotation(args[0])

            if origin in (typing.Union, types.UnionType) and args:
                # Prefer a pydantic model if one exists in the union.
                for arg in args:
                    resolved = self._resolve_annotation(arg)
                    if hasattr(resolved, "model_fields"):
                        return resolved
                return self._resolve_annotation(args[0])

        return annotation

    def _get_list_item_type(self, model_class: object) -> object | None:
        """Extract the item type from a list annotation.

        Args:
            model_class: A type annotation like list[SomeModel]

        Returns:
            The item type, or None if not determinable
        """
        import typing

        if hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
            origin = typing.get_origin(model_class)
            args = typing.get_args(model_class)

            if origin is list and args:
                return self._resolve_annotation(args[0])

        return None
