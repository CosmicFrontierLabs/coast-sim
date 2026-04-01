Mission Configuration
=====================

The :class:`~conops.config.MissionConfig` class is the central configuration object in COASTSim.
It aggregates all spacecraft subsystem configurations into a single, serializable Pydantic model
that can be passed to DITL simulations.

Overview
--------

``MissionConfig`` serves as the complete definition of your spacecraft, including:

* Spacecraft bus properties (power, attitude control, communications)
* Solar panel configuration and power generation
* Payload instruments and data generation
* Battery specifications
* Pointing constraints (Sun, Moon, Earth limb avoidance)
* Ground station network
* Onboard data recorder
* Fault management system
* Visualization settings

Creating a Configuration
------------------------

There are several ways to create a ``MissionConfig``:

**1. Default Configuration**

Create a configuration with all default values:

.. code-block:: python

   from conops.config import MissionConfig

   config = MissionConfig()

**2. Programmatic Configuration**

Build a configuration by specifying individual components:

.. code-block:: python

   from conops.config import (
       MissionConfig,
       SpacecraftBus,
       SolarPanelSet,
       SolarPanel,
       Payload,
       Instrument,
       Battery,
       Constraint,
       GroundStationRegistry,
       GroundStation,
       OnboardRecorder,
       FaultManagement,
   )

   config = MissionConfig(
       name="My Space Telescope",
       spacecraft_bus=SpacecraftBus(...),
       solar_panel=SolarPanelSet(...),
       payload=Payload(...),
       battery=Battery(...),
       constraint=Constraint(...),
       ground_stations=GroundStationRegistry(...),
       recorder=OnboardRecorder(...),
       fault_management=FaultManagement(...),
   )

**3. Load from JSON File**

Load a pre-defined configuration from a JSON file:

.. code-block:: python

   config = MissionConfig.from_json_file("spacecraft_config.json")

**4. Load from YAML File**

Load a pre-defined configuration from a YAML file (typically more human-readable than JSON):

.. code-block:: python

   config = MissionConfig.from_yaml_file("spacecraft_config.yaml")

**5. Save to JSON File**

Save a configuration to JSON for version control or sharing:

.. code-block:: python

   config.to_json_file("spacecraft_config.json")

**6. Save to YAML File**

Save a configuration to YAML with helpful annotations explaining units and purpose:

.. code-block:: python

   config.to_yaml_file("spacecraft_config.yaml")

The YAML output includes comprehensive comments explaining:

- Default units for physical quantities (power in Watts, time in seconds, etc.)
- Purpose and meaning of configuration settings
- Valid ranges or constraints where applicable

This makes YAML configurations particularly useful for:

- Human review and editing
- Documentation and configuration examples
- Onboarding new team members
- Version control with readable diffs

Configuration Components
------------------------

name
~~~~

A human-readable name for the mission configuration.

.. code-block:: python

   name: str = "Default Config"

Example:

.. code-block:: python

   config = MissionConfig(name="STROBE-X Observatory")

spacecraft_bus
~~~~~~~~~~~~~~

The :class:`~conops.config.SpacecraftBus` defines the spacecraft bus subsystems.

**Attributes:**

* ``name`` (str): Bus identifier
* ``power_draw`` (:class:`~conops.config.PowerDraw`): Power consumption characteristics
* ``attitude_control`` (:class:`~conops.config.AttitudeControlSystem`): ACS configuration
* ``communications`` (:class:`~conops.config.CommunicationsSystem`): Optional comms system
* ``heater`` (:class:`~conops.config.Heater`): Optional thermal heater
* ``data_generation`` (:class:`~conops.config.DataGeneration`): Bus-level data generation
* ``star_trackers`` (:class:`~conops.config.StarTrackerConfiguration`): Optional star tracker configuration

.. code-block:: python

   from conops.config import SpacecraftBus, PowerDraw, AttitudeControlSystem, Heater

   spacecraft_bus = SpacecraftBus(
       name="Observatory Bus",
       power_draw=PowerDraw(
           nominal_power=50.0,      # Watts - normal operations
           peak_power=300.0,        # Watts - maximum draw
           power_mode={0: 70.0, 1: 100.0},  # Mode-specific power
           eclipse_power=75.0,      # Power draw during eclipse
       ),
       attitude_control=AttitudeControlSystem(
           slew_acceleration=0.01,  # deg/s² - angular acceleration
           max_slew_rate=0.3,       # deg/s - maximum slew rate
           slew_accuracy=0.01,      # deg - pointing accuracy
           settle_time=10.0,        # seconds - time to settle after slew
       ),
       heater=Heater(
           name="Bus Heaters",
           power_draw=PowerDraw(
               nominal_power=25.0,
               eclipse_power=75.0,  # Higher power in eclipse
           ),
       ),
   )

solar_panel
~~~~~~~~~~~

The :class:`~conops.config.SolarPanelSet` defines the solar array configuration.

**SolarPanelSet Attributes:**

* ``name`` (str): Array identifier
* ``panels`` (list[SolarPanel]): List of individual panel configurations
* ``conversion_efficiency`` (float): Default array-level efficiency (0-1)

**SolarPanel Attributes:**

* ``name`` (str): Panel identifier
* ``gimbled`` (bool): Whether the panel can track the Sun
* ``normal`` (tuple[float, float, float]): Panel normal vector in spacecraft body frame

  - +x is the spacecraft pointing direction (boresight)
  - +y is the spacecraft "up" direction
  - +z completes the right-handed coordinate system
  - Should be a unit vector for proper illumination calculations

* ``max_power`` (float): Maximum power output at full illumination (Watts)
* ``conversion_efficiency`` (float | None): Per-panel efficiency override

.. code-block:: python

   from conops.config import SolarPanelSet, SolarPanel

   solar_panel = SolarPanelSet(
       name="Main Solar Array",
       panels=[
           SolarPanel(
               name="Panel +Y",
               gimbled=False,
               normal=(0.0, 1.0, 0.0),  # Side-mounted
               max_power=400.0,
               conversion_efficiency=0.94,
           ),
           SolarPanel(
               name="Panel -Y",
               gimbled=False,
               normal=(0.0, -1.0, 0.0),  # Opposite side
               max_power=400.0,
               conversion_efficiency=0.94,
           ),
       ],
       conversion_efficiency=0.95,
   )

Solar Panel Vector Helper Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defining panel normal vectors manually can be error-prone. The
:func:`~conops.config.solar_panel.create_solar_panel_vector` helper function
simplifies this by generating unit normal vectors based on mount type and cant angles.

**Supported Mount Types:**

* ``'sidemount'``: Panel faces +Y (spacecraft "up")
* ``'aftmount'``: Panel faces -X (spacecraft "back")
* ``'boresight'``: Panel faces +X (spacecraft forward/pointing direction)

**Cant Angles:**

* ``cant_z``: Rotation around Z-axis (yaw) in degrees
* ``cant_perp``: Rotation around perpendicular axis (pitch) in degrees

  - For 'sidemount': rotates around X-axis
  - For 'aftmount': rotates around Y-axis
  - For 'boresight': rotates around Y-axis

**Examples:**

.. code-block:: python

   from conops.config import SolarPanelSet, SolarPanel
   from conops.config.solar_panel import create_solar_panel_vector

   # Simple side-mounted panel (no cant)
   normal = create_solar_panel_vector('sidemount')
   # Result: (0.0, 1.0, 0.0)

   # Side-mounted panel with 30° yaw
   normal = create_solar_panel_vector('sidemount', cant_z=30.0)

   # Aft-mounted panel with 45° pitch backward slant
   normal = create_solar_panel_vector('aftmount', cant_perp=-45.0)

   # Boresight panel tilted backward 45° (forward-facing with backward slant)
   normal = create_solar_panel_vector('boresight', cant_perp=-45.0)

   # Complex orientation: boresight with 30° yaw left and 45° backward pitch
   normal = create_solar_panel_vector('boresight', cant_z=30.0, cant_perp=-45.0)

   # Use in panel definition
   solar_panel = SolarPanelSet(
       name="Main Solar Array",
       panels=[
           SolarPanel(
               name="Panel +Y",
               gimbled=False,
               normal=create_solar_panel_vector('sidemount', cant_z=10.0),
               max_power=400.0,
               conversion_efficiency=0.94,
           ),
           SolarPanel(
               name="Panel Boresight Aft",
               gimbled=False,
               normal=create_solar_panel_vector('boresight', cant_perp=-30.0),
               max_power=200.0,
               conversion_efficiency=0.94,
           ),
       ],
       conversion_efficiency=0.95,
   )

star_tracker
~~~~~~~~~~~~

The :class:`~conops.config.StarTrackerConfiguration` configures the star tracker system.
Star trackers provide attitude determination by identifying star fields and are subject
to avoidance constraints (e.g., never look within N° of the Sun).

**StarTrackerConfiguration Attributes:**

* ``star_trackers`` (list[StarTracker]): Individual star tracker configurations
* ``min_functional_trackers`` (int): Minimum trackers not in soft violation for pointing to be valid (science-quality check).
  Hard constraints are always enforced regardless of this value.
* ``modes_require_lock`` (list[ACSMode] | None): ACS modes in which star tracker **soft** constraints are enforced as a
  science-quality check. ``None`` (default) enforces soft constraints in all modes. An empty list disables soft constraint
  checks entirely. For example, ``[ACSMode.SCIENCE]`` only applies soft constraints during science observations.

  .. note::

     **Hard** constraints are absolute health-and-safety keep-outs (e.g. sensor blinding) and are *always* enforced in
     every mode, regardless of ``modes_require_lock``. ``modes_require_lock`` only gates the science-quality **soft**
     constraint checks.

**StarTracker Attributes:**

* ``name`` (str): Tracker identifier
* ``orientation`` (:class:`~conops.config.StarTrackerOrientation`): Boresight direction in spacecraft body frame
* ``hard_constraint`` (optional): Constraint defining regions where the tracker *cannot* operate (e.g. Sun avoidance).
  Always enforced. Violations are recorded in the ``star_tracker_hard_violations`` telemetry field.
* ``soft_constraint`` (optional): Constraint defining regions of degraded performance (science-quality check).
  Enforced only in modes listed in ``StarTrackerConfiguration.modes_require_lock``. Violations are recorded in
  ``star_tracker_soft_violations``.

**StarTrackerOrientation Attributes:**

* ``boresight`` (tuple[float, float, float]): Boresight direction as a unit vector in spacecraft body frame

  - +x is the spacecraft pointing direction (forward/boresight)
  - +y is the spacecraft "up" direction
  - +z completes the right-handed coordinate system

Star Tracker Vector Helper Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`~conops.config.star_tracker.create_star_tracker_vector` helper converts
roll, pitch, and yaw Euler angles to a boresight vector, mirroring the solar panel
helper:

.. code-block:: python

   from conops.config.star_tracker import create_star_tracker_vector

   # Star tracker pointing along spacecraft boresight (+X)
   boresight = create_star_tracker_vector(roll_deg=0, pitch_deg=0, yaw_deg=0)
   # Result: (1.0, 0.0, 0.0)

   # Star tracker rotated 90° in pitch to point "up" (+Z)
   boresight = create_star_tracker_vector(roll_deg=0, pitch_deg=90, yaw_deg=0)

   # Star tracker angled 45° to the side
   boresight = create_star_tracker_vector(roll_deg=0, pitch_deg=0, yaw_deg=45)

Euler angles use the ZYX convention: yaw about Z, then pitch about Y, then roll about X.

**Example:**

.. code-block:: python

   from conops.config import (
       SpacecraftBus,
       StarTracker,
       StarTrackerConfiguration,
       StarTrackerOrientation,
   )
   from conops.config.star_tracker import create_star_tracker_vector
   from conops.common import ACSMode
   from rust_ephem import SunConstraint, EarthLimbConstraint

   # Build two star trackers at different orientations
   st1 = StarTracker(
       name="ST1",
       orientation=StarTrackerOrientation(
           boresight=create_star_tracker_vector(pitch_deg=45),  # 45° off boresight
       ),
       hard_constraint=SunConstraint(min_angle=30.0),   # Always enforced: never look within 30° of Sun
       soft_constraint=SunConstraint(min_angle=45.0),   # Science-quality: degraded within 45° of Sun
   )

   st2 = StarTracker(
       name="ST2",
       orientation=StarTrackerOrientation(
           boresight=create_star_tracker_vector(pitch_deg=-45),  # Opposite side
       ),
       hard_constraint=SunConstraint(min_angle=30.0),
   )

   star_trackers = StarTrackerConfiguration(
       star_trackers=[st1, st2],
       min_functional_trackers=1,      # At least one must satisfy soft constraint
       modes_require_lock=[ACSMode.SCIENCE],  # Enforce soft constraints in science mode only
   )

   # Attach to spacecraft bus
   spacecraft_bus = SpacecraftBus(
       name="Observatory Bus",
       star_trackers=star_trackers,
       # ... other bus fields ...
   )

The ACS monitors star tracker constraints at each timestep and records:

* ``star_tracker_hard_violations``: Number of trackers violating their hard constraint (always monitored)
* ``star_tracker_soft_violations``: Whether any tracker is in its soft constraint zone
* ``star_tracker_functional_count``: Number of trackers at full science quality (not degraded by soft constraints)

Hard violations are health-and-safety events and always cause a pointing-invalid result. Soft violations
only affect pointing validity in modes listed in ``StarTrackerConfiguration.modes_require_lock``.

When star trackers are configured, ``MissionConfig.init_fault_management_defaults()`` automatically adds
a ``star_tracker_functional_count`` threshold (``direction="below"``, both yellow and red set to
``num_trackers - 1``) so that degradation of any tracker to soft constraints triggers a fault alert.

payload
~~~~~~~

The :class:`~conops.config.Payload` contains the science instruments.

**Payload Attributes:**

* ``instruments`` (list[Instrument]): List of instrument configurations

**Instrument Attributes:**

* ``name`` (str): Instrument identifier
* ``power_draw`` (:class:`~conops.config.PowerDraw`): Power consumption
* ``heater`` (:class:`~conops.config.Heater`): Optional thermal heater
* ``data_generation`` (:class:`~conops.config.DataGeneration`): Data output characteristics

.. code-block:: python

   from conops.config import Payload, Instrument, PowerDraw, DataGeneration

   payload = Payload(
       instruments=[
           Instrument(
               name="X-ray Telescope",
               power_draw=PowerDraw(
                   nominal_power=100.0,
                   peak_power=150.0,
                   power_mode={0: 120.0, 1: 80.0},
               ),
               data_generation=DataGeneration(
                   rate_gbps=0.001,  # 1 Mbps continuous
               ),
           ),
           Instrument(
               name="Star Tracker",
               power_draw=PowerDraw(nominal_power=15.0),
               data_generation=DataGeneration(
                   per_observation_gb=0.01,  # Fixed per observation
               ),
           ),
       ]
   )

battery
~~~~~~~

The :class:`~conops.config.Battery` models the spacecraft battery system.

**Attributes:**

* ``name`` (str): Battery identifier
* ``amphour`` (float): Battery capacity in amp-hours
* ``voltage`` (float): Battery voltage (Volts)
* ``watthour`` (float): Total energy capacity (Watt-hours, auto-calculated)
* ``max_depth_of_discharge`` (float): Maximum allowed DoD (0-1)
* ``recharge_threshold`` (float): SOC level to end emergency recharge (0-1)
* ``charge_level`` (float): Current charge in Watt-hours

.. code-block:: python

   from conops.config import Battery

   battery = Battery(
       name="Primary Battery",
       amphour=20.0,
       voltage=28.0,
       watthour=560.0,  # Optional, calculated from amphour * voltage
       max_depth_of_discharge=0.4,  # Allow 40% discharge
       recharge_threshold=0.95,     # Recharge until 95% SOC
   )

constraint
~~~~~~~~~~

The :class:`~conops.config.Constraint` defines pointing constraints for the spacecraft.

**Attributes:**

* ``sun_constraint``: Minimum angle from the Sun
* ``anti_sun_constraint``: Maximum angle from anti-Sun direction
* ``moon_constraint``: Minimum angle from the Moon
* ``earth_constraint``: Minimum angle from Earth limb
* ``panel_constraint``: Solar panel pointing constraint
* ``ephem``: Ephemeris object (set at runtime, not serialized)

.. code-block:: python

   import rust_ephem
   from conops.config import Constraint

   constraint = Constraint(
       sun_constraint=rust_ephem.SunConstraint(min_angle=45.0),
       moon_constraint=rust_ephem.MoonConstraint(min_angle=20.0),
       earth_constraint=rust_ephem.EarthLimbConstraint(min_angle=15.0),
       panel_constraint=(
           rust_ephem.SunConstraint(min_angle=45.0, max_angle=135.0)
           & ~rust_ephem.EclipseConstraint()
       ),
   )

   # Set ephemeris at runtime
   constraint.ephem = ephemeris

ground_stations
~~~~~~~~~~~~~~~

The :class:`~conops.config.GroundStationRegistry` contains all ground stations.

**GroundStationRegistry Methods:**

* ``add(station)``: Add a ground station
* ``get(code)``: Get station by code
* ``codes()``: List all station codes
* ``default()``: Get pre-populated registry

**GroundStation Attributes:**

* ``code`` (str): Short identifier (e.g., "MAL", "SGS")
* ``name`` (str): Human-readable name
* ``latitude_deg`` (float): Latitude in degrees
* ``longitude_deg`` (float): Longitude in degrees
* ``elevation_m`` (float): Elevation in meters
* ``min_elevation_deg`` (float): Minimum pass elevation
* ``schedule_probability`` (float): Probability of scheduling (0-1)
* ``bands`` (list[BandCapability]): Supported frequency bands
* ``gain_db`` (float | None): Antenna gain in dB

.. code-block:: python

   from conops.config import GroundStationRegistry, GroundStation, BandCapability

   ground_stations = GroundStationRegistry(
       stations=[
           GroundStation(
               code="MAL",
               name="Malindi",
               latitude_deg=-3.22,
               longitude_deg=40.12,
               elevation_m=0.0,
               min_elevation_deg=10.0,
               schedule_probability=1.0,
               bands=[
                   BandCapability(
                       band="S",
                       uplink_rate_mbps=2.0,
                       downlink_rate_mbps=10.0,
                   ),
               ],
           ),
           GroundStation(
               code="SGS",
               name="Svalbard",
               latitude_deg=78.229,
               longitude_deg=15.407,
               min_elevation_deg=5.0,
               schedule_probability=0.8,
               bands=[
                   BandCapability(band="X", downlink_rate_mbps=150.0),
               ],
           ),
       ]
   )

recorder
~~~~~~~~

The :class:`~conops.config.OnboardRecorder` simulates onboard data storage.

**Attributes:**

* ``name`` (str): Recorder identifier
* ``capacity_gb`` (float): Maximum capacity in Gigabits
* ``current_volume_gb`` (float): Current stored data in Gigabits
* ``yellow_threshold`` (float): Warning threshold (fraction, 0-1)
* ``red_threshold`` (float): Critical threshold (fraction, 0-1)

.. code-block:: python

   from conops.config import OnboardRecorder

   recorder = OnboardRecorder(
       name="Solid State Recorder",
       capacity_gb=128.0,
       current_volume_gb=0.0,
       yellow_threshold=0.7,  # 70% full warning
       red_threshold=0.9,     # 90% full critical
   )

fault_management
~~~~~~~~~~~~~~~~

The :class:`~conops.config.FaultManagement` monitors parameters and triggers safe mode.

**Attributes:**

* ``thresholds`` (list[FaultThreshold]): Parameter thresholds
* ``red_limit_constraints`` (list[FaultConstraint]): Pointing constraints
* ``safe_mode_on_red`` (bool): Trigger safe mode on RED conditions

**FaultThreshold Attributes:**

* ``name`` (str): Parameter name to monitor
* ``yellow`` (float): Yellow (warning) threshold
* ``red`` (float): Red (critical) threshold
* ``direction`` (str): "below" or "above"

.. code-block:: python

   from conops.config import FaultManagement, FaultThreshold
   import rust_ephem

   fault_management = FaultManagement(
       thresholds=[
           FaultThreshold(
               name="battery_level",
               yellow=0.5,
               red=0.4,
               direction="below",  # Alert when value drops below threshold
           ),
           FaultThreshold(
               name="recorder_fill_fraction",
               yellow=0.7,
               red=0.9,
               direction="above",  # Alert when value exceeds threshold
           ),
       ],
       safe_mode_on_red=True,
   )

   # Add red limit constraint for Sun avoidance
   fault_management.add_red_limit_constraint(
       name="Sun Avoidance",
       constraint=rust_ephem.SunConstraint(min_angle=20.0) & ~rust_ephem.EclipseConstraint(),
       time_threshold_seconds=120.0,
       description="Avoid pointing within 20° of Sun for more than 2 minutes",
   )

observation_categories
~~~~~~~~~~~~~~~~~~~~~~

The :class:`~conops.config.ObservationCategories` defines how observations are categorized
based on their target ID (obsid) for visualization purposes.

**Attributes:**

* ``categories`` (list[ObservationCategory]): Category definitions
* ``default_name`` (str): Default category name
* ``default_color`` (str): Default visualization color

.. code-block:: python

   from conops.config import ObservationCategories, ObservationCategory

   categories = ObservationCategories(
       categories=[
           ObservationCategory(
               name="Science",
               obsid_min=10000,
               obsid_max=30000,
               color="tab:blue",
           ),
           ObservationCategory(
               name="Calibration",
               obsid_min=90000,
               obsid_max=91000,
               color="gray",
           ),
       ],
       default_name="Other",
       default_color="tab:purple",
   )

   # Or use defaults
   categories = ObservationCategories.default_categories()

visualization
~~~~~~~~~~~~~

The :class:`~conops.config.VisualizationConfig` controls plot appearance.
This field is excluded from JSON serialization.

**Attributes:**

* ``mode_colors`` (dict): Colors for ACS modes
* ``font_family`` (str): Font for plots
* ``title_font_size`` (int): Title font size
* ``label_font_size`` (int): Axis label font size
* ``figsize`` (tuple): Default figure size
* ``timeline_figsize`` (tuple): DITL timeline figure size
* ``data_telemetry_figsize`` (tuple): Data management plot size

.. code-block:: python

   from conops.config import VisualizationConfig

   visualization = VisualizationConfig(
       mode_colors={
           "SCIENCE": "green",
           "SLEWING": "orange",
           "SAA": "purple",
           "PASS": "cyan",
           "CHARGING": "yellow",
           "SAFE": "red",
       },
       font_family="DejaVu Sans",
       title_font_size=14,
       figsize=(16, 10),
   )

Complete Programmatic Example
-----------------------------

Here is a complete example of creating a ``MissionConfig`` programmatically:

.. code-block:: python

   from conops.config import (
       MissionConfig,
       SpacecraftBus,
       SolarPanelSet,
       SolarPanel,
       Payload,
       Instrument,
       Battery,
       Constraint,
       GroundStationRegistry,
       GroundStation,
       OnboardRecorder,
       FaultManagement,
       PowerDraw,
       DataGeneration,
       BandCapability,
       AttitudeControlSystem,
       Heater,
   )
   from conops.common import ACSMode
   import rust_ephem

   # Create fault management with custom thresholds
   fault_management = FaultManagement()
   fault_management.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")

   # Create the complete configuration
   config = MissionConfig(
       name="Example Observatory",
       spacecraft_bus=SpacecraftBus(
           name="Observatory Bus",
           power_draw=PowerDraw(
               nominal_power=50.0,
               peak_power=300.0,
               power_mode={0: 70.0, 1: 100.0},
               eclipse_power=75.0
           ),
           attitude_control=AttitudeControlSystem(
               slew_acceleration=0.01,
               max_slew_rate=0.3,
               slew_accuracy=0.01,
               settle_time=10.0
           ),
           heater=Heater(
               name="Bus Heaters",
               power_draw=PowerDraw(
                   nominal_power=25.0,
                   eclipse_power=75.0
               )
           )
       ),
       solar_panel=SolarPanelSet(
           name="Solar Array",
           panels=[
               SolarPanel(
                   name="Panel A",
                   gimbled=False,
                   sidemount=True,
                   max_power=500.0,
                   conversion_efficiency=0.94
               )
           ],
           conversion_efficiency=0.95
       ),
       payload=Payload(
           instruments=[
               Instrument(
                   name="Main Instrument",
                   power_draw=PowerDraw(nominal_power=100.0, peak_power=150.0),
                   data_generation=DataGeneration(rate_gbps=0.001)
               )
           ]
       ),
       battery=Battery(
           amphour=20.0,
           voltage=28.0,
           max_depth_of_discharge=0.4,
           recharge_threshold=0.95
       ),
       constraint=Constraint(
           sun_constraint=rust_ephem.SunConstraint(min_angle=45.0),
           moon_constraint=rust_ephem.MoonConstraint(min_angle=20.0),
           earth_constraint=rust_ephem.EarthLimbConstraint(min_angle=15.0)
       ),
       ground_stations=GroundStationRegistry(
           stations=[
               GroundStation(
                   code="GND",
                   name="Ground Station",
                   latitude_deg=35.0,
                   longitude_deg=-106.0,
                   min_elevation_deg=10.0,
                   bands=[BandCapability(band="S", downlink_rate_mbps=10.0)]
               )
           ]
       ),
       recorder=OnboardRecorder(
           capacity_gb=64.0,
           yellow_threshold=0.7,
           red_threshold=0.9
       ),
       fault_management=fault_management
   )

Automatic Fault Thresholds
--------------------------

When a ``MissionConfig`` is created, it automatically initializes default fault thresholds
based on the battery and recorder configuration:

1. **battery_level**: Yellow at ``1.0 - max_depth_of_discharge``, Red 10% below that
2. **recorder_fill_fraction**: Uses the recorder's ``yellow_threshold`` and ``red_threshold``
3. **star_tracker_functional_count**: When star trackers are configured, both yellow and red are set to
   ``num_trackers - 1`` with ``direction="below"``. This fires the moment any tracker enters a hard
   constraint zone (``functional_count`` drops from ``num_trackers`` to ``num_trackers - 1``), making
   any hard violation immediately critical. No threshold is added if no star trackers are configured.

You can override these by adding custom thresholds to the ``FaultManagement`` instance before calling
``init_fault_management_defaults()`` (defaults are skipped if a threshold for that parameter already exists).

.. code-block:: python

   from conops.config import MissionConfig, FaultManagement
   from conops.common import ACSMode

   # Create config (automatically adds default thresholds on first use)
   config = MissionConfig()

   # Override the auto-configured battery threshold
   config.fault_management.add_threshold(
       "battery_level",
       yellow=0.35,
       red=0.25,
       direction="below",
   )

Using with DITL Simulation
--------------------------

Pass the configuration to DITL for simulation:

.. code-block:: python

   from conops.ditl import DITL, QueueDITL
   from datetime import datetime, timedelta

   # Create or load configuration
   config = MissionConfig.from_json_file("config.json")

   # Set ephemeris on constraint (required at runtime)
   config.constraint.ephem = ephemeris

   # Run simulation
   ditl = DITL(
       config=config,
       ephem=ephemeris,
       begin=datetime(2025, 1, 1),
       end=datetime(2025, 1, 2),
   )
   ditl.calc()

   # Or use QueueDITL for target-driven simulation
   queue_ditl = QueueDITL(
       config=config,
       ephem=ephemeris,
       begin=datetime(2025, 1, 1),
       end=datetime(2025, 1, 2),
   )
   queue_ditl.run()

API Reference
-------------

For detailed API documentation, see:

* :class:`~conops.config.MissionConfig`
* :class:`~conops.config.SpacecraftBus`
* :class:`~conops.config.SolarPanelSet`
* :class:`~conops.config.SolarPanel`
* :class:`~conops.config.Payload`
* :class:`~conops.config.Instrument`
* :class:`~conops.config.Battery`
* :class:`~conops.config.Constraint`
* :class:`~conops.config.GroundStationRegistry`
* :class:`~conops.config.GroundStation`
* :class:`~conops.config.OnboardRecorder`
* :class:`~conops.config.FaultManagement`
* :class:`~conops.config.AttitudeControlSystem`
* :class:`~conops.config.PowerDraw`
* :class:`~conops.config.DataGeneration`
* :class:`~conops.config.CommunicationsSystem`
* :class:`~conops.config.ObservationCategories`
* :class:`~conops.config.VisualizationConfig`
