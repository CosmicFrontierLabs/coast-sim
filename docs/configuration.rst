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

**4. Save to JSON File**

Save a configuration to JSON for version control or sharing:

.. code-block:: python

   config.to_json_file("spacecraft_config.json")

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
* ``sidemount`` (bool): Whether panel normal is perpendicular to boresight
* ``cant_x`` (float): Cant angle around X-axis (degrees)
* ``cant_y`` (float): Cant angle around Y-axis (degrees)
* ``azimuth_deg`` (float): Structural placement angle around boresight (degrees)
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
               sidemount=True,
               cant_x=0.0,
               azimuth_deg=0.0,
               max_power=400.0,
               conversion_efficiency=0.94,
           ),
           SolarPanel(
               name="Panel -Y",
               gimbled=False,
               sidemount=True,
               cant_x=0.0,
               azimuth_deg=180.0,
               max_power=400.0,
               conversion_efficiency=0.94,
           ),
       ],
       conversion_efficiency=0.95,
   )

payload
~~~~~~~

The :class:`~conops.config.Payload` contains the science instruments.

**Payload Attributes:**

* ``payload`` (list[Instrument]): List of instrument configurations

**Instrument Attributes:**

* ``name`` (str): Instrument identifier
* ``power_draw`` (:class:`~conops.config.PowerDraw`): Power consumption
* ``heater`` (:class:`~conops.config.Heater`): Optional thermal heater
* ``data_generation`` (:class:`~conops.config.DataGeneration`): Data output characteristics

.. code-block:: python

   from conops.config import Payload, Instrument, PowerDraw, DataGeneration

   payload = Payload(
       payload=[
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

Complete JSON Example
---------------------

Here is a complete example of a ``MissionConfig`` in JSON format:

.. code-block:: json

   {
       "name": "Example Observatory",
       "spacecraft_bus": {
           "name": "Observatory Bus",
           "power_draw": {
               "nominal_power": 50.0,
               "peak_power": 300.0,
               "power_mode": {"0": 70.0, "1": 100.0},
               "eclipse_power": 75.0
           },
           "attitude_control": {
               "slew_acceleration": 0.01,
               "max_slew_rate": 0.3,
               "slew_accuracy": 0.01,
               "settle_time": 10.0
           },
           "heater": {
               "name": "Bus Heaters",
               "power_draw": {
                   "nominal_power": 25.0,
                   "eclipse_power": 75.0
               }
           }
       },
       "solar_panel": {
           "name": "Solar Array",
           "panels": [
               {
                   "name": "Panel A",
                   "gimbled": false,
                   "sidemount": true,
                   "max_power": 500.0,
                   "conversion_efficiency": 0.94
               }
           ],
           "conversion_efficiency": 0.95
       },
       "payload": {
           "payload": [
               {
                   "name": "Main Instrument",
                   "power_draw": {
                       "nominal_power": 100.0,
                       "peak_power": 150.0
                   },
                   "data_generation": {
                       "rate_gbps": 0.001
                   }
               }
           ]
       },
       "battery": {
           "amphour": 20.0,
           "voltage": 28.0,
           "max_depth_of_discharge": 0.4,
           "recharge_threshold": 0.95
       },
       "constraint": {
           "sun_constraint": {
               "type": "sun",
               "min_angle": 45.0
           },
           "moon_constraint": {
               "type": "moon",
               "min_angle": 20.0
           },
           "earth_constraint": {
               "type": "earth_limb",
               "min_angle": 15.0
           }
       },
       "ground_stations": {
           "stations": [
               {
                   "code": "GND",
                   "name": "Ground Station",
                   "latitude_deg": 35.0,
                   "longitude_deg": -106.0,
                   "min_elevation_deg": 10.0,
                   "bands": [{"band": "S", "downlink_rate_mbps": 10.0}]
               }
           ]
       },
       "recorder": {
           "capacity_gb": 64.0,
           "yellow_threshold": 0.7,
           "red_threshold": 0.9
       },
       "fault_management": {
           "thresholds": [
               {"name": "battery_level", "yellow": 0.5, "red": 0.4, "direction": "below"}
           ],
           "safe_mode_on_red": true
       }
   }

Automatic Fault Thresholds
--------------------------

When a ``MissionConfig`` is created, it automatically initializes default fault thresholds
based on the battery and recorder configuration:

1. **battery_level**: Yellow at ``1.0 - max_depth_of_discharge``, Red 10% below that
2. **recorder_fill_fraction**: Uses the recorder's ``yellow_threshold`` and ``red_threshold``

You can override these by providing custom thresholds in the ``fault_management`` configuration.

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
