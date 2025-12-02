Examples
========

COASTSim includes comprehensive example Jupyter notebooks in the ``examples/`` directory
that demonstrate various features and workflows.

Example Notebooks
-----------------

Example Spacecraft DITL
~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Example_Spacecraft_DITL.ipynb``

This notebook demonstrates a complete spacecraft DITL simulation with:

* Custom spacecraft configuration
* Power modeling (battery and solar panels)
* Attitude control system simulation
* Observation scheduling with constraints
* Ground station pass calculations
* South Atlantic Anomaly (SAA) avoidance
* Timeline visualization and analysis

Key features demonstrated:

.. code-block:: python

   # Configure spacecraft components
   battery = Battery(capacity=100.0, initial_charge=80.0)
   solar_panels = SolarPanelSet(panels=[...])
   acs = AttitudeControlSystem(slew_acceleration=0.01)

   # Set up instruments
   instruments = InstrumentSet(instruments=[...])

   # Create spacecraft bus
   bus = SpacecraftBus(
       battery=battery,
       solar_panels=solar_panels,
       acs=acs,
       instruments=instruments
   )

   # Run simulation
   ditl = QueueDITL(config, ephemeris, begin, end)
   ditl.run()

Example DITL from JSON
~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Example_DITL_from_JSON.ipynb``

This notebook shows a simplified workflow using JSON configuration files:

* Loading spacecraft configuration from JSON
* Quick DITL setup and execution
* Basic analysis and visualization

This approach is ideal for rapid prototyping and testing different configurations:

.. code-block:: python

   from conops import MissionConfig, QueueDITL

   # Load everything from configuration
   config = MissionConfig.from_json("example_config.json")

   # Run simulation
   ditl = QueueDITL(config, ephemeris, begin, end)
   ditl.run()

   # Analyze
   ditl.plot()

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/example_config.json``

A complete JSON configuration file showing all available parameters:

* Spacecraft bus configuration
* Solar panel specifications
* Instrument definitions
* Battery parameters
* Attitude control system settings
* Ground station network
* Constraint definitions

Example TLE File
~~~~~~~~~~~~~~~~

**Location**: ``examples/example.tle``

Sample Two-Line Element (TLE) file for orbit propagation testing.

Running the Examples
--------------------

To run the example notebooks:

.. code-block:: bash

   cd examples
   jupyter notebook

Then open any of the ``.ipynb`` files in your browser.

Common Use Cases
----------------

Mission Planning
~~~~~~~~~~~~~~~~

Use COASTSim to evaluate operational scenarios before launch:

* Test different target scheduling strategies
* Validate power budget over multiple orbits
* Assess ground station coverage
* Optimize observation efficiency

Performance Analysis
~~~~~~~~~~~~~~~~~~~~

Assess spacecraft performance:

* Power generation vs. consumption analysis
* Thermal constraints during different operational modes
* Data downlink capacity and scheduling
* Slew time and observation efficiency

Constraint Validation
~~~~~~~~~~~~~~~~~~~~~

Verify observational constraints are satisfied:

* Sun angle avoidance
* Moon exclusion zones
* Earth limb constraints
* South Atlantic Anomaly avoidance
* Custom geometric constraints

Schedule Optimization
~~~~~~~~~~~~~~~~~~~~~

Test different scheduling algorithms:

* Priority-based scheduling
* Greedy vs. optimal schedulers
* Target visibility windows
* Constraint satisfaction

Data Management
~~~~~~~~~~~~~~~

Configure and simulate onboard data storage, generation, and downlink:

.. code-block:: python

   from conops.config import (
       DataGeneration,
       Instrument,
       Payload,
       OnboardRecorder,
       Antenna,
       GroundStation
   )

   # Configure instruments with data generation
   camera = Instrument(
       name="High-Res Camera",
       power_draw=PowerDraw(nominal_power=100),
       data_generation=DataGeneration(rate_gbps=0.5)  # 0.5 Gbps continuous
   )

   spectrometer = Instrument(
       name="Spectrometer",
       power_draw=PowerDraw(nominal_power=50),
       data_generation=DataGeneration(per_observation_gb=2.0)  # 2 Gb per obs
   )

   payload = Payload(payload=[camera, spectrometer])

   # Configure onboard data recorder
   recorder = OnboardRecorder(
       name="Solid State Recorder",
       capacity_gb=128.0,  # 128 Gigabits capacity
       yellow_threshold=0.7,  # 70% full warning
       red_threshold=0.9  # 90% full critical
   )

   # Configure ground station with downlink capability
   station = GroundStation(
       code="GND",
       name="Ground Station",
       latitude_deg=35.0,
       longitude_deg=-106.0,
       antenna=Antenna(
           bands=["X"],
           max_data_rate_mbps=100.0  # 100 Mbps downlink
       )
   )

   # Add to configuration
   config = MissionConfig(
       spacecraft_bus=spacecraft_bus,
       payload=payload,
       recorder=recorder,
       ground_stations=GroundStationRegistry(stations=[station]),
       # ... other config
   )

   # Run simulation - data automatically generated and downlinked
   ditl = DITL(config=config)
   ditl.calc()

   # Analyze data management
   import matplotlib.pyplot as plt

   plt.figure(figsize=(12, 8))

   # Plot recorder fill level
   plt.subplot(3, 1, 1)
   plt.plot(ditl.utime, ditl.recorder_fill_fraction)
   plt.ylabel('Recorder Fill Fraction')
   plt.title('Onboard Data Recorder Status')

   # Plot data generation
   plt.subplot(3, 1, 2)
   plt.plot(ditl.utime, ditl.data_generated_gb)
   plt.ylabel('Data Generated (Gb)')

   # Plot data downlink
   plt.subplot(3, 1, 3)
   plt.plot(ditl.utime, ditl.data_downlinked_gb)
   plt.ylabel('Data Downlinked (Gb)')
   plt.xlabel('Time')

   plt.tight_layout()
   plt.show()

Key features:

* **Rate-based generation**: Instruments generate data continuously at specified Gbps
* **Per-observation generation**: Instruments generate fixed data amount per observation
* **Recorder management**: Automatic capacity limiting and alert thresholds
* **Downlink simulation**: Data automatically downlinked during ground station passes
* **Alert integration**: Recorder fill level monitored by fault management system

Creating Your Own Examples
---------------------------

When creating your own simulations:

1. Start with the example configuration as a template
2. Modify spacecraft parameters to match your mission
3. Define your target list or observation queue
4. Set appropriate constraints for your mission
5. Run the simulation over your desired time period
6. Analyze and visualize the results

For detailed API documentation, see :doc:`api/modules`.
