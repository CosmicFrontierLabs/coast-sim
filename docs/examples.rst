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

Dynamic Merit and Targets of Opportunity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Dynamic_Merit_Example.ipynb``

This notebook demonstrates how the queue scheduler chooses its next target, turning on
each dynamic merit term in turn:

* Urgency, which observes a short-window request that priority order would lose
* Tiers, which give a class of work absolute priority
* Cadence, which revisits monitoring targets on schedule
* Completion deficit, which keeps programs near their allocated share of time
* Targets of Opportunity as ordinary short-window requests, including when they
  interrupt and when they wait

See :doc:`configuration` (Dynamic Merit) and :doc:`target_of_opportunity` for reference.

Rolling-Horizon Replanning
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Rolling_Horizon_DITL_Example.ipynb``

This notebook simulates a ground-planned mission with ``RollingHorizonDITL``:

* Planning 12 hours ahead and replanning every 6 hours, with a commit lead time
* A Target of Opportunity that triggers a rapid replan and interrupts an observation
* What each replan kept, dropped and added, and how long it took
* Checking that the simulation executed the plan it followed
* A comparison with ``QueueDITL`` dispatch on the same targets

See :doc:`planning` (Rolling-horizon replanning) for reference.

Local Search Planning
~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Local_Search_Scheduler_Example.ipynb``

This notebook compares ``LocalSearchScheduler`` with ``PriorityScheduler`` on 100 random
targets:

* One-day plans from each, executed to check that they run exactly as planned
* Where the time goes: science, slewing and idle time, and the two timelines
* Merit-weighted science through the day, and which targets gained or lost time
* How the result and the planning time grow with the number of changes tried
* A benchmark of ``RollingHorizonDITL`` runs with each scheduler over several target sets

See :doc:`planning` (Improving plans by local search) for reference.

CP-SAT Planning
~~~~~~~~~~~~~~~

**Location**: ``examples/CP_SAT_Scheduler_Example.ipynb``

This notebook compares ``CpSatScheduler`` with ``LocalSearchScheduler``, with
``PriorityScheduler`` as the baseline, on 100 random targets (it needs the ``cpsat``
extra):

* One-day plans from each, with the same planning time, executed to check that they
  run exactly as planned
* Science, merit-weighted science, idle time and targets completed
* The gain in each of CP-SAT's chunks, and through the day
* Which targets gained or lost time
* How each scheduler's result grows with planning time
* A benchmark of ``RollingHorizonDITL`` runs with each scheduler over several target sets

See :doc:`planning` (Optimizing plans with CP-SAT) for reference.

Planning Within the Recorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/Recorder_Constraint_Example.ipynb``

This notebook plans a day for a spacecraft with a 32 Gb recorder, housekeeping data,
fifty-two faint targets and eight bright ones whose single 20-minute visit is 5 Gb
each:

* One-day plans with and without the recorder's planning limit, executed to show the
  recorder staying within the limit in one, with the bright visits rationed around
  the contacts, and filling, losing data and safing the spacecraft in the other
* The scheduler's ledger: what each contact downlinks and the recorder's peak before
  the next one
* Which targets are observed under the limit, and how local search fills the time
  the recorder left idle, trading on merit-weighted science within the limit
* Keeping margin with a lower ``planning_limit``, and rolling-horizon replans that
  start from the recorder's actual volume

See :doc:`planning` (Planning within the recorder) and :doc:`data_management` for
reference.

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
       ScienceData,
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
       data_generation=DataGeneration(science=ScienceData(rate_gbps=0.5))  # 0.5 Gbps while collecting
   )

   spectrometer = Instrument(
       name="Spectrometer",
       power_draw=PowerDraw(nominal_power=50),
       data_generation=DataGeneration(science=ScienceData(per_observation_gb=2.0))  # 2 Gb per obs
   )

   payload = Payload(instruments=[camera, spectrometer])

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
