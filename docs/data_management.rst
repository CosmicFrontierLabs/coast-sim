Data Management
===============

COASTSim provides comprehensive data management capabilities for simulating onboard data storage,
generation, and downlink operations. This system allows mission planners to evaluate:

* Data recorder sizing requirements
* Downlink pass frequency and duration needs
* Data generation rates from science instruments
* Storage capacity utilization over mission timelines
* Alert thresholds for recorder fullness

Overview
--------

The data management system consists of three main components:

1. **OnboardRecorder**: Simulates the spacecraft's solid-state recorder (SSR) or data storage device
2. **DataGeneration**: Models how the bus and each instrument produce data, and how targets change it
3. **Downlink Operations**: Simulates data transfer during ground station passes

These components work together seamlessly in DITL simulations to track data flow throughout
the spacecraft's operations.

OnboardRecorder
---------------

The :class:`~conops.config.recorder.OnboardRecorder` class simulates an onboard data storage
device with configurable capacity and alert thresholds.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from conops.config import OnboardRecorder

   # Create a recorder with 128 Gigabit capacity
   recorder = OnboardRecorder(
       name="Solid State Recorder",
       capacity_gb=128.0,          # Maximum capacity in Gigabits
       current_volume_gb=0.0,      # Starting empty
       yellow_threshold=0.7,       # Warning at 70% full
       red_threshold=0.9           # Critical at 90% full
   )

Key Features
~~~~~~~~~~~~

**Capacity Management**

The recorder automatically caps data storage at the configured capacity. When the recorder
is full, additional data generation is lost (simulating data that cannot be stored).

.. code-block:: python

   # Add data to recorder
   data_stored = recorder.add_data(10.5)  # Returns amount actually stored

   # Check current state
   print(f"Current volume: {recorder.current_volume_gb} Gb")
   print(f"Fill fraction: {recorder.get_fill_fraction()}")
   print(f"Available space: {recorder.available_capacity()} Gb")

**Alert System**

The recorder provides three alert levels based on fill fraction:

* **0 (Green)**: Below yellow threshold - normal operations
* **1 (Yellow)**: At or above yellow threshold - warning state
* **2 (Red)**: At or above red threshold - critical state

.. code-block:: python

   alert_level = recorder.get_alert_level()
   if alert_level == 2:
       print("CRITICAL: Recorder nearly full!")
   elif alert_level == 1:
       print("WARNING: Recorder filling up")

**Data Removal**

During ground station passes, data is removed from the recorder to simulate downlink:

.. code-block:: python

   # Remove data during downlink
   data_downlinked = recorder.remove_data(25.0)  # Returns amount actually removed
   print(f"Downlinked {data_downlinked} Gb")

DataGeneration
--------------

Data comes in two kinds, and what drives each is different:

* **Science data** depends on *what is observed*, and is generated only while
  observing it: :class:`~conops.config.ScienceData` is ``rate_gbps`` for every
  second collected, and ``per_observation_gb``, a fixed amount each observation
  generates whatever its length, such as a frame readout or headers. An instrument
  sets what it generates by default, and a target can set its own for each
  instrument (below).
* **Housekeeping data** depends on *what the spacecraft is doing*, and never stops:
  :class:`~conops.config.HousekeepingData` is ``rate_gbps`` at all times, the
  engineering telemetry (temperatures, voltages, attitude, status) generated
  whatever is observed. ``rate_mode`` sets a different rate in particular ACS modes,
  given by name or number, the way :class:`~conops.config.PowerDraw` sets power by
  mode.

The spacecraft bus generates housekeeping data only, so its ``data_generation`` is a
``HousekeepingData``. An instrument generates both, so its ``data_generation`` is a
:class:`~conops.config.DataGeneration` with a ``science`` and a ``housekeeping`` part.

.. code-block:: python

   from conops.config import (
       DataGeneration,
       HousekeepingData,
       Instrument,
       MissionConfig,
       Payload,
       ScienceData,
       SpacecraftBus,
   )

   config = MissionConfig(
       spacecraft_bus=SpacecraftBus(
           # 1 Mbps of housekeeping, 4 Mbps of diagnostics in safe mode
           data_generation=HousekeepingData(
               rate_gbps=0.001, rate_mode={"SAFE": 0.004}
           ),
       ),
       payload=Payload(
           instruments=[
               # 0.5 Gbps while collecting, and its own housekeeping
               Instrument(
                   name="camera",
                   data_generation=DataGeneration(
                       science=ScienceData(rate_gbps=0.5),
                       housekeeping=HousekeepingData(rate_gbps=0.0002),
                   ),
               ),
               # 5 Gb for each observation, whatever its length; off in the SAA
               Instrument(
                   name="spectrometer",
                   data_generation=DataGeneration(
                       science=ScienceData(per_observation_gb=5.0),
                       housekeeping=HousekeepingData(
                           rate_gbps=0.0001, rate_mode={"SAA": 0.0}
                       ),
                   ),
               ),
           ]
       ),
   )

Each part computes its own data with ``data_generated``. Science data is what an
observation generates having collected for some seconds: its rate for those seconds
plus its fixed amount. Housekeeping data is the rate for the length of the interval,
in its ACS mode. An instrument's ``DataGeneration`` adds the two; with only a
duration, the observation collects for all of it.

.. code-block:: python

   camera = config.payload.instruments[0]
   camera.data_generation.science.data_generated(60)      # 0.5 * 60 = 30 Gb
   camera.data_generation.housekeeping.data_generated(60)  # 0.0002 * 60 = 0.012 Gb
   camera.data_generation.data_generated(60)              # 30.012 Gb

   spectrometer = config.payload.instruments[1]
   spectrometer.data_generation.science.data_generated(60)   # 5 Gb, whatever the length
   spectrometer.data_generation.data_generated(60)           # 5 + 0.0001 * 60 = 5.006 Gb

:meth:`Payload.data_generated <conops.config.Payload.data_generated>`,
:meth:`SpacecraftBus.data_generated <conops.config.SpacecraftBus.data_generated>` and
:meth:`MissionConfig.data_generated <conops.config.MissionConfig.data_generated>` sum
what their generators produce over an interval, given the ACS mode and the seconds
collected. ``housekeeping_generated`` and ``science_generated`` give the two parts:

.. code-block:: python

   from conops.common.enums import ACSMode

   # A 60 s step in safe mode
   config.housekeeping_generated(60, mode=ACSMode.SAFE)
   # An observation that has collected for 45 s
   config.science_generated(45)
   # Both: an observation collecting for 45 s of a 60 s interval
   config.data_generated(60, mode=ACSMode.SCIENCE, collection_seconds=45)

For many simulations only the science rate matters, so as a shorthand an instrument's
science fields may be given directly: ``DataGeneration(rate_gbps=0.5)``, or
``data_generation: {rate_gbps: 0.5}`` in a file, is its ``science`` part.

Target Science Data
~~~~~~~~~~~~~~~~~~~

What an observation generates depends on what is observed: a bright source fills a
photon-counting detector faster than a faint one. A target's ``science_data`` sets,
for each instrument by name, the :class:`~conops.config.ScienceData` that instrument
generates while observing it, replacing the instrument's ``rate_gbps`` and
``per_observation_gb``. Instruments it does not name generate their own, and
housekeeping data is unchanged.

.. code-block:: python

   from conops.config import ScienceData

   queue.add(
       ra=83.63,
       dec=22.01,
       obsid=1001,
       name="Crab",
       science_data={
           "camera": ScienceData(rate_gbps=1.2),
           "spectrometer": ScienceData(per_observation_gb=12.0),
       },
   )

   ditl.submit_too(
       obsid=2001, ra=180.0, dec=45.0, merit=10000.0, exptime=3600,
       name="GRB 250101A",
       science_data={"camera": ScienceData(rate_gbps=2.0)},
   )

Schedulers copy a target's ``science_data`` onto its plan entries, and it is saved
with the plan. A name that matches no configured instrument is an error when the
target is created with a configuration.

In a DITL simulation, each step generates housekeeping data for the whole step, in the
step's ACS mode. For science data it keeps what the observation in progress has
generated so far, its rate for the seconds it has collected plus its fixed amount, and
adds the step's increase, so the fixed amount is counted once. A later visit to the
same target is a new observation.

Downlink Operations
-------------------

Data downlink occurs automatically during ground station passes. The downlink rate
is configured in the ground station's antenna parameters.

Ground Station Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from conops.config import GroundStation, Antenna, GroundStationRegistry

   # Configure station with X-band downlink at 100 Mbps
   station = GroundStation(
       code="GND",
       name="Ground Station",
       latitude_deg=35.0,
       longitude_deg=-106.0,
       antenna=Antenna(
           bands=["X"],
           max_data_rate_mbps=100.0  # 100 Mbps = 0.1 Gbps
       )
   )

   registry = GroundStationRegistry(stations=[station])

Downlink Calculation
~~~~~~~~~~~~~~~~~~~~

During a ground station pass in DITL simulation:

1. The simulation identifies the active pass
2. Takes the rate the spacecraft gets from the station (:func:`~conops.config.groundstation.effective_downlink_rate_mbps`)
3. Converts rate to Gbps (divide by 1000)
4. Calculates data volume: ``rate × seconds in contact`` this step
5. Removes that amount from the recorder, alongside what the step generates

.. code-block:: python

   # Example: 100 Mbps for 60 seconds
   downlink_rate_gbps = 100.0 / 1000.0  # 0.1 Gbps
   time_step = 60  # seconds
   data_downlinked = downlink_rate_gbps * time_step  # 6.0 Gb

Housekeeping data goes on being generated through a contact, and the link carries it
along with what is already stored, so within a step the two are applied together
rather than one after the other (:meth:`~conops.config.OnboardRecorder.transfer`):
the recorder's volume moves at the net rate, it cannot overflow while the link removes
data faster than it arrives, and data generated while the recorder is empty goes
straight down the link. A contact that begins or ends partway through a step is
credited for the seconds in contact only: what is generated before it is stored
first, and what is generated after it stays on the recorder. The result is the same
whatever the step size.

Ground Station Pass Plan Entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting with version 0.2.0, :class:`~conops.ditl.queue_ditl.QueueDITL` automatically
creates plan entries for commanded ground station passes. These ``GSP`` (Ground Station Pass)
entries appear in the exported observation plan alongside science observations.

**Pass Deconfliction**

When multiple ground stations are visible simultaneously (overlapping pass windows),
COASTSim automatically selects the pass with the highest expected data volume:

.. code-block:: python

   # Selection metric: downlink_rate × pass_duration
   # Prefers: higher data volume > longer pass > earlier start time

Overlapping opportunities that are not selected are logged but not executed, preventing
accidental "tail contacts" where a second pass starts during an ongoing pass.

**Pass Metadata**

Each GSP entry includes:

* ``station``: Ground station code (e.g., ``"SGS"``, ``"TRO"``)
* ``contact_begin``: Actual pass start time
* ``contact_end``: Actual pass end time
* ``slewtime``: Time spent slewing to point at the station

The GSP entry's ``begin`` time marks when the spacecraft reserves the pass window
(potentially including slew preparation), while ``contact_begin`` marks when data
downlink actually starts.

**Example:**

.. code-block:: python

   from conops import QueueDITL

   ditl = QueueDITL(config=config, ephem=ephem, queue=queue)
   ditl.calc()

   # Find ground station passes in the plan
   for entry in ditl.plan:
       if entry.obstype == ObsType.GSP:
           downlink_time = entry.contact_end - entry.contact_begin
           rate = config.ground_stations.get(entry.station).get_overall_max_downlink()
           volume = rate * downlink_time / 1000.0  # Convert to Gb
           print(f"{entry.station}: {downlink_time}s at {rate} Mbps = {volume:.1f} Gb")

See :doc:`plan_serialization` for complete details on GSP entry format and
:doc:`communications` for pass selection behavior.

Integration with DITL
----------------------

The data management system is automatically integrated into DITL simulations.

Configuration
~~~~~~~~~~~~~

Add the recorder to your spacecraft configuration:

.. code-block:: python

   from conops.config import MissionConfig

   config = MissionConfig(
       name="Science Mission",
       spacecraft_bus=spacecraft_bus,
       solar_panel=solar_panel,
       payload=payload,
       battery=battery,
       constraint=constraint,
       ground_stations=ground_stations,
       recorder=recorder  # Add recorder here
   )

Telemetry Output
~~~~~~~~~~~~~~~~

During simulation, DITL automatically records data management telemetry in structured format:

.. code-block:: python

   from conops.ditl import QueueDITL
   from conops import MissionConfig

   config = MissionConfig()
   ditl = QueueDITL(config=config)
   ditl.calc()

   # Access data management telemetry through structured telemetry system
   telemetry = ditl.telemetry

   # Housekeeping data (recorded at each timestep)
   recorder_volume = telemetry.housekeeping.recorder_volume_gb      # Current volume at each timestep
   fill_fraction = telemetry.housekeeping.recorder_fill_fraction    # Fill level (0-1) at each timestep
   alert_level = telemetry.housekeeping.recorder_alert              # Alert level (0/1/2) at each timestep

   # Payload data (recorded when data is generated)
   data_events = telemetry.data  # List of PayloadData records
   data_sizes = [pd.data_size_gb for pd in data_events]

For backward compatibility, the legacy array access is still available:

.. code-block:: python

   # Legacy access (still supported)
   recorder_volume = ditl.recorder_volume_gb
   fill_fraction = ditl.recorder_fill_fraction
   alert_level = ditl.recorder_alert
   data_generated = ditl.data_generated_gb
   data_downlinked = ditl.data_downlinked_gb

Visualization Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from conops.visualization import plot_data_management_telemetry

   # Use the built-in plotting function
   fig, axes = plot_data_management_telemetry(ditl)
   plt.show()

   # Or create custom plots using telemetry data
   fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

   # Get timestamps for plotting
   times = telemetry.housekeeping.timestamp

   # Recorder volume
   axes[0].plot(times, telemetry.housekeeping.recorder_volume_gb, 'b-')
   axes[0].set_ylabel('Volume (Gb)')
   axes[0].set_title('Onboard Recorder Status')
   axes[0].grid(True)

   # Fill fraction with thresholds
   axes[1].plot(times, telemetry.housekeeping.recorder_fill_fraction, 'g-', label='Fill Level')
   axes[1].axhline(y=0.7, color='orange', linestyle='--', label='Yellow Threshold')
   axes[1].axhline(y=0.9, color='red', linestyle='--', label='Red Threshold')
   axes[1].set_ylabel('Fill Fraction')
   axes[1].legend()
   axes[1].grid(True)

   # Data generation events
   if telemetry.data:
       data_times = [pd.timestamp for pd in telemetry.data]
       data_sizes = [pd.data_size_gb for pd in telemetry.data]
       axes[2].bar(data_times, data_sizes, width=0.01, alpha=0.7, color='purple')
       axes[2].set_ylabel('Generated (Gb)')
       axes[2].set_title('Data Generation Events')
       axes[2].grid(True)

   # Alert levels
   colors = ['green', 'orange', 'red']  # 0=normal, 1=warning, 2=critical
   for i, alert in enumerate(telemetry.housekeeping.recorder_alert):
       if alert > 0:  # Only plot non-zero alerts
           axes[3].scatter(times[i], alert, c=colors[alert], s=20, alpha=0.8)
   axes[3].set_ylabel('Alert Level')
   axes[3].set_title('Recorder Alerts')
   axes[3].set_yticks([0, 1, 2])
   axes[3].set_yticklabels(['Normal', 'Warning', 'Critical'])
   axes[3].grid(True)

   plt.tight_layout()
   plt.show()

Fault Management Integration
-----------------------------

The recorder's fill fraction is automatically monitored by the fault management system
when configured.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from conops.config import FaultManagement

   fault_mgmt = FaultManagement()

   config = MissionConfig(
       # ... other config ...
       recorder=recorder,
       fault_management=fault_mgmt
   )

   # Initialize fault management with default thresholds
   config.init_fault_management_defaults()

This automatically adds a ``recorder_fill_fraction`` threshold using the recorder's
yellow and red threshold values. The fault management system will:

* Monitor the fill fraction at each timestep
* Track state transitions (GREEN → YELLOW → RED)
* Log threshold violations
* Optionally trigger safe mode on RED alerts (if configured)

Manual Threshold Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also manually configure recorder thresholds:

.. code-block:: python

   fault_mgmt.add_threshold(
       name="recorder_fill_fraction",
       yellow=0.75,      # Custom yellow threshold
       red=0.95,         # Custom red threshold
       direction="above"  # Alert when value goes above thresholds
   )

Use Cases and Examples
----------------------

Sizing the Onboard Recorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine the minimum recorder capacity needed for your mission:

.. code-block:: python

   # Run simulation with different recorder sizes
   capacities = [32, 64, 128, 256]  # Gigabits

   for capacity in capacities:
       recorder = OnboardRecorder(capacity_gb=capacity)
       config.recorder = recorder

       ditl = DITL(config=config)
       ditl.calc()

       max_fill = max(ditl.recorder_fill_fraction)
       print(f"Capacity: {capacity} Gb, Max Fill: {max_fill:.1%}")

       if max_fill < 0.9:
           print(f"✓ {capacity} Gb is sufficient")
           break

Evaluating Downlink Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determine how many ground station passes are needed:

.. code-block:: python

   # Track cumulative data
   cumulative_generated = 0
   cumulative_downlinked = 0

   for i in range(len(ditl.utime)):
       cumulative_generated += ditl.data_generated_gb[i]
       cumulative_downlinked += ditl.data_downlinked_gb[i]

   print(f"Total data generated: {cumulative_generated:.1f} Gb")
   print(f"Total data downlinked: {cumulative_downlinked:.1f} Gb")
   print(f"Net accumulation: {cumulative_generated - cumulative_downlinked:.1f} Gb")

   # Calculate required passes
   passes = len([p for p in ditl.executed_passes.passes])
   print(f"Ground station passes: {passes}")

Optimizing Observation Schedules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance science observations with data downlink availability:

.. code-block:: python

   # Analyze periods when recorder is full
   full_periods = []
   for i, fill in enumerate(ditl.recorder_fill_fraction):
       if fill >= 1.0:
           full_periods.append(ditl.utime[i])

   if full_periods:
       print(f"WARNING: Recorder full for {len(full_periods)} timesteps")
       print("Consider:")
       print("  - Adding more ground station passes")
       print("  - Increasing downlink rate")
       print("  - Reducing instrument data rates")
       print("  - Increasing recorder capacity")

Best Practices
--------------

1. **Start with Conservative Estimates**

   Begin with generous recorder capacity and downlink capabilities, then optimize
   based on simulation results.

2. **Monitor Alert Levels**

   Pay attention to yellow and red alerts during simulation. Persistent red alerts
   indicate insufficient downlink capability or excessive data generation.

3. **Account for Growth**

   Size your recorder with margin for:

   * Mission lifetime degradation
   * Potential instrument mode additions
   * Missed passes due to weather or station outages

4. **Balance Generation and Downlink**

   The ideal scenario maintains recorder fill between 30-70%, providing buffer
   for both data accumulation and unexpected gaps in downlink opportunities.

5. **Validate with Different Scenarios**

   Test your configuration with:

   * Best case: All passes succeed
   * Worst case: 20-30% of passes fail
   * Realistic case: Mix of successful and failed passes

Troubleshooting
---------------

Recorder Frequently Full
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Red alerts, data loss, fill fraction at 100%

**Solutions**:

* Increase recorder capacity
* Add more ground station passes
* Increase antenna downlink rate
* Reduce instrument data generation rates
* Adjust observation scheduling to prioritize downlink windows

Recorder Always Empty
~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Fill fraction near 0%, unused capacity

**Solutions**:

* Reduce recorder capacity (weight/cost savings)
* Reduce number of ground station contacts
* Consider lower downlink rates
* Optimize pass scheduling

Insufficient Downlink
~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Recorder steadily filling, not enough passes to clear data

**Solutions**:

* Calculate required downlink rate: ``total_data / mission_duration``
* Add ground stations at different longitudes for more frequent passes
* Increase antenna data rate
* Consider data compression (simulate with reduced generation rates)

API Reference
-------------

For detailed API documentation, see:

* :class:`~conops.config.recorder.OnboardRecorder`
* :class:`~conops.config.instrument.DataGeneration`
* :class:`~conops.config.instrument.Instrument`
* :class:`~conops.config.instrument.Payload`
* :class:`~conops.config.groundstation.Antenna`
* :class:`~conops.ditl.ditl.DITL`
* :class:`~conops.ditl.queue_ditl.QueueDITL`
