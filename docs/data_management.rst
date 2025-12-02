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
2. **DataGeneration**: Models how instruments produce data during observations
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

The :class:`~conops.config.instrument.DataGeneration` class models how instruments
produce data. It supports two generation modes:

Rate-Based Generation
~~~~~~~~~~~~~~~~~~~~~

Instruments generate data continuously at a specified rate in Gigabits per second (Gbps).
This is appropriate for instruments like cameras or spectrometers that produce continuous
data streams.

.. code-block:: python

   from conops.config import DataGeneration, Instrument

   # Camera generating 0.5 Gbps continuously
   camera_data = DataGeneration(rate_gbps=0.5)

   camera = Instrument(
       name="High-Resolution Camera",
       data_generation=camera_data
   )

   # Calculate data generated over 60 seconds
   data = camera.data_generation.data_generated(60)  # Returns 30.0 Gb

Per-Observation Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instruments generate a fixed amount of data per observation, regardless of duration.
This is appropriate for instruments that capture discrete snapshots or measurements.

.. code-block:: python

   # Spectrometer generating 5 Gb per observation
   spec_data = DataGeneration(per_observation_gb=5.0)

   spectrometer = Instrument(
       name="X-ray Spectrometer",
       data_generation=spec_data
   )

   # Returns 5.0 Gb regardless of duration
   data = spectrometer.data_generation.data_generated(60)

Payload-Level Data Rates
~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~conops.config.instrument.Payload` class aggregates data generation
across all instruments:

.. code-block:: python

   from conops.config import Payload

   payload = Payload(payload=[camera, spectrometer])

   # Get total data rate across all instruments
   total_rate = payload.total_data_rate_gbps()  # 0.5 Gbps from camera

   # Calculate total data generated over a period
   total_data = payload.data_generated(60)  # Camera: 30 Gb + Spec: 5 Gb = 35 Gb

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
2. Retrieves the antenna's ``max_data_rate_mbps``
3. Converts rate to Gbps (divide by 1000)
4. Calculates data volume: ``rate × time_step``
5. Removes that amount from the recorder

.. code-block:: python

   # Example: 100 Mbps for 60 seconds
   downlink_rate_gbps = 100.0 / 1000.0  # 0.1 Gbps
   time_step = 60  # seconds
   data_downlinked = downlink_rate_gbps * time_step  # 6.0 Gb

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

During simulation, DITL tracks data management telemetry:

.. code-block:: python

   from conops.ditl import DITL

   ditl = DITL(config=config)
   ditl.calc()

   # Access data management telemetry
   recorder_volume = ditl.recorder_volume_gb      # Current volume at each timestep
   fill_fraction = ditl.recorder_fill_fraction    # Fill level (0-1) at each timestep
   alert_level = ditl.recorder_alert              # Alert level (0/1/2) at each timestep
   data_generated = ditl.data_generated_gb        # Data generated at each timestep
   data_downlinked = ditl.data_downlinked_gb      # Data downlinked at each timestep

Visualization Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from datetime import datetime

   # Convert timestamps to datetime for plotting
   times = [datetime.fromtimestamp(t) for t in ditl.utime]

   fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

   # Recorder volume
   axes[0].plot(times, ditl.recorder_volume_gb, 'b-')
   axes[0].set_ylabel('Volume (Gb)')
   axes[0].set_title('Onboard Recorder Status')
   axes[0].grid(True)

   # Fill fraction with thresholds
   axes[1].plot(times, ditl.recorder_fill_fraction, 'g-', label='Fill Level')
   axes[1].axhline(y=0.7, color='orange', linestyle='--', label='Yellow Threshold')
   axes[1].axhline(y=0.9, color='red', linestyle='--', label='Red Threshold')
   axes[1].set_ylabel('Fill Fraction')
   axes[1].legend()
   axes[1].grid(True)

   # Data generation
   axes[2].plot(times, ditl.data_generated_gb, 'purple', alpha=0.7)
   axes[2].set_ylabel('Generated (Gb)')
   axes[2].set_title('Data Generation Rate')
   axes[2].grid(True)

   # Data downlink
   axes[3].plot(times, ditl.data_downlinked_gb, 'red', alpha=0.7)
   axes[3].set_ylabel('Downlinked (Gb)')
   axes[3].set_xlabel('Time')
   axes[3].set_title('Data Downlink Rate')
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
