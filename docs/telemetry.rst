Telemetry System
================

COASTSim's telemetry system provides structured access to simulation data through typed data models.
The telemetry system captures spacecraft state information during DITL simulations, making it easier
to analyze, visualize, and export simulation results.

Overview
--------

The telemetry system consists of three main components:

1. **Housekeeping Records**: Timestamped snapshots of spacecraft state at each simulation timestep
2. **Payload Data Records**: Records of data generation events during observations
3. **Telemetry Container**: A structured container that holds both housekeeping and payload data

This replaces the previous approach of storing telemetry as separate parallel arrays, providing
better type safety, easier data access, and more structured analysis capabilities.

Housekeeping Data
-----------------

Housekeeping records capture the complete spacecraft state at each simulation timestep.

.. code-block:: python

   from conops.ditl.telemetry import Housekeeping

   # Example housekeeping record
   hk = Housekeeping(
       timestamp=datetime(2025, 1, 1, 12, 0, 0),
       ra=45.0,                    # Right ascension in degrees
       dec=-23.5,                  # Declination in degrees
       roll=0.0,                   # Roll angle in degrees
       acs_mode="SCIENCE",         # ACS mode
       panel_illumination=0.85,    # Solar panel illumination (0-1)
       power_usage=125.0,          # Total power usage in W
       power_bus=45.0,             # Spacecraft bus power in W
       power_payload=80.0,         # Payload power in W
       battery_level=0.75,         # Battery state of charge (0-1)
       charge_state=1,             # Charging state (0=discharging, 1=charging)
       battery_alert=0,            # Battery alert level (0/1/2)
       obsid=1001,                 # Current observation ID
       recorder_volume_gb=25.3,    # Data volume in recorder (Gb)
       recorder_fill_fraction=0.2, # Recorder fill level (0-1)
       recorder_alert=0,           # Recorder alert level (0/1/2)
       sun_angle_deg=75.3,         # Angular distance to Sun in degrees
       earth_angle_deg=62.1,       # Angular distance to Earth in degrees
       moon_angle_deg=118.5,       # Angular distance to Moon in degrees
       in_constraint=None,         # Name of violated constraint, or None
   )

Housekeeping Fields
~~~~~~~~~~~~~~~~~~~

**Attitude and Pointing**
   - ``timestamp``: UTC timestamp of the record
   - ``ra``: Right ascension in degrees
   - ``dec``: Declination in degrees
   - ``roll``: Roll angle in degrees

**ACS State**
   - ``acs_mode``: Current ACS mode (SCIENCE, SLEWING, SAFE, SAA, etc.)

**Power System**
   - ``panel_illumination``: Solar panel illumination fraction (0-1)
   - ``power_usage``: Total spacecraft power consumption in W
   - ``power_bus``: Spacecraft bus power consumption in W
   - ``power_payload``: Payload power consumption in W
   - ``battery_level``: Battery state of charge (0-1)
   - ``charge_state``: Battery charging state (0=discharging, 1=charging)
   - ``battery_alert``: Battery alert level (0=normal, 1=warning, 2=critical)

**Data Management**
   - ``obsid``: Current observation ID
   - ``recorder_volume_gb``: Current data volume in recorder (Gb)
   - ``recorder_fill_fraction``: Recorder fill fraction (0-1)
   - ``recorder_alert``: Recorder alert level (0=normal, 1=warning, 2=critical)

**Constraint Geometry**
   - ``sun_angle_deg``: Angular distance from current pointing to Sun in degrees
   - ``earth_angle_deg``: Angular distance from current pointing to Earth in degrees
   - ``moon_angle_deg``: Angular distance from current pointing to Moon in degrees
   - ``in_constraint``: Name of the constraint currently being violated (e.g. ``"Sun"``,
     ``"Earth Limb"``, ``"Moon"``, ``"Panel"``, ``"Anti-Sun"``, ``"ST Hard"``, ``"ST Soft"``),
     or ``None`` when no constraint is active.
   - ``for_solid_angle_sr``: Instantaneous field-of-regard solid angle in steradians.
     This value is optional and is ``None`` unless FOR calculation is enabled.
   - ``in_eclipse``: Whether spacecraft is in eclipse

**Star Tracker Health**
   - ``star_tracker_hard_violations``: Number of trackers violating their hard constraint
   - ``star_tracker_soft_violations``: Whether any tracker is in its soft constraint zone
   - ``star_tracker_functional_count``: Number of trackers at full science quality (not in soft constraint)

Payload Data
------------

Payload data records capture data generation events during observations.

.. code-block:: python

   from conops.ditl.telemetry import PayloadData
   from datetime import datetime

   # Example payload data record
   pd = PayloadData(
       timestamp=datetime(2025, 1, 1, 12, 15, 0),
       data_size_gb=2.5  # Size of data generated in Gb
   )

Telemetry Container
-------------------

The :class:`~conops.ditl.telemetry.Telemetry` class provides a structured container for
both housekeeping and payload data records.

.. code-block:: python

   from conops.ditl.telemetry import Telemetry, HousekeepingList, Housekeeping, PayloadData

   # Create telemetry container
   telemetry = Telemetry(
       housekeeping=HousekeepingList([hk1, hk2, hk3]),  # List of housekeeping records
       data=[pd1, pd2]  # List of payload data records
   )

Accessing Telemetry Data
~~~~~~~~~~~~~~~~~~~~~~~~~

The telemetry container provides convenient access to data fields:

.. code-block:: python

   # Access housekeeping data
   timestamps = telemetry.housekeeping.timestamp
   ra_values = telemetry.housekeeping.ra
   dec_values = telemetry.housekeeping.dec
   acs_modes = telemetry.housekeeping.acs_mode
   battery_levels = telemetry.housekeeping.battery_level

   # Angular distances and constraint geometry
   sun_angles = telemetry.housekeeping.sun_angle_deg
   earth_angles = telemetry.housekeeping.earth_angle_deg
   moon_angles = telemetry.housekeeping.moon_angle_deg
   constraints = telemetry.housekeeping.in_constraint  # list of str|None

   # Star tracker health
   hard_violations = telemetry.housekeeping.star_tracker_hard_violations
   functional_count = telemetry.housekeeping.star_tracker_functional_count

   # Access payload data
   data_timestamps = [pd.timestamp for pd in telemetry.data]
   data_sizes = [pd.data_size_gb for pd in telemetry.data]

Field Extraction Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~conops.ditl.telemetry.Housekeeping` class provides utility methods for
extracting fields from multiple records:

.. code-block:: python

   from conops.ditl.telemetry import Housekeeping

   # Extract single field from multiple records
   timestamps = Housekeeping.extract_field(housekeeping_records, 'timestamp')
   battery_levels = Housekeeping.extract_field(housekeeping_records, 'battery_level')

   # Extract multiple fields at once
   fields = Housekeeping.extract_fields(housekeeping_records, ['ra', 'dec', 'acs_mode'])

Integration with DITL
----------------------

During DITL simulation, telemetry data is automatically recorded and stored in the
``ditl.telemetry`` attribute.

.. code-block:: python

   from conops.ditl import QueueDITL
   from conops import MissionConfig

   config = MissionConfig()
   # FOR calculation is disabled by default.
   # Enable it only when this telemetry is needed.
   ditl = QueueDITL(config=config, calculate_field_of_regard=True)
   ditl.calc()  # Run simulation

   # Access telemetry data
   telemetry = ditl.telemetry
   housekeeping = telemetry.housekeeping
   payload_data = telemetry.data

   # Analyze results
   print(f"Simulation covered {len(housekeeping)} timesteps")
   print(f"Generated {len(payload_data)} data records")
   print(f"Average battery level: {sum(housekeeping.battery_level) / len(housekeeping):.2f}")

Visualization with Telemetry
-----------------------------

All COASTSim visualization functions now work with the structured telemetry system:

.. code-block:: python

   from conops.visualization import plot_ditl_telemetry, plot_data_management_telemetry

   # Plot basic telemetry timeline
   fig, axes = plot_ditl_telemetry(ditl)

   # Plot data management telemetry
   fig2, axes2 = plot_data_management_telemetry(ditl)

Exporting Telemetry Data
-------------------------

Telemetry data can be easily exported for external analysis:

.. code-block:: python

   import pandas as pd

   # Convert housekeeping to DataFrame
   hk_df = pd.DataFrame({
       'timestamp': housekeeping.timestamp,
       'ra': housekeeping.ra,
       'dec': housekeeping.dec,
       'acs_mode': housekeeping.acs_mode,
       'battery_level': housekeeping.battery_level,
       'recorder_fill_fraction': housekeeping.recorder_fill_fraction
   })

   # Convert payload data to DataFrame
   pd_df = pd.DataFrame({
       'timestamp': [pd.timestamp for pd in payload_data],
       'data_size_gb': [pd.data_size_gb for pd in payload_data]
   })

   # Export to CSV
   hk_df.to_csv('housekeeping.csv', index=False)
   pd_df.to_csv('payload_data.csv', index=False)

Migration from Legacy Arrays
-----------------------------

If you have existing code that accesses telemetry as parallel arrays (``ditl.ra``, ``ditl.dec``, etc.),
you can migrate to the new telemetry system:

.. code-block:: python

   # Legacy access (still works)
   ra_values = ditl.ra
   dec_values = ditl.dec
   battery_levels = ditl.batterylevel

   # New structured access
   ra_values = ditl.telemetry.housekeeping.ra
   dec_values = ditl.telemetry.housekeeping.dec
   battery_levels = ditl.telemetry.housekeeping.battery_level

The legacy array access is maintained for backward compatibility, but the new telemetry
system provides better type safety and more structured data access.
