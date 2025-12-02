Fault Management
================

Overview
--------

The Fault Management system provides extensible monitoring and response capabilities for spacecraft operations. It monitors configured parameters against yellow and red thresholds, tracks time spent in each fault state, and can automatically trigger safe mode when critical (RED) conditions occur.

Key Features
------------

* **Multi-parameter monitoring**: Track multiple spacecraft parameters simultaneously
* **Configurable thresholds**: Set yellow (warning) and red (critical) limits for each parameter
* **Spacecraft red limit constraints**: Define health and safety pointing constraints with time-based safe mode triggering
* **Bidirectional thresholds**: Support for both "below" and "above" threshold types
* **Time tracking**: Accumulate duration spent in yellow and red states, or in constraint violations
* **Automatic safe mode**: Trigger irreversible safe mode on RED conditions or sustained constraint violations
* **Extensible architecture**: Easily add new monitored parameters or constraints

Configuration
-------------

JSON Configuration
^^^^^^^^^^^^^^^^^^

Add the ``fault_management`` section to your spacecraft configuration:

.. code-block:: json

   {
       "fault_management": {
           "thresholds": [
               {
                   "name": "battery_level",
                   "yellow": 0.5,
                   "red": 0.4,
                   "direction": "below"
               },
               {
                   "name": "temperature",
                   "yellow": 50.0,
                   "red": 60.0,
                   "direction": "above"
               },
               {
                   "name": "power_draw",
                   "yellow": 450.0,
                   "red": 500.0,
                   "direction": "above"
               }
           ],
           "red_limit_constraints": [],
           "states": {},
           "safe_mode_on_red": true,
           "events": []
       }
   }

Threshold Parameters
^^^^^^^^^^^^^^^^^^^^

Each threshold requires:

* ``name``: Unique identifier for the parameter
* ``yellow``: Warning threshold value
* ``red``: Critical threshold value (must be more severe than yellow)
* ``direction``: Either ``"below"`` or ``"above"``

  * ``"below"``: Fault triggered when value ≤ threshold (e.g., battery level)
  * ``"above"``: Fault triggered when value ≥ threshold (e.g., temperature, power)

Spacecraft Red Limit Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to threshold-based monitoring, you can define spacecraft-level red limit constraints for health and safety. These are typically looser than instrument constraints (which are optimized for data quality) and exist purely to protect spacecraft hardware.

Red limit constraints can be added to the JSON configuration:

.. code-block:: json

   {
       "fault_management": {
           "thresholds": [
               {
                   "name": "battery_level",
                   "yellow": 0.5,
                   "red": 0.4,
                   "direction": "below"
               }
           ],
           "red_limit_constraints": [
               {
                   "name": "spacecraft_sun_limit",
                   "constraint": {
                       "type": "sun",
                       "min_angle": 30.0
                   },
                   "time_threshold_seconds": 300.0,
                   "description": "Spacecraft must not point within 30° of Sun for more than 5 minutes"
               },
               {
                   "name": "spacecraft_earth_limit",
                   "constraint": {
                       "type": "earth_limb",
                       "min_angle": 10.0
                   },
                   "time_threshold_seconds": 600.0,
                   "description": "Spacecraft must not point within 10° of Earth limb for more than 10 minutes"
               }
           ],
           "states": {},
           "safe_mode_on_red": true,
           "events": []
       }
   }

Red Limit Constraint Parameters
""""""""""""""""""""""""""""""""

Each red limit constraint requires:

* ``name``: Unique identifier for the constraint
* ``constraint``: rust_ephem constraint definition (sun, earth_limb, moon, etc.)
* ``time_threshold_seconds``: Maximum continuous time in violation before triggering safe mode (set to ``null`` for monitoring only)
* ``description``: Human-readable description of the constraint purpose

Programmatic Usage
------------------

Creating Fault Management
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.fault_management import FaultManagement

   # Create fault management system
   fm = FaultManagement()

   # Add thresholds programmatically
    fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
    fm.add_threshold("temperature", yellow=50.0, red=60.0, direction="above")
    fm.add_threshold("power_draw", yellow=450.0, red=500.0, direction="above")

    # Thresholds are stored as a list; find one by name:
    battery_thresh = next(t for t in fm.thresholds if t.name == "battery_level")

Adding Red Limit Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Red limit constraints can be added programmatically using ``add_red_limit_constraint()``:

.. code-block:: python

   import rust_ephem
   from conops.fault_management import FaultManagement

   fm = FaultManagement()

   # Add spacecraft sun constraint - thermal protection
   fm.add_red_limit_constraint(
       name="spacecraft_sun_limit",
       constraint=rust_ephem.SunConstraint(min_angle=30.0),
       time_threshold_seconds=300.0,  # 5 minutes
       description="Prevent thermal damage from prolonged sun exposure"
   )

   # Add spacecraft earth constraint - stray light protection
   fm.add_red_limit_constraint(
       name="spacecraft_earth_limit",
       constraint=rust_ephem.EarthLimbConstraint(min_angle=10.0),
       time_threshold_seconds=600.0,  # 10 minutes
       description="Prevent stray light contamination"
   )

   # Add monitoring-only constraint (no safe mode trigger)
   fm.add_red_limit_constraint(
       name="spacecraft_moon_monitor",
       constraint=rust_ephem.MoonConstraint(min_angle=5.0),
       time_threshold_seconds=None,  # No automatic safe mode
       description="Monitor moon proximity (informational only)"
   )

Checking Parameters
^^^^^^^^^^^^^^^^^^^

Call ``check()`` each simulation cycle to evaluate monitored parameters and red limit constraints:

.. code-block:: python

   # Check parameters and constraints (typically called in simulation loop)
   classifications = fm.check(
       values={
           "battery_level": battery.battery_level,
           "temperature": thermal.current_temp,
           "power_draw": power_system.total_draw
       },
       utime=current_time,
       step_size=simulation_step_size,  # seconds
       acs=spacecraft_acs,
       # For red limit constraint checking, also provide:
       ephem=spacecraft_ephemeris,
       ra=current_pointing_ra,
       dec=current_pointing_dec
   )

   # classifications = {"battery_level": "yellow", "temperature": "nominal", ...}
   # Red limit constraints are checked automatically if ephem/ra/dec provided

Retrieving Statistics
^^^^^^^^^^^^^^^^^^^^^

Get accumulated time in each fault state and constraint violation statistics:

.. code-block:: python

   stats = fm.statistics()

   # For threshold-based parameters:
   # {
   #     "battery_level": {
   #         "yellow_seconds": 120.0,
   #         "red_seconds": 0.0,
   #         "current": "yellow"
   #     },
   #     "temperature": {
   #         "yellow_seconds": 45.0,
   #         "red_seconds": 30.0,
   #         "current": "red"
   #     }
   # }

   # For red limit constraints:
   # {
   #     "spacecraft_sun_limit": {
   #         "in_violation": False,
   #         "total_violation_seconds": 180.0,
   #         "continuous_violation_seconds": 0.0
   #     }
   # }

Separating Statistics by Type
""""""""""""""""""""""""""""""

To separate threshold-based and constraint-based statistics:

.. code-block:: python

   stats = fm.statistics()

   # Get red limit constraint stats
   constraint_stats = {
       name: data for name, data in stats.items()
       if any(c.name == name for c in fm.red_limit_constraints)
   }

   threshold_stats = {
       name: data for name, data in stats.items()
       if any(t.name == name for t in fm.thresholds)
   }

Integration with QueueDITL
--------------------------

The fault management system is automatically integrated into the ``QueueDITL`` simulation loop when configured. It checks parameters after each power update:

.. code-block:: python

   from conops.config import MissionConfig
   from conops.queue_ditl import QueueDITL

   # Load config with fault_management section
   config = MissionConfig.from_json("config_with_fault_management.json")

   # Initialize defaults (adds battery_level threshold if not present)
   config.init_fault_management_defaults()

   # Run simulation
   ditl = QueueDITL(config, target_queue, begin, end, tle_file)
   ditl.run()

   # Check fault statistics after simulation
   if config.fault_management:
       stats = config.fault_management.statistics()
       print(f"Fault statistics: {stats}")

Safe Mode Behavior
------------------

When ``safe_mode_on_red`` is ``true`` (default), any parameter reaching RED state **or** any red limit constraint exceeding its time threshold will:

1. **Set flag**: The ``safe_mode_requested`` flag is set to ``True``
2. **DITL checks flag**: The QueueDITL loop detects the flag and enqueues an ``ENTER_SAFE_MODE`` command
3. **Irreversible operation**: Safe mode cannot be exited once entered
4. **Sun pointing**: Spacecraft points solar panels at Sun for maximum power
5. **Command queue cleared**: All pending commands are discarded
6. **Emergency power**: System operates in minimal power configuration

Red Limit Constraint Triggering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Red limit constraints track **continuous** violation time. The constraint must be violated continuously for the entire ``time_threshold_seconds`` duration before safe mode is triggered. If the constraint is satisfied (even briefly), the continuous violation counter resets to zero.

This allows for:

* **Transient violations**: Brief constraint violations during slews or maneuvers
* **Grace periods**: Reasonable allowance for operational flexibility
* **Critical protection**: Sustained violations that could cause hardware damage trigger safe mode

Set ``time_threshold_seconds`` to ``null`` to create monitoring-only constraints that track violations but never trigger safe mode.

Example Configuration File
---------------------------

Complete example configurations are available in:

* ``examples/example_config_with_fault_management.json`` - Threshold-based monitoring
* ``examples/example_config_with_red_limits.json`` - Red limit constraints
* ``examples/example_spacecraft_red_limits.py`` - Programmatic red limit configuration

The threshold-based example demonstrates monitoring of:

* **battery_level**: Warning at 50%, critical at 40%
* **temperature**: Warning at 50°C, critical at 60°C
* **power_draw**: Warning at 450W, critical at 500W

The red limit example demonstrates spacecraft health and safety constraints:

* **spacecraft_sun_limit**: 30° exclusion zone, 5 minute threshold (thermal protection)
* **spacecraft_earth_limit**: 10° exclusion zone, 10 minute threshold (stray light protection)

Event Log
---------

All significant fault management transitions are recorded in an in-memory ``events`` list on ``FaultManagement``.

``FaultEvent`` fields:

* ``utime`` – Unix timestamp (float)
* ``event_type`` – One of ``threshold_transition``, ``constraint_violation``, ``safe_mode_trigger``
* ``name`` – Threshold / constraint name
* ``cause`` – Human-readable description
* ``metadata`` – Optional contextual dict (subset of keys; may include current value, thresholds, RA/Dec, violation durations)

Example:

.. code-block:: python

    # After running fm.check(...)
    for evt in fm.events:
         print(evt)  # Uses concise __str__ representation

Filtering events:

.. code-block:: python

    safe_mode_events = [e for e in fm.events if e.event_type == "safe_mode_trigger"]
    sun_constraint_events = [e for e in fm.events if e.name == "spacecraft_sun_limit"]

The event log is append-only for the duration of a simulation; clear with ``fm.events.clear()`` if needed between runs.

Adding Custom Parameters
------------------------

To monitor additional parameters:

1. Add threshold to configuration (JSON or programmatically)
2. Pass parameter value in ``check()`` values dictionary
3. System automatically tracks state and accumulates duration

Example - monitoring data buffer usage:

.. code-block:: python

   # Add threshold
   fm.add_threshold("data_buffer", yellow=0.8, red=0.95, direction="above")

   # Check in simulation loop
   fm.check(
       values={
           "battery_level": battery.battery_level,
           "data_buffer": data_system.buffer_usage_fraction
       },
       utime=utime,
       step_size=step_size,
       acs=acs
   )

API Reference
-------------

See :mod:`conops.fault_management` for detailed API documentation.

Best Practices
--------------

Threshold-Based Monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Yellow before Red**: Set yellow thresholds as early warnings before critical limits
* **Test thresholds**: Validate threshold values don't cause premature safe mode triggers
* **Monitor statistics**: Review accumulated yellow/red time after simulations
* **Battery monitoring**: Always include battery_level monitoring for power-critical missions

Red Limit Constraints
^^^^^^^^^^^^^^^^^^^^^^

* **Looser than science constraints**: Red limits should be less restrictive than instrument constraints
* **Hardware protection focus**: Design constraints around thermal limits, detector saturation, etc.
* **Appropriate time thresholds**: Allow transient violations during normal operations (slews, etc.)
* **Test time thresholds**: Verify thresholds don't trigger during routine maneuvers
* **Use monitoring mode**: Set ``time_threshold_seconds: null`` to track violations without triggering safe mode

General
^^^^^^^

* **Safe mode policy**: Consider setting ``safe_mode_on_red: false`` for analysis runs where you want to observe fault behavior without intervention
* **Separate concerns**: Use thresholds for subsystem health (battery, temperature), red limits for pointing safety
