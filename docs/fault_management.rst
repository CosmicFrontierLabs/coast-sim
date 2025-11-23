Fault Management
================

Overview
--------

The Fault Management system provides extensible monitoring and response capabilities for spacecraft operations. It monitors configured parameters against yellow and red thresholds, tracks time spent in each fault state, and can automatically trigger safe mode when critical (RED) conditions occur.

Key Features
------------

* **Multi-parameter monitoring**: Track multiple spacecraft parameters simultaneously
* **Configurable thresholds**: Set yellow (warning) and red (critical) limits for each parameter
* **Bidirectional thresholds**: Support for both "below" and "above" threshold types
* **Time tracking**: Accumulate duration spent in yellow and red states
* **Automatic safe mode**: Trigger irreversible safe mode on RED conditions
* **Extensible architecture**: Easily add new monitored parameters

Configuration
-------------

JSON Configuration
^^^^^^^^^^^^^^^^^^

Add the ``fault_management`` section to your spacecraft configuration:

.. code-block:: json

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
               },
               "power_draw": {
                   "name": "power_draw",
                   "yellow": 450.0,
                   "red": 500.0,
                   "direction": "above"
               }
           },
           "states": {},
           "safe_mode_on_red": true
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

Checking Parameters
^^^^^^^^^^^^^^^^^^^

Call ``check()`` each simulation cycle to evaluate monitored parameters:

.. code-block:: python

   # Check parameters (typically called in simulation loop)
   classifications = fm.check(
       values={
           "battery_level": battery.battery_level,
           "temperature": thermal.current_temp,
           "power_draw": power_system.total_draw
       },
       utime=current_time,
       step_size=simulation_step_size,  # seconds
       acs=spacecraft_acs
   )

   # classifications = {"battery_level": "yellow", "temperature": "nominal", ...}

Retrieving Statistics
^^^^^^^^^^^^^^^^^^^^^

Get accumulated time in each fault state:

.. code-block:: python

   stats = fm.statistics()
   # Returns:
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

Integration with QueueDITL
--------------------------

The fault management system is automatically integrated into the ``QueueDITL`` simulation loop when configured. It checks parameters after each power update:

.. code-block:: python

   from conops.config import Config
   from conops.queue_ditl import QueueDITL

   # Load config with fault_management section
   config = Config.from_json("config_with_fault_management.json")

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

When ``safe_mode_on_red`` is ``true`` (default), any parameter reaching RED state will:

1. **Set flag**: The ``safe_mode_requested`` flag is set to ``True``
2. **DITL checks flag**: The QueueDITL loop detects the flag and enqueues an ``ENTER_SAFE_MODE`` command
3. **Irreversible operation**: Safe mode cannot be exited once entered
4. **Sun pointing**: Spacecraft points solar panels at Sun for maximum power
5. **Command queue cleared**: All pending commands are discarded
6. **Emergency power**: System operates in minimal power configuration

Example Configuration File
---------------------------

A complete example configuration with fault management is available in:

``examples/example_config_with_fault_management.json``

This demonstrates monitoring of:

* **battery_level**: Warning at 50%, critical at 40%
* **temperature**: Warning at 50°C, critical at 60°C
* **power_draw**: Warning at 450W, critical at 500W

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

* **Yellow before Red**: Set yellow thresholds as early warnings before critical limits
* **Test thresholds**: Validate threshold values don't cause premature safe mode triggers
* **Monitor statistics**: Review accumulated yellow/red time after simulations
* **Battery monitoring**: Always include battery_level monitoring for power-critical missions
* **Safe mode policy**: Consider setting ``safe_mode_on_red: false`` for analysis runs where you want to observe fault behavior without intervention
