Quick Start Guide
=================

This guide will help you get started with COASTSim quickly.

---------------------

Here's a simple example of running a Day-In-The-Life (DITL) simulation:

.. code-block:: python

   from datetime import datetime, timedelta
   from conops import Config, QueueDITL
   from rust_ephem import TLEEphemeris

   # Load configuration
   config = Config.from_json("example_config.json")

   # Set simulation period
   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)

   # Compute orbit ephemeris
   ephemeris = TLEEphemeris(tle="example.tle", begin=begin, end=end)

   # Run DITL simulation
   ditl = QueueDITL(config, ephemeris, begin, end)
   ditl.run()

   # Analyze results
   ditl.plot()
   ditl.print_statistics()
Logging DITL Events
-------------------

DITL simulations now log structured events internally instead of printing.
Access the log via ``ditl.log``:

.. code-block:: python

     # Inspect events collected during the run
     for event in ditl.log:
             print(event.timestamp, event.event_type, event.description)

     # Persist logs for many runs using the standard library (optional)
     from conops.ditl import DITLLogStore
     store = DITLLogStore("ditl_logs.sqlite")
     ditl.log.run_id = "my-run-001"
     ditl.log.store = store
     # Events are persisted as they are logged; you can also bulk flush
     ditl.log.flush_to_store()

     # Later, query events by run
     events = store.fetch_events("my-run-001", event_type="PASS")

Configuration-Based Approach

Create a JSON configuration file defining your spacecraft parameters:

.. code-block:: json

   {
       "name": "My Space Telescope",
       "spacecraft_bus": {
           "power_draw": {
               "nominal_power": 50.0,
               "peak_power": 300.0
           },
           "attitude_control": {
               "slew_acceleration": 0.01,
               "max_slew_rate": 1.0
           }
       },
       "solar_panel": {
           "panels": [
               {
                   "name": "Main Solar Panel",
                   "gimbled": false,
                   "sidemount": true,
                   "cant_x": 0.0,
                   "cant_y": 0.0,
                   "azimuth_deg": 0.0,
                   "max_power": 500.0,
                   "conversion_efficiency": 0.9
               }
           ]
       },
       "payload": {
           "payload": [
               {
                   "name": "Telescope Camera",
                   "power_draw": {
                       "nominal_power": 100.0,
                       "peak_power": 150.0
                   }
               }
           ]
       }
   }

Then load and use it:

.. code-block:: python

   from conops.config import Config

   config = Config.from_json("my_spacecraft_config.json")

Key Components
--------------

Ephemeris
~~~~~~~~~

Compute spacecraft orbit:

.. code-block:: python

   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)
   ephemeris = TLEEphemeris(tle="spacecraft.tle", begin=begin, end=end)

Pointing
~~~~~~~~

Define observation targets:

.. code-block:: python

   from conops import Pointing

   target = Pointing(ra=180.0, dec=45.0, roll=0.0)

Queue Scheduler
~~~~~~~~~~~~~~~

Manage observation targets:

.. code-block:: python

   from conops import Queue

   queue = Queue()
   queue.add_target(target, priority=10, duration=300)

Constraints
~~~~~~~~~~~

Apply observational constraints:

.. code-block:: python

   from rust_ephem import SunConstraint, MoonConstraint

   sun_constraint = SunConstraint(min_angle=45.0)
   moon_constraint = MoonConstraint(min_angle=30.0)

Module Structure
----------------

COASTSim is organized into several key modules:

* `conops.common`: Shared utilities, enums (ACSMode, ChargeState, ACSCommandType), and common functions
* `conops.config`: Configuration classes for spacecraft components (battery, solar panels, instruments, etc.)
* `conops.targets`: Target management classes (Pointing, Queue, Plan, PlanEntry)
* `conops.schedulers`: Scheduling algorithms (DumbScheduler, DumbQueueScheduler)
* `conops.simulation`: Core simulation components (ACS, DITL classes, constraints, etc.)
* `conops.ditl`: Day-In-The-Life simulation classes (DITL, DITLMixin, QueueDITL)

All classes are available directly from the ``conops`` package:

.. code-block:: python

   from conops import (
       Config, ACS, DITL, QueueDITL, Pointing, Queue,
       DumbScheduler, ACSMode, ACSCommandType
   )

Next Steps
----------

* Check out the :doc:`examples` for detailed use cases
* Explore the :doc:`api/modules` for complete API reference
* See the ``examples/`` directory for Jupyter notebooks with complete workflows
