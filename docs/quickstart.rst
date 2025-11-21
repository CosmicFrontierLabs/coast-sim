Quick Start Guide
=================

This guide will help you get started with COASTSim quickly.

Basic DITL Simulation
---------------------

Here's a simple example of running a Day-In-The-Life (DITL) simulation:

.. code-block:: python

   from datetime import datetime, timedelta
   from conops.config import Config
   from conops.queue_ditl import QueueDITL
   from conops.ephemeris import compute_tle_ephemeris

   # Load configuration
   config = Config.from_json("example_config.json")

   # Set simulation period
   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)

   # Compute orbit ephemeris
   ephemeris = compute_tle_ephemeris("example.tle", begin, end)

   # Run DITL simulation
   ditl = QueueDITL(config, ephemeris, begin, end)
   ditl.run()

   # Analyze results
   ditl.plot()
   ditl.print_statistics()

Configuration-Based Approach
-----------------------------

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
           "panels": [...]
       },
       "instruments": {
           "instruments": [...]
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

   from conops.ephemeris import compute_tle_ephemeris
   from datetime import datetime, timedelta

   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)
   ephemeris = compute_tle_ephemeris("spacecraft.tle", begin, end)

Pointing
~~~~~~~~

Define observation targets:

.. code-block:: python

   from conops.pointing import Pointing

   target = Pointing(ra=180.0, dec=45.0, roll=0.0)

Queue Scheduler
~~~~~~~~~~~~~~~

Manage observation targets:

.. code-block:: python

   from conops.queue_scheduler import Queue

   queue = Queue()
   queue.add_target(target, priority=10, duration=300)

Constraints
~~~~~~~~~~~

Apply observational constraints:

.. code-block:: python

   from rust_ephem import SunConstraint, MoonConstraint

   sun_constraint = SunConstraint(min_angle=45.0)
   moon_constraint = MoonConstraint(min_angle=30.0)

Next Steps
----------

* Check out the :doc:`examples` for detailed use cases
* Explore the :doc:`api/modules` for complete API reference
* See the ``examples/`` directory for Jupyter notebooks with complete workflows
