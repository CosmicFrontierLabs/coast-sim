COASTSim: ConOps Astronomical Space Telescope Simulator
========================================================

Welcome to COASTSim's documentation!

COASTSim is a comprehensive Python module for simulating Concept of Operations (ConOps)
for space telescopes and astronomical spacecraft missions. It enables mission planners
and engineers to simulate Day-In-The-Life (DITL) scenarios, evaluate spacecraft performance,
optimize observation schedules, and validate operational constraints before launch.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   examples
   api/modules
   contributing

Features
--------

* **Spacecraft Bus Simulation**: Model power systems, attitude control, and thermal management
* **Orbit Propagation**: TLE-based ephemeris computation and orbit tracking
* **Observation Planning**: Target queue management and scheduling algorithms
* **Instrument Modeling**: Multi-instrument configurations with power and pointing requirements
* **Constraint Management**: Sun, Moon, Earth limb, and custom geometric constraints
* **Power Budget Analysis**: Solar panel modeling, battery management, and emergency charging scenarios
* **Ground Station Passes**: Communication window calculations and data downlink planning
* **Attitude Control System**: Slew modeling, pointing accuracy, and settle time simulation
* **South Atlantic Anomaly (SAA) Avoidance**: Radiation belt constraint handling
* **DITL Generation**: Comprehensive day-in-the-life timeline simulation

Quick Example
-------------

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
