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
   configuration
   examples
   visualization
   sky_pointing_visualization
   communications
   data_management
   fault_management
   target_of_opportunity
   api/modules
   contributing
   README

Features
--------

* **Spacecraft Bus Simulation**: Model power systems, attitude control, and thermal management
* **Orbit Propagation**: TLE-based ephemeris computation and orbit tracking
* **Observation Planning**: Target queue management and scheduling algorithms
* **Instrument Modeling**: Multi-instrument configurations with power and pointing requirements
* **Data Management**: Onboard data storage simulation with generation and downlink modeling
* **Constraint Management**: Sun, Moon, Earth limb, and custom geometric constraints
* **Fault Management**: Extensible parameter monitoring with yellow/red thresholds and automatic safe mode
* **Power Budget Analysis**: Solar panel modeling, battery management, and emergency charging scenarios
* **Ground Station Passes**: Communication window calculations and data downlink planning
* **Attitude Control System**: Slew modeling, pointing accuracy, and settle time simulation
* **South Atlantic Anomaly (SAA) Avoidance**: Radiation belt constraint handling
* **DITL Generation**: Comprehensive day-in-the-life timeline simulation

Quick Example
-------------

.. code-block:: python

   from datetime import datetime, timedelta
   from matplotlib import pyplot as plt
   import numpy as np
   from rust_ephem import TLEEphemeris

   from conops import MissionConfig, QueueDITL
   from conops.visualization import plot_ditl_timeline, plot_sky_pointing

   # Load configuration - use the default MissionConfig for this example
   config = MissionConfig()

   # Set simulation period
   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)

   # Compute orbit ephemeris
   ephemeris = TLEEphemeris(
      begin=begin,
      end=end,
      step_size=60,
      tle="examples/example.tle",
   )

   # Create DITL object
   ditl = QueueDITL(config=config, ephem=ephemeris)

   # Add 1000 random observations to the observation queue
   for i in range(1000):
      ditl.queue.add(
         ra=np.random.uniform(0, 360),
         dec=np.random.uniform(-90, 90),
         exptime=1000,
         obsid=10000 + i,
      )


   # Run DITL simulation
   ditl.calc()

   # Analyze results visually and statistically
   _ = plot_sky_pointing(ditl, figsize=(10, 5))
   plt.show()
   _ = plot_ditl_timeline(ditl, figsize=(8, 6))
   plt.show()
   ditl.print_statistics()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
