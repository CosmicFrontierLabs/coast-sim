Quick Start Guide
=================

This guide will help you get started with COASTSim quickly.

---------------------

Here's a simple example of running a Day-In-The-Life (DITL) simulation:

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

     store.close()

Configuration-Based Approach
----------------------------

Configure your spacecraft parameters using the ``MissionConfig`` class.

.. code-block:: python

   from conops.config import MissionConfig

   # Create a default configuration
   config = MissionConfig()

   # Customize parameters as needed
   config.spacecraft_bus.power_draw.nominal_power = 75.0  # in Watts
   config.solar_panel.panels[0].max_power = 600.0  # in Watts

``MissionConfig`` encapsulates all spacecraft parameters for simulations. You can
save this to a json file using ``config.to_json_file("filename.json")``.

You can use this output to store your configuration or edit it manually. Here
is an example of a JSON configuration file defining your spacecraft parameters:

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

   from conops.config import MissionConfig

   config = MissionConfig.from_json_file("my_spacecraft_config.json")

Key Components
--------------

Ephemeris
~~~~~~~~~

COASTSim relies on the `rust-ephem` library to compute spacecraft orbit, it
will accept any ephemeris supported by that library. For example, to use TLEs:

.. code-block:: python

   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   begin = datetime(2025, 11, 1)
   end = begin + timedelta(days=1)
   ephemeris = TLEEphemeris(tle="spacecraft.tle", begin=begin, end=end)


Queue Scheduler
~~~~~~~~~~~~~~~

The `QueueDITL` class, which is the most commonly used DITL simulator,
implements a simple queue-based scheduling algorithm. Targets are scheduled
prioritized by merit. Targets of equal merit have a randomized order, ensuring
each simulation run can yield different results, aiding in Monte Carlo analyses.

Here is an example of adding targets to the observation queue:

.. code-block:: python

    ditl = QueueDITL(config=config, ephem=ephemeris)
    ditl.queue.add(
        ra=266,
        dec=-29,
        exptime=1000,
        obsid=10000,
        merit=55
    )



Constraints
~~~~~~~~~~~

Spacecraft pointing constraints can be defined using constraint classes from
the `rust-ephem` library. For example, to create Sun and Moon angle constraints

.. code-block:: python

   from rust_ephem import SunConstraint, MoonConstraint

   sun_constraint = SunConstraint(min_angle=45.0)
   moon_constraint = MoonConstraint(min_angle=30.0)

These can then be added to MissionConfig:

.. code-block:: python

   from conops.config import MissionConfig, Constraint

   config = MissionConfig()
   config.constraints = Constraint(
       sun_constraint=sun_constraint,
       moon_constraint=moon_constraint
   )

`rust-ephem` provides us the ability to combine multiple constraints using logical
operations (AND, OR, NOT) to create complex constraint conditions. For example
if we only want to avoid pointing close to the Sun when it's daytime, we can
do:

.. code-block:: python

   from conops.config import MissionConfig, Constraint
   from rust_ephem import SunConstraint, MoonConstraint, EclipseConstraint

   config = MissionConfig()
   config.constraints = Constraint(
       sun_constraint=SunConstraint(min_angle=45.0) & ~EclipseConstraint(),
       moon_constraint=MoonConstraint(min_angle=30.0)
   )

The above ``sun_constraint`` only applies when the spacecraft is not in eclipse
(i.e. in the Earth's shadow).

Module Structure
----------------

COASTSim is organized into several key modules:

* `conops.common`: Shared utilities, enums (ACSMode, ChargeState, ACSCommandType), and common functions
* `conops.config`: Configuration classes for spacecraft components (battery, solar panels, instruments, etc.)
* `conops.targets`: Target management classes (Pointing, Queue, Plan, PlanEntry)
* `conops.schedulers`: Scheduling algorithms (DumbScheduler, DumbQueueScheduler)
* `conops.simulation`: Core simulation components (ACS, DITL classes, constraints, etc.)
* `conops.ditl`: Day-In-The-Life simulation classes (DITL, DITLMixin, QueueDITL)
* `conops.visualization`: Visualization utilities for plotting results (sky pointing, timelines, etc.)

All classes are available directly from the ``conops`` package:

.. code-block:: python

   from conops import (
       MissionConfig, ACS, DITL, QueueDITL, Pointing, Queue,
       DumbScheduler, ACSMode, ACSCommandType
   )

Next Steps
----------

* Check out the :doc:`examples` for detailed use cases
* Explore the :doc:`api/modules` for complete API reference
* See the ``examples/`` directory for Jupyter notebooks with complete workflows
