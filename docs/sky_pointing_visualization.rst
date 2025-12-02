Sky Pointing Visualization
==========================

An interactive visualization tool for spacecraft pointing visualization on a mollweide sky projection.

Features
--------

* **Interactive Mollweide Projection**: Full-sky view with spacecraft pointing and observations
* **Scheduled Observations**: All planned targets marked on the sky with color-coding by priority
* **Constraint Regions**: Shaded areas showing Sun, Moon, Earth, Anti-Sun, and Solar Panel constraints
* **Current Pointing**: Highlighted marker showing where the spacecraft is currently pointing
* **Time Controls**:

  * Slider to jump to any time
  * Previous/Next buttons to step through time
  * Play button to animate through the entire DITL

* **Frame Export**: Save individual frames for creating animations or presentations

Quick Start
-----------

.. code-block:: python

   from conops import DITL, MissionConfig
   from conops.visualization import plot_sky_pointing
   import matplotlib.pyplot as plt

   # Load configuration and create DITL
   config = MissionConfig.from_json('config.json')
   ditl = DITL(config)

   # Run simulation
   ditl.calc()

   # Create interactive visualization
   fig, ax, controller = plot_sky_pointing(
       ditl,
       figsize=(14, 8),
       show_controls=True,
   )

   plt.show()

Usage
-----

Basic Visualization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simple static plot at the first time step
   fig, ax, _ = plot_sky_pointing(
       ditl,
       show_controls=False,  # No interactive controls
   )

Interactive Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Full interactive plot with controls
   %matplotlib widget  # For Jupyter notebooks

   fig, ax, controller = plot_sky_pointing(
       ditl,
       figsize=(14, 8),
       n_grid_points=100,  # Higher = more accurate constraints, slower
       show_controls=True,
       constraint_alpha=0.3,  # Transparency of constraint regions
   )

Customization
^^^^^^^^^^^^^

.. code-block:: python

   fig, ax, controller = plot_sky_pointing(
       ditl,
       figsize=(16, 10),           # Larger figure
       n_grid_points=50,            # Faster rendering
       time_step_seconds=120,       # Step by 2 minutes
       constraint_alpha=0.2,        # More transparent constraints
   )

Saving Frames
^^^^^^^^^^^^^

.. code-block:: python

   from conops.visualization import save_sky_pointing_frames

   # Save every 10th frame
   saved_files = save_sky_pointing_frames(
       ditl,
       output_dir='./sky_frames',
       n_grid_points=50,
       frame_interval=10,
   )

   # Create animation with imageio
   import imageio
   frames = [imageio.imread(f) for f in saved_files]
   imageio.mimsave('sky_pointing.gif', frames, fps=10)

Visualization Elements
----------------------

Observation Markers
^^^^^^^^^^^^^^^^^^^

Scheduled observations are shown as colored circles:

* **Red (large)**: TOO/GRB targets (obsid >= 1,000,000)
* **Orange (medium)**: High priority targets (obsid >= 20,000)
* **Yellow (medium)**: Survey targets (obsid >= 10,000)
* **Light Blue (small)**: Standard targets

Current Pointing
^^^^^^^^^^^^^^^^

* **Red Star**: Current spacecraft pointing direction
* **Red Circle**: Emphasizes the current pointing region

Constraint Regions
^^^^^^^^^^^^^^^^^^

Color-coded shaded areas showing forbidden pointing regions:

* **Yellow**: Sun constraint zone (too close to Sun)
* **Gray**: Moon constraint zone (too close to Moon)
* **Blue**: Earth limb constraint (pointing too close to Earth)
* **Orange**: Anti-Sun constraint (solar panel limitations)
* **Green**: Solar panel constraint (power generation limitations)

Celestial Bodies
^^^^^^^^^^^^^^^^

Large circular markers show the current position of:

* **Sun** (yellow with black edge)
* **Moon** (gray with black edge)
* **Earth** (blue with black edge)

Interactive Controls
--------------------

When ``show_controls=True``:

* **Time Slider**: Drag to jump to any point in the simulation
* **< Prev Button**: Step backward one time step
* **Next > Button**: Step forward one time step
* **Play Button**: Automatically advance through time (becomes "Pause" when playing)

Performance Tips
----------------

1. **Lower n_grid_points** (e.g., 30-50) for faster rendering during exploration
2. **Increase n_grid_points** (e.g., 100-200) for publication-quality figures
3. **Use frame_interval > 1** when saving many frames to reduce file count
4. The constraint calculation is the slowest part; consider caching results for animations

Example Workflows
-----------------

Interactive Exploration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use in Jupyter notebook for interactive exploration
   %matplotlib widget

   fig, ax, ctrl = plot_sky_pointing(ditl, n_grid_points=50)

   # Play through the simulation
   # Use the Play button or call programmatically:
   # ctrl.start_animation()

Publication Figures
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # High-quality static image at specific time
   fig, ax, ctrl = plot_sky_pointing(
       ditl,
       show_controls=False,
       n_grid_points=150,
       figsize=(12, 8),
   )

   # Update to specific time
   specific_time = ditl.utime[500]  # e.g., 500th timestep
   ctrl.update_plot(specific_time)

   # Save figure
   fig.savefig('spacecraft_pointing.pdf', dpi=300, bbox_inches='tight')

Movie Export
^^^^^^^^^^^^

You can export sky pointing visualizations as animated movies showing how the spacecraft
pointing and constraints evolve throughout the DITL simulation. The ``save_sky_pointing_movie()``
function supports multiple output formats:

.. code-block:: python

   from conops.visualization import save_sky_pointing_movie

   # Export as MP4 video (requires ffmpeg)
   save_sky_pointing_movie(
       ditl,
       "pointing.mp4",
       fps=15,  # frames per second
       frame_interval=5,  # use every 5th time step
       n_grid_points=30,  # constraint grid resolution
       dpi=100  # output resolution
   )

   # Export as animated GIF (requires pillow)
   save_sky_pointing_movie(
       ditl,
       "pointing.gif",
       fps=5,
       frame_interval=10
   )
API Reference
-------------

plot_sky_pointing()
^^^^^^^^^^^^^^^^^^^

Main visualization function.

**Parameters:**

* ``ditl``: DITL or QueueDITL object (must have run ``calc()`` first)
* ``figsize``: tuple, figure size (width, height) in inches
* ``n_grid_points``: int, resolution for constraint calculations (default: 100)
* ``show_controls``: bool, add interactive controls (default: True)
* ``time_step_seconds``: float, time step for controls (default: ditl.step_size)
* ``constraint_alpha``: float, transparency for constraint regions (default: 0.3)

**Returns:**

* ``fig``: matplotlib Figure
* ``ax``: matplotlib Axes (mollweide projection)
* ``controller``: SkyPointingController or None

save_sky_pointing_frames()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save individual frames to disk.

**Parameters:**

* ``ditl``: DITL or QueueDITL object
* ``output_dir``: str, directory to save frames
* ``figsize``: tuple, figure size
* ``n_grid_points``: int, constraint resolution
* ``frame_interval``: int, save every Nth frame (default: 1)

**Returns:**

* List of saved file paths

Implementation Notes
--------------------

* Requires a completed DITL simulation with ephemeris data
* Mollweide projection shows the entire sky with minimal distortion
* RA runs from 0° to 360° (or -180° to 180° in the projection)
* Dec runs from -90° (south) to +90° (north)
* Constraint regions are computed at each timestep based on orbital position
* For very long simulations, consider using ``frame_interval > 1`` to reduce computation time

Troubleshooting
---------------

**ValueError: "DITL simulation has no pointings"**

Run ``ditl.calc()`` before calling ``plot_sky_pointing()``

**ValueError: "DITL constraint has no ephemeris set"**

Ensure your MissionConfig has ephemeris data loaded

**Slow rendering**

* Reduce ``n_grid_points`` (try 30-50 for exploration)
* The constraint calculation checks many sky positions; this is expected

**Interactive controls not working in Jupyter**

* Use ``%matplotlib widget`` or ``%matplotlib notebook``
* May need to install ``ipympl``: ``pip install ipympl``

See Also
--------

* ``plot_ditl_timeline()``: Timeline view of spacecraft operations
* ``plot_ditl_telemetry()``: Telemetry plots (power, battery, etc.)
* Example notebook: ``examples/Example_Sky_Pointing.ipynb``


**Parameters:**

* ``fps`` — frames per second in output movie (controls playback speed)
* ``frame_interval`` — use every Nth time step (1 = use all frames)
* ``n_grid_points`` — grid resolution for constraint regions (lower = faster rendering)
* ``dpi`` — output resolution (higher = larger file size, better quality)
* ``codec`` — video codec for MP4/AVI (e.g., 'h264', 'mpeg4')
* ``bitrate`` — video bitrate in kbps (higher = better quality, larger file)
* ``show_progress`` — whether to display a progress bar using tqdm (default: True)

**Requirements:**

* MP4 and AVI formats require ffmpeg to be installed on your system
* GIF format requires the pillow library (usually bundled with matplotlib)
* Progress bar requires the tqdm library (optional, will fall back gracefully if not available)

API Documentation
-----------------

For detailed API documentation, see:

* :py:func:`conops.visualization.plot_sky_pointing`
* :py:func:`conops.visualization.save_sky_pointing_frames`
* :py:func:`conops.visualization.save_sky_pointing_movie`
* :py:class:`conops.visualization.SkyPointingController`
