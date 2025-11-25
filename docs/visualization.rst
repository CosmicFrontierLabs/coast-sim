Visualization
=============

COASTSim includes plotting utilities for DITL simulation outputs and telemetry.
These plotting functions are implemented under `conops.visualization` and accept
an optional `config` parameter. When omitted, the functions use `ditl.config.visualization`
if available, or sensible defaults from the `VisualizationConfig` model.

VisualizationConfig
-------------------

The `VisualizationConfig` model lives in `conops.config.visualization` and exposes the
configuration for fonts, colors, and timeline options.

Important fields:

- `font_family` (str): Font family used across plot titles, labels, and legends (default: `Helvetica`).
- `title_font_size` (int)
- `label_font_size` (int)
- `legend_font_size` (int)
- `tick_font_size` (int)
- `mode_colors` (dict[str, str]): Mode color mapping for ACS modes (e.g. `SCIENCE`, `SLEWING`, `SAA`).

Key plotting utilities
----------------------

The following plotting functions are available under `conops.visualization`:

- `plot_ditl_telemetry()` — basic multi-panel timeline showing RA/Dec, ACS mode, battery,
  power, and observation IDs.
- `plot_data_management_telemetry()` — recorder volume, fill fraction, generated/downlinked data, and alerts.
- `plot_acs_mode_distribution()` — pie chart showing the distribution of time spent in each ACS mode.
- `plot_ditl_timeline()` — timeline with orbit numbers, observations, slews, SAA, and eclipses.
- `plot_sky_pointing()` — an interactive Mollweide sky projection with current pointing and constraints.
- `save_sky_pointing_movie()` — export the entire DITL sky pointing visualization as a movie (MP4, AVI, or GIF).

Examples and advanced usage
---------------------------

The visualization functions accept an optional `config` parameter or get the
configuration from `ditl.config.visualization` if available.

Example:

.. code-block:: python

   from conops import Config, QueueDITL
   from conops.visualization import plot_ditl_telemetry, plot_acs_mode_distribution
   from conops.config.visualization import VisualizationConfig
   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   cfg = Config.from_json_file("examples/example_config.json")

   # Customize visual style
   cfg.visualization.font_family = "Helvetica"
   cfg.visualization.title_font_size = 14
   cfg.visualization.mode_colors["SAA"] = "#800080"

   begin = datetime.utcnow()
   end = begin + timedelta(days=1)
   ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)

   ditl = QueueDITL(config=cfg)
   ditl.ephem = ephem
   ditl.calc()

   fig, axes = plot_ditl_telemetry(ditl)
   fig2, ax2 = plot_acs_mode_distribution(ditl, config=cfg.visualization)

Movie Export
------------

You can export sky pointing visualizations as animated movies showing how the spacecraft
pointing and constraints evolve throughout the DITL simulation. The `save_sky_pointing_movie()`
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

**Parameters:**

- `fps` — frames per second in output movie (controls playback speed)
- `frame_interval` — use every Nth time step (1 = use all frames)
- `n_grid_points` — grid resolution for constraint regions (lower = faster rendering)
- `dpi` — output resolution (higher = larger file size, better quality)
- `codec` — video codec for MP4/AVI (e.g., 'h264', 'mpeg4')
- `bitrate` — video bitrate in kbps (higher = better quality, larger file)
- `show_progress` — whether to display a progress bar using tqdm (default: True)

**Requirements:**

- MP4 and AVI formats require ffmpeg to be installed on your system
- GIF format requires the pillow library (usually bundled with matplotlib)
- Progress bar requires the tqdm library (optional, will fall back gracefully if not available)

Fonts and fallbacks
-------------------

Matplotlib will fall back if a requested font family is not installed on the system.
To guarantee a specific font across platforms, provide a `FontProperties` object
with a path to the font file. Alternatively, ensure the font is installed on your system.

.. code-block:: python

   from matplotlib.font_manager import FontProperties
   fp = FontProperties(fname='/Library/Fonts/Helvetica.ttc')
   ax.set_title("Title", fontproperties=fp)

Example images
--------------

.. image:: /_static/visualization_acs_mode_distribution.png
   :alt: ACS mode distribution
   :align: center

.. image:: /_static/visualization_ditl_telemetry.png
   :alt: DITL telemetry example
   :align: center

Further references
------------------

See the API docs for the visualization functions under :mod:`conops.visualization`.
