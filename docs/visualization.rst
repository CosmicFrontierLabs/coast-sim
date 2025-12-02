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

   from conops import MissionConfig, QueueDITL
   from conops.visualization import plot_ditl_telemetry, plot_acs_mode_distribution
   from conops.config.visualization import VisualizationConfig
   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   cfg = MissionConfig.from_json_file("examples/example_config.json")

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

Working with DITL Logs
----------------------

All timeline and telemetry visualizations can be driven from the structured
event log ``ditl.log``. For batch processing across many runs, persist logs
with ``DITLLogStore`` and load events by ``run_id``:

.. code-block:: python

   from conops.ditl import DITLLogStore
   store = DITLLogStore("ditl_logs.sqlite")
   events = store.fetch_events("my-run-001")
   # feed events to your visualization pipeline as needed
   store.close()

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
