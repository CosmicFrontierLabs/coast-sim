3-D Spacecraft Configuration Visualizer
========================================

``plot_spacecraft_3d`` renders an interactive, rotatable 3-D model of a
spacecraft directly from its :class:`~conops.config.MissionConfig`.  The
figure is produced with Plotly and can be rotated, zoomed, and panned in
any browser or Jupyter notebook without running a full DITL simulation.

Features
--------

* **Spacecraft bus** — lit, solid-shaded box with configurable dimensions.
* **Telescope assembly** — cylindrical baffle tube with aperture ring, mid-tube
  baffle rings, and secondary mirror housing.  Sized automatically from
  ``optics.aperture_m`` and ``optics.tube_length_m``.
* **Solar panels** — deep-blue solar cell faces on a silver frame.  Uses
  :class:`~conops.config.geometry.PanelGeometry` when configured, otherwise
  places panels on the bus face whose outward normal best matches the panel
  normal vector.  Panel area is estimated from ``max_power``.
* **Radiators** — white/MLI panels placed on the bus face matching
  ``orientation.normal``, sized by ``width_m`` × ``height_m``.
* **Star trackers** — small prism bodies with a coloured aperture face, one
  per configured tracker.  Each gets a distinct boresight-arrow colour.
* **Body-frame axes** — +X / +Y / +Z triad with labelled arrows (red, green,
  blue).
* **Normal / boresight arrows** — outward-normal arrows for solar panels and
  radiators; boresight arrows for the telescope and each star tracker.

Coordinate system
-----------------

All geometry is expressed in the spacecraft body frame:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Axis
     - Meaning
   * - +X
     - Boresight / telescope pointing direction
   * - +Y
     - Spacecraft "up"
   * - +Z
     - Completes the right-handed system

Quick start
-----------

.. code-block:: python

   from conops.config import MissionConfig
   from conops.visualization import plot_spacecraft_3d

   config = MissionConfig.from_json_file("examples/example_config.json")
   fig = plot_spacecraft_3d(config)
   fig.show()

No simulation run is needed — the visualizer reads geometry directly from
the configuration.

Usage examples
--------------

Minimal default spacecraft
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.config import MissionConfig
   from conops.visualization import plot_spacecraft_3d

   # Default config gives a bus, one solar panel, one radiator, and one star tracker
   fig = plot_spacecraft_3d(MissionConfig())
   fig.show()

Custom Ritchey-Chrétien telescope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.config import MissionConfig
   from conops.config.instrument import Telescope, TelescopeConfig, TelescopeType, Payload
   from conops.config.solar_panel import SolarPanel, SolarPanelSet, create_solar_panel_vector
   from conops.visualization import plot_spacecraft_3d

   config = MissionConfig(
       payload=Payload(instruments=[
           Telescope(
               name="Primary",
               boresight=(1.0, 0.0, 0.0),
               optics=TelescopeConfig(
                   aperture_m=0.6,
                   tube_length_m=1.6,
                   telescope_type=TelescopeType.RITCHEY_CHRETIEN,
               ),
           )
       ]),
       solar_panel=SolarPanelSet(panels=[
           SolarPanel(name="Panel +Y", normal=create_solar_panel_vector("sidemount"),         max_power=1200.0),
           SolarPanel(name="Panel -Y", normal=create_solar_panel_vector("sidemount", cant_z=180.0), max_power=1200.0),
       ]),
   )

   fig = plot_spacecraft_3d(config, title="RC Telescope — Dual Wing Config")
   fig.show()

Overriding the bus box size
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # bus_half_dims = (half_X, half_Y, half_Z) in metres
   fig = plot_spacecraft_3d(config, bus_half_dims=(1.2, 0.65, 0.65))
   fig.show()

Hiding normal vectors or axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Clean view without arrows
   fig = plot_spacecraft_3d(config, show_normals=False, show_axes=False)
   fig.show()

Embedding in a Jupyter notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # The figure is a standard go.Figure — display inline with:
   fig = plot_spacecraft_3d(config)
   fig  # Jupyter evaluates this cell and renders the Plotly widget

Saving to HTML
^^^^^^^^^^^^^^

.. code-block:: python

   fig = plot_spacecraft_3d(config)
   fig.write_html("spacecraft_model.html")

Visual elements reference
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Element
     - Colour
     - Source in config
   * - Spacecraft bus
     - Silver-gray
     - ``bus_half_dims`` parameter (default 1.0 × 0.55 × 0.55 m)
   * - Telescope tube
     - Dark gunmetal
     - ``payload.instruments`` (``instrument_type = "Telescope"``)
   * - Aperture / baffle rings
     - Gold
     - ``optics.aperture_m``, ``optics.tube_length_m``
   * - Solar cell faces
     - Deep blue
     - ``solar_panel.panels[*].normal``, ``PanelGeometry``
   * - Solar panel frames
     - Silver
     - Derived from panel geometry
   * - Radiators
     - White / MLI
     - ``spacecraft_bus.radiators.radiators[*]``
   * - Star tracker body
     - Dark gray
     - ``spacecraft_bus.star_trackers.star_trackers[*]``
   * - Star tracker aperture
     - Per-tracker colour
     - One of orange / hotpink / cyan / limegreen / gold / violet
   * - +X axis
     - Red
     - Hard-coded body frame
   * - +Y axis
     - Limegreen
     - Hard-coded body frame
   * - +Z axis
     - Dodgerblue
     - Hard-coded body frame
   * - Panel / radiator normals
     - Yellow
     - ``normal`` / ``orientation.normal``
   * - Telescope boresight
     - Cyan
     - ``boresight``
   * - Star tracker boresights
     - Per-tracker colour
     - ``orientation.boresight``

Panel placement rules
---------------------

When a :class:`~conops.config.geometry.PanelGeometry` is attached to a solar
panel or radiator, its ``center_m``, ``u``, ``v``, ``width_m``, and
``height_m`` fields are used directly.

When no ``PanelGeometry`` is set the visualizer falls back to automatic face
placement:

1. Dot-product the component normal against the six bus face normals.
2. Select the face with the largest dot product.
3. Place the component centre just outside that face (3 mm gap) and orient the
   panel spanning directions perpendicular to the face normal.

Solar panel size is estimated from ``max_power`` assuming 150 W/m² (solar
constant × typical conversion efficiency).  Radiator size comes directly from
``width_m`` and ``height_m``.

API reference
-------------

.. autofunction:: conops.visualization.plot_spacecraft_3d

See also
--------

* :doc:`configuration` — all geometric configuration fields in detail.
* :doc:`visualization` — overview of all COASTSim visualization utilities.
* :doc:`sky_pointing_visualization` — interactive sky-pointing globe.
* :py:func:`conops.visualization.plot_sky_pointing_globe` — celestial-sphere
  globe driven by a completed DITL simulation.
