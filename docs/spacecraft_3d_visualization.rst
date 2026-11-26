3-D Spacecraft Configuration Visualizer
========================================

``plot_spacecraft_3d`` renders an interactive, rotatable 3-D model of a
spacecraft directly from its :class:`~conops.config.MissionConfig`.  The
figure is produced with Plotly and can be rotated, zoomed, and panned in
any browser or Jupyter notebook without running a full DITL simulation.

Features
--------

* **Spacecraft surfaces** — when ``spacecraft_bus.surfaces`` has plates, render
  their configured centers, normals and areas in the declared body frame. A
  plate's optional ``geometry`` supplies its rectangular outline; otherwise a
  square of equal area illustrates it. Area and normal alone cannot recover a
  facet's aspect ratio or the outline of a scarfed baffle. The configured model
  replaces the synthetic bus. Radiators, star trackers, wheels and panels
  without geometry of their own are placed on the box the hull plates bound,
  unless ``bus_half_dims`` is given: each face of that box is as far out as the
  furthest plate facing that way, so a wing seen edge-on or a hull away from
  the body origin does not move them off it. Along an axis no plate faces, the
  box takes the plates' own extent. The telescope tube is still drawn,
  with its boresight where ``show_normals`` is set. The tube is left out only
  where plates are marked as that telescope's by their ``instrument`` field, as
  :func:`~conops.config.surfaces.telescope_surfaces` marks the aperture; the
  boresight then starts at the telescope's foremost plate, and parts are not
  placed on those plates. An ``instrument`` name must be one the payload has: a
  configuration with any other is refused, and the plot warns of one assigned to
  a configuration afterwards. Coincident panel front/back plates are represented
  once by the panel geometry. The plates share one legend entry, "Surfaces",
  and their normals another. Plot bounds include all components.
* **Spacecraft bus fallback** — lit, solid-shaded box with configurable dimensions
  when no surfaces are configured.
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
* **Momentum wheels** — a copper disc per wheel in
  ``spacecraft_bus.attitude_control.stored_momentum.wheels``, turned to its spin
  axis and labelled with its capacity.  Drawn at ``position_m`` when configured,
  otherwise inside the bus along the spin axis. The configured surfaces or
  fallback bus box become translucent so the wheels show through.
* **Body-frame axes** — +X / +Y / +Z triad with labelled arrows (red, green,
  blue).
* **Normal / boresight arrows** — outward-normal arrows for solar panels and
  radiators; boresight arrows for the telescope and each star tracker.

Coordinate system
-----------------

All geometry is expressed directly in ``spacecraft_bus.frame``; vectors are not
rotated a second time. The boresight label follows the declared frame. The
following axes describe the default frame:

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
                   aperture_m=0.5,
                   focal_length_m=3.5,
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

Momentum wheels
^^^^^^^^^^^^^^^

.. code-block:: python

   from conops.config import MomentumWheel

   momentum = config.spacecraft_bus.attitude_control.stored_momentum
   momentum.wheels = [
       MomentumWheel(name="RW1", spin_axis=(1, 0, 0), capacity_n_m_s=4.0),
       MomentumWheel(name="RW2", spin_axis=(0, 1, 0), capacity_n_m_s=4.0),
       MomentumWheel(name="RW3", spin_axis=(0, 0, 1), capacity_n_m_s=4.0,
                     position_m=(-0.4, 0.0, 0.2)),
   ]
   fig = plot_spacecraft_3d(config)
   fig.show()

   # The wheels hidden, and the bus solid again
   fig = plot_spacecraft_3d(config, show_wheels=False)

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
   * - Momentum wheels
     - Copper
     - ``spacecraft_bus.attitude_control.stored_momentum.wheels[*]``
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
3. Place the component centre just outside that face (gap is component-specific; e.g. 20 mm for solar panels and 5 mm for radiators) and orient the
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
