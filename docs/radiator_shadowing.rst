Radiator Thermal Modelling and Panel Shadowing
===============================================

Overview
--------

COASTSim models body-mounted radiator panels with physically-based thermal
calculations that include:

* **Sun and Earth exposure** — cosine-law geometric fractions based on the
  radiator's surface normal and the spacecraft pointing.
* **Net heat dissipation** — Stefan–Boltzmann emitted flux minus absorbed
  solar and Earth-IR flux, scaled by area and efficiency.
* **Hard keep-out constraints** — optional boresight-offset constraints that
  flag radiator orientations incompatible with thermal or optical limits.
* **Inter-component panel shadowing** — when a solar panel is mounted adjacent
  to a radiator, its physical extent can cast a shadow that reduces the
  radiator's effective solar heat load.  This requires explicit 3-D geometry
  (position and spanning vectors) for both the panel and the radiator.

Coordinate Frame
----------------

All geometry uses the spacecraft body frame:

* ``+X`` — spacecraft boresight (pointing direction)
* ``+Y`` — spacecraft "up"
* ``+Z`` — completes the right-handed system

Panel normals and ``PanelGeometry`` spanning vectors are expressed as unit
vectors in this frame.

Basic Radiator Configuration
-----------------------------

A radiator is defined by its surface normal, physical dimensions, and thermal
properties.  The ``orientation.normal`` unit vector points outward from the
radiating face; heat is rejected in that direction.

.. code-block:: python

   from conops.config import Radiator, RadiatorConfiguration, RadiatorOrientation

   # Single radiator on the -Y face (anti-sun side for a +Y solar panel)
   rad = Radiator(
       name="Bus Radiator",
       orientation=RadiatorOrientation(normal=(0.0, -1.0, 0.0)),
       width_m=0.8,
       height_m=0.6,
       emissivity=0.85,
       absorptivity=0.20,
       radiator_temperature_k=310.0,
   )

   config = RadiatorConfiguration(radiators=[rad])

The ``heat_dissipation_w`` method computes net heat flow (W) from sun and
Earth exposure fractions:

.. code-block:: python

   # With known exposure fractions (e.g. from exposure_factors())
   net_heat_w = rad.heat_dissipation_w(sun_exposure=0.3, earth_exposure=0.1)
   print(f"Net heat rejection: {net_heat_w:.1f} W")

Aggregate metrics for all radiators at a given simulation instant:

.. code-block:: python

   metrics = config.exposure_metrics(
       ra_deg=45.0, dec_deg=-20.0, utime=unix_time, ephem=ephemeris
   )
   print(metrics["heat_dissipation_w"])   # total W across all radiators
   print(metrics["sun_exposure"])         # area-weighted mean exposure [0–1]
   for r in metrics["per_radiator"]:
       print(r["name"], r["heat_dissipation_w"])

Panel Shadowing
---------------

When a solar panel is physically adjacent to a radiator — mounted
perpendicular to it, for example — the panel's structural extent can cast a
shadow on the radiator face.  This reduces the solar heat load absorbed by the
radiator and must be accounted for in accurate thermal analyses.

Geometry Model
~~~~~~~~~~~~~~

Both the solar panel and the radiator must be given explicit 3-D geometry via
:class:`~conops.config.geometry.PanelGeometry`.  This model describes a
rectangle in body-frame space by its centre position and two orthogonal
spanning unit vectors:

.. code-block:: text

   Panel points:  center_m  ±  u * width_m/2  ±  v * height_m/2

The outward normal is implicitly ``u × v`` (right-hand rule).  Ensure this is
consistent with the ``orientation.normal`` of the owning ``Radiator``, and
with the ``normal`` field of the owning ``SolarPanel``.

Perpendicular-Mount Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical scenario: a solar panel in the XZ plane (facing ``+Y``) with a
radiator attached to the ``+X`` edge, extending in the ``−Y`` direction and
facing outward (``+X``).

.. code-block:: python

   import numpy as np
   from conops.config import (
       PanelGeometry,
       Radiator,
       RadiatorConfiguration,
       RadiatorOrientation,
       SolarPanel,
       SolarPanelSet,
   )

   # --- Solar panel ---------------------------------------------------------
   # 2 m × 1 m panel in the XZ plane (y = 0), facing +Y.
   # u = (1,0,0) and v = (0,0,1) so  u × v = (0,−1,0); the component's
   # orientation.normal = (0,+1,0) is the physically meaningful face normal.
   solar_panel = SolarPanel(
       name="Wing Panel",
       normal=(0.0, 1.0, 0.0),           # faces +Y (toward sun)
       max_power=1200.0,
       geometry=PanelGeometry(
           center_m=(0.0, 0.0, 0.0),
           u=(1.0, 0.0, 0.0),            # width  along +X
           v=(0.0, 0.0, 1.0),            # height along +Z
           width_m=2.0,
           height_m=1.0,
       ),
   )

   # --- Radiator ------------------------------------------------------------
   # 0.8 m × 1 m panel on the +X edge of the solar panel, facing +X.
   # Mounted so its inner edge shares the panel's +X boundary (x = 1 m).
   radiator = Radiator(
       name="Side Radiator",
       orientation=RadiatorOrientation(normal=(1.0, 0.0, 0.0)),  # faces +X
       width_m=0.8,
       height_m=1.0,
       shadowed_by=["Wing Panel"],       # reference solar panel by name
       geometry=PanelGeometry(
           center_m=(1.0, -0.4, 0.0),   # centre 0.4 m in −Y from the edge
           u=(0.0, 1.0, 0.0),           # width  along ±Y
           v=(0.0, 0.0, 1.0),           # height along ±Z
           width_m=0.8,
           height_m=1.0,
       ),
   )

   panel_set = SolarPanelSet(panels=[solar_panel])
   rad_config = RadiatorConfiguration(radiators=[radiator])

When :meth:`~conops.config.RadiatorConfiguration.exposure_metrics` is called
(either directly or through the simulation loop), it receives a mapping of
solar-panel names to their :class:`~conops.config.geometry.PanelGeometry`.
For each radiator whose ``shadowed_by`` list overlaps that mapping, the shadow
fraction is computed and applied:

.. code-block:: python

   # Build the panel geometry look-up that the simulation would construct
   panel_geometries = {
       p.name: p.geometry
       for p in panel_set.panels
       if p.geometry is not None
   }

   metrics = rad_config.exposure_metrics(
       ra_deg=ra, dec_deg=dec, utime=t, ephem=ephem,
       solar_panel_geometries=panel_geometries,
   )

   for r in metrics["per_radiator"]:
       print(f"{r['name']}: sun_exposure={r['sun_exposure']:.3f}, "
             f"heat={r['heat_dissipation_w']:.1f} W")

The simulation ACS (``:class:`~conops.simulation.acs.ACS```) performs this
look-up automatically on every timestep; no extra wiring is needed when both
``geometry`` and ``shadowed_by`` are configured.

Shadow Computation
~~~~~~~~~~~~~~~~~~

The shadow fraction is calculated by
:func:`~conops.config.geometry.compute_shadow_fraction`:

1. **Project occluder corners** — for each corner of the solar panel, cast a
   ray in the anti-sun direction (``−s``) to find where it lands on the
   radiator's plane.  Corners on the wrong side of the plane (``t < 0``) are
   discarded.
2. **Build shadow polygon** — the projected corners form a parallelogram
   (shadow outline) in the radiator's local 2-D frame ``(u, v)``.
3. **Intersect with receiver** — clip the shadow polygon against the
   radiator rectangle using `Shapely`_.  For multiple solar panels the
   shadows are unioned before clipping.
4. **Compute fraction** — ``shadow_fraction = intersection_area / radiator_area``.
5. **Apply to exposure** — ``effective_sun_exposure *= (1 − shadow_fraction)``.

.. note::
   When the sun direction is nearly parallel to the radiator face
   (``|s · n_rad| < 10⁻⁹``) no shadow is projected and the fraction is zero.
   This is correct: the radiator's direct sun exposure is already negligible
   in that orientation.

.. _Shapely: https://shapely.readthedocs.io

Geometry Consistency Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **``u × v`` vs ``orientation.normal``** — :class:`~conops.config.geometry.PanelGeometry`
  computes its internal normal as ``u × v`` for projection maths.  This may
  differ in sign from the component's ``orientation.normal`` (which governs
  the exposure dot-product).  The shadow code handles both orientations
  automatically; however, keeping them consistent avoids confusion.

* **Orthogonality** — ``u`` and ``v`` should be orthogonal.  They are each
  validated as unit vectors.  Non-orthogonal spanning vectors will still
  produce a valid Shapely polygon but the area calculation will be slightly
  off.

* **``shadowed_by`` names** — the strings in ``Radiator.shadowed_by`` must
  match ``SolarPanel.name`` exactly.  Unrecognised names are silently
  ignored.

* **Shadow without geometry** — if either the radiator or the referenced
  solar panel lacks a ``geometry`` field, no shadow fraction is computed and
  the radiator's sun exposure is unchanged.  This means adding geometry is
  always an opt-in refinement; existing configurations are unaffected.

Multiple Radiators and Panels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any number of radiators can reference any number of panels.  All shadows on
a single radiator are unioned before computing the fraction, so overlapping
occluders are handled correctly.

.. code-block:: python

   rad_payload = Radiator(
       name="Payload Radiator",
       orientation=RadiatorOrientation(normal=(-1.0, 0.0, 0.0)),
       subsystem="payload",
       shadowed_by=["Port Wing", "Starboard Wing"],
       geometry=PanelGeometry(
           center_m=(-0.5, 0.0, 0.0),
           u=(0.0, 1.0, 0.0),
           v=(0.0, 0.0, 1.0),
           width_m=1.0, height_m=0.6,
       ),
   )

Hard Keep-Out Constraints
--------------------------

A radiator can have an optional
:class:`~conops.config.constraint.Constraint` that defines orientations where
pointing is prohibited (e.g. to prevent a heat-pipe radiator from facing the
sun for extended periods).

.. code-block:: python

   from conops.config import Constraint, Radiator, RadiatorOrientation
   import rust_ephem

   constraint = Constraint(
       constraint=rust_ephem.SunConstraint(min_angle=45.0)
   )
   rad = Radiator(
       name="Sensitive Radiator",
       orientation=RadiatorOrientation(normal=(0.0, -1.0, 0.0)),
       hard_constraint=constraint,
   )

Violations are counted by
:meth:`~conops.config.RadiatorConfiguration.radiators_violating_hard_constraints`
and logged by the ACS at every timestep where any radiator is in violation.

API Reference
-------------

See the following API pages for complete method signatures and parameter
descriptions:

* :mod:`conops.config.geometry` — :class:`~conops.config.geometry.PanelGeometry`
  and :func:`~conops.config.geometry.compute_shadow_fraction`
* :mod:`conops.config.radiator` — :class:`~conops.config.radiator.Radiator`,
  :class:`~conops.config.radiator.RadiatorConfiguration`
