Plan Serialisation
==================

COASTSim can save an observation plan produced by a DITL run to a portable JSON file and
reload it later.  The serialisation layer is built on `Pydantic v2`_ and lives in
:mod:`conops.targets.plan_schema`.  Convenience methods are also available directly on
:class:`~conops.targets.Plan` so you rarely need to import the schema classes explicitly.

Overview
--------

Two Pydantic models handle the conversion:

* :class:`~conops.targets.plan_schema.PlanEntrySchema` — represents a single observation
  entry (a :class:`~conops.targets.PlanEntry` or :class:`~conops.targets.Pointing`).
* :class:`~conops.targets.plan_schema.PlanSchema` — top-level container that bundles metadata
  (version, timestamps, entry count) with the list of entries.

Both models support ``model_validate(..., from_attributes=True)``, so they accept plain Python
objects produced by the scheduler without any intermediate conversion step.

Quick Start
-----------

**Save a plan after a DITL run**

The simplest approach is to call :meth:`~conops.targets.Plan.save` directly on the plan:

.. code-block:: python

   # `ditl` is a QueueDITL (or similar) instance that has already been run
   saved_path = ditl.plan.save("plan_20251201.json")
   print(f"Saved to {saved_path}")

You can also go via :class:`~conops.targets.plan_schema.PlanSchema` if you need access to the
metadata fields before writing:

.. code-block:: python

   from conops.targets import PlanSchema

   schema = PlanSchema.from_plan(ditl.plan)
   print(f"Saving {schema.num_entries} entries (version {schema.version})")
   schema.save("plan_20251201.json")

**Load it back**

:meth:`~conops.targets.Plan.load` is a class method on :class:`~conops.targets.Plan` that
returns a :class:`~conops.targets.plan_schema.PlanSchema` (preserving all metadata):

.. code-block:: python

   schema = Plan.load("plan_20251201.json")
   print(schema.version)          # schema format version (integer)
   print(schema.num_entries)      # number of plan entries
   print(schema.entries[0].name)  # first target name

Or equivalently via :class:`~conops.targets.plan_schema.PlanSchema` directly:

.. code-block:: python

   from conops.targets import PlanSchema

   schema = PlanSchema.load("plan_20251201.json")

**Round-trip via model_validate**

.. code-block:: python

   schema = PlanSchema.model_validate(ditl.plan, from_attributes=True)

JSON File Format
----------------

The JSON file contains a metadata envelope followed by the entry list.

.. code-block:: json

   {
     "version": "0.1.3",
     "created_at": "2025-12-01T00:00:00+00:00",
     "start": 1764547200.0,
     "end": 1764633540.0,
     "num_entries": 42,
     "entries": [
       {
         "name": "TEST_001",
         "ra": 83.82,
         "dec": -5.39,
         "roll": 0.0,
         "begin": 1764547200.0,
         "end": 1764548200.0,
         "merit": 95.0,
         "slewtime": 120,
         "insaa": 0,
         "obsid": 1001,
         "obstype": "AT",
         "slewdist": 10.3,
         "ss_min": 300.0,
         "ss_max": 1000000.0,
         "exptime": 880,
         "exporig": 1000,
         "isat": false,
         "done": true,
         "exposure": 880
       }
     ]
   }

Metadata Fields
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``version``
     - string
     - COASTSim package version that produced the file.
   * - ``created_at``
     - string
     - ISO-8601 UTC timestamp of when the :class:`~conops.targets.plan_schema.PlanSchema`
       instance was created/validated (not updated by :meth:`~conops.targets.plan_schema.PlanSchema.save`).
   * - ``start``
     - float
     - Unix timestamp of the first entry's ``begin`` time (0 if the plan is empty).
   * - ``end``
     - float
     - Unix timestamp of the last entry's ``end`` time (0 if the plan is empty).
   * - ``num_entries``
     - int
     - Total number of entries in the file.

Entry Fields
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``name``
     - string
     - Human-readable target name (e.g. ``"Crab Nebula"``).
   * - ``ra``
     - float
     - Right ascension in degrees (J2000).
   * - ``dec``
     - float
     - Declination in degrees (J2000).
   * - ``roll``
     - float
     - Spacecraft roll angle in degrees (``-1`` = unset).
   * - ``begin``
     - float
     - Start of the visibility window (Unix time).
   * - ``end``
     - float
     - End of the visibility window (Unix time).
   * - ``merit``
     - float
     - Scheduler merit/priority figure of merit.
   * - ``slewtime``
     - int
     - Slew duration in seconds.
   * - ``insaa``
     - int
     - Time spent in the South Atlantic Anomaly during the window (seconds).
   * - ``obsid``
     - int
     - Numeric observation identifier.
   * - ``obstype``
     - string
     - Observation type (e.g. ``"AT"`` for Astronomical Target, ``"PPT"`` for Preprogrammed Target).
   * - ``slewdist``
     - float
     - Angular slew distance in degrees.
   * - ``ss_min``
     - float
     - Minimum Sun-spacecraft separation angle encountered (degrees).
   * - ``ss_max``
     - float
     - Maximum Sun-spacecraft separation angle encountered (degrees).
   * - ``exptime``
     - int
     - Exposure time in seconds (may be shorter than original if interrupted).
   * - ``exporig``
     - int
     - Originally requested exposure time in seconds.
   * - ``isat``
     - bool
     - ``true`` if the detector was saturated during the exposure.
   * - ``done``
     - bool
     - ``true`` if the observation completed successfully.
   * - ``exposure``
     - int
     - Net science exposure time: ``end − begin − slewtime − insaa`` (seconds).

Backward Compatibility
----------------------

:meth:`~conops.targets.plan_schema.PlanSchema.save` creates any missing parent directories
automatically, so you can pass a nested path without creating it first.

:meth:`~conops.targets.plan_schema.PlanSchema.load` accepts files written by older versions of
COASTSim that predate ``PlanSchema``.  Fields not present in the file (e.g. ``created_at``,
``num_entries``) are filled with schema defaults.  Legacy files must already use the field
names documented above (including ``exporig``); there is currently no automatic renaming or
aliasing of deprecated keys.

The ``from_attributes=True`` model configuration means the schema can also validate against
any object that exposes the expected attributes, not just plain dicts.

API Reference
-------------

.. automethod:: conops.targets.plan.Plan.save

.. automethod:: conops.targets.plan.Plan.load

.. autoclass:: conops.targets.plan_schema.PlanEntrySchema
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conops.targets.plan_schema.PlanSchema
   :members:
   :undoc-members:
   :show-inheritance:

.. _Pydantic v2: https://docs.pydantic.dev/latest/
