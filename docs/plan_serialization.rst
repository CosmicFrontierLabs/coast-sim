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
     "version": 3,
     "coast_sim_version": "0.1.3",
     "created_at": "2025-12-01T00:00:00+00:00",
     "start": "2025-12-01T00:00:00+00:00",
     "end": "2025-12-01T23:59:00+00:00",
     "num_entries": 42,
     "entries": [
       {
         "name": "TEST_001",
         "ra": 83.82,
         "dec": -5.39,
         "roll": 0.0,
         "begin": "2025-12-01T00:00:00+00:00",
         "end": "2025-12-01T00:16:40+00:00",
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
       },
       {
         "name": "SGS_PASS",
         "ra": 120.0,
         "dec": 45.0,
         "roll": 0.0,
         "begin": "2025-12-01T00:18:00+00:00",
         "end": "2025-12-01T00:28:00+00:00",
         "merit": 101.0,
         "slewtime": 120,
         "insaa": 0,
         "obsid": 65535,
         "obstype": "GSP",
         "slewdist": 5.2,
         "ss_min": 45.0,
         "ss_max": 180.0,
         "exptime": 480,
         "exporig": 600,
         "isat": false,
         "done": true,
         "exposure": 480,
         "station": "SGS",
         "contact_begin": "2025-12-01T00:20:00+00:00",
         "contact_end": "2025-12-01T00:28:00+00:00",
         "track_start_ra": 120.0,
         "track_start_dec": 45.0,
         "track_end_ra": 231.67,
         "track_end_dec": -0.38
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
     - int
     - Integer plan-file revision counter.  Starts at 0 and is incremented
       automatically each time a new plan is saved to the same directory for
       the same time window (see :ref:`auto-versioning` below).
   * - ``coast_sim_version``
     - string
     - COASTSim package version that produced the file (e.g. ``"0.1.3"``).
   * - ``created_at``
     - string
     - ISO-8601 UTC timestamp of when the :class:`~conops.targets.plan_schema.PlanSchema`
       instance was created/validated (not updated by :meth:`~conops.targets.plan_schema.PlanSchema.save`).
   * - ``start``
     - string
     - ISO-8601 UTC timestamp of the first entry's ``begin`` time
       (``"1970-01-01T00:00:00+00:00"`` if the plan is empty).
   * - ``end``
     - string
     - ISO-8601 UTC timestamp of the last entry's ``end`` time
       (``"1970-01-01T00:00:00+00:00"`` if the plan is empty).
   * - ``num_entries``
     - int
     - Total number of entries in the file.
   * - ``attitude_timeseries_file``
     - string | null
     - Filename (no path) of the sibling attitude-timeseries JSON file written alongside
       this plan, or ``null`` / absent when no timeseries was exported.
       See :ref:`attitude-timeseries` for the file format.

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
     - string
     - Start of the observation window (ISO-8601 UTC).
   * - ``end``
     - string
     - End of the observation window (ISO-8601 UTC).
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
     - Observation type. Valid values: ``"AT"`` (Astronomical Target), ``"PPT"`` (Preprogrammed Target),
       ``"TOO"`` (Target of Opportunity), ``"SAFE"`` (Safe mode pointing), ``"CHARGE"`` (Emergency charging),
       ``"GSP"`` (Ground Station Pass).
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
     - Net science exposure time in seconds. For ``AT``/``PPT``/``TOO`` entries:
       ``end − begin − slewtime − insaa``. For ``GSP`` entries: the actual downlink
       contact duration, computed as ``contact_end − max(contact_begin, begin)``.
   * - ``station``
     - string | null
     - Ground station code for ``GSP`` (Ground Station Pass) entries. Only present for
       ``obstype="GSP"``. Identifies which station is used for the commanded pass.
   * - ``contact_begin``
     - string | null
     - ISO-8601 UTC timestamp of when the ground station contact window begins. Only present
       for ``obstype="GSP"``. This is the actual pass start time; ``begin`` is the reservation
       time (which may include slew preparation).
   * - ``contact_end``
     - string | null
     - ISO-8601 UTC timestamp of when the ground station contact window ends. Only present
       for ``obstype="GSP"``. Typically matches ``end``.
   * - ``track_start_ra``
     - float | null
     - Ground-station tracking right ascension at ``contact_begin`` in degrees. Only present
       for ``obstype="GSP"``. For GSP entries, ``ra`` uses the same pass-start convention.
   * - ``track_start_dec``
     - float | null
     - Ground-station tracking declination at ``contact_begin`` in degrees. Only present
       for ``obstype="GSP"``. For GSP entries, ``dec`` uses the same pass-start convention.
   * - ``track_end_ra``
     - float | null
     - Ground-station tracking right ascension used by ACS at pass end, in degrees. This is
       derived from the final tracking sample and matches ``Pass.ra_dec(contact_end)``. Only
       present for ``obstype="GSP"``. Use this with ``track_end_dec`` when inspecting slews
       after a pass.
   * - ``track_end_dec``
     - float | null
     - Ground-station tracking declination used by ACS at pass end, in degrees. This is
       derived from the final tracking sample and matches ``Pass.ra_dec(contact_end)``. Only
       present for ``obstype="GSP"``. Use this with ``track_end_ra`` when inspecting slews
       after a pass.

Ground Station Pass (GSP) Entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ground station pass entries (``obstype="GSP"``) represent commanded communication windows
with ground stations. Unlike science observations, these entries capture the reservation
and execution of data downlink passes.

**Key characteristics of GSP entries:**

* **Automatic creation**: Created by :class:`~conops.ditl.queue_ditl.QueueDITL` when the
  spacecraft enters a ground station visibility window.
* **Pass reservation**: The ``begin`` time marks when the spacecraft reserves the window
  (potentially including slew preparation), while ``contact_begin`` marks the actual start
  of the ground station contact.
* **Metadata fields**: The ``station``, ``contact_begin``, and ``contact_end`` fields are
  only present for GSP entries (``null`` or omitted for other observation types).
* **Tracking attitude**: GSP entries track the ground station through the contact. The
  generic ``ra`` and ``dec`` fields are the pass-start pointing, matching
  ``track_start_ra`` and ``track_start_dec``. Use ``track_end_ra`` and
  ``track_end_dec`` to inspect the spacecraft pointing at the end of the pass. The end
  fields are derived from the final tracking sample, matching ACS
  ``Pass.ra_dec(contact_end)`` behavior.
* **Deconfliction**: When multiple ground stations are visible simultaneously, COASTSim
  automatically selects the pass with the highest expected data volume (downlink rate × duration).
  Dropped overlapping opportunities are logged but not exported to the plan.
* **Visualization**: GSP entries are excluded from the science observation bands in
  :func:`~conops.visualization.mpl.ditl_timeline.plot_ditl_timeline` and
  :func:`~conops.visualization.plotly.ditl_timeline.plot_ditl_timeline`.
  Ground station passes are still shown in the timeline's dedicated "Ground Contact" row.
* **Safe mode**: No GSP entries are created when the spacecraft is in SAFE mode.

**Example GSP entry:**

.. code-block:: json

   {
     "name": "TRO_PASS",
     "obstype": "GSP",
     "station": "TRO",
     "begin": "2025-12-01T12:00:00+00:00",
     "contact_begin": "2025-12-01T12:02:00+00:00",
     "contact_end": "2025-12-01T12:12:00+00:00",
     "end": "2025-12-01T12:12:00+00:00",
     "track_start_ra": 120.0,
     "track_start_dec": 45.0,
     "track_end_ra": 231.67,
     "track_end_dec": -0.38,
     "slewtime": 120,
     "exptime": 600,
     "obsid": 65535
   }

In this example, the spacecraft begins reserving the pass window at 12:00:00 (``begin``),
uses 2 minutes for slew preparation (``slewtime``), and the actual ground station contact
runs from 12:02:00 to 12:12:00 (``contact_begin`` to ``contact_end``).

.. _auto-versioning:

Auto-versioning
~~~~~~~~~~~~~~~

When ``path`` is a directory (or ends with ``/``), :meth:`~conops.targets.Plan.save`
scans the directory for existing files matching
``plan_<start>_<end>_v<N>.json`` and sets ``version`` to ``max(N) + 1``
(or ``0`` if no matching files exist).  Saving to an explicit file path
leaves ``version`` unchanged.

.. _attitude-timeseries:

Attitude Timeseries
~~~~~~~~~~~~~~~~~~~

Both :class:`~conops.ditl.ditl.DITL` and :class:`~conops.ditl.queue_ditl.QueueDITL`
automatically attach the executed spacecraft attitude timeline to the plan at the end of
``calc()``/``run()``.  When the plan is then saved, this timeseries is written to a sibling
JSON file alongside the main plan file.

**File naming**: the sibling file is placed in the same directory as the plan and named
``<plan_stem>_attitude_timeseries.json``.  For example, if the plan is saved as
``plan_20251201T000000Z_20251201T235900Z_v3.json`` the attitude file is
``plan_20251201T000000Z_20251201T235900Z_v3_attitude_timeseries.json``.

The plan file records the sibling filename in the ``attitude_timeseries_file`` metadata
field so that consumers can locate it without scanning the directory.

**Attitude Timeseries File Format**

.. code-block:: json

   {
     "version": 0,
     "coast_sim_version": "0.6.1",
     "created_at": "2025-12-01T00:00:00+00:00",
     "plan_file": "plan_20251201T000000Z_20251201T235900Z_v3.json",
     "plan_version": 3,
     "plan_start": "2025-12-01T00:00:00+00:00",
     "plan_end": "2025-12-01T23:59:00+00:00",
     "num_samples": 2,
     "samples": [
       {
         "utime": 1748736000.0,
         "timestamp": "2025-12-01T00:00:00+00:00",
         "ra": 83.82,
         "dec": -5.39,
         "roll": 0.0,
         "mode": "SCIENCE",
         "obsid": 1001,
         "quat_w": 0.9998,
         "quat_x": 0.0050,
         "quat_y": -0.0120,
         "quat_z": 0.0150
       },
       {
         "utime": 1748736060.0,
         "timestamp": "2025-12-01T00:01:00+00:00",
         "ra": 83.82,
         "dec": -5.39,
         "roll": 0.0,
         "mode": "SCIENCE",
         "obsid": 1001,
         "quat_w": 0.9998,
         "quat_x": 0.0051,
         "quat_y": -0.0121,
         "quat_z": 0.0151
       }
     ]
   }

**Attitude Timeseries Metadata Fields**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``version``
     - int
     - Schema format version (currently always ``0``).
   * - ``coast_sim_version``
     - string
     - COASTSim package version that produced the file.
   * - ``created_at``
     - string
     - ISO-8601 UTC timestamp of when the file was created.
   * - ``plan_file``
     - string | null
     - Filename of the associated plan JSON file.
   * - ``plan_version``
     - int | null
     - ``version`` value of the associated plan.
   * - ``plan_start``
     - string | null
     - ISO-8601 UTC timestamp matching the plan's ``start`` field.
   * - ``plan_end``
     - string | null
     - ISO-8601 UTC timestamp matching the plan's ``end`` field.
   * - ``num_samples``
     - int
     - Number of attitude samples in ``samples``.

**Attitude Sample Fields**

Each element of ``samples`` is an :class:`~conops.targets.plan_schema.AttitudeSampleSchema`:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Field
     - Type
     - Description
   * - ``utime``
     - float
     - Unix timestamp (seconds since epoch, UTC).
   * - ``timestamp``
     - string
     - ISO-8601 UTC timestamp — human-readable form of ``utime``.
   * - ``ra``
     - float | null
     - Right ascension of the spacecraft boresight in degrees (J2000).
   * - ``dec``
     - float | null
     - Declination of the spacecraft boresight in degrees (J2000).
   * - ``roll``
     - float | null
     - Spacecraft roll angle in degrees.
   * - ``mode``
     - string | null
     - ACS mode name at this sample (e.g. ``"SCIENCE"``, ``"SLEWING"``, ``"PASS"``).
   * - ``obsid``
     - int | null
     - Observation ID active at this sample.
   * - ``quat_w``
     - float | null
     - Spacecraft attitude quaternion W component.
   * - ``quat_x``
     - float | null
     - Spacecraft attitude quaternion X component.
   * - ``quat_y``
     - float | null
     - Spacecraft attitude quaternion Y component.
   * - ``quat_z``
     - float | null
     - Spacecraft attitude quaternion Z component.

**Loading the Attitude Timeseries**

.. code-block:: python

   from conops.targets import PlanSchema, AttitudeTimeseriesSchema
   from pathlib import Path

   schema = PlanSchema.load("plan_20251201T000000Z_20251201T235900Z_v3.json")

   if schema.attitude_timeseries_file:
       plan_dir = Path("plan_20251201T000000Z_20251201T235900Z_v3.json").parent
       timeseries_path = plan_dir / schema.attitude_timeseries_file
       raw = __import__("json").loads(timeseries_path.read_text())
       timeseries = AttitudeTimeseriesSchema.model_validate(raw)
       print(f"Loaded {timeseries.num_samples} attitude samples")

Backward Compatibility
----------------------

:meth:`~conops.targets.plan_schema.PlanSchema.save` creates any missing parent directories
automatically, so you can pass a nested path without creating it first.

:meth:`~conops.targets.plan_schema.PlanSchema.load` accepts files written by older versions of
COASTSim that predate ``PlanSchema``.  Fields not present in the file (e.g. ``created_at``,
``num_entries``) are filled with schema defaults; ``num_entries`` is always recomputed from
the actual entry list after loading.  Legacy files that store ``start``, ``end``, ``begin``,
or ``end`` as numeric Unix timestamps (float/int) are accepted and converted to
:class:`~datetime.datetime` objects internally — only the on-disk format changed to ISO-8601.
Legacy files that stored ``version`` as a semantic-version string (e.g. ``"0.1.3"``) are
coerced to ``0``.  Legacy files must already use the field names documented above (including
``exporig``); there is currently no automatic renaming or aliasing of deprecated keys.

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

.. autoclass:: conops.targets.plan_schema.AttitudeTimeseriesSchema
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conops.targets.plan_schema.AttitudeSampleSchema
   :members:
   :undoc-members:
   :show-inheritance:

.. _Pydantic v2: https://docs.pydantic.dev/latest/
