Engineering attitude exports
============================

A retained simulation can sample its executed attitude at a finer interval than
its command or telemetry step. Sampling evaluates the flown slew, tracking and
hold trajectories, including interrupted motion. It does not interpolate between
coarse telemetry quaternions or regenerate a commanded plan.

Export a retained run
---------------------

After a successful ``run.calc()``, sample the interval needed for analysis::

    from conops.targets import Plan

    attitude = run.sample_attitude(interval_seconds=1.0)
    attitude.save_stk("executed_attitude.a", reference_frame="J2000")
    attitude.save_euler_csv("executed_attitude.csv", reference_frame="J2000")

    run.plan.attitude_timeseries = attitude
    saved = run.plan.save("executed_plan.json")
    loaded = Plan.load(saved, load_timeseries=True)
    loaded.attitude_timeseries.save_stk("reopened_attitude.a")

``begin`` and ``end`` accept UTC datetimes, ISO strings or Unix seconds.
Both endpoints are included. Motion boundaries and executed handoffs are also
included unless ``include_transitions=False``. Consequently a nominal one-second
series may contain additional samples between its regular ticks.

The requested interval must have been flown and remain covered by the ephemeris.
The default final time is the last time physically advanced by the attitude
executor; it may precede the configured run end by one simulation step. Sampling
before execution or beyond the retained interval raises ``ValueError``. Export
or save the fine series before discarding the run: loading a coarse sidecar
cannot reconstruct finer motion.

Frames and file conventions
---------------------------

Quaternions describe the physical spacecraft body frame declared by
``SpacecraftBus.frame``. Every body-frame configuration vector and hardware
mounting must use that same frame. The auxiliary RA/Dec/roll fields keep their
existing body-X parameterization; use the quaternion for the complete physical
orientation.

* STK ``.a`` files use ``AttitudeTimeQuatScalarFirst``, elapsed UTC seconds from
  ``ScenarioEpoch``, and scalar-first Shuster quaternions. The Hamilton vector
  components are conjugated to preserve the reference-to-body rotation in
  STK's convention. Adjacent quaternion signs remain continuous.
* Euler CSV files contain intrinsic 1-2-3 angles in degrees, with the convention
  ``R_reference_to_body = (Rx(a) Ry(b) Rz(c)).T``. At gimbal lock the final angle
  is zero and the ``gimbal_lock`` column is true; the equivalent rotation is
  preserved. The body +X and +Z angles to the Sun are included when the series
  contains a Sun vector.
* ``reference_frame="ICRF"`` uses the GCRS reference axes directly.
  ``reference_frame="J2000"`` converts to the FK5/IAU76/80 realization used by
  STK through its Earth-fixed relation. UTC, TT and UT1 are kept distinct;
  sampled series retain UT1-UTC for repeatable re-export. Older telemetry
  sidecars without UT1-UTC can use ICRF, but J2000 export raises instead of
  inventing a conversion. See STK's `central-body reference frames
  <https://help.agi.com/stk/content/stk/referenceFramesCentralBodies.htm>`_.

Each export has a ``.metadata.json`` companion recording the rotation convention,
reference and spacecraft frames, sampling source and interval, and origin offset.
``origin_offset_m`` records a translation in metadata only; it does not modify
attitude. Supply the actual vector components when an analysis requires them.
No displacement direction is inferred from a distance.

Sidecars and downstream checks
------------------------------

``Plan.load(path, load_timeseries=True)`` reads linked sibling attitude and orbit
sidecars; normal plan-only loading remains the default. Sidecar names cannot
escape the plan's directory. A reloaded fine series can be exported without the
original live executor.

Rotation round trips, quaternion conventions, frame transforms, gimbal lock,
interrupted trajectories and a real planned flight are covered by library tests.
Import settings and axis/origin assumptions still need checking in the intended
STK and thermal-analysis installations.
