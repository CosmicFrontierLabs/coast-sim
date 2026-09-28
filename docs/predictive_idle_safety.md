# Predictive idle safety

An inertial attitude that is safe now need not remain safe as the orbit evolves.
`QueueDITL` forecasts the configured IDLE constraints at the ephemeris cadence
and reserves time to leave each science attitude before it becomes unsafe.

The reserve is a conservative 180-degree, rest-to-rest maneuver evaluated with
the smallest configured acceleration and rate semiaxes (or scalar limits),
including fixed settling and two scheduler timesteps. This bounds direct
quaternion slews; it is not a guarantee that an obstacle-avoiding route exists.
Science admission and collection deadlines use this same reserve.

The first violating sample brackets a crossing in the preceding timestep; it
is not an exact crossing time. A sample at or just beyond the simulation end
therefore still requires escape when that preceding interval overlaps the run.
Recovery holds must also end before the potentially unsafe interval. Crossings
whose entire preceding interval is after the run do not require post-run escape.

If no science target is available when the current hold reaches its departure
deadline, the scheduler searches deterministic nearby attitudes and roll
alternatives. A candidate must have:

- a finite slew obeying the configured directional motion limits;
- no sampled SLEWING-scope violation on its trajectory;
- an IDLE-safe hold after arrival for another escape reserve plus
  `spacecraft_bus.attitude_control.idle_min_hold_s` (default 300 seconds),
  clipped to the simulation end;
- enough time for the next planned ground-station acquisition, if any.

The selected maneuver is executed through the ordinary ACS command queue as
an `IDLE`-destination slew. It contributes normal SLEWING telemetry and power.
Discretionary science/TOO selection does not interrupt protective
motion. Subsequent holds retain the commanded attitude and roll.

Already-unsafe initialization, no feasible recovery in the finite search, or
an unsafe executed IDLE hold raises an error. ACS no longer changes attitude
instantaneously or dispatches an unvalidated solar-pointing safe-mode command
to conceal that failure. Callers must supply a safe initial body attitude.

This is a conservative planning policy, not a flight controller or proof of
continuous-time collision avoidance. Tests and acceptance checks remain at
the ephemeris cadence; finer sampling must be assessed for a specific study.
The initial attitude, reserve policy and hold dwell affect science yield,
energy and disturbance momentum and should be recorded with study results.
Changing constraints during a run is unsupported; create a fresh run/planner
after configuration edits. An explicitly empty IDLE scope disables its safety
forecast and must not be interpreted as a safe operating policy.
