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

Delayed science commands are rounded up to their actual scheduler execution
tick before path and observation validation. The attitude held while waiting
(including after an in-progress slew) must remain safe through that tick;
departure cannot be later than the sample preceding the first violation.

If no science target is available when the current hold reaches its departure
deadline, the scheduler searches deterministic nearby attitudes and roll
alternatives. A candidate must have:

- a finite slew obeying the configured directional motion limits;
- no sampled SLEWING-scope violation on its trajectory;
- an IDLE-safe hold after arrival for another escape reserve plus
  `spacecraft_bus.attitude_control.idle_min_hold_s` (default 300 seconds),
  clipped to the simulation end;
- enough time for every available tracking profile of the next planned
  ground-station acquisition, using the same admission deadlines as science
  (including routed-slew bounds and the pass trigger buffer).

The selected maneuver is executed through the ordinary ACS command queue as
an `IDLE`-destination slew. It contributes normal SLEWING telemetry and power.
Discretionary science/TOO selection does not interrupt protective
motion. Subsequent holds retain the commanded attitude and roll.

Already-unsafe initialization, no feasible recovery in the finite search, or
an unsafe executed IDLE hold latches an `idle_safety` RED fault in
`config.fault_management.states`. The first `operational_fault` event retains
the timestamp, cause, and body attitude (RA/Dec/roll in degrees). Repeated
reports do not restart recovery or flood the event log.

With the default `safe_mode_on_red: true` policy, the fault requests SAFE
through fault management and immediately executes the ordinary SAFE command
path. SAFE entry clears pending commands and initiates a finite slew from the
current attitude. It does not substitute a new reported attitude. Science and
TOO selection stop, and any active observation is closed. The simulation
continues recording telemetry through its end in safehold. Explicitly setting
`safe_mode_on_red: false` keeps the fault visible but disables automatic SAFE
entry, consistent with the other fault-management policies.

`QueueDITL.calc()` returns `False` after an idle-safety fault, including in
monitor-only mode: the retained plan and telemetry are diagnostic outputs,
not a valid science plan. Both end-of-run audits still run. Any attitude-rate
or plan-execution failure is recorded as a further operational fault instead
of converting this diagnosed failed run into a Python exception. The public
validation methods still expose the violations; nominal runs retain their
strict exception behavior for execution-validation failures.

SAFE is an operational response, not proof of a keepout-safe escape. The
standard solar-pointing SAFE guidance does not gain a path-validity guarantee
from this policy. An unsafe initial condition remains a failed study input,
and any violations during recovery remain visible. Continuous rate and
acceleration enforcement across all guidance modes is a separate execution
layer; this predictive policy does not supply that model.

This is a conservative planning policy, not a flight controller or proof of
continuous-time collision avoidance. Tests and acceptance checks remain at
the ephemeris cadence; finer sampling must be assessed for a specific study.
The initial attitude, reserve policy and hold dwell affect science yield,
energy and disturbance momentum and should be recorded with study results.
Changing constraints during a run is unsupported; create a fresh run/planner
after configuration edits. An explicitly empty IDLE scope disables its safety
forecast and must not be interpreted as a safe operating policy.
