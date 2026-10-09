Physical attitude execution
===========================

ACS operating mode and physical motion are separate. Mode handlers and guidance
may request new targets; only the attitude executor advances orientation and
body-frame angular velocity. Each update advances the previously installed
trajectory before processing commands. Assigning ``last_slew`` or changing modes
does not move the spacecraft.

Execution contract
------------------

* Every motion uses a quaternion trajectory with the configured scalar or
  directional rate and acceleration limits. Ordinary slews share the planner's
  analytic rest-to-rest motion profile; tracking preserves rate through knots.
* Trajectories snapshot geometry and motion coefficients. Mutating a command's
  slew object cannot rewrite installed motion. Changing configuration during a
  run remains unsupported; start a new run after configuration changes.
* Handoffs must match both orientation and body angular velocity, including at
  nonzero rates. An interrupting slew or end-of-tracking command brakes along
  the current rate axis within the directional acceleration envelope, then
  holds until the next execution tick. The next target slew starts from that
  actual stopping attitude; it does not wait for the old target to be reached.
* SAFE entry changes mode immediately, but its repointing follows that same
  handoff policy. Charging and SAFE solar guidance also request bounded turns;
  an unreachable target is tracked with lag, not an instantaneous correction.
  Queued commands take priority over these discretionary guidance corrections.
* Ground-contact attitudes use normalized quaternion Hermite curves, with shared
  knot rates estimated from neighboring shortest-arc secants. Constant-spin
  intervals use an exact analytic rotation. Admission and execution check rate
  and acceleration over entire intervals using subdivided Bernstein bounds,
  rather than relying on time samples. Failure to certify a curve rejects it.
* Acquisition must already meet the selected profile's pointing tolerance.
  Current ingress slews arrive at rest: the first tracking interval must allow
  acceleration from that boundary rate, but internal knots need not stop.
  Execution matches the actual initial rate, and a final braking arc prevents
  an instantaneous stop when the profile ends. Infeasible trajectories are
  rejected rather than stretching a contact window silently.
* After initialization, ``acs.ra``, ``acs.dec`` and ``acs.roll`` are read-only.
  Set an initial attitude before the first update, then use ACS commands.
  Without an explicit initial roll, solar-optimal roll is selected once at the
  initial boundary. ``angular_velocity_body`` reports degrees per second.

The scheduler predicts from this installed physical state, including tracking
and any newly commanded braking arc, rather than extrapolating the last target's
metadata. Repeating an update at the
same timestamp cannot advance motion. Coarse telemetry samples do not make the
underlying analytic trajectory instantaneous.

Quaternion handoffs
-------------------

Execution compares and transfers quaternions directly. Slew waypoints retain
their native quaternions, and ACS supplies its exact initial ``AttitudeState``
when constructing a slew, a tracking profile, or a guidance turn. RA/Dec/roll
are coordinates for reporting and target requests, not the physical handoff
state: their round trip is ill-conditioned at the celestial poles.

Use ``AttitudeTrajectory.turn_from_state(state, target, limits)`` to continue
from executed motion at rest. ``turn`` remains a convenience for independently
specified RA/Dec/roll endpoints. A rest-to-rest turn cannot replace nonzero
angular velocity; request a physical stop first.

Handoffs allow only numerical error: ``1e-8`` degrees in quaternion separation
and ``1e-9`` degrees/second in the norm of the body-rate difference. These are
fixed numerical tolerances, far below the operating motion limits, not a budget
for discontinuous maneuvers. Quaternion sign reversal represents the same
orientation and is accepted.

Execution faults
----------------

Trajectory builders and the executor raise ``AttitudeExecutionError`` when
their physical contract cannot be satisfied. ACS catches these failures during
command execution and dwell guidance, records an ``execution_fault`` in fault
management and an ERROR in the DITL log, and cancels pending commands and science
or contact activity. The rejected command is not recorded as executed.

The recovery requests a bounded brake from the current quaternion and body
rate. Fault management applies its existing ``safe_mode_on_red`` policy; with
the default policy, ACS enters SAFE and queues its solar-pointing slew after
braking. With automatic SAFE disabled, it brakes and holds without automatically
commanding a new target. Neither policy accepts the rejected trajectory.

If a brake cannot be constructed, the last installed trajectory is retained.
If SAFE motion itself is rejected, discretionary guidance is disabled for the
remainder of the run rather than retried every tick. Both failures are recorded;
SAFE operating mode is not a claim that solar pointing was achieved. Existing
constraint checks still run. Initialization errors, nonmonotonic time, unrelated
programming errors, and final plan-validation failures remain explicit errors;
fault recovery does not certify an invalid plan for export.

Model limits
------------

This is a kinematic execution boundary, not a full spacecraft dynamics engine.
It enforces angular rate and acceleration, but does not model actuator torque,
wheel momentum, flexible modes, jerk limits or pointing-control error. Acceleration
can change discontinuously within the specified bound; angular velocity cannot.

The contact model preserves sampled attitudes and continuous angular velocity,
but is not an exact antenna-tracking solution between knots. Its terminal brake
can move beyond the last pointing sample. Established constant-spin tracking is
independent of knot spacing; acquisition from rest and curved profiles are not
generally cadence-independent. Changing guidance cadence can also change the
requested trajectory. Sampling an already installed trajectory does not change it.

The phase search and interval certificates are conservative, not globally optimal:
failure to find or certify a feasible curve does not prove no such curve exists.
Braking follows the current rate axis; it is not a time-optimal controller for
arbitrary multi-axis dynamics.

Keepout and visibility checks remain separate planning/validation concerns.
Kinematic feasibility does not prove continuous-time clearance, nor guarantee
that an emergency transition or its waiting attitude satisfies every keepout.
The independent recorded-attitude rate check remains a secondary diagnostic.
