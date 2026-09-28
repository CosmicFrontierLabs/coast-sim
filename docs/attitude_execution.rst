Physical attitude execution
===========================

ACS operating mode and physical motion are separate. Mode handlers and guidance
may request new targets; only the attitude executor advances orientation and
body-frame angular velocity. Each update advances the previously installed
trajectory before processing commands. Assigning ``last_slew`` or changing modes
does not move the spacecraft.

Execution contract
------------------

* Every turn is an analytic quaternion trajectory with the configured scalar or
  directional rate and acceleration limits. Planning and execution share the
  same rest-to-rest motion profile.
* Trajectories snapshot geometry and motion coefficients. Mutating a command's
  slew object cannot rewrite installed motion. Changing configuration during a
  run remains unsupported; start a new run after configuration changes.
* Handoffs must match orientation and occur at zero angular velocity. An
  interrupting slew or end-of-tracking command finishes the current turn, then
  holds at that rest boundary until the next execution tick. It does not reset
  velocity. This is conservative preemption, not an optimal braking maneuver.
* SAFE entry changes mode immediately, but its repointing follows that same
  handoff policy. Charging and SAFE solar guidance also request bounded turns;
  an unreachable target is tracked with lag, not an instantaneous correction.
  Queued commands take priority over these discretionary guidance corrections.
* Ground-contact attitudes are connected by time-stretched rest-to-rest turns.
  Acquisition must already meet the selected profile's pointing tolerance, and
  every interval must be kinematically feasible. Infeasible execution raises
  ``AttitudeExecutionError`` rather than stretching a contact window silently.
* After initialization, ``acs.ra``, ``acs.dec`` and ``acs.roll`` are read-only.
  Set an initial attitude before the first update, then use ACS commands.
  Without an explicit initial roll, solar-optimal roll is selected once at the
  initial boundary. ``angular_velocity_body`` reports degrees per second.

The scheduler predicts from this installed physical state, including tracking,
rather than extrapolating the last target's metadata. Repeating an update at the
same timestamp cannot advance motion. Coarse telemetry samples do not make the
underlying analytic trajectory instantaneous.

Model limits
------------

This is a kinematic execution boundary, not a full spacecraft dynamics engine.
It enforces angular rate and acceleration, but does not model actuator torque,
wheel momentum, flexible modes, jerk limits or pointing-control error. Acceleration
can change discontinuously within the specified bound; angular velocity cannot.

The contact model stops at each profile knot. It preserves sampled contact
attitudes, but is not an exact continuously tracking antenna solution between
knots. Likewise, changing guidance or profile cadence can change the requested
trajectory; only sampling an already installed trajectory is cadence-independent.

Keepout and visibility checks remain separate planning/validation concerns.
Kinematic feasibility does not prove continuous-time clearance, nor guarantee
that an emergency transition or its waiting attitude satisfies every keepout.
The independent recorded-attitude rate check remains a secondary diagnostic.
