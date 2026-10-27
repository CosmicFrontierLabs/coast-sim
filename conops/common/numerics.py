"""Numerical tolerances and small structural constants shared across the
simulation, named for what they decide rather than for their values."""

ROUNDOFF = 1e-12
"""A length, dot product or interval below this is floating-point roundoff:
a vector this short has no direction, and a quantity this small is zero."""

NEGLIGIBLE_NORM = 1e-15
"""A vector norm below this is taken as exactly zero where a division by it
would follow."""

POLE_NORM = 1e-10
"""An axis whose projection on the equatorial plane is shorter than this
lies along the pole, where a position angle is undefined."""

REST_RATE_DEG_S = 1e-10
"""An angular rate below this, in degrees per second, is rest."""

TIME_TOLERANCE_S = 1e-6
"""Two times within this many seconds are the same time."""

ANGLE_TOLERANCE_DEG = 1e-6
"""Two angles within this many degrees are the same angle."""

ATTITUDE_TOLERANCE_DEG = 1e-8
"""Two attitudes within this quaternion distance, in degrees, are the same
attitude: a hold does not change the physical attitude."""

ROLL_TOLERANCE_DEG = 1e-9
"""Two rolls within this many degrees are the same roll."""

UNIT_MAGNITUDE_TOLERANCE = 0.01
"""A configured direction vector must have a magnitude within this of one."""

VECTOR_COMPONENTS = 3
"""The components of a position, velocity or direction vector."""

MIN_SAMPLES = 2
"""The fewest samples a track, a profile or an interval can have."""

HALF_TURN_DEG = 180.0
FULL_TURN_DEG = 360.0
POLE_DEC_DEG = 90.0
