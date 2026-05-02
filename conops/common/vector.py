from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from pyproj import Geod


def radec2vec(ra: float, dec: float) -> npt.NDArray[np.float64]:
    """Convert RA/Dec angle (in radians) to a vector"""

    v1 = np.cos(dec) * np.cos(ra)
    v2 = np.cos(dec) * np.sin(ra)
    v3 = np.sin(dec)

    return np.array([v1, v2, v3], dtype=np.float64)


def scbodyvector(
    ra: float, dec: float, roll: float, eciarr: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """For a given RA,Dec and Roll, and vector, return that vector that in
    the spacecraft body coordinate system"""

    # Precalculate, to cut by half the number of trig commands we do (optimising)
    croll = np.cos(-roll)
    sroll = np.sin(-roll)
    cra = np.cos(ra)
    sra = np.sin(ra)
    cdec = np.cos(-dec)
    sdec = np.sin(-dec)

    # Direction Cosine matrix (new sleeker version)
    rot1: npt.NDArray[np.float64] = np.array(
        ((1, 0, 0), (0, croll, sroll), (0, -sroll, croll))
    )
    rot2: npt.NDArray[np.float64] = np.array(
        ((cdec, 0, -sdec), (0, 1, 0), (sdec, 0, cdec))
    )
    rot3: npt.NDArray[np.float64] = np.array(((cra, sra, 0), (-sra, cra, 0), (0, 0, 1)))

    # Multiply them all up
    a: npt.NDArray[np.float64] = np.dot(rot1, rot2)
    b: npt.NDArray[np.float64] = np.dot(a, rot3)
    body: npt.NDArray[np.float64] = np.dot(b, eciarr)
    return body


def vecnorm(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a vector to unit length. If the vector has zero length, returns the original vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v  # Return original vector if it's effectively zero-length
    return v / norm


def rotvec(n: int, a: float, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rotate a vector v by angle a (radians) around axis n (1=x,2=y,3=z).
    Preserves the original sign convention (rotation uses -a)."""
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")

    v = np.asarray(v, dtype=float).copy()
    k = np.zeros(3)
    k[n - 1] = 1.0

    c = np.cos(a)
    s = -np.sin(a)  # match original sign convention

    result: npt.NDArray[np.float64] = (
        v * c + np.cross(k, v) * s + k * (np.dot(k, v)) * (1 - c)
    )
    return result


def separation(
    one: npt.NDArray[np.float64] | list[float],
    two: npt.NDArray[np.float64] | list[float],
) -> float:
    """Calculate the angular distance between two RA,Dec values.
    Both Ra/Dec values are given as an array of form [ra,dec] where
    RA and Dec are in radians. Form of function mimics pyephem library
    version except result is simply in radians."""

    onevec = radec2vec(one[0], one[1])
    twovec = radec2vec(two[0], two[1])

    # Flatten vectors to ensure they're 1D for dot product
    onevec = np.atleast_1d(onevec).flatten().astype(float)
    twovec = np.atleast_1d(twovec).flatten().astype(float)

    # Normalize and handle degenerate zero-length vectors
    n1 = np.linalg.norm(onevec)
    n2 = np.linalg.norm(twovec)
    if n1 < 1e-15 or n2 < 1e-15:
        # If either vector is effectively zero-length, treat separation as zero
        return 0.0

    # Compute cosine of the angle and clip to [-1, 1] to avoid rounding errors producing NaN
    cosang = np.dot(onevec / n1, twovec / n2)
    cosang = np.clip(cosang, -1.0, 1.0)

    return float(np.arccos(cosang))


def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Calculate the angular distance between two RA,Dec values in degrees."""
    ra1_rad = np.deg2rad(ra1)
    dec1_rad = np.deg2rad(dec1)
    ra2_rad = np.deg2rad(ra2)
    dec2_rad = np.deg2rad(dec2)

    sep_rad = separation([ra1_rad, dec1_rad], [ra2_rad, dec2_rad])
    return float(np.rad2deg(sep_rad))


def great_circle(
    ra1: float, dec1: float, ra2: float, dec2: float, npts: int = 100
) -> tuple[list[float], list[float]]:
    """Return Great Circle Path between two coordinates"""
    g = Geod(ellps="sphere")

    lonlats = g.npts(ra1 - 180, dec1, ra2 - 180, dec2, npts)

    ras, decs = np.array(lonlats).transpose()

    ras += 180
    ras = np.append(ra1, ras)
    ras = np.append(ras, ra2)
    decs = np.append(dec1, decs)
    decs = np.append(decs, dec2)
    return ras.tolist(), decs.tolist()


def roll_over_angle(
    angles: npt.NDArray[np.float64] | list[float],
) -> npt.NDArray[np.float64]:
    """Make a list of angles that include a roll over (e.g. 359.9 - 0.1) into a smooth distribution.

    Uses 180° threshold to ensure interpolation always takes the shortest path
    around the circle. This is critical for slew paths that cross near the poles
    where RA can change rapidly.
    """
    outangles = list()
    last = -1.0
    flip = 0.0
    diff = 0.0

    for i in range(len(angles)):
        if last != -1:
            diff = angles[i] + flip - last
            # Use 180° threshold to always take shortest path around circle
            if diff > 180:
                flip -= 360
            elif diff < -180:
                flip += 360
        raf = angles[i] + flip
        last = raf
        outangles.append(raf)

    return np.array(outangles)


def vec2radec(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a vector to Ra/Dec (in radians).

    RA is always returned in [0, 2π).
    """
    # Normalize once
    norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    # Dec from z component
    dec = np.arcsin(v[2] / norm)

    # RA from x,y using arctan2 (handles all quadrants correctly)
    # arctan2 returns [-π, π], so add 2π and mod to get [0, 2π)
    ra = np.arctan2(v[1], v[0]) % (2 * np.pi)

    return np.array([ra, dec])


def normal_to_euler_deg(
    normal: tuple[float, float, float] | npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Convert a body-frame normal vector to (roll, pitch, yaw) in degrees.

    The mapping is equivalent to the prior radiator-local helper:
    - roll is fixed at 0 deg
    - pitch = atan2(z, hypot(x, y))
    - yaw = atan2(y, x)
    """
    x, y, z = vecnorm(np.asarray(normal, dtype=np.float64))
    yaw_deg = float(np.rad2deg(np.arctan2(y, x)))
    pitch_deg = float(np.rad2deg(np.arctan2(z, np.hypot(x, y))))
    return 0.0, pitch_deg, yaw_deg


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------
# Convention: q = [w, x, y, z] (scalar-first).
# Attitude quaternion represents the rotation from ECI to spacecraft body
# frame, matching the direction-cosine convention used in scbodyvector():
#   R = R_x(-roll) @ R_y(dec) @ R_z(-ra)   (all angles in radians)
# Body X = boresight, Body Z = "up" (defines roll).


def _rot_to_quat(rot: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a 3×3 rotation matrix to quaternion [w, x, y, z] (Shepperd)."""
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / float(np.sqrt(trace + 1.0))
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]))
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * float(np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]))
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * float(np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]))
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_rot(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion [w, x, y, z] to 3×3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def attitude_to_quat(
    ra_deg: float, dec_deg: float, roll_deg: float
) -> npt.NDArray[np.float64]:
    """Convert spacecraft attitude (RA, Dec, Roll) in degrees to quaternion [w, x, y, z].

    The attitude quaternion encodes the rotation R = R_x(-roll) @ R_y(dec) @ R_z(-ra)
    that transforms ECI vectors into spacecraft body frame.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    roll = np.deg2rad(roll_deg)
    # Elementary rotation quaternions (axis-angle: q = [cos(θ/2), axis*sin(θ/2)])
    q_z = np.array(
        [np.cos(ra / 2), 0.0, 0.0, -np.sin(ra / 2)], dtype=np.float64
    )  # R_z(-ra)
    q_y = np.array(
        [np.cos(dec / 2), 0.0, np.sin(dec / 2), 0.0], dtype=np.float64
    )  # R_y(+dec)
    q_x = np.array(
        [np.cos(roll / 2), -np.sin(roll / 2), 0.0, 0.0], dtype=np.float64
    )  # R_x(-roll)
    # Compose: q = q_x ⊗ q_y ⊗ q_z  (rightmost applied first)
    return _quat_mul(_quat_mul(q_x, q_y), q_z)


def _quat_mul(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Quaternion product a ⊗ b, both [w, x, y, z]."""
    aw, ax, ay, az = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bw, bx, by, bz = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_attitude(q: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to (RA, Dec, Roll) in degrees.

    Inverse of attitude_to_quat().
    """
    rot = _quat_to_rot(q)
    # Row 0 of rot is the boresight (body X) direction in ECI
    bx, by, bz = float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2])
    dec_rad = float(np.arcsin(np.clip(bz, -1.0, 1.0)))
    ra_rad = float(np.arctan2(by, bx)) % (2 * np.pi)

    # Row 2 of rot is the body-Z (up) direction in ECI
    b = np.array([bx, by, bz])
    body_z_eci = np.array([float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2])])
    north = np.array([0.0, 0.0, 1.0])
    n_proj = north - np.dot(north, b) * b
    n_norm = float(np.linalg.norm(n_proj))
    if n_norm < 1e-10:
        # Boresight near celestial pole – use RA=0 direction as reference
        north = np.array([1.0, 0.0, 0.0])
        n_proj = north - np.dot(north, b) * b
        n_norm = float(np.linalg.norm(n_proj))
    n_hat = n_proj / n_norm
    e_hat = np.cross(b, n_hat)  # east direction in boresight-perpendicular plane
    roll_rad = float(np.arctan2(np.dot(body_z_eci, e_hat), np.dot(body_z_eci, n_hat)))

    return (
        float(np.rad2deg(ra_rad)),
        float(np.rad2deg(dec_rad)),
        float(np.rad2deg(roll_rad)),
    )


def quat_slerp(
    q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64], t: float
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation between two attitude quaternions.

    Args:
        q1: Start quaternion [w, x, y, z].
        q2: End quaternion [w, x, y, z].
        t:  Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit quaternion.
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    # Always take the shorter arc through SO(3)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        # Quaternions nearly identical – linear blend then normalise
        result: npt.NDArray[np.float64] = q1 + t * (q2 - q1)
        return result / float(np.linalg.norm(result))
    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    interp: npt.NDArray[np.float64] = (
        np.sin((1.0 - t) * theta_0) * q1 + np.sin(t * theta_0) * q2
    ) / sin_theta_0
    return interp


def quaternion_slew_path(
    ra1: float,
    dec1: float,
    roll1: float,
    ra2: float,
    dec2: float,
    roll2: float,
    steps: int = 100,
) -> tuple[list[float], list[float], list[float]]:
    """Compute a slew path via quaternion SLERP.

    Returns uniformly-spaced intermediate attitudes (including endpoints) as
    separate RA, Dec, and Roll lists (all in degrees).
    """
    q1 = attitude_to_quat(ra1, dec1, roll1)
    q2 = attitude_to_quat(ra2, dec2, roll2)
    ras, decs, rolls = [], [], []
    for i in range(steps + 1):
        t = i / steps
        q = quat_slerp(q1, q2, t)
        ra, dec, roll = quat_to_attitude(q)
        ras.append(ra)
        decs.append(dec)
        rolls.append(roll)
    return ras, decs, rolls


# ---------------------------------------------------------------------------
# Constraint-avoiding slew path
# ---------------------------------------------------------------------------


def constraint_avoiding_waypoint(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    time: float,
    constraint_check_fn: Callable[[float, float, float], bool],
    margin_deg: float = 5.0,
    samples: int = 50,
) -> tuple[float, float] | None:
    """Return a waypoint (RA, Dec) that routes a slew around constraint violations.

    If the direct great-circle arc from (ra1, dec1) to (ra2, dec2) would violate
    the provided constraint at the given time, a single waypoint is computed that
    routes around the constraint.  Two candidate waypoints (one on each side of
    the arc) are considered; the one giving the shorter total detour is returned.

    Returns None when no constraint violation is detected on the direct arc.

    Args:
        ra1, dec1:       Start pointing (degrees).
        ra2, dec2:       End pointing (degrees).
        time:            Unix timestamp for constraint evaluation.
        constraint_check_fn: Callable with signature (ra, dec, time) -> bool
                         that returns True if constraint is violated.
        margin_deg:      Extra buffer beyond the constraint boundary (degrees).
        samples:         Number of points to sample along the arc for violation check.

    Returns:
        (waypoint_ra, waypoint_dec) in degrees if a violation is found, None otherwise.
    """
    a = vecnorm(radec2vec(np.deg2rad(ra1), np.deg2rad(dec1)))
    b = vecnorm(radec2vec(np.deg2rad(ra2), np.deg2rad(dec2)))

    n = vecnorm(np.cross(a, b))
    if np.linalg.norm(n) < 1e-12:
        # Start and end are identical or antipodal
        return None

    # Sample the direct arc to detect violations
    # Use SLERP for uniform angular spacing
    total_angle = float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))
    if total_angle < 1e-6:
        # Arc too short to matter
        return None

    # Sample the arc and check for violations
    violation_found = False
    closest_violator = None

    for i in range(samples + 1):
        t = i / samples
        # SLERP between start and end vectors
        if total_angle < 0.01:
            # Near-identical points, use linear interpolation
            sample_vec = vecnorm(a + t * (b - a))
        else:
            sin_total = np.sin(total_angle)
            sample_vec = (
                np.sin((1 - t) * total_angle) * a + np.sin(t * total_angle) * b
            ) / sin_total

        sample_radec = vec2radec(sample_vec)
        sample_ra = float(np.rad2deg(sample_radec[0]))
        sample_dec = float(np.rad2deg(sample_radec[1]))

        # Check if this point violates the constraint
        if constraint_check_fn(sample_ra, sample_dec, time):
            violation_found = True
            # Track the point with minimum distance from the arc plane
            # (for now, just use the first violation)
            if closest_violator is None:
                closest_violator = sample_vec
            break

    if not violation_found or closest_violator is None:
        return None

    # A violation was found. Compute waypoints perpendicular to the arc plane.
    # Use the closest violator as the reference point on the arc
    c = vecnorm(closest_violator)

    # Compute two candidate waypoints: one on each side of the arc plane
    # Offset by margin_deg from the current position
    offset_rad = np.deg2rad(margin_deg)

    # Two perpendicular directions relative to the arc plane
    away_dir1 = vecnorm(np.cross(n, c))
    away_dir2 = -away_dir1

    w1 = vecnorm(np.cos(offset_rad) * c + np.sin(offset_rad) * away_dir1)
    w2 = vecnorm(np.cos(offset_rad) * c + np.sin(offset_rad) * away_dir2)

    def path_length_deg(w: npt.NDArray[np.float64]) -> float:
        """Calculate total path length through waypoint w."""
        d1 = float(np.rad2deg(np.arccos(np.clip(float(np.dot(a, w)), -1.0, 1.0))))
        d2 = float(np.rad2deg(np.arccos(np.clip(float(np.dot(w, b)), -1.0, 1.0))))
        return d1 + d2

    # Choose the waypoint with the shorter total path
    w = w1 if path_length_deg(w1) <= path_length_deg(w2) else w2

    # Convert back to RA/Dec in degrees
    w_ra_dec = vec2radec(w)
    return float(np.rad2deg(w_ra_dec[0])), float(np.rad2deg(w_ra_dec[1]))
