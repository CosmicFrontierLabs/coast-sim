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
# Sun-avoiding slew path
# ---------------------------------------------------------------------------


def _closest_approach_on_arc(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Find the point on the great-circle arc A→B closest to unit vector S.

    Returns (closest_point, within_arc).  *within_arc* is True when the
    closest point lies strictly between A and B on the arc (not at an endpoint).
    """
    # Normal to the plane of the great circle
    n = vecnorm(np.cross(a, b))
    if np.linalg.norm(n) < 1e-12:
        # A and B are antipodal or identical
        return a, False
    # Project S onto the great-circle plane, then normalise → closest direction
    s_proj = s - np.dot(s, n) * n
    s_proj_norm = float(np.linalg.norm(s_proj))
    if s_proj_norm < 1e-12:
        # Sun is at the great-circle pole – every arc point is equidistant
        return a, False
    c = s_proj / s_proj_norm  # closest point on the (infinite) great circle

    # Check whether c is within the arc A→B using the signed cross-product test
    within = (
        float(np.dot(np.cross(a, c), n)) >= 0 and float(np.dot(np.cross(c, b), n)) >= 0
    )
    # The antipodal point -c is the farthest; pick the nearer one if needed
    if not within:
        c_anti = -c
        within_anti = (
            float(np.dot(np.cross(a, c_anti), n)) >= 0
            and float(np.dot(np.cross(c_anti, b), n)) >= 0
        )
        if within_anti:
            return c_anti, True
    return c, within


def sun_avoiding_path(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    sun_ra: float,
    sun_dec: float,
    min_sun_angle: float,
    steps: int = 100,
    margin_deg: float = 5.0,
) -> tuple[list[float], list[float]]:
    """Compute a slew path that avoids the Sun exclusion zone.

    If the direct great-circle arc from (ra1, dec1) to (ra2, dec2) passes
    within *min_sun_angle* degrees of the Sun at (sun_ra, sun_dec), a single
    waypoint is inserted on the boundary of the exclusion zone + *margin_deg*
    to route the spacecraft around the Sun.  If no violation is detected the
    function falls back to a plain great-circle path.

    Waypoint geometry: the waypoint is placed at angular distance
    (min_sun_angle + margin_deg) from the Sun, offset perpendicular to the
    great-circle arc in the direction that avoids the Sun.  Two candidate
    offsets are considered (one on each side of the arc) and the shorter
    total detour is selected.

    Args:
        ra1, dec1:       Start pointing (degrees).
        ra2, dec2:       End pointing (degrees).
        sun_ra, sun_dec: Sun position (degrees).
        min_sun_angle:   Sun exclusion radius (degrees).
        steps:           Number of waypoints in the output path.
        margin_deg:      Extra buffer beyond min_sun_angle for the waypoint.

    Returns:
        Tuple (ra_list, dec_list) of the computed path (degrees).
    """
    a = vecnorm(radec2vec(np.deg2rad(ra1), np.deg2rad(dec1)))
    b = vecnorm(radec2vec(np.deg2rad(ra2), np.deg2rad(dec2)))
    s = vecnorm(radec2vec(np.deg2rad(sun_ra), np.deg2rad(sun_dec)))

    n = vecnorm(np.cross(a, b))
    if np.linalg.norm(n) < 1e-12:
        # A and B are antipodal or identical – can't determine arc normal
        return great_circle(ra1, dec1, ra2, dec2, steps)

    c, within_arc = _closest_approach_on_arc(a, b, s)
    min_dist_deg = float(np.rad2deg(np.arccos(np.clip(float(np.dot(s, c)), -1.0, 1.0))))

    if not within_arc or min_dist_deg >= min_sun_angle:
        return great_circle(ra1, dec1, ra2, dec2, steps)

    # Angular distance from S to the great circle (same as min_dist_deg)
    d_min_rad = np.deg2rad(min_dist_deg)
    threshold_rad = np.deg2rad(min_sun_angle + margin_deg)
    beta = float(threshold_rad - d_min_rad)  # offset angle off the arc

    # Perpendicular direction to the arc at C, pointing away from the Sun
    dot_sn = float(np.dot(s, n))
    away_dir = -n if dot_sn > 0 else n  # unit vector, perpendicular to arc, away from S

    # Two candidate waypoints:
    #   W1: step β away from arc in the "away from sun" direction (shorter offset)
    #   W2: step β+2*d_min away from arc in the "toward sun" direction (other side)
    beta2 = float(threshold_rad + d_min_rad)
    toward_dir = -away_dir

    w1 = np.cos(beta) * c + np.sin(beta) * away_dir  # already unit (C ⊥ away_dir)
    w2 = np.cos(beta2) * c + np.sin(beta2) * toward_dir  # already unit (C ⊥ toward_dir)

    def path_length_deg(w: npt.NDArray[np.float64]) -> float:
        d1 = float(np.rad2deg(np.arccos(np.clip(float(np.dot(a, w)), -1.0, 1.0))))
        d2 = float(np.rad2deg(np.arccos(np.clip(float(np.dot(w, b)), -1.0, 1.0))))
        return d1 + d2

    w = w1 if path_length_deg(w1) <= path_length_deg(w2) else w2
    w_ra_dec = vec2radec(w)
    w_ra = float(np.rad2deg(w_ra_dec[0]))
    w_dec = float(np.rad2deg(w_ra_dec[1]))

    half = steps // 2
    seg1_ra, seg1_dec = great_circle(ra1, dec1, w_ra, w_dec, half)
    seg2_ra, seg2_dec = great_circle(w_ra, w_dec, ra2, dec2, max(steps - half, 1))
    return seg1_ra + seg2_ra[1:], seg1_dec + seg2_dec[1:]
