import numpy as np
import numpy.typing as npt
from pyproj import Geod


def radec2vec(
    ra: float | npt.NDArray[np.float64], dec: float | npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Convert RA/Dec angle (in radians) to a unit vector.

    Args:
        ra: Right ascension in radians (scalar or array)
        dec: Declination in radians (scalar or array)

    Returns:
        Unit vector(s). Shape (3,) for scalar input, (n, 3) for array input.
    """
    ra_arr = np.atleast_1d(ra)
    dec_arr = np.atleast_1d(dec)

    v1 = np.cos(dec_arr) * np.cos(ra_arr)
    v2 = np.cos(dec_arr) * np.sin(ra_arr)
    v3 = np.sin(dec_arr)

    result = np.stack([v1, v2, v3], axis=-1).astype(np.float64)
    # Return shape (3,) for scalar input, (n, 3) for array input
    if np.isscalar(ra) and np.isscalar(dec):
        return result.flatten()
    return result


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
) -> float | npt.NDArray[np.float64]:
    """Calculate the angular distance between two RA,Dec values.

    Both Ra/Dec values are given as an array of form [ra, dec] where
    RA and Dec are in radians. Supports both scalar and array inputs.

    Args:
        one: [ra, dec] in radians. ra/dec can be scalars or arrays.
        two: [ra, dec] in radians. ra/dec can be scalars or arrays.

    Returns:
        Angular separation in radians. Scalar if inputs are scalar,
        array if either input contains arrays.
    """
    ra1, dec1 = one[0], one[1]
    ra2, dec2 = two[0], two[1]

    # Check if inputs are scalar or array
    scalar_input = (
        np.isscalar(ra1)
        and np.isscalar(dec1)
        and np.isscalar(ra2)
        and np.isscalar(dec2)
    )

    onevec = radec2vec(ra1, dec1)
    twovec = radec2vec(ra2, dec2)

    if scalar_input:
        # Original scalar path
        onevec = np.atleast_1d(onevec).flatten().astype(float)
        twovec = np.atleast_1d(twovec).flatten().astype(float)

        n1 = np.linalg.norm(onevec)
        n2 = np.linalg.norm(twovec)
        if n1 < 1e-15 or n2 < 1e-15:
            return 0.0

        cosang = np.dot(onevec / n1, twovec / n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        return float(np.arccos(cosang))

    # Vectorized path for array inputs
    # Ensure both are 2D arrays of shape (n, 3)
    onevec = np.atleast_2d(onevec)
    twovec = np.atleast_2d(twovec)

    # Broadcast to same shape if needed (one scalar, one array)
    if onevec.shape[0] == 1 and twovec.shape[0] > 1:
        onevec = np.broadcast_to(onevec, twovec.shape)
    elif twovec.shape[0] == 1 and onevec.shape[0] > 1:
        twovec = np.broadcast_to(twovec, onevec.shape)

    # Compute norms along last axis
    n1 = np.linalg.norm(onevec, axis=-1, keepdims=True)
    n2 = np.linalg.norm(twovec, axis=-1, keepdims=True)

    # Handle zero-length vectors
    zero_mask = (n1.flatten() < 1e-15) | (n2.flatten() < 1e-15)

    # Normalize
    onevec_norm = np.where(n1 > 1e-15, onevec / n1, onevec)
    twovec_norm = np.where(n2 > 1e-15, twovec / n2, twovec)

    # Dot product along last axis
    cosang = np.sum(onevec_norm * twovec_norm, axis=-1)
    cosang = np.clip(cosang, -1.0, 1.0)

    result = np.arccos(cosang)
    result = np.where(zero_mask, 0.0, result)

    return result


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
