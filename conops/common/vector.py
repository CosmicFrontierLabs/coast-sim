import numpy as np
import numpy.typing as npt
from pyproj import Geod  # type: ignore[import-untyped]


def radec2vec(ra: float, dec: float) -> npt.NDArray:
    """Convert RA/Dec angle (in radians) to a vector"""

    v1 = np.cos(dec) * np.cos(ra)
    v2 = np.cos(dec) * np.sin(ra)
    v3 = np.sin(dec)

    return np.array([v1, v2, v3])


def scbodyvector(
    ra: float, dec: float, roll: float, eciarr: npt.NDArray
) -> npt.NDArray:
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
    rot1 = np.array(((1, 0, 0), (0, croll, sroll), (0, -sroll, croll)))
    rot2 = np.array(((cdec, 0, -sdec), (0, 1, 0), (sdec, 0, cdec)))
    rot3 = np.array(((cra, sra, 0), (-sra, cra, 0), (0, 0, 1)))

    # Multiply them all up
    a = np.dot(rot1, rot2)
    b = np.dot(a, rot3)
    body = np.dot(b, eciarr)
    return body


def rotvec(n: int, a: float, v: np.ndarray) -> np.ndarray:
    """Rotate a vector v by angle a (radians) around axis n (1=x,2=y,3=z).
    Preserves the original sign convention (rotation uses -a)."""
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")

    v = np.asarray(v, dtype=float).copy()
    k = np.zeros(3)
    k[n - 1] = 1.0

    c = np.cos(a)
    s = -np.sin(a)  # match original sign convention

    return v * c + np.cross(k, v) * s + k * (np.dot(k, v)) * (1 - c)


def separation(one: npt.NDArray | list[float], two: npt.NDArray | list[float]) -> float:
    """Calculate the angular distance between two RA,Dec values.
    Both Ra/Dec values are given as an array of form [ra,dec] where
    RA and Dec are in radians. Form of function mimics pyephem library
    version except result is simply in radians."""

    onevec = radec2vec(one[0], one[1])
    twovec = radec2vec(two[0], two[1])

    # Flatten vectors to ensure they're 1D for dot product
    onevec = np.atleast_1d(onevec).flatten()
    twovec = np.atleast_1d(twovec).flatten()

    return np.arccos(np.dot(onevec, twovec))


def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Calculate the angular distance between two RA,Dec values in degrees."""
    ra1_rad = np.deg2rad(ra1)
    dec1_rad = np.deg2rad(dec1)
    ra2_rad = np.deg2rad(ra2)
    dec2_rad = np.deg2rad(dec2)

    sep_rad = separation([ra1_rad, dec1_rad], [ra2_rad, dec2_rad])
    return np.rad2deg(sep_rad)


def great_circle(
    ra1: float, dec1: float, ra2: float, dec2: float, npts: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Return Great Circle Path between two coordinates"""
    g = Geod(ellps="sphere")

    lonlats = g.npts(ra1 - 180, dec1, ra2 - 180, dec2, npts)

    ras, decs = np.array(lonlats).transpose()

    ras += 180
    ras = np.append(ra1, ras)
    ras = np.append(ras, ra2)
    decs = np.append(dec1, decs)
    decs = np.append(decs, dec2)
    return ras, decs


def roll_over_angle(angles: npt.NDArray | list[float]) -> npt.NDArray:
    """Make a list of angles that include a roll over (e.g. 359.9 - 0.1) into a smooth distribution"""
    outangles = list()
    last = -1.0
    flip = 0.0
    diff = 0.0

    for i in range(len(angles)):
        if last != -1:
            diff = angles[i] + flip - last
            if diff > 300:
                flip = -360
            elif diff < -300:
                flip = 360
        raf = angles[i] + flip
        last = raf
        outangles.append(raf)

    return np.array(outangles)


def vec2radec(v: npt.NDArray[np.float64]) -> npt.NDArray:
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
