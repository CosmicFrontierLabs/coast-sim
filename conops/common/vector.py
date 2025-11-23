import numpy as np
from pyproj import Geod  # type: ignore[import-untyped]


def radec2vec(ra, dec):
    """Convert RA/Dec angle (in radians) to a vector"""

    v1 = np.cos(dec) * np.cos(ra)
    v2 = np.cos(dec) * np.sin(ra)
    v3 = np.sin(dec)

    return np.array([v1, v2, v3])


def scbodyvector(ra, dec, roll, eciarr):
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


def separation(one, two):
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


def great_circle(ra1, dec1, ra2, dec2, npts=100):
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


def roll_over_angle(angles):
    """Make a list of angles that include a roll over (e.g. 359.9 - 0.1) into a smooth distribution"""
    outangles = list()
    last = -1
    flip = 0
    diff = 0

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
