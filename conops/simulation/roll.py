import numpy as np
import rust_ephem

from ..common import dtutcfromtimestamp, rotvec, scbodyvector
from ..config import DTOR, SolarPanelSet

"""Roll computation helpers."""


def optimum_roll(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    solar_panel: SolarPanelSet | None = None,
) -> float:
    """Calculate the optimum roll angle (degrees in [0,360)).

    - If `solar_panel` is None: return closed-form optimum that minimizes the Sun's
      Z-component in the body frame (good for side-mounted arrays).
    - If provided: maximize weighted total power using actual panel normals, sizes,
      and efficiencies by scanning roll in 1° increments.
    """
    # Fetch ephemeris index and Sun vector
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun[index].cartesian.xyz.to_value("km")  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra, dec, 0.0, sunvec)

    if solar_panel is None:
        # Analytic optimum: choose roll that minimizes the Z-component of Sun
        y0 = s_body_0[1]
        z0 = s_body_0[2]
        roll_rad = np.arctan2(-y0, z0)
        return float((roll_rad / DTOR) % 360.0)

    # Weighted optimization using actual panel geometry (vectorized)
    panels = solar_panel._effective_panels()
    base_normals = []
    weights = []  # max_power * efficiency
    azimuths = []  # radians per panel
    for p in panels:
        n = np.array([0.0, 1.0, 0.0]) if p.sidemount else np.array([0.0, 0.0, -1.0])
        if p.cant_x:
            n = rotvec(1, p.cant_x * DTOR, n)
        if p.cant_y:
            n = rotvec(2, p.cant_y * DTOR, n)
        base_normals.append(n)
        eff = (
            p.conversion_efficiency
            if p.conversion_efficiency is not None
            else solar_panel.conversion_efficiency
        )
        weights.append(p.max_power * eff)
        azimuths.append((p.azimuth_deg or 0.0) * DTOR)

    # Convert lists to arrays
    n_mat = np.asarray(base_normals, dtype=float)  # shape (P,3)
    w_vec = np.asarray(weights, dtype=float)  # shape (P,)
    phi = np.asarray(azimuths, dtype=float)  # shape (P,)
    s = np.asarray(s_body_0, dtype=float)  # shape (3,)

    # Precompute per-panel coefficients for rotation about X:
    # illum(theta) = (nx*sx) + cos(theta)*(ny*sy + nz*sz) + sin(theta)*(nz*sy - ny*sz)
    a_coef = n_mat[:, 0] * s[0]
    b_coef = n_mat[:, 1] * s[1] + n_mat[:, 2] * s[2]
    c_coef = n_mat[:, 2] * s[1] - n_mat[:, 1] * s[2]

    # Apply per-panel azimuth as rotation about X to (b,c)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    b_adj = b_coef * cphi + c_coef * sphi
    c_adj = c_coef * cphi - b_coef * sphi

    # Angles 0..359 degrees
    deg = np.arange(360.0, dtype=float)
    ang = deg * DTOR
    cos_t = np.cos(ang)  # (360,)
    sin_t = np.sin(ang)  # (360,)

    # Illumination per angle and panel: (360,P)
    # Broadcasting: cos_t[:,None]*B[None,:] etc.
    illum = (
        a_coef[None, :]
        + cos_t[:, None] * b_adj[None, :]
        + sin_t[:, None] * c_adj[None, :]
    )
    illum = np.maximum(illum, 0.0)

    # Total weighted power per angle: (360,)
    totals = illum * w_vec[None, :]
    totals = totals.sum(axis=1)

    # Argmax over angles
    best_idx = int(np.argmax(totals))
    return float(deg[best_idx])


def optimum_roll_sidemount(
    ra: float, dec: float, utime: float, ephem: rust_ephem.Ephemeris
) -> float:
    """Calculate the optimum Roll angle (in degrees) for a given Ra, Dec
    and Unix Time"""
    # Analytic optimum: choose roll that minimizes the Z-component of the
    # Sun vector in the spacecraft body frame (roll=free about X).
    # This maximizes illumination for side-mounted panels (and general canted
    # panels derived from -Z toward +X), independent of panel cant magnitude.

    # Fetch ephemeris index and Sun vector
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun[index].cartesian.xyz.to_value("km")  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra, dec, 0.0, sunvec)
    y0 = s_body_0[1]
    z0 = s_body_0[2]

    # Rotate about X by roll to minimize z' = -sin(roll)*y0 + cos(roll)*z0
    # d(z')/d(roll) = 0 -> cos(roll)*y0 + sin(roll)*z0 = 0
    # => tan(roll) = -y0 / z0 -> roll = atan2(-y0, z0)
    roll_rad = np.arctan2(-y0, z0)

    # Return degrees in [0, 360)
    roll_deg = (roll_rad / DTOR) % 360.0
    return float(roll_deg)
