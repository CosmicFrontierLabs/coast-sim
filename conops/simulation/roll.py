import numpy as np
import rust_ephem

from ..common import dtutcfromtimestamp, scbodyvector
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
    # Fetch ephemeris index and Sun vector from pre-computed arrays
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra * DTOR, dec * DTOR, 0.0, sunvec)

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
    for p in panels:
        base_normals.append(np.array(p.normal, dtype=float))
        eff = (
            p.conversion_efficiency
            if p.conversion_efficiency is not None
            else solar_panel.conversion_efficiency
        )
        weights.append(p.max_power * eff)

    # Convert lists to arrays
    n_mat = np.asarray(base_normals, dtype=float)  # shape (P,3)
    w_vec = np.asarray(weights, dtype=float)  # shape (P,)
    s = np.asarray(s_body_0, dtype=float)  # shape (3,)

    # For roll angle theta about X-axis, the rotated normal is:
    # n'_x = n_x
    # n'_y = n_y*cos(theta) - n_z*sin(theta)
    # n'_z = n_y*sin(theta) + n_z*cos(theta)
    # Illumination = n' · s_normalized

    # Normalize sun vector
    s_norm = s / np.linalg.norm(s)

    # Precompute per-panel coefficients for rotation about X:
    # illum(theta) = (nx*sx) + cos(theta)*(ny*sy + nz*sz) + sin(theta)*(nz*sy - ny*sz)
    a_coef = n_mat[:, 0] * s_norm[0]
    b_coef = n_mat[:, 1] * s_norm[1] + n_mat[:, 2] * s_norm[2]
    c_coef = n_mat[:, 1] * s_norm[2] - n_mat[:, 2] * s_norm[1]

    # Angles 0..359 degrees
    deg = np.arange(360.0, dtype=float)
    ang = deg * DTOR
    cos_t = np.cos(ang)  # (360,)
    sin_t = np.sin(ang)  # (360,)

    # Illumination per angle and panel: (360,P)
    # Broadcasting: cos_t[:,None]*B[None,:] etc.
    illum = (
        a_coef[None, :]
        + cos_t[:, None] * b_coef[None, :]
        + sin_t[:, None] * c_coef[None, :]
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

    # Fetch ephemeris index and Sun vector from pre-computed arrays
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra * DTOR, dec * DTOR, 0.0, sunvec)
    y0 = s_body_0[1]
    z0 = s_body_0[2]

    # Rotate about X by roll to minimize z' = -sin(roll)*y0 + cos(roll)*z0
    # d(z')/d(roll) = 0 -> cos(roll)*y0 + sin(roll)*z0 = 0
    # => tan(roll) = -y0 / z0 -> roll = atan2(-y0, z0)
    roll_rad = np.arctan2(-y0, z0)

    # Return degrees in [0, 360)
    roll_deg = (roll_rad / DTOR) % 360.0
    return float(roll_deg)
