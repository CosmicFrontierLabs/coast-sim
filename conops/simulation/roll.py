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

    - If `solar_panel` is None: return the closed-form optimum that **maximises
      the Sun's Y-component** in the spacecraft body frame (i.e. maximises
      illumination on a +Y-normal panel), obtained by differentiating
      ``s_y(θ) = s_y0·cos(θ) − s_z0·sin(θ)`` and solving.
    - If provided: maximise the total weighted power across all panels by
      scanning roll in 1° increments.
    """
    # Fetch ephemeris index and Sun vector from pre-computed arrays
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra * DTOR, dec * DTOR, 0.0, sunvec)

    if solar_panel is None:
        # Analytic optimum for side-mounted panel (0,1,0): max y_body = cos(θ)*y0 - sin(θ)*z0
        # d/dθ = 0 → θ = atan2(-z0, y0)
        y0 = s_body_0[1]
        z0 = s_body_0[2]
        roll_rad = np.arctan2(-z0, y0)
        return float((roll_rad / DTOR) % 360.0)

    # Weighted optimization using actual panel geometry (vectorized)
    panels = solar_panel.panels
    if not panels:
        # No panels configured — fall back to analytic
        y0 = s_body_0[1]
        z0 = s_body_0[2]
        roll_rad = np.arctan2(-z0, y0)
        return float((roll_rad / DTOR) % 360.0)
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

    # For a spacecraft roll of θ about the body +X (boresight) axis the Sun
    # vector expressed in the body frame evolves as (right-hand rule):
    #   s_body_y(θ) = s_y · cos(θ) − s_z · sin(θ)
    #   s_body_z(θ) = s_y · sin(θ) + s_z · cos(θ)
    # (s_x is unchanged; s_y, s_z are the roll=0 body-frame components)
    #
    # Panel illumination = n · s_body(θ)  (panel normal n is fixed in the body frame):
    #   illum(θ) = n_x·s_x + cos(θ)·(n_y·s_y + n_z·s_z) + sin(θ)·(n_z·s_y − n_y·s_z)

    # Normalize sun vector
    s_norm = s / np.linalg.norm(s)

    # Precompute per-panel coefficients:
    #   illum(θ) = a + cos(θ)·b + sin(θ)·c
    # where a = nx·sx, b = ny·sy + nz·sz, c = nz·sy − ny·sz
    a_coef = n_mat[:, 0] * s_norm[0]
    b_coef = n_mat[:, 1] * s_norm[1] + n_mat[:, 2] * s_norm[2]
    c_coef = n_mat[:, 2] * s_norm[1] - n_mat[:, 1] * s_norm[2]

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

    # Maximize y_body = cos(roll)*y0 - sin(roll)*z0
    # d/d(roll) = 0 -> -sin(roll)*y0 - cos(roll)*z0 = 0
    # => tan(roll) = -z0 / y0 -> roll = atan2(-z0, y0)
    roll_rad = np.arctan2(-z0, y0)

    # Return degrees in [0, 360)
    roll_deg = (roll_rad / DTOR) % 360.0
    return float(roll_deg)
