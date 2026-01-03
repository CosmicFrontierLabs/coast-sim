from __future__ import annotations

import numpy as np

from .reaction_wheel import ReactionWheel


def build_wheel_orientation_matrix(
    wheels: list[ReactionWheel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the wheel orientation matrix and static wheel properties.

    This function extracts the static (unchanging) properties from wheels:
    - Orientation unit vectors (columns of E matrix)
    - Maximum momentum per wheel
    - Maximum torque per wheel

    Call this once at initialization and reuse the results.

    Args:
        wheels: List of ReactionWheel instances.

    Returns:
        Tuple of (e_mat, max_moms, max_torques) where:
            e_mat: (3, N) matrix of wheel axis unit vectors
            max_moms: (N,) array of max momentum values
            max_torques: (N,) array of max torque values
    """
    if not wheels:
        return np.zeros((3, 0)), np.zeros(0), np.zeros(0)

    n = len(wheels)
    e_mat = np.zeros((3, n), dtype=float)
    max_moms = np.zeros(n, dtype=float)
    max_torques = np.zeros(n, dtype=float)

    for i, w in enumerate(wheels):
        try:
            v = np.array(w.orientation, dtype=float)
        except Exception:
            v = np.array([1.0, 0.0, 0.0])
        vn = np.linalg.norm(v)
        if vn <= 0:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = v / vn
        e_mat[:, i] = v
        max_moms[i] = float(getattr(w, "max_momentum", 0.0))
        max_torques[i] = float(getattr(w, "max_torque", 0.0))

    return e_mat, max_moms, max_torques


def allocate_wheel_torques(
    wheels: list[ReactionWheel],
    t_desired: np.ndarray,
    dt: float,
    use_weights: bool = False,
    bias_gain: float = 0.0,
    mom_margin: float = 0.99,
    e_mat: np.ndarray | None = None,
    max_moms: np.ndarray | None = None,
    max_torques: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve wheel torques for a desired body torque and apply headroom clamps.

    Args:
        wheels: List of ReactionWheel instances.
        t_desired: Desired torque vector (3D).
        dt: Time step in seconds.
        use_weights: If True, penalize high-momentum wheels in allocation.
        bias_gain: Null-space gain for driving momentum toward zero.
        mom_margin: Momentum margin (0-1) for headroom clamping.
        e_mat: Optional precomputed wheel orientation matrix (3, N).
        max_moms: Optional precomputed max momentum array (N,).
        max_torques: Optional precomputed max torque array (N,).

    Returns:
        Tuple of (raw_torques, allowed_torques, actual_torque, clamped).
    """
    if not wheels or dt <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(3), False

    n = len(wheels)

    # Use precomputed matrix if provided, otherwise build it
    if e_mat is None or max_moms is None or max_torques is None:
        e_mat, max_moms, max_torques = build_wheel_orientation_matrix(wheels)

    if e_mat.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(3), False

    # Get current momentum (dynamic, changes each timestep)
    curr_moms = np.array(
        [float(getattr(w, "current_momentum", 0.0)) for w in wheels], dtype=float
    )

    # Solve least-squares for wheel torques (scalar per wheel) that produce t_desired.
    try:
        if use_weights:
            # Penalize wheels with higher stored momentum to spread load.
            weights = []
            for max_m, curr_m in zip(max_moms, curr_moms):
                frac = abs(curr_m) / max_m if max_m > 0 else 0.0
                weights.append(1.0 + 4.0 * frac * frac)
            w_mat = np.diag(weights)
            lam = 1e-2
            cols = e_mat.shape[1]
            reg = np.sqrt(lam) * w_mat
            a_mat = np.vstack([e_mat, reg])
            b_vec = np.concatenate([t_desired, np.zeros(cols, dtype=float)])
            taus, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        else:
            taus, *_ = np.linalg.lstsq(e_mat, t_desired, rcond=None)
    except Exception:
        # fallback: assign all torque to first wheel only (legacy behavior)
        taus = np.zeros((len(wheels),), dtype=float)
        if len(taus) > 0:
            taus[0] = float(np.linalg.norm(t_desired))

    # Null-space bias: drive wheel momentum toward zero without changing net torque.
    if bias_gain > 0.0:
        try:
            e_pinv = np.linalg.pinv(e_mat)
            n = e_mat.shape[1]
            null_proj = np.eye(n) - (e_pinv @ e_mat)
            h_vec = np.array(curr_moms, dtype=float)
            taus = taus + null_proj.dot(-bias_gain * h_vec)
        except Exception:
            pass

    # Clamp per-wheel torques by peak torque and momentum headroom over dt
    taus_allowed = np.zeros_like(taus)
    clamped = False
    for i, (mt, mm, cm) in enumerate(zip(max_torques, max_moms, curr_moms)):
        avail = max(0.0, (mm * mom_margin) - abs(cm))
        max_by_mom = avail / dt if dt > 0 else 0.0
        # If torque would reduce stored momentum magnitude, don't limit by momentum headroom
        if cm != 0 and (taus[i] * cm) < 0:
            limit = mt
        else:
            limit = min(mt, max_by_mom)
        allowed = np.sign(taus[i]) * min(abs(taus[i]), limit)
        taus_allowed[i] = allowed
        if abs(allowed) + 1e-9 < abs(taus[i]):
            clamped = True

    # Compute actual produced torque vector
    t_actual = e_mat.dot(taus_allowed) if e_mat.size else np.zeros(3)

    return taus, taus_allowed, t_actual, clamped
