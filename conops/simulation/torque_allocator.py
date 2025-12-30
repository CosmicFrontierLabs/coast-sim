from __future__ import annotations

import numpy as np

from .reaction_wheel import ReactionWheel


def allocate_wheel_torques(
    wheels: list[ReactionWheel],
    t_desired: np.ndarray,
    dt: float,
    use_weights: bool = False,
    bias_gain: float = 0.0,
    mom_margin: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve wheel torques for a desired body torque and apply headroom clamps."""
    if not wheels or dt <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(3), False

    # Build wheel orientation matrix E (3 x N), columns are wheel axis unit vectors
    e_cols: list[np.ndarray] = []
    curr_moms: list[float] = []
    max_moms: list[float] = []
    max_torques: list[float] = []
    for w in wheels:
        try:
            v = np.array(w.orientation, dtype=float)
        except Exception:
            v = np.array([1.0, 0.0, 0.0])
        vn = np.linalg.norm(v)
        if vn <= 0:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = v / vn
        e_cols.append(v)
        curr_moms.append(float(getattr(w, "current_momentum", 0.0)))
        max_moms.append(float(getattr(w, "max_momentum", 0.0)))
        max_torques.append(float(getattr(w, "max_torque", 0.0)))

    e_mat = np.column_stack(e_cols) if e_cols else np.zeros((3, 0))
    if e_mat.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(3), False

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
