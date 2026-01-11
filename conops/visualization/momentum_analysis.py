"""Momentum conservation analysis and visualization for DITL simulations."""

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from ..common import ACSMode
from ..config.visualization import VisualizationConfig
from ..simulation.disturbance import DisturbanceModel

if TYPE_CHECKING:
    from ..ditl.ditl_mixin import DITLMixin


def plot_ditl_momentum(
    ditl: "DITLMixin",
    figsize: tuple[float, float] = (14, 16),
    config: VisualizationConfig | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot momentum conservation analysis for a DITL simulation.

    Creates an 8-panel figure showing:
    - Per-wheel momentum over time
    - Wheel fill fraction (saturation indicator)
    - Body slew rate (angular velocity from pointing changes)
    - External disturbance torques breakdown (log scale)
    - Instantaneous torques (disturbance vs MTQ potential/bleed + sign match)
    - Cumulative impulse (disturbance vs MTQ bleed/potential)
    - Conservation residual (wheel momentum vs external impulse)
    - Mode timeline (color-coded operational modes)

    Args:
        ditl: DITLMixin instance containing simulation telemetry data.
        figsize: Tuple of (width, height) for the figure size. Default: (14, 16)
        config: VisualizationConfig object. If None, uses ditl.config.visualization
            if available.

    Returns:
        tuple: (fig, axes) - The matplotlib figure and list of axes objects.

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_ditl_momentum
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig, axes = plot_ditl_momentum(ditl)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt

    # Resolve config
    if not isinstance(config, VisualizationConfig):
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()

    # Font settings
    font_family = config.font_family
    title_font_size = config.title_font_size
    label_font_size = config.label_font_size
    tick_font_size = config.tick_font_size
    legend_font_size = config.legend_font_size
    title_prop = FontProperties(family=font_family, size=title_font_size, weight="bold")

    # Time array
    utime = np.array(ditl.utime)
    hours = (utime - utime[0]) / 3600
    dt = ditl.step_size

    # Extract telemetry
    wheel_frac = np.array(getattr(ditl, "wheel_momentum_fraction", []))
    wheel_frac_raw = np.array(getattr(ditl, "wheel_momentum_fraction_raw", []))
    wheel_history = getattr(ditl, "wheel_momentum_history", {})
    mtq_torque = np.array(getattr(ditl, "mtq_torque_mag", []))
    mtq_bleed = np.array(getattr(ditl, "mtq_bleed_torque_mag", []))
    mtq_torque_vec = np.array(getattr(ditl, "mtq_torque_vec_history", []), dtype=float)
    has_mtq_torque_vec = mtq_torque_vec.shape == (len(hours), 3)
    if mtq_torque.size != len(hours) and has_mtq_torque_vec:
        mtq_torque = np.linalg.norm(mtq_torque_vec, axis=1)
    dist_torque = np.array(getattr(ditl, "disturbance_total", []))
    mode = np.array(ditl.mode)
    mode_colors = {
        ACSMode.SCIENCE.value: "green",
        ACSMode.SLEWING.value: "blue",
        ACSMode.PASS.value: "purple",
        ACSMode.CHARGING.value: "orange",
        ACSMode.SAFE.value: "red",
        ACSMode.SAA.value: "gray",
        ACSMode.DESAT.value: "cyan",
    }

    # Disturbance breakdown
    dist_gg = np.array(getattr(ditl, "disturbance_gg", []), dtype=float)
    dist_drag = np.array(getattr(ditl, "disturbance_drag", []), dtype=float)
    dist_srp = np.array(getattr(ditl, "disturbance_srp", []), dtype=float)
    dist_mag = np.array(getattr(ditl, "disturbance_mag", []), dtype=float)

    # Compute body slew rate from RA/Dec changes
    ra_arr = np.array(getattr(ditl, "ra", []), dtype=float)
    dec_arr = np.array(getattr(ditl, "dec", []), dtype=float)
    roll_arr = np.array(getattr(ditl, "roll", []), dtype=float)
    body_rate = np.zeros(len(hours))
    if len(ra_arr) == len(hours) and len(dec_arr) == len(hours) and dt > 0:
        ra_rad = np.deg2rad(ra_arr)
        dec_rad = np.deg2rad(dec_arr)
        for i in range(1, len(hours)):
            # Great-circle distance between consecutive pointings
            r0, d0 = ra_rad[i - 1], dec_rad[i - 1]
            r1, d1 = ra_rad[i], dec_rad[i]
            cosc = np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(r1 - r0)
            cosc = np.clip(cosc, -1.0, 1.0)
            dist_deg = np.rad2deg(np.arccos(cosc))
            body_rate[i] = dist_deg / dt  # deg/s

    body_momentum_kin = None
    if (
        len(ra_arr) == len(hours)
        and len(dec_arr) == len(hours)
        and len(roll_arr) == len(hours)
        and dt > 0
    ):

        def _build_rotation_matrix_with_roll(
            ra_deg: float, dec_deg: float, roll_deg: float
        ) -> np.ndarray:
            r_ib = DisturbanceModel._build_rotation_matrix(ra_deg, dec_deg)
            try:
                roll_rad = np.deg2rad(float(roll_deg))
            except Exception:
                return r_ib
            x_b, y_b, z_b = r_ib
            c = np.cos(roll_rad)
            s = np.sin(roll_rad)
            x_r = c * x_b + s * y_b
            y_r = -s * x_b + c * y_b
            return np.vstack([x_r, y_r, z_b])

        def _rotation_vector_from_matrix(r_mat: np.ndarray) -> np.ndarray:
            try:
                tr = float(np.trace(r_mat))
            except Exception:
                return np.zeros(3, dtype=float)
            cos_angle = (tr - 1.0) / 2.0
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = float(np.arccos(cos_angle))
            if angle < 1e-8:
                return 0.5 * np.array(
                    [
                        r_mat[2, 1] - r_mat[1, 2],
                        r_mat[0, 2] - r_mat[2, 0],
                        r_mat[1, 0] - r_mat[0, 1],
                    ],
                    dtype=float,
                )
            sin_angle = float(np.sin(angle))
            if abs(sin_angle) < 1e-8:
                diag = np.maximum(np.diag(r_mat) + 1.0, 0.0)
                axis = np.sqrt(diag / 2.0)
                axis = np.array(axis, dtype=float)
                axis[0] = np.copysign(axis[0], r_mat[2, 1] - r_mat[1, 2])
                axis[1] = np.copysign(axis[1], r_mat[0, 2] - r_mat[2, 0])
                axis[2] = np.copysign(axis[2], r_mat[1, 0] - r_mat[0, 1])
                return axis * angle
            axis = np.array(
                [
                    r_mat[2, 1] - r_mat[1, 2],
                    r_mat[0, 2] - r_mat[2, 0],
                    r_mat[1, 0] - r_mat[0, 1],
                ],
                dtype=float,
            )
            axis = axis / (2.0 * sin_angle)
            return axis * angle

        moi_cfg = None
        try:
            moi_cfg = getattr(ditl.acs.acs_config, "spacecraft_moi", None)
        except Exception:
            moi_cfg = None
        if moi_cfg is None:
            try:
                moi_cfg = ditl.config.spacecraft_bus.attitude_control.spacecraft_moi
            except Exception:
                moi_cfg = None
        inertia_matrix = (
            DisturbanceModel._build_inertia(moi_cfg) if moi_cfg is not None else None
        )
        if inertia_matrix is not None:
            body_momentum_kin = np.zeros((len(hours), 3), dtype=float)
            r_prev = None
            for i in range(len(hours)):
                r_now = _build_rotation_matrix_with_roll(
                    ra_arr[i], dec_arr[i], roll_arr[i]
                )
                if r_prev is None:
                    r_prev = r_now
                    continue
                r_rel = r_now @ r_prev.T
                rotvec = _rotation_vector_from_matrix(r_rel)
                omega_body = rotvec / dt
                body_momentum_kin[i] = inertia_matrix @ omega_body
                r_prev = r_now

    dist_vec = np.array(getattr(ditl, "disturbance_vec", []), dtype=float)
    body_momentum_hist = np.array(
        getattr(ditl, "body_momentum_history", []), dtype=float
    )
    external_impulse_hist = np.array(
        getattr(ditl, "external_impulse_history", []), dtype=float
    )

    mtq_active = None
    if mtq_torque.size == len(hours):
        mtq_active = mtq_torque > 0
    elif mtq_bleed.size == len(hours):
        mtq_active = mtq_bleed > 0
    elif has_mtq_torque_vec:
        mtq_active = np.linalg.norm(mtq_torque_vec, axis=1) > 0

    wheel_axes: dict[str, np.ndarray] = {}
    wheel_mom_vec = None
    if wheel_history and hasattr(ditl, "acs") and hasattr(ditl.acs, "reaction_wheels"):
        for w in ditl.acs.reaction_wheels:
            name = getattr(w, "name", "")
            hist = wheel_history.get(name)
            if not name or hist is None or len(hist) != len(hours):
                continue
            axis = np.array(getattr(w, "orientation", (1.0, 0.0, 0.0)), dtype=float)
            an = np.linalg.norm(axis)
            wheel_axes[name] = (
                axis / an if an > 0 else np.array([1.0, 0.0, 0.0], dtype=float)
            )
        if wheel_axes:
            wheel_mom_vec = np.zeros((len(hours), 3), dtype=float)
            for name, axis in wheel_axes.items():
                hist = np.array(wheel_history[name], dtype=float)
                wheel_mom_vec += hist[:, None] * axis

    mtq_sign_match: dict[str, np.ndarray] = {}
    mtq_bleed_impulse = None
    mtq_bleed_impulse_est = None
    if (
        dt > 0
        and dist_vec.shape == (len(hours), 3)
        and external_impulse_hist.shape == (len(hours), 3)
    ):
        delta_ext = np.diff(
            external_impulse_hist, axis=0, prepend=external_impulse_hist[:1]
        )
        mtq_bleed_impulse = dist_vec * dt - delta_ext
    if dt > 0 and wheel_axes:
        if has_mtq_torque_vec:
            mtq_sign_match = {
                name: np.zeros(len(hours), dtype=bool) for name in wheel_axes
            }
            mtq_bleed_impulse_est = np.zeros((len(hours), 3), dtype=float)
            for i in range(1, len(hours)):
                if mtq_active is not None and not mtq_active[i]:
                    continue
                t_mtq = mtq_torque_vec[i]
                if not np.isfinite(t_mtq).all():
                    continue
                if not np.any(t_mtq):
                    continue
                for name, axis in wheel_axes.items():
                    mom = float(wheel_history[name][i - 1])
                    if mom == 0.0:
                        continue
                    tau_w = float(np.dot(t_mtq, axis))
                    if tau_w == 0.0:
                        continue
                    dm = tau_w * dt
                    if (mom > 0 and dm <= 0) or (mom < 0 and dm >= 0):
                        continue
                    new_mom = mom - dm
                    if mom > 0:
                        new_mom = max(0.0, new_mom)
                    else:
                        new_mom = min(0.0, new_mom)
                    actual_dm = mom - new_mom
                    if actual_dm == 0.0:
                        continue
                    mtq_sign_match[name][i] = True
                    mtq_bleed_impulse_est[i] += actual_dm * axis
        elif (
            mtq_active is not None
            and len(ra_arr) == len(hours)
            and len(dec_arr) == len(hours)
            and hasattr(ditl, "acs")
            and hasattr(ditl.acs, "magnetorquers")
            and hasattr(ditl.acs, "disturbance_model")
        ):
            mtq_vecs = []
            for mtq in ditl.acs.magnetorquers:
                try:
                    v = np.array(mtq.get("orientation", (1.0, 0.0, 0.0)), dtype=float)
                except Exception:
                    v = np.array([1.0, 0.0, 0.0], dtype=float)
                vn = np.linalg.norm(v)
                v = v / vn if vn > 0 else np.array([1.0, 0.0, 0.0], dtype=float)
                dipole_val = mtq.get("dipole_strength") or mtq.get("dipole") or 0.0
                try:
                    dipole = float(dipole_val)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    dipole = 0.0
                mtq_vecs.append(v * dipole)
            if mtq_vecs:
                mtq_sign_match = {
                    name: np.zeros(len(hours), dtype=bool) for name in wheel_axes
                }
                mtq_bleed_impulse_est = np.zeros((len(hours), 3), dtype=float)
                for i in range(1, len(hours)):
                    if not mtq_active[i]:
                        continue
                    try:
                        b_body, _ = ditl.acs.disturbance_model.local_bfield_vector(
                            utime[i], ra_arr[i], dec_arr[i]
                        )
                    except Exception:
                        continue
                    t_mtq = np.zeros(3, dtype=float)
                    for m_vec in mtq_vecs:
                        t_mtq += np.cross(m_vec, b_body)
                    for name, axis in wheel_axes.items():
                        mom = float(wheel_history[name][i - 1])
                        if mom == 0.0:
                            continue
                        tau_w = float(np.dot(t_mtq, axis))
                        if tau_w == 0.0:
                            continue
                        dm = tau_w * dt
                        if (mom > 0 and dm <= 0) or (mom < 0 and dm >= 0):
                            continue
                        new_mom = mom - dm
                        if mom > 0:
                            new_mom = max(0.0, new_mom)
                        else:
                            new_mom = min(0.0, new_mom)
                        actual_dm = mom - new_mom
                        if actual_dm == 0.0:
                            continue
                        mtq_sign_match[name][i] = True
                        mtq_bleed_impulse_est[i] += actual_dm * axis

    # Compute cumulative impulse
    if dist_vec.shape == (len(hours), 3) and dt > 0:
        dist_impulse_vec = np.cumsum(dist_vec, axis=0) * dt
        dist_impulse_cum = np.linalg.norm(dist_impulse_vec, axis=1)
    elif dist_torque.size == len(hours):
        dist_impulse_vec = None
        dist_impulse_cum = np.cumsum(dist_torque) * dt
    else:
        dist_impulse_vec = None
        dist_impulse_cum = np.zeros_like(hours)

    mtq_bleed_impulse_use = mtq_bleed_impulse
    if mtq_bleed_impulse_use is None:
        mtq_bleed_impulse_use = mtq_bleed_impulse_est
    if mtq_bleed_impulse_use is not None:
        mtq_impulse_cum = np.linalg.norm(
            np.cumsum(mtq_bleed_impulse_use, axis=0), axis=1
        )
        mtq_impulse_label = "MTQ bleed"
    elif mtq_bleed.size == len(hours):
        mtq_impulse_cum = np.cumsum(mtq_bleed) * dt
        mtq_impulse_label = "MTQ bleed"
    elif mtq_torque.size == len(hours):
        mtq_impulse_cum = np.cumsum(mtq_torque) * dt
        mtq_impulse_label = "MTQ potential"
    else:
        mtq_impulse_cum = np.zeros_like(hours)
        mtq_impulse_label = "MTQ"

    if mtq_torque.size == len(hours) and (
        mtq_bleed.size == len(hours) or mtq_bleed_impulse_use is not None
    ):
        mtq_potential_impulse_cum = np.cumsum(mtq_torque) * dt
    else:
        mtq_potential_impulse_cum = None

    # Get momentum warnings count
    n_warnings = 0
    if hasattr(ditl, "acs") and hasattr(ditl.acs, "get_momentum_warnings"):
        n_warnings = len(ditl.acs.get_momentum_warnings())

    # Create figure with 8 panels
    fig, axes_arr = plt.subplots(8, 1, figsize=figsize, sharex=True)
    axes: list[Axes] = list(axes_arr)

    # Panel 1: Per-wheel momentum
    ax = axes[0]
    if wheel_history:
        for name, hist in wheel_history.items():
            if len(hist) == len(hours):
                ax.plot(hours, hist, linewidth=0.5, alpha=0.8, label=name)
        # Get max momentum from config if available
        max_mom = 1.0
        if (
            hasattr(ditl, "acs")
            and hasattr(ditl.acs, "reaction_wheels")
            and ditl.acs.reaction_wheels
        ):
            max_mom = float(getattr(ditl.acs.reaction_wheels[0], "max_momentum", 1.0))
        ax.axhline(
            max_mom, color="red", linestyle="--", alpha=0.5, label="Max capacity"
        )
        ax.axhline(-max_mom, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel(
            "Per-Wheel\nMomentum (Nms)",
            fontsize=label_font_size,
            fontfamily=font_family,
        )
        ax.legend(loc="upper right", fontsize=legend_font_size, ncol=3)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No per-wheel momentum history available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
        ax.set_ylabel(
            "Per-Wheel\nMomentum", fontsize=label_font_size, fontfamily=font_family
        )

    # Panel 2: Wheel fill fraction
    ax = axes[1]
    if wheel_frac_raw.size == len(hours):
        ax.fill_between(
            hours, 0, wheel_frac_raw * 100, alpha=0.3, color="blue", label="Raw fill"
        )
        ax.plot(hours, wheel_frac * 100, "b-", linewidth=0.5, label="Effective fill")
        ax.axhline(95, color="orange", linestyle="--", alpha=0.7, label="95% threshold")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=legend_font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No wheel fraction data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel("Wheel Fill\n(%)", fontsize=label_font_size, fontfamily=font_family)
    ax.grid(True, alpha=0.3)

    # Panel 3: Body slew rate
    ax = axes[2]
    if body_rate.size == len(hours):
        # Color by mode
        ax.plot(hours, body_rate, "k-", linewidth=0.3, alpha=0.5, label="Body rate")
        # Highlight pass tracking rate
        pass_mask = mode == ACSMode.PASS
        if np.any(pass_mask):
            ax.scatter(
                hours[pass_mask],
                body_rate[pass_mask],
                c="purple",
                s=2,
                alpha=0.8,
                label="Pass tracking",
                zorder=5,
            )
        # Highlight slewing
        slew_mask = mode == ACSMode.SLEWING
        if np.any(slew_mask):
            ax.scatter(
                hours[slew_mask],
                body_rate[slew_mask],
                c="blue",
                s=1,
                alpha=0.5,
                label="Slewing",
                zorder=4,
            )
        ax.legend(loc="upper right", fontsize=legend_font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No pointing data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel(
        "Body Rate\n(deg/s)", fontsize=label_font_size, fontfamily=font_family
    )
    ax.grid(True, alpha=0.3)

    # Panel 4: External disturbance torques breakdown (log scale)
    ax = axes[3]
    has_disturbance_data = (
        dist_torque.size == len(hours)
        and dist_gg.size == len(hours)
        and dist_drag.size == len(hours)
        and dist_srp.size == len(hours)
        and dist_mag.size == len(hours)
    )
    if has_disturbance_data:
        ax.plot(hours, dist_torque, label="Total", color="C0", linewidth=1)
        ax.plot(hours, dist_gg, label="Gravity gradient", color="C1", alpha=0.7)
        ax.plot(hours, dist_drag, label="Aero drag", color="C2", alpha=0.7)
        ax.plot(hours, dist_srp, label="Solar pressure", color="C3", alpha=0.7)
        ax.plot(hours, dist_mag, label="Residual magnetic", color="C4", alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1e-4)
        ax.legend(loc="upper right", fontsize=legend_font_size, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "No disturbance breakdown data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel(
        "Disturbance\nTorque (N·m)", fontsize=label_font_size, fontfamily=font_family
    )
    ax.grid(True, alpha=0.2)

    # Panel 5: Instantaneous MTQ torques
    ax = axes[4]
    has_mtq_torque = mtq_torque.size == len(hours)
    has_mtq_bleed = mtq_bleed.size == len(hours)
    if has_mtq_torque or has_mtq_bleed:
        mtq_plot = mtq_torque if has_mtq_torque else None
        bleed_plot = mtq_bleed if has_mtq_bleed else None
        max_y = 0.0
        for series in (mtq_plot, bleed_plot):
            if series is None:
                continue
            if series.size == len(hours):
                max_y = max(max_y, float(np.nanmax(np.abs(series))))
        if max_y <= 0:
            max_y = 1e-6
        sign_y_base = -0.5 * max_y
        sign_y_step = 0.15 * max_y

        if has_mtq_torque and mtq_plot is not None:
            ax.plot(
                hours,
                mtq_plot,
                "b-",
                linewidth=0.3,
                alpha=1.0,
                label="mtq potential",
            )
        if has_mtq_bleed and bleed_plot is not None:
            ax.plot(
                hours,
                bleed_plot,
                "g-",
                linewidth=0.3,
                alpha=1.0,
                label="mtq bleed",
            )

        palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
        wheel_colors = {
            "rw1": "#1b9e77",
            "rw2": "#d95f02",
            "rw3": "#7570b3",
            "rw4": "#e7298a",
        }
        if mtq_sign_match and mtq_active is not None:
            label_names = list(mtq_sign_match.keys())
            for idx, name in enumerate(label_names):
                mask = mtq_sign_match[name]
                active_mask = mask & mtq_active
                if not np.any(active_mask):
                    continue
                y = sign_y_base + idx * sign_y_step
                color = wheel_colors.get(name, palette[idx % len(palette)])
                ax.scatter(
                    hours[active_mask],
                    np.full(np.sum(active_mask), y),
                    s=6,
                    alpha=1.0,
                    color=color,
                )
        y_min = 0.0
        if mtq_sign_match:
            y_min = min(sign_y_base - 0.5 * sign_y_step, 0.0)
        y_max = max_y * 1.1
        if y_max <= y_min:
            y_max = max_y if max_y > 0 else y_min + 1e-6
        ax.set_ylim(y_min, y_max)
        legend = ax.legend(
            loc="upper right",
            fontsize=legend_font_size,
            ncol=1,
            framealpha=1.0,
            facecolor="white",
        )
        if legend.get_frame() is not None:
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_facecolor("white")
        if mtq_sign_match:
            x_anchor = 0.98
            try:
                fig = ax.figure  # type: ignore[assignment]
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
                bbox = legend.get_window_extent(renderer=renderer)
                x0, y0 = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))
                x1, _ = ax.transAxes.inverted().transform((bbox.x1, bbox.y0))
                x_anchor = min(0.98, x1)
            except Exception:
                pass
            label_names = list(mtq_sign_match.keys())
            for idx, name in enumerate(label_names):
                y = sign_y_base + idx * sign_y_step
                color = wheel_colors.get(name, palette[idx % len(palette)])
                ax.scatter(
                    [x_anchor - 0.02],
                    [y],
                    transform=ax.get_yaxis_transform(),
                    s=20,
                    color=color,
                    clip_on=False,
                )
                ax.text(
                    x_anchor,
                    y,
                    name,
                    transform=ax.get_yaxis_transform(),
                    color=color,
                    fontsize=tick_font_size,
                    ha="right",
                    va="center",
                )
    else:
        ax.text(
            0.5,
            0.5,
            "No torque data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel("Torque\n(N·m)", fontsize=label_font_size, fontfamily=font_family)
    ax.grid(True, alpha=0.3)

    # Panel 6: Cumulative impulse (momentum budget)
    ax = axes[5]
    eps = 1e-9
    dist_plot = np.maximum(dist_impulse_cum, eps)
    mtq_plot = np.maximum(mtq_impulse_cum, eps)
    ax.plot(hours, dist_plot, color="0.4", linewidth=1)
    ax.plot(hours, mtq_plot, color="#2ca02c", linewidth=1)
    if mtq_potential_impulse_cum is not None:
        pot_plot = np.maximum(mtq_potential_impulse_cum, eps)
        ax.plot(
            hours,
            pot_plot,
            color="#ff7f0e",
            linewidth=0.8,
            alpha=0.4,
            linestyle="--",
        )
    ax.set_ylabel(
        "Impulse\n(Nms, log)", fontsize=label_font_size, fontfamily=font_family
    )
    ax.set_yscale("log")
    y_max = max(float(np.nanmax(dist_plot)), float(np.nanmax(mtq_plot)))
    y_min = np.inf
    for series in (dist_plot, mtq_plot):
        positive = series[series > eps]
        if positive.size:
            y_min = min(y_min, float(np.nanmin(positive)))
    if mtq_potential_impulse_cum is not None:
        positive = pot_plot[pot_plot > eps]
        if positive.size:
            y_min = min(y_min, float(np.nanmin(positive)))
        y_max = max(y_max, float(np.nanmax(pot_plot)))
    if not np.isfinite(y_min):
        y_min = eps
    y_min = max(y_min * 0.2, eps)
    max_ratio = 1e6
    if y_max / y_min > max_ratio:
        y_min = max(y_max / max_ratio, eps)
    y_max = max(y_max * 1.5, y_min * 10)
    ax.set_ylim(y_min, y_max)
    if len(hours) > 1:
        x_label = hours[-1] - 0.02 * (hours[-1] - hours[0])
        ax.text(
            x_label,
            dist_plot[-1],
            "dist",
            color="0.4",
            fontsize=legend_font_size,
            ha="right",
            va="center",
        )
        ax.text(
            x_label,
            mtq_plot[-1],
            mtq_impulse_label,
            color="#2ca02c",
            fontsize=legend_font_size,
            ha="right",
            va="center",
        )
        if mtq_potential_impulse_cum is not None:
            ax.text(
                x_label,
                pot_plot[-1],
                "MTQ potential",
                color="#ff7f0e",
                fontsize=legend_font_size,
                ha="right",
                va="center",
            )
    ax.grid(True, alpha=0.3)

    # Panel 7: Conservation check
    ax = axes[6]
    eps_resid = 1e-9
    residual_plot = None
    residual_raw = None
    diff_plot = None
    tol_band = None
    kin_step_plot = None
    body_momentum_source = None
    if body_momentum_kin is not None and body_momentum_kin.shape == (len(hours), 3):
        body_momentum_source = body_momentum_kin
    elif body_momentum_hist.shape == (len(hours), 3):
        body_momentum_source = body_momentum_hist

    total_mom_vec = None
    if wheel_mom_vec is not None and body_momentum_source is not None:
        total_mom_vec = wheel_mom_vec + body_momentum_source

    ext_impulse_internal = None
    if external_impulse_hist.shape == (len(hours), 3):
        ext_impulse_internal = external_impulse_hist

    ext_impulse_independent = None
    if dist_impulse_vec is not None:
        ext_impulse_independent = dist_impulse_vec.copy()
        if mtq_bleed_impulse_use is not None:
            ext_impulse_independent -= np.cumsum(mtq_bleed_impulse_use, axis=0)

    if total_mom_vec is not None and ext_impulse_independent is not None:
        h0 = total_mom_vec[0]
        residual_vec = total_mom_vec - (h0 + ext_impulse_independent)
        residual_mag = np.linalg.norm(residual_vec, axis=1)
        residual_raw = residual_mag
        residual_plot = np.maximum(residual_mag, eps_resid)
        ax.plot(
            hours,
            residual_plot,
            color="black",
            linewidth=1,
            label="residual (dist+mtq)",
        )
        if ext_impulse_internal is not None:
            diff_vec = ext_impulse_internal - ext_impulse_independent
            diff_mag = np.linalg.norm(diff_vec, axis=1)
            diff_plot = np.maximum(diff_mag, eps_resid)
            show_diff = True
            if residual_mag.size:
                try:
                    p95_resid = float(np.quantile(residual_mag, 0.95))
                    if np.max(diff_mag) < 0.01 * p95_resid:
                        show_diff = False
                except Exception:
                    pass
            if show_diff:
                ax.plot(
                    hours,
                    diff_plot,
                    color="#7b3294",
                    linewidth=0.8,
                    linestyle="--",
                    label="ext impulse diff",
                )
            else:
                diff_plot = None
        tol_frac = None
        if (
            hasattr(ditl, "acs")
            and hasattr(ditl.acs, "wheel_dynamics")
            and hasattr(ditl.acs.wheel_dynamics, "_conservation_tolerance")
        ):
            tol_frac = float(ditl.acs.wheel_dynamics._conservation_tolerance)
        if tol_frac is not None:
            total_mom_mag = np.linalg.norm(total_mom_vec, axis=1)
            tol_band = np.maximum(total_mom_mag, 1e-6) * tol_frac
            ax.fill_between(
                hours,
                np.full_like(tol_band, eps_resid),
                tol_band,
                color="0.8",
                alpha=0.3,
                linewidth=0,
                label="tolerance",
            )
        if body_momentum_kin is not None and body_momentum_kin.shape == (
            len(hours),
            3,
        ):
            delta_h = np.diff(body_momentum_kin, axis=0)
            delta_mag = np.linalg.norm(delta_h, axis=1)
            if delta_mag.size:
                delta_mag = np.insert(delta_mag, 0, delta_mag[0])
            else:
                delta_mag = np.zeros(len(hours), dtype=float)
            window_seconds = 600.0
            if dt > 0:
                window = max(3, int(round(window_seconds / dt)))
            else:
                window = 3
            window = min(window, len(hours))
            step_scale = np.full(len(delta_mag), np.nan, dtype=float)
            for i in range(len(delta_mag)):
                start = max(0, i - window + 1)
                window_vals = delta_mag[start : i + 1]
                window_vals = window_vals[np.isfinite(window_vals)]
                if window_vals.size:
                    step_scale[i] = float(np.quantile(window_vals, 0.9))
            kin_step_plot = np.maximum(step_scale, eps_resid)
            ax.plot(
                hours,
                kin_step_plot,
                color="0.5",
                linewidth=0.8,
                alpha=0.6,
                linestyle="--",
                label="kin step p90",
                zorder=1,
            )
    elif total_mom_vec is not None and ext_impulse_internal is not None:
        h0 = total_mom_vec[0]
        residual_vec = total_mom_vec - (h0 + ext_impulse_internal)
        residual_mag = np.linalg.norm(residual_vec, axis=1)
        residual_raw = residual_mag
        residual_plot = np.maximum(residual_mag, eps_resid)
        ax.plot(
            hours,
            residual_plot,
            color="black",
            linewidth=1,
            label="residual (internal)",
        )
        tol_frac = None
        if (
            hasattr(ditl, "acs")
            and hasattr(ditl.acs, "wheel_dynamics")
            and hasattr(ditl.acs.wheel_dynamics, "_conservation_tolerance")
        ):
            tol_frac = float(ditl.acs.wheel_dynamics._conservation_tolerance)
        if tol_frac is not None:
            total_mom_mag = np.linalg.norm(total_mom_vec, axis=1)
            tol_band = np.maximum(total_mom_mag, 1e-6) * tol_frac
            ax.fill_between(
                hours,
                np.full_like(tol_band, eps_resid),
                tol_band,
                color="0.8",
                alpha=0.3,
                linewidth=0,
                label="tolerance",
            )
    elif wheel_frac_raw.size == len(hours):
        max_mom = 1.0
        if (
            hasattr(ditl, "acs")
            and hasattr(ditl.acs, "reaction_wheels")
            and ditl.acs.reaction_wheels
        ):
            max_mom = float(getattr(ditl.acs.reaction_wheels[0], "max_momentum", 1.0))
        est_wheel_h = wheel_frac_raw * max_mom
        net_external = dist_impulse_cum - mtq_impulse_cum
        residual_mag = np.abs(est_wheel_h - (est_wheel_h[0] + net_external))
        residual_raw = residual_mag
        residual_plot = np.maximum(residual_mag, eps_resid)
        ax.plot(hours, residual_plot, color="black", linewidth=1, label="residual")
        tol_frac = None
        if (
            hasattr(ditl, "acs")
            and hasattr(ditl.acs, "wheel_dynamics")
            and hasattr(ditl.acs.wheel_dynamics, "_conservation_tolerance")
        ):
            tol_frac = float(ditl.acs.wheel_dynamics._conservation_tolerance)
        if tol_frac is not None:
            tol_band = np.maximum(est_wheel_h, 1e-6) * tol_frac
            ax.fill_between(
                hours,
                np.full_like(tol_band, eps_resid),
                tol_band,
                color="0.8",
                alpha=0.3,
                linewidth=0,
                label="tolerance",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No conservation data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    if residual_raw is not None and mode.size == len(hours):
        whisker_x = 0.985
        tick_half = 0.006
        whisker_transform = ax.get_yaxis_transform()
        quantiles = (0.1, 0.5, 0.9)
        for acs_mode in ACSMode:
            mask = mode == acs_mode
            if not np.any(mask):
                continue
            values = residual_raw[mask]
            values = values[np.isfinite(values) & (values > eps_resid)]
            if values.size < 2:
                continue
            low = float(np.quantile(values, quantiles[0]))
            med = float(np.quantile(values, quantiles[1]))
            high = float(np.quantile(values, quantiles[2]))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                continue
            low = max(low, eps_resid)
            color = mode_colors.get(acs_mode.value, "0.8")
            ax.plot(
                [whisker_x, whisker_x],
                [low, high],
                color=color,
                linewidth=2.0,
                transform=whisker_transform,
                solid_capstyle="round",
                zorder=2,
            )
            if np.isfinite(med):
                ax.plot(
                    [whisker_x - tick_half, whisker_x + tick_half],
                    [med, med],
                    color=color,
                    linewidth=2.0,
                    transform=whisker_transform,
                    solid_capstyle="round",
                    zorder=2,
                )
    if residual_plot is not None:
        ax.set_yscale("log")
        y_min = np.inf
        y_max = 0.0

        def _robust_min(arr: np.ndarray, q: float = 0.1) -> float | None:
            positive = arr[arr > eps_resid]
            if positive.size:
                return float(np.quantile(positive, q))
            return None

        def _robust_max(arr: np.ndarray) -> float | None:
            positive = arr[arr > eps_resid]
            if positive.size:
                return float(np.nanmax(positive))
            return None

        if residual_raw is not None:
            rmin = _robust_min(residual_raw)
            rmax = _robust_max(residual_raw)
            if rmin is not None:
                y_min = min(y_min, rmin)
            if rmax is not None:
                y_max = max(y_max, rmax)
        if not np.isfinite(y_min):
            y_min = float(np.nanmin(residual_plot))
            y_max = float(np.nanmax(residual_plot))
        if diff_plot is not None:
            dmin = _robust_min(diff_plot)
            dmax = _robust_max(diff_plot)
            if dmin is not None:
                y_min = min(y_min, dmin)
            if dmax is not None:
                y_max = max(y_max, dmax)
        if kin_step_plot is not None:
            kmin = _robust_min(kin_step_plot)
            kmax = _robust_max(kin_step_plot)
            if kmin is not None:
                y_min = min(y_min, kmin)
            if kmax is not None:
                y_max = max(y_max, kmax)
        if tol_band is not None:
            tmin = _robust_min(tol_band)
            tmax = _robust_max(tol_band)
            if tmin is not None:
                y_min = min(y_min, tmin)
            if tmax is not None:
                y_max = max(y_max, tmax)
        y_min = max(y_min * 0.5, eps_resid)
        max_ratio = 1e3
        if y_max / y_min > max_ratio:
            y_min = max(y_max / max_ratio, eps_resid)
        y_max = max(y_max * 1.5, y_min * 10)
        ax.set_ylim(y_min, y_max)
        if total_mom_vec is not None:
            denom = np.maximum(np.linalg.norm(total_mom_vec, axis=1), 1e-6)
            rel = residual_plot / denom
            rel_mean = float(np.mean(rel)) * 100.0
            rel_p95 = float(np.quantile(rel, 0.95)) * 100.0
            text_lines = [f"residual/|H|: mean {rel_mean:.1f}%, p95 {rel_p95:.1f}%"]
            total_mom_mag = np.linalg.norm(total_mom_vec, axis=1)
            max_h = float(np.nanmax(total_mom_mag)) if total_mom_mag.size else 0.0
            if np.isfinite(max_h) and max_h > 0:
                gate_threshold = max(0.05 * max_h, 1e-6)
                gate_mask = total_mom_mag > gate_threshold
                if np.any(gate_mask):
                    rel_gate = residual_plot[gate_mask] / np.maximum(
                        total_mom_mag[gate_mask], 1e-6
                    )
                    rel_gate_mean = float(np.mean(rel_gate)) * 100.0
                    rel_gate_p95 = float(np.quantile(rel_gate, 0.95)) * 100.0
                    text_lines.append(
                        f"H>5% max: mean {rel_gate_mean:.1f}%, p95 {rel_gate_p95:.1f}%"
                    )
            ax.text(
                0.02,
                0.05,
                "\n".join(text_lines),
                transform=ax.transAxes,
                fontsize=tick_font_size,
                color="0.3",
                ha="left",
                va="bottom",
            )
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper right", fontsize=legend_font_size)
    ax.set_ylabel(
        "Residual\n(Nms, log)", fontsize=label_font_size, fontfamily=font_family
    )
    ax.grid(True, alpha=0.3)

    # Panel 8: Mode timeline (color-coded)
    ax = axes[7]
    for acs_mode in ACSMode:
        mask = mode == acs_mode
        if np.any(mask):
            ax.fill_between(
                hours,
                0,
                1,
                where=mask,
                alpha=0.5,
                color=mode_colors.get(acs_mode.value, "gray"),
                label=acs_mode.name,
            )
    ax.set_ylabel("Mode", fontsize=label_font_size, fontfamily=font_family)
    ax.set_xlabel("Time (hours)", fontsize=label_font_size, fontfamily=font_family)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=legend_font_size, ncol=4)

    # Title
    config_name = getattr(ditl.config, "name", "DITL Simulation")
    warning_text = (
        f"({n_warnings} momentum warnings)" if n_warnings > 0 else "(0 warnings)"
    )
    fig.suptitle(
        f"Momentum Conservation Analysis: {config_name}\n{warning_text}",
        fontproperties=title_prop,
    )

    # Set tick font sizes for all axes
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)

    plt.tight_layout()
    return fig, axes
