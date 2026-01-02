"""Debug test replicating ditl-adcs.ipynb cells 1-2 to investigate momentum warnings.

This test uses the exact configuration from the notebook to reproduce and diagnose
momentum consistency/conservation warnings during DITL simulation.
"""

import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from conops import MissionConfig, Queue, QueueDITL
from conops.common import ACSMode

# Path to example TLE file
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
TLE_PATH = EXAMPLES_DIR / "example.tle"


def make_ditl_adcs_config(ephem: object, mtq_dipole: float = 32.0) -> MissionConfig:
    """Create config matching ditl-adcs.ipynb cell 1."""
    # Load from example_config.json like the notebook does
    example_config = EXAMPLES_DIR / "example_config.json"
    if example_config.exists():
        cfg = MissionConfig.from_json_file(str(example_config))
    else:
        cfg = MissionConfig()
    acs = cfg.spacecraft_bus.attitude_control

    # Spacecraft properties from notebook
    acs.spacecraft_moi = (45.0, 45.0, 45.0)
    acs.wheel_enabled = False

    # Magnetorquers
    acs.magnetorquers = [
        {
            "name": "mtq_x",
            "orientation": (1, 0, 0),
            "dipole_strength": mtq_dipole,
            "power_draw": 5.0,
        },
        {
            "name": "mtq_y",
            "orientation": (0, 1, 0),
            "dipole_strength": mtq_dipole,
            "power_draw": 5.0,
        },
        {
            "name": "mtq_z",
            "orientation": (0, 0, 1),
            "dipole_strength": mtq_dipole,
            "power_draw": 5.0,
        },
    ]
    acs.magnetorquer_bfield_T = 3e-5

    # Disturbance sources
    acs.cp_offset_body = (0.25, 0.0, 0.0)
    acs.residual_magnetic_moment = (0.05, 0, 0)
    acs.drag_area_m2 = 1.3
    acs.drag_coeff = 2.2
    acs.solar_area_m2 = 1.3
    acs.solar_reflectivity = 1.3
    acs.use_msis_density = True

    # Pyramid wheel array (4 wheels)
    s = math.sqrt(2 / 3)
    c = math.sqrt(1 / 3)
    acs.wheels = [
        {
            "name": "rw1",
            "orientation": (+s, 0.0, +c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
        {
            "name": "rw2",
            "orientation": (-s, 0.0, +c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
        {
            "name": "rw3",
            "orientation": (0.0, +s, -c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
        {
            "name": "rw4",
            "orientation": (0.0, -s, -c),
            "max_torque": 0.06,
            "max_momentum": 1.0,
        },
    ]

    cfg.constraint.ephem = ephem
    return cfg


def make_mock_ephem(begin: datetime, end: datetime, step_size: float = 10.0):
    """Create a mock ephemeris for testing."""
    duration_s = (end - begin).total_seconds()
    n_steps = int(duration_s / step_size) + 1

    ephem = MagicMock()
    ephem.step_size = step_size
    ephem.begin = begin
    ephem.end = end

    # Earth direction (nadir)
    earth_mock = MagicMock()
    earth_mock.ra = MagicMock(deg=0.0)
    earth_mock.dec = MagicMock(deg=0.0)
    ephem.earth = [earth_mock] * n_steps

    # Sun direction
    sun_mock = MagicMock()
    sun_mock.ra = MagicMock(deg=45.0)
    sun_mock.dec = MagicMock(deg=23.5)
    ephem.sun = [sun_mock] * n_steps

    # Position/velocity (LEO ~400km)
    ephem.gcrs_pv = MagicMock()
    ephem.gcrs_pv.position = [[6778e3, 0, 0]] * n_steps  # 400km altitude
    ephem.gcrs_pv.velocity = [[0, 7700, 0]] * n_steps

    # Lat/lon for MSIS
    ephem.lat = [0.0] * n_steps
    ephem.long = [0.0] * n_steps

    ephem.index = lambda dt: min(
        int((dt.timestamp() - begin.timestamp()) / step_size), n_steps - 1
    )

    return ephem


class TestDITLADCSMomentumWarnings:
    """Test class to debug momentum warnings from ditl-adcs notebook."""

    @pytest.fixture
    def short_ditl_setup(self):
        """Create a short DITL run (1 hour) for quick debugging.

        Note: This 1-hour window (00:00-01:00) has no ground station passes visible.
        The first passes occur ~7-8 hours into the day. Use notebook_ditl_setup for
        testing with ground station passes.
        """
        # Skip if TLE file not available
        if not TLE_PATH.exists():
            pytest.skip(f"TLE file not found: {TLE_PATH}")

        from rust_ephem import TLEEphemeris

        begin = datetime(2025, 1, 1)
        end = begin + timedelta(hours=1)
        ephem = TLEEphemeris(tle=str(TLE_PATH), begin=begin, end=end, step_size=10)

        cfg = make_ditl_adcs_config(ephem)
        cfg.constraint.ephem = ephem

        # Create a simple queue with a few targets
        np.random.seed(43)
        n_targets = 20
        target_ra = np.random.uniform(0, 360, n_targets)
        target_dec = np.random.uniform(-90, 90, n_targets)

        queue = Queue(ephem=ephem, config=cfg)
        queue.slew_distance_weight = 0.0
        for i in range(n_targets):
            queue.add(
                merit=40,
                ra=target_ra[i],
                dec=target_dec[i],
                obsid=10000 + i,
                name=f"pointing_{10000 + i}",
                exptime=100,
                ss_min=30,
            )

        ditl = QueueDITL(config=cfg, ephem=ephem, begin=begin, end=end, queue=queue)
        ditl.acs._wheel_mom_margin = 1.0  # Disable margin
        ditl.step_size = 10

        return ditl, cfg

    @pytest.fixture
    def notebook_ditl_setup(self):
        """Create DITL matching exactly the ditl-adcs.ipynb notebook (12h, 1000 targets)."""
        if not TLE_PATH.exists():
            pytest.skip(f"TLE file not found: {TLE_PATH}")

        from rust_ephem import TLEEphemeris

        # Exact notebook parameters
        begin = datetime(2025, 1, 1)
        end = begin + timedelta(hours=12)
        ephem = TLEEphemeris(tle=str(TLE_PATH), begin=begin, end=end, step_size=10)

        cfg = make_ditl_adcs_config(ephem)
        cfg.constraint.ephem = ephem

        # Guarantee ground station passes are scheduled (bypass random filtering)
        if cfg.ground_stations is not None:
            for station in cfg.ground_stations.stations:
                station.schedule_probability = 1.0

        # Exact notebook target setup
        np.random.seed(43)
        n_targets = 1000
        target_ra = np.random.uniform(0, 360, n_targets)
        target_dec = np.random.uniform(-90, 90, n_targets)

        queue = Queue(ephem=ephem, config=cfg)
        queue.slew_distance_weight = 0.0
        for i in range(n_targets):
            queue.add(
                merit=40,
                ra=target_ra[i],
                dec=target_dec[i],
                obsid=10000 + i,
                name=f"pointing_{10000 + i}",
                exptime=1000,  # Notebook uses 1000s
                ss_min=300,  # Notebook uses 300s
            )

        ditl = QueueDITL(config=cfg, ephem=ephem, begin=begin, end=end, queue=queue)
        ditl.acs._wheel_mom_margin = 1.0
        ditl.step_size = 10

        return ditl, cfg

    def test_collect_momentum_warnings(self, short_ditl_setup):
        """Run DITL and collect all momentum-related warnings."""
        ditl, cfg = short_ditl_setup

        # Run simulation
        ditl.calc()

        # Collect warnings from ACS
        warnings = ditl.acs.get_momentum_warnings()

        print(f"\n{'=' * 60}")
        print("MOMENTUM WARNINGS ANALYSIS")
        print(f"{'=' * 60}")
        print(f"Total warnings: {len(warnings)}")

        # Categorize warnings
        consistency_warnings = [w for w in warnings if "consistency" in w.lower()]
        conservation_warnings = [w for w in warnings if "conservation" in w.lower()]
        saturation_warnings = [w for w in warnings if "saturation" in w.lower()]
        other_warnings = [
            w
            for w in warnings
            if w
            not in consistency_warnings + conservation_warnings + saturation_warnings
        ]

        print(f"\nConsistency warnings: {len(consistency_warnings)}")
        print(f"Conservation warnings: {len(conservation_warnings)}")
        print(f"Saturation warnings: {len(saturation_warnings)}")
        print(f"Other warnings: {len(other_warnings)}")

        # Print first few of each type
        for category, items in [
            ("CONSISTENCY", consistency_warnings),
            ("CONSERVATION", conservation_warnings),
            ("SATURATION", saturation_warnings),
            ("OTHER", other_warnings),
        ]:
            if items:
                print(f"\n--- {category} (first 5) ---")
                for w in items[:5]:
                    print(f"  {w}")

        # Print mode distribution
        mode_arr = np.array(ditl.mode)
        print(f"\n{'=' * 60}")
        print("MODE DISTRIBUTION")
        print(f"{'=' * 60}")
        for m in ACSMode:
            frac = np.mean(mode_arr == m) * 100
            if frac > 0:
                print(f"  {m.name}: {frac:.1f}%")

        # Wheel momentum analysis
        wm_raw = np.array(getattr(ditl, "wheel_momentum_fraction_raw", []))
        if wm_raw.size:
            print(f"\n{'=' * 60}")
            print("WHEEL MOMENTUM STATS")
            print(f"{'=' * 60}")
            print(f"  Max: {wm_raw.max():.3f}")
            print(f"  Mean: {wm_raw.mean():.3f}")
            print(f"  P95: {np.quantile(wm_raw, 0.95):.3f}")
            print(f"  Steps >= 0.95: {(wm_raw >= 0.95).sum()}")
            print(f"  Steps >= 1.0: {(wm_raw >= 1.0).sum()}")

        # This test is for debugging - always pass but show output
        print(f"\n{'=' * 60}")
        print("TEST COMPLETE - Review output above for diagnostics")
        print(f"{'=' * 60}")

    def test_slew_by_slew_momentum_tracking(self, short_ditl_setup):
        """Track momentum changes slew-by-slew to find where drift occurs."""
        ditl, cfg = short_ditl_setup

        # Hook into ACS to capture momentum at slew boundaries
        acs = ditl.acs
        slew_momentum_log = []

        original_update = acs._update_wheel_momentum

        def tracking_update(utime):
            h_before = acs._get_total_wheel_momentum().copy()
            original_update(utime)
            h_after = acs._get_total_wheel_momentum()

            is_slewing = acs.current_slew is not None and acs.current_slew.is_slewing(
                utime
            )
            slew_momentum_log.append(
                {
                    "utime": utime,
                    "h_before": h_before.copy(),
                    "h_after": h_after.copy(),
                    "delta": h_after - h_before,
                    "is_slewing": is_slewing,
                    "mode": acs.acsmode,
                }
            )

        acs._update_wheel_momentum = tracking_update

        # Run simulation
        ditl.calc()

        # Analyze slew transitions
        print(f"\n{'=' * 60}")
        print("SLEW-BY-SLEW MOMENTUM TRACKING")
        print(f"{'=' * 60}")

        # Find slew start/end transitions
        prev_slewing = False
        slew_count = 0
        for i, entry in enumerate(slew_momentum_log):
            curr_slewing = entry["is_slewing"]

            if not prev_slewing and curr_slewing:
                # Slew started
                slew_count += 1
                h_start = entry["h_before"]
                print(f"\nSlew {slew_count} START at t={entry['utime']:.1f}")
                print(
                    f"  H_start: [{h_start[0]:.6f}, {h_start[1]:.6f}, {h_start[2]:.6f}] Nms"
                )

            elif prev_slewing and not curr_slewing:
                # Slew ended
                h_end = slew_momentum_log[i - 1]["h_after"]
                delta_slew = (
                    np.linalg.norm(h_end - h_start) if "h_start" in dir() else 0
                )
                print(f"Slew {slew_count} END")
                print(f"  H_end: [{h_end[0]:.6f}, {h_end[1]:.6f}, {h_end[2]:.6f}] Nms")
                print(f"  |ΔH|: {delta_slew:.6f} Nms")

            prev_slewing = curr_slewing

            if slew_count >= 5:  # Limit output
                print("\n... (showing first 5 slews)")
                break

        # Summary stats
        deltas = [
            np.linalg.norm(e["delta"]) for e in slew_momentum_log if e["is_slewing"]
        ]
        if deltas:
            print(f"\n{'=' * 60}")
            print("MOMENTUM DELTA STATS (during slews)")
            print(f"{'=' * 60}")
            print(f"  Mean |ΔH| per step: {np.mean(deltas):.6f} Nms")
            print(f"  Max |ΔH| per step: {np.max(deltas):.6f} Nms")
            print(f"  Total steps during slews: {len(deltas)}")

    def test_conservation_check_details(self, short_ditl_setup):
        """Examine what triggers conservation violations."""
        ditl, cfg = short_ditl_setup

        # Run simulation
        ditl.calc()

        # Get detailed conservation info from wheel dynamics
        wd = ditl.acs.wheel_dynamics

        print(f"\n{'=' * 60}")
        print("CONSERVATION CHECK DETAILS")
        print(f"{'=' * 60}")

        # Current momentum state
        h_total = ditl.acs._get_total_wheel_momentum()
        print(
            f"\nFinal wheel momentum: [{h_total[0]:.6f}, {h_total[1]:.6f}, {h_total[2]:.6f}] Nms"
        )
        print(f"  |H|: {np.linalg.norm(h_total):.6f} Nms")

        # Check individual wheel states
        print("\nPer-wheel momentum:")
        for w in ditl.acs.reaction_wheels:
            frac = abs(w.current_momentum) / w.max_momentum if w.max_momentum > 0 else 0
            print(f"  {w.name}: {w.current_momentum:.6f} Nms ({frac * 100:.1f}%)")

        # External torque accumulation
        ext_impulse = getattr(wd, "_external_impulse", np.zeros(3))
        print(f"\nAccumulated external impulse: {ext_impulse}")

        # Check for any clamping events
        clamp_count = getattr(ditl.acs, "_clamp_count", 0)
        print(f"\nTorque clamping events: {clamp_count}")

    def test_saturation_analysis(self, short_ditl_setup):
        """Analyze why wheels are saturating."""
        ditl, cfg = short_ditl_setup

        # Run simulation
        ditl.calc()

        # Get configuration parameters
        acs_cfg = cfg.spacecraft_bus.attitude_control
        moi = np.mean(acs_cfg.spacecraft_moi)
        max_slew_rate = acs_cfg.max_slew_rate
        accel = acs_cfg.slew_acceleration

        print(f"\n{'=' * 60}")
        print("SATURATION ANALYSIS")
        print(f"{'=' * 60}")

        # Theoretical peak momentum for a slew at max rate
        omega_peak = max_slew_rate * np.pi / 180  # rad/s
        h_peak_theory = moi * omega_peak
        print("\nConfiguration:")
        print(f"  MOI: {moi} kg·m²")
        print(f"  Max slew rate: {max_slew_rate} deg/s")
        print(f"  Slew acceleration: {accel} deg/s²")
        print(f"  Theoretical peak H per axis: {h_peak_theory:.4f} Nms")

        # Wheel capacity
        if ditl.acs.reaction_wheels:
            wheel_cap = ditl.acs.reaction_wheels[0].max_momentum
            print(f"\n  Wheel capacity (each): {wheel_cap} Nms")
            print(f"  Number of wheels: {len(ditl.acs.reaction_wheels)}")

            # For pyramid config, effective capacity along any axis
            # is roughly wheel_cap (projection factor ~0.816 for pyramid)
            proj = 0.816
            effective_cap = wheel_cap * proj * 4 / 3  # 4 wheels, ~3 axis coverage
            print(f"  Effective capacity per axis (approx): {effective_cap:.4f} Nms")
            print(
                f"  Margin: {(effective_cap - h_peak_theory) / h_peak_theory * 100:.1f}%"
            )

        # Saturation statistics
        wm_raw = np.array(getattr(ditl, "wheel_momentum_fraction_raw", []))
        if wm_raw.size:
            # When did first saturation occur?
            sat_mask = wm_raw >= 0.95
            if sat_mask.any():
                first_sat_idx = np.argmax(sat_mask)
                first_sat_time = first_sat_idx * ditl.step_size
                print(
                    f"\n  First saturation (>=95%): step {first_sat_idx} ({first_sat_time:.0f}s)"
                )
                print(f"  Total saturated steps: {sat_mask.sum()}")
                print(f"  Saturation fraction: {sat_mask.mean() * 100:.1f}%")

        # Slew statistics
        from conops.common import ACSCommandType

        slew_cmds = [
            c
            for c in ditl.acs.executed_commands
            if c.command_type == ACSCommandType.SLEW_TO_TARGET and c.slew
        ]
        if slew_cmds:
            dists = [c.slew.slewdist for c in slew_cmds]
            times = [c.slew.slewtime for c in slew_cmds]
            print(f"\n  Slews executed: {len(slew_cmds)}")
            print(
                f"  Slew distances: mean={np.mean(dists):.1f}°, max={np.max(dists):.1f}°"
            )
            print(f"  Slew times: mean={np.mean(times):.1f}s, max={np.max(times):.1f}s")

        # Desat events
        desat_events = getattr(ditl.acs, "desat_events", 0)
        print(f"\n  Desat events: {desat_events}")

        # MTQ telemetry
        mtq_power = np.array(getattr(ditl, "mtq_power", []))
        if mtq_power.size:
            print("\n  MTQ power stats:")
            print(f"    Mean: {mtq_power.mean():.2f} W")
            print(f"    Max: {mtq_power.max():.2f} W")
            print(f"    Steps with MTQ on: {(mtq_power > 0).sum()}")
            print(f"    MTQ on fraction: {(mtq_power > 0).mean() * 100:.1f}%")

            # MTQ during different modes
            mode_arr = np.array(ditl.mode)
            for m in [ACSMode.SLEWING, ACSMode.SCIENCE, ACSMode.PASS]:
                mask = mode_arr == m
                if mask.any():
                    mtq_on = (mtq_power[mask] > 0).mean() * 100
                    print(f"    MTQ on during {m.name}: {mtq_on:.1f}%")

        # MTQ vs disturbance torque comparison
        mtq_torque = np.array(getattr(ditl, "mtq_torque_mag", []))
        dist_torque = np.array(getattr(ditl, "disturbance_total", []))
        if mtq_torque.size and dist_torque.size:
            print("\n  Torque comparison:")
            print(f"    Mean |τ_mtq|: {mtq_torque.mean():.2e} N·m")
            print(f"    Mean |τ_dist|: {dist_torque.mean():.2e} N·m")
            print(f"    Ratio MTQ/dist: {mtq_torque.mean() / dist_torque.mean():.1f}x")

            # Net impulse over simulation
            dt = ditl.step_size
            mtq_impulse = mtq_torque.sum() * dt
            dist_impulse = dist_torque.sum() * dt
            print(f"    Total MTQ impulse: {mtq_impulse:.4f} N·m·s")
            print(f"    Total disturbance impulse: {dist_impulse:.4f} N·m·s")
            print(f"    Net accumulation: {dist_impulse - mtq_impulse:.4f} N·m·s")

        # Mode distribution
        mode_arr = np.array(ditl.mode)
        print("\n  Mode distribution:")
        for m in ACSMode:
            frac = np.mean(mode_arr == m) * 100
            if frac > 0:
                print(f"    {m.name}: {frac:.1f}%")

        print(f"\n{'=' * 60}")
        print("DIAGNOSIS")
        print(f"{'=' * 60}")
        if h_peak_theory > wheel_cap * 0.5:
            print("  [!] Peak slew momentum is >50% of single wheel capacity")
        if wm_raw.size and sat_mask.mean() > 0.1:
            print("  [!] Wheels saturated >10% of the time")
        if desat_events == 0 and sat_mask.sum() > 0:
            print("  [!] No desat events despite saturation")
        slew_frac = np.mean(mode_arr == ACSMode.SLEWING)
        if slew_frac > 0.5:
            print(f"  [!] Slewing {slew_frac * 100:.0f}% of time - high maneuver load")

    def test_pass_tracking_rate(self, notebook_ditl_setup):
        """Verify spacecraft body rate matches required ground station tracking rate during passes."""
        ditl, cfg = notebook_ditl_setup

        # Run simulation
        ditl.calc()

        # Check we have passes
        passes = ditl.acs.passrequests.passes
        assert len(passes) > 0, (
            "No passes scheduled - test requires ground station passes"
        )

        mode_arr = np.array(ditl.mode)
        ra_arr = np.array(ditl.ra)
        dec_arr = np.array(ditl.dec)
        utime_arr = np.array(ditl.utime)
        dt = ditl.step_size

        print(f"\n{'=' * 60}")
        print("GROUND STATION TRACKING RATE VERIFICATION")
        print(f"{'=' * 60}")
        print(f"Number of passes: {len(passes)}")

        # Analyze each pass
        for pass_idx, gspass in enumerate(passes):
            print(f"\n--- Pass {pass_idx + 1}: {gspass.station} ---")
            print(
                f"  Time: {gspass.begin} to {gspass.end} ({gspass.length / 60:.1f} min)"
            )

            # Find timesteps during this pass
            pass_mask = (utime_arr >= gspass.begin) & (utime_arr <= gspass.end)
            pass_mask &= mode_arr == ACSMode.PASS

            if not pass_mask.any():
                print("  WARNING: No PASS mode timesteps found during this pass window")
                continue

            pass_indices = np.where(pass_mask)[0]
            print(f"  Pass mode timesteps: {len(pass_indices)}")

            # Compute actual tracking rates from DITL pointing
            actual_rates = []
            required_rates = []
            rate_errors = []

            for i in range(1, len(pass_indices)):
                idx = pass_indices[i]
                idx_prev = pass_indices[i - 1]

                # Skip if not consecutive timesteps
                if idx != idx_prev + 1:
                    continue

                # Actual pointing change
                ra0, dec0 = ra_arr[idx_prev], dec_arr[idx_prev]
                ra1, dec1 = ra_arr[idx], dec_arr[idx]

                # Compute angular distance (same formula as ACS uses)
                r0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
                r1, d1 = np.deg2rad(ra1), np.deg2rad(dec1)
                cosc = np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(
                    r1 - r0
                )
                cosc = np.clip(cosc, -1.0, 1.0)
                actual_dist_deg = np.rad2deg(np.arccos(cosc))
                actual_rate = actual_dist_deg / dt  # deg/s

                # Required pointing from Pass object
                t = utime_arr[idx]
                t_prev = utime_arr[idx_prev]
                req_ra0, req_dec0 = gspass.ra_dec(t_prev)
                req_ra1, req_dec1 = gspass.ra_dec(t)

                if req_ra0 is None or req_ra1 is None:
                    continue

                # Compute required angular distance
                r0, d0 = np.deg2rad(req_ra0), np.deg2rad(req_dec0)
                r1, d1 = np.deg2rad(req_ra1), np.deg2rad(req_dec1)
                cosc = np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(
                    r1 - r0
                )
                cosc = np.clip(cosc, -1.0, 1.0)
                req_dist_deg = np.rad2deg(np.arccos(cosc))
                required_rate = req_dist_deg / dt  # deg/s

                actual_rates.append(actual_rate)
                required_rates.append(required_rate)
                rate_errors.append(abs(actual_rate - required_rate))

            if not actual_rates:
                print("  WARNING: No valid rate samples computed")
                continue

            actual_rates = np.array(actual_rates)
            required_rates = np.array(required_rates)
            rate_errors = np.array(rate_errors)

            print("\n  Tracking Rate Statistics:")
            print(
                f"    Required rate: mean={required_rates.mean():.4f} deg/s, max={required_rates.max():.4f} deg/s"
            )
            print(
                f"    Actual rate:   mean={actual_rates.mean():.4f} deg/s, max={actual_rates.max():.4f} deg/s"
            )
            print(
                f"    Rate error:    mean={rate_errors.mean():.6f} deg/s, max={rate_errors.max():.6f} deg/s"
            )

            # Verify actual rate matches required rate within tolerance
            # Tolerance: 0.01 deg/s (about 36 arcsec/s)
            rate_tolerance = 0.01  # deg/s
            max_error = rate_errors.max()
            mean_error = rate_errors.mean()

            print(f"\n  Verification (tolerance={rate_tolerance} deg/s):")
            print(
                f"    Max error:  {max_error:.6f} deg/s - {'PASS' if max_error < rate_tolerance else 'FAIL'}"
            )
            print(
                f"    Mean error: {mean_error:.6f} deg/s - {'PASS' if mean_error < rate_tolerance else 'FAIL'}"
            )

            # Assert tracking is accurate
            assert max_error < rate_tolerance, (
                f"Pass {pass_idx + 1} ({gspass.station}): tracking rate error {max_error:.6f} deg/s "
                f"exceeds tolerance {rate_tolerance} deg/s"
            )

        print(f"\n{'=' * 60}")
        print("ALL PASSES VERIFIED - Tracking rates match required rates")
        print(f"{'=' * 60}")

    def test_no_pointing_or_momentum_warnings(self, notebook_ditl_setup):
        """Verify 12-hour DITL produces zero pointing and momentum warnings.

        This test ensures physical consistency of the simulation:
        - Pointing rate never exceeds wheel physics limits during slews
        - Pointing rate never exceeds wheel physics limits during pass tracking
        - No momentum conservation violations
        - No wheel saturation that would violate physics
        """
        ditl, cfg = notebook_ditl_setup

        # Run simulation
        ditl.calc()

        # Check pointing warnings
        pointing_warnings = ditl.acs.get_pointing_warnings()
        if pointing_warnings:
            print(f"\nPointing warnings ({len(pointing_warnings)}):")
            for w in pointing_warnings[:5]:
                print(f"  {w}")
            if len(pointing_warnings) > 5:
                print(f"  ... and {len(pointing_warnings) - 5} more")

        assert len(pointing_warnings) == 0, (
            f"Expected 0 pointing warnings, got {len(pointing_warnings)}. "
            f"First: {pointing_warnings[0] if pointing_warnings else 'N/A'}"
        )

        # Check momentum warnings
        momentum_warnings = ditl.acs.get_momentum_warnings()
        if momentum_warnings:
            print(f"\nMomentum warnings ({len(momentum_warnings)}):")
            for w in momentum_warnings[:5]:
                print(f"  {w}")
            if len(momentum_warnings) > 5:
                print(f"  ... and {len(momentum_warnings) - 5} more")

        assert len(momentum_warnings) == 0, (
            f"Expected 0 momentum warnings, got {len(momentum_warnings)}. "
            f"First: {momentum_warnings[0] if momentum_warnings else 'N/A'}"
        )

        print("\nSUCCESS: No pointing or momentum warnings in 12-hour DITL")

    def test_pointing_momentum_consistency_catches_teleport(self):
        """Verify pointing-momentum consistency check catches non-physical teleportation.

        This test simulates the bug where pointing jumped without corresponding
        wheel momentum changes. The consistency check should catch this.
        """
        from conops.simulation.acs import ACS

        begin = datetime(2025, 1, 1)
        end = begin + timedelta(minutes=5)
        ephem = make_mock_ephem(begin, end, step_size=10.0)

        cfg = make_ditl_adcs_config(ephem)

        # Create ACS directly without full constraint setup
        acs = ACS(config=cfg, log=None)

        # Manually initialize wheel momentum tracking state
        t0 = begin.timestamp()
        initial_ra, initial_dec = 45.0, 30.0
        acs.ra = initial_ra
        acs.dec = initial_dec
        acs._last_pointing_ra = initial_ra
        acs._last_pointing_dec = initial_dec
        acs._last_pointing_utime = t0

        # Get initial wheel momentum
        h_initial = acs._get_total_wheel_momentum().copy()
        acs._last_wheel_momentum = h_initial.copy()
        acs._wheel_momentum_before_update = h_initial.copy()

        # Now simulate a "teleport" - change pointing by 30 deg without wheel change
        t1 = t0 + 10.0
        acs.ra = initial_ra + 30.0  # Jump 30 degrees in RA

        # Wheel momentum stays the same (no update happened)
        acs._wheel_momentum_before_update = h_initial.copy()

        # Call the consistency check - should catch the mismatch
        acs._validate_pointing_momentum_consistency(t1)

        # Check that a warning was generated
        warnings = acs.get_pointing_warnings()
        momentum_warnings = [w for w in warnings if "momentum" in w.lower()]

        print(f"\nPointing warnings after simulated teleport: {len(warnings)}")
        for w in warnings:
            print(f"  {w}")

        assert len(momentum_warnings) > 0, (
            "Expected pointing-momentum consistency check to catch teleportation, "
            "but no warning was generated"
        )

        # Verify the warning mentions the inconsistency
        assert any("inconsistency" in w.lower() for w in momentum_warnings), (
            f"Warning should mention inconsistency: {momentum_warnings}"
        )

        print("\nSUCCESS: Pointing-momentum consistency check caught teleportation")


def test_single_slew_momentum_detail():
    """Detailed trace of a single slew with notebook config."""
    begin = datetime(2025, 1, 1)
    end = begin + timedelta(minutes=10)
    ephem = make_mock_ephem(begin, end, step_size=0.1)  # Fine timestep

    cfg = make_ditl_adcs_config(ephem)

    # Disable disturbances for clean test
    acs_cfg = cfg.spacecraft_bus.attitude_control
    acs_cfg.cp_offset_body = (0.0, 0.0, 0.0)
    acs_cfg.residual_magnetic_moment = (0.0, 0.0, 0.0)
    acs_cfg.drag_area_m2 = 0.0
    acs_cfg.solar_area_m2 = 0.0

    from conops.simulation.acs import ACS

    acs = ACS(config=cfg, log=None)

    # Create a test slew
    slew = MagicMock()
    slew.slewdist = 30.0  # deg - moderate slew
    slew.slewstart = begin.timestamp() + 10.0
    slew.rotation_axis = (0.0, 0.0, 1.0)  # Z-axis
    slew._accel_override = None
    slew._vmax_override = None

    # Compute motion time
    accel = acs_cfg.slew_acceleration
    vmax = acs_cfg.max_slew_rate
    motion_time = cfg.spacecraft_bus.attitude_control.motion_time(
        30.0, accel=accel, vmax=vmax
    )
    slew.slewtime = motion_time + acs_cfg.settle_time
    slew.slewend = slew.slewstart + slew.slewtime
    slew.is_slewing = lambda t: slew.slewstart <= t < slew.slewend

    # Calculate accel/cruise phase times
    t_accel = vmax / accel  # Time to reach max rate
    d_accel = 0.5 * accel * t_accel**2  # Distance during accel
    d_cruise = slew.slewdist - 2 * d_accel
    t_cruise = d_cruise / vmax if vmax > 0 else 0.0

    print(f"\n{'=' * 60}")
    print("SINGLE SLEW DETAILED TRACE (ditl-adcs config)")
    print(f"{'=' * 60}")
    print(f"Slew: {slew.slewdist} deg about Z-axis")
    print(f"Accel: {accel} deg/s², Max rate: {vmax} deg/s")
    print(f"Motion time: {motion_time:.2f}s, Settle: {acs_cfg.settle_time}s")
    print(
        f"  t_accel: {t_accel:.2f}s, t_cruise: {t_cruise:.2f}s, t_decel: {t_accel:.2f}s"
    )
    print(f"MOI: {acs_cfg.spacecraft_moi}")

    # Initial state
    h_initial = acs._get_total_wheel_momentum().copy()
    print(f"\nInitial H: {h_initial}")

    # Step through slew with fine timestep to catch accel/decel phases
    dt = 0.1  # Fine enough to catch 0.5s accel phase
    acs._last_pointing_time = slew.slewstart
    acs.current_slew = slew
    acs.last_slew = slew
    acs._was_slewing = False

    print(f"\n{'t':>6} {'phase':>8} {'accel':>8} {'H_mag':>12} {'ΔH_mag':>12}")
    print("-" * 54)

    h_peak = 0.0
    sample_count = 0
    for t in np.arange(slew.slewstart + dt, slew.slewend + dt, dt):
        h_before = acs._get_total_wheel_momentum().copy()

        # Determine phase
        t_rel = t - slew.slewstart
        if t_rel <= t_accel:
            phase = "accel"
        elif t_rel <= t_accel + t_cruise:
            phase = "cruise"
        elif t_rel <= motion_time:
            phase = "decel"
        else:
            phase = "settle"

        accel_val = acs._slew_accel_profile(slew, t)

        acs._update_wheel_momentum(t)
        acs._last_pointing_time = t

        h_after = acs._get_total_wheel_momentum()
        h_mag = np.linalg.norm(h_after)
        delta_h_mag = h_mag - np.linalg.norm(h_before)
        h_peak = max(h_peak, h_mag)

        # Print during accel/decel phases, and at phase transitions
        is_interesting = (
            phase in ("accel", "decel")
            or t_rel <= 2 * dt
            or abs(t_rel - t_accel) < dt
            or abs(t_rel - (t_accel + t_cruise)) < dt
            or abs(t_rel - motion_time) < dt
            or t_rel >= motion_time - dt
        )
        if is_interesting and sample_count < 50:
            print(
                f"{t_rel:6.2f} {phase:>8} {accel_val:8.3f} {h_mag:12.6f} {delta_h_mag:12.6f}"
            )
            sample_count += 1

    h_final = acs._get_total_wheel_momentum()
    delta = np.linalg.norm(h_final - h_initial)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"H_initial: {h_initial}")
    print(f"H_final:   {h_final}")
    print(f"H_peak:    {h_peak:.6f} Nms")
    print(f"|ΔH|:      {delta:.6f} Nms")
    print(f"Status:    {'PASS' if delta < 0.01 else 'FAIL - momentum not conserved'}")

    assert delta < 0.01, f"Slew momentum not returning to baseline: delta={delta:.6f}"


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
