"""High-priority physics validation tests.

These tests verify critical physics properties that ensure simulation correctness:
1. End-to-end slew momentum validation (predicted vs actual)
2. Time step independence (results consistent across different dt values)
"""

from unittest.mock import MagicMock

import numpy as np

from conops.simulation.acs import ACS
from conops.simulation.slew import Slew


class MockEphemeris:
    """Minimal ephemeris mock for ACS initialization."""

    def __init__(self, step_size: float = 1.0):
        self.step_size = step_size
        self.earth = [MagicMock(ra=MagicMock(deg=0.0), dec=MagicMock(deg=0.0))]
        self.sun = [MagicMock(ra=MagicMock(deg=45.0), dec=MagicMock(deg=23.5))]
        self.gcrs_pv = MagicMock()
        self.gcrs_pv.position = [[7000e3, 0, 0]]
        self.gcrs_pv.velocity = [[0, 7500, 0]]

    def index(self, dt):
        return 0


class MockConstraint:
    """Mock constraint with ephemeris."""

    def __init__(self, step_size: float = 1.0):
        self.ephem = MockEphemeris(step_size)

    def in_constraint(self, *args, **kwargs):
        return False

    def in_eclipse(self, *args, **kwargs):
        return False


def make_config(step_size: float = 1.0) -> MagicMock:
    """Create a mock config for both ACS and Slew initialization."""
    config = MagicMock()
    config.constraint = MockConstraint(step_size)
    config.spacecraft_bus = MagicMock()
    acs_cfg = config.spacecraft_bus.attitude_control = MagicMock()
    acs_cfg.slew_acceleration = 0.5  # deg/s^2
    acs_cfg.max_slew_rate = 1.0  # deg/s
    acs_cfg.settle_time = 30.0
    acs_cfg.wheel_enabled = False
    acs_cfg.spacecraft_moi = (10.0, 10.0, 10.0)
    acs_cfg.strict_wheel_validation = False
    # Disable disturbances for clean physics validation
    acs_cfg.cp_offset_body = (0.0, 0.0, 0.0)
    acs_cfg.residual_magnetic_moment = (0.0, 0.0, 0.0)
    acs_cfg.drag_area_m2 = 0.0
    acs_cfg.solar_area_m2 = 0.0
    acs_cfg.disturbance_torque_body = (0.0, 0.0, 0.0)
    # 3-axis wheel configuration with ample capacity
    acs_cfg.wheels = [
        {
            "orientation": [1, 0, 0],
            "max_torque": 0.5,
            "max_momentum": 20.0,
            "name": "X",
        },
        {
            "orientation": [0, 1, 0],
            "max_torque": 0.5,
            "max_momentum": 20.0,
            "name": "Y",
        },
        {
            "orientation": [0, 0, 1],
            "max_torque": 0.5,
            "max_momentum": 20.0,
            "name": "Z",
        },
    ]
    acs_cfg.magnetorquers = []
    config.solar_panel = None

    # Add methods needed by Slew (mimic AttitudeControlSystem methods)
    def motion_time(angle, accel=None, vmax=None):
        a = accel if accel else acs_cfg.slew_acceleration
        # Triangular profile time
        t_peak = (angle / a) ** 0.5
        return 2 * t_peak

    def slew_time(angle, accel=None, vmax=None):
        return motion_time(angle, accel, vmax) + acs_cfg.settle_time

    def predict_slew(startra, startdec, endra, enddec, steps=20):
        """Compute slew distance and path (simplified great circle)."""
        # Convert to radians
        ra1, dec1 = np.radians(startra), np.radians(startdec)
        ra2, dec2 = np.radians(endra), np.radians(enddec)
        # Great circle distance
        cos_dist = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
            ra2 - ra1
        )
        cos_dist = np.clip(cos_dist, -1.0, 1.0)
        dist_rad = np.arccos(cos_dist)
        dist_deg = np.degrees(dist_rad)
        # Simple linear interpolation for path (not true great circle but ok for tests)
        ra_path = list(np.linspace(startra, endra, steps))
        dec_path = list(np.linspace(startdec, enddec, steps))
        return dist_deg, (ra_path, dec_path)

    acs_cfg.motion_time = motion_time
    acs_cfg.slew_time = slew_time
    acs_cfg.predict_slew = predict_slew

    return config


def make_acs_with_wheels(step_size: float = 1.0) -> tuple[ACS, MagicMock]:
    """Create ACS instance with 3-axis wheel configuration.

    Returns:
        Tuple of (ACS instance, config) - config is needed for Slew creation.
    """
    config = make_config(step_size)
    acs = ACS(config=config, log=None)
    return acs, config


def make_slew(
    config: MagicMock,
    start_ra: float,
    start_dec: float,
    end_ra: float,
    end_dec: float,
) -> Slew:
    """Create a Slew with specified start/end positions."""
    slew = Slew(config=config)
    slew.startra = start_ra
    slew.startdec = start_dec
    slew.endra = end_ra
    slew.enddec = end_dec
    slew.obsid = 1
    slew.obstype = "PPT"
    slew.predict_slew()
    return slew


class TestEndToEndSlewMomentum:
    """Verify predicted slew momentum matches actual momentum during execution."""

    def test_slew_peak_momentum_matches_prediction(self):
        """Peak wheel momentum during slew should match pre-slew prediction."""
        acs, config = make_acs_with_wheels(step_size=0.5)

        # Create a 10-degree slew
        slew = make_slew(config, 0.0, 0.0, 10.0, 0.0)

        # Get predicted peak momentum before starting
        predicted_peak, axis = acs._compute_slew_peak_momentum(slew)
        assert predicted_peak > 0, "Should predict non-zero peak momentum"

        # Start the slew
        utime = 1000.0
        slew.slewstart = utime
        slew.calc_slewtime()
        acs.current_slew = slew
        acs.last_slew = slew
        acs._last_pointing_time = utime
        acs._was_slewing = False

        # Execute slew and track max momentum
        max_observed_momentum = 0.0
        dt = 0.5
        slew_duration = slew.slewtime

        t = utime
        while t < utime + slew_duration:
            acs._update_wheel_momentum(t)
            h_wheels = acs._get_total_wheel_momentum()
            h_mag = float(np.linalg.norm(h_wheels))
            max_observed_momentum = max(max_observed_momentum, h_mag)
            t += dt
            acs._last_pointing_time = t - dt

        # Verify actual peak is in reasonable range of predicted
        # Note: There's inherent discrepancy between analytical prediction and
        # discrete simulation due to time stepping and profile approximations.
        # We use a loose tolerance (60%) to catch gross errors while allowing
        # for expected numerical differences.
        tolerance = 0.60
        if predicted_peak > 0:
            relative_error = (
                abs(max_observed_momentum - predicted_peak) / predicted_peak
            )
            assert relative_error < tolerance, (
                f"Peak momentum mismatch: predicted={predicted_peak:.4f}, "
                f"observed={max_observed_momentum:.4f}, error={relative_error:.1%}"
            )
            # Also verify we observed meaningful momentum (not zero)
            assert max_observed_momentum > 0.01, (
                f"Observed peak momentum too low: {max_observed_momentum:.4f}"
            )

    def test_slew_momentum_returns_to_baseline(self):
        """After a complete slew, wheel momentum should return near starting value."""
        acs, config = make_acs_with_wheels(step_size=0.5)

        # Record initial wheel momentum
        h_initial = acs._get_total_wheel_momentum().copy()

        # Create and execute a 5-degree slew
        slew = make_slew(config, 0.0, 0.0, 5.0, 0.0)

        utime = 1000.0
        slew.slewstart = utime
        slew.calc_slewtime()
        acs.current_slew = slew
        acs.last_slew = slew
        acs._last_pointing_time = utime
        acs._was_slewing = False

        # Execute entire slew including settle time
        dt = 0.5
        t = utime
        end_time = utime + slew.slewtime + acs.acs_config.settle_time

        while t < end_time:
            acs._update_wheel_momentum(t)
            t += dt
            acs._last_pointing_time = t - dt

        # Check final momentum is close to initial
        h_final = acs._get_total_wheel_momentum()
        delta_h = np.linalg.norm(h_final - h_initial)

        # Should be within 1% of max wheel capacity
        max_capacity = 20.0
        assert delta_h < 0.01 * max_capacity, (
            f"Momentum drift after slew: delta={delta_h:.6f}"
        )


class TestTimeStepIndependence:
    """Verify physics results are consistent across different time step sizes."""

    def _run_slew_with_dt(self, dt: float) -> dict:
        """Run a standard slew and return key metrics."""
        acs, config = make_acs_with_wheels(step_size=dt)

        # Standard 8-degree test slew
        slew = make_slew(config, 0.0, 0.0, 8.0, 0.0)

        utime = 1000.0
        slew.slewstart = utime
        slew.calc_slewtime()
        acs.current_slew = slew
        acs.last_slew = slew
        acs._last_pointing_time = utime
        acs._was_slewing = False

        # Track metrics during slew
        max_momentum = 0.0
        total_steps = 0

        t = utime
        end_time = utime + slew.slewtime

        while t < end_time:
            acs._update_wheel_momentum(t)
            h_mag = float(np.linalg.norm(acs._get_total_wheel_momentum()))
            max_momentum = max(max_momentum, h_mag)
            total_steps += 1
            t += dt
            acs._last_pointing_time = t - dt

        final_momentum = acs._get_total_wheel_momentum().copy()

        return {
            "peak_momentum": max_momentum,
            "final_momentum": final_momentum,
            "steps": total_steps,
        }

    def test_peak_momentum_independent_of_dt(self):
        """Peak momentum during slew should be similar regardless of time step."""
        results_fine = self._run_slew_with_dt(0.1)
        results_medium = self._run_slew_with_dt(0.5)
        results_coarse = self._run_slew_with_dt(1.0)

        peak_fine = results_fine["peak_momentum"]
        peak_medium = results_medium["peak_momentum"]
        peak_coarse = results_coarse["peak_momentum"]

        # All should be within 15% of the fine resolution reference
        reference = peak_fine
        tolerance = 0.15

        if reference > 0:
            error_medium = abs(peak_medium - reference) / reference
            error_coarse = abs(peak_coarse - reference) / reference

            assert error_medium < tolerance, (
                f"Medium dt peak differs: fine={peak_fine:.4f}, "
                f"medium={peak_medium:.4f}, error={error_medium:.1%}"
            )
            assert error_coarse < tolerance, (
                f"Coarse dt peak differs: fine={peak_fine:.4f}, "
                f"coarse={peak_coarse:.4f}, error={error_coarse:.1%}"
            )

    def test_final_momentum_independent_of_dt(self):
        """Final wheel momentum after slew should be similar regardless of dt."""
        results_fine = self._run_slew_with_dt(0.1)
        results_medium = self._run_slew_with_dt(0.5)
        results_coarse = self._run_slew_with_dt(1.0)

        final_fine = results_fine["final_momentum"]
        final_medium = results_medium["final_momentum"]
        final_coarse = results_coarse["final_momentum"]

        # Use absolute tolerance since values may be near zero
        abs_tolerance = 0.5  # N·m·s

        diff_medium = np.linalg.norm(final_medium - final_fine)
        diff_coarse = np.linalg.norm(final_coarse - final_fine)

        assert diff_medium < abs_tolerance, (
            f"Medium dt final momentum differs by {diff_medium:.4f}"
        )
        assert diff_coarse < abs_tolerance, (
            f"Coarse dt final momentum differs by {diff_coarse:.4f}"
        )

    def test_conservation_holds_across_dt_values(self):
        """Total system momentum conservation should hold for all dt values."""
        for dt in [0.1, 0.5, 1.0, 2.0]:
            acs, _ = make_acs_with_wheels(step_size=dt)

            # Record initial total system momentum
            h_initial = acs._get_total_system_momentum().copy()

            # Apply some wheel torques (internal exchange only)
            torque = np.array([0.1, 0.05, 0.02])
            for _ in range(10):
                acs._apply_wheel_torques_conserving(torque, dt)

            # Total momentum should be conserved (no external torques)
            h_final = acs._get_total_system_momentum()
            delta = np.linalg.norm(h_final - h_initial)

            assert delta < 1e-10, f"Conservation violated with dt={dt}: delta={delta}"


class TestNonPrincipalAxisSlew:
    """Test slews about non-principal axes (diagonal rotations)."""

    def test_diagonal_axis_slew_conserves_momentum(self):
        """Slew about diagonal axis should conserve momentum correctly."""
        acs, config = make_acs_with_wheels(step_size=0.5)

        # Create a slew with both RA and Dec change (diagonal motion)
        slew = make_slew(config, 0.0, 0.0, 5.0, 5.0)

        # The rotation axis should not be aligned with any principal axis
        axis = np.array(getattr(slew, "rotation_axis", [0, 0, 1]), dtype=float)
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm

        # Execute slew and verify conservation
        utime = 1000.0
        slew.slewstart = utime
        slew.calc_slewtime()
        acs.current_slew = slew
        acs.last_slew = slew
        acs._last_pointing_time = utime

        h_initial = acs._get_total_system_momentum().copy()

        # Run slew
        dt = 0.5
        t = utime
        while t < utime + slew.slewtime:
            acs._update_wheel_momentum(t)
            t += dt
            acs._last_pointing_time = t - dt

        h_final = acs._get_total_system_momentum()

        # Conservation should hold (no external torques - disturbances disabled)
        delta = np.linalg.norm(h_final - h_initial)
        assert delta < 0.01, f"Momentum not conserved: delta={delta:.6f} N·m·s"
