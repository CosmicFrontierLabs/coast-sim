"""Tests for disturbance torque model physics."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops.simulation.disturbance import DisturbanceConfig, DisturbanceModel


class MockEphemeris:
    """Mock ephemeris providing controlled orbital state."""

    def __init__(
        self,
        position_m: tuple[float, float, float] = (6878e3, 0.0, 0.0),
        velocity_m_s: tuple[float, float, float] = (0.0, 7.5e3, 0.0),
        lat_deg: float = 0.0,
        lon_deg: float = 0.0,
    ):
        self.gcrs_pv = Mock()
        self.gcrs_pv.position = [position_m]
        self.gcrs_pv.velocity = [velocity_m_s]
        self.lat = [lat_deg]
        self.long = [lon_deg]
        self.latitude_deg = [lat_deg]
        self.longitude_deg = [lon_deg]
        # Mock sun position at 1 AU along +X (rust-ephem 0.3.0+ sun_pv API)
        self.sun_pv = Mock()
        self.sun_pv.position = [np.array([1.496e11, 0.0, 0.0])]

    def index(self, dt):
        return 0


class TestAtmosphericDensity:
    """Tests for atmospheric density lookup table."""

    def test_density_at_sea_level(self):
        """Sea level density should be ~1.225 kg/m^3."""
        ephem = MockEphemeris()
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        rho = model._table_density(0.0)
        assert rho == pytest.approx(1.225, rel=0.01)

    def test_density_at_400km(self):
        """400 km altitude density should match table value."""
        ephem = MockEphemeris()
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        # New table: (400, 3.725e-12, 59.4)
        rho = model._table_density(400e3)
        assert rho == pytest.approx(3.725e-12, rel=0.1)

    def test_density_interpolation(self):
        """Density between table points uses exponential model."""
        ephem = MockEphemeris()
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        # 450 km uses band (450, 1.585e-12, 62.2)
        # At exactly 450 km: rho = 1.585e-12 * exp(0) = 1.585e-12
        rho = model._table_density(450e3)
        assert rho == pytest.approx(1.585e-12, rel=0.1)

    def test_density_above_table_max(self):
        """Density above 1000 km extrapolates with final scale height."""
        ephem = MockEphemeris()
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        # Uses (1000, 3.019e-15, 268.0): rho = 3.019e-15 * exp(-(1500-1000)/268)
        rho = model._table_density(1500e3)
        import numpy as np

        expected = 3.019e-15 * np.exp(-500 / 268.0)
        assert rho == pytest.approx(expected, rel=0.01)

    def test_density_decreases_with_altitude(self):
        """Density should monotonically decrease with altitude."""
        ephem = MockEphemeris()
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        altitudes = [0, 100e3, 200e3, 300e3, 400e3, 500e3]
        densities = [model._table_density(alt) for alt in altitudes]

        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1]


class TestGravityGradientTorque:
    """Tests for gravity gradient torque calculation."""

    def test_gg_torque_zero_for_spherical_inertia(self):
        """GG torque is zero for spherically symmetric spacecraft."""
        # Spacecraft at 500 km altitude along +X
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        # Spherical inertia: I = diag(1, 1, 1)
        moi = (1.0, 1.0, 1.0)
        torque, components = model.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=moi
        )

        # GG torque should be zero for spherical inertia
        assert components["gg"] == pytest.approx(0.0, abs=1e-12)

    def test_gg_torque_nonzero_for_asymmetric_inertia(self):
        """GG torque is nonzero for asymmetric spacecraft."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        # Asymmetric inertia
        moi = (10.0, 5.0, 1.0)
        torque, components = model.compute(
            utime=0.0, ra_deg=45.0, dec_deg=30.0, in_eclipse=False, moi_cfg=moi
        )

        # GG torque should be nonzero
        assert components["gg"] > 0

    def test_gg_torque_scales_with_altitude(self):
        """GG torque decreases with altitude (1/r^3 dependence)."""
        cfg = DisturbanceConfig()
        moi = (10.0, 5.0, 1.0)

        # Low altitude: 300 km
        r_low = 6378e3 + 300e3
        ephem_low = MockEphemeris(position_m=(r_low, 0.0, 0.0))
        model_low = DisturbanceModel(ephem_low, cfg)
        _, comp_low = model_low.compute(
            utime=0.0, ra_deg=45.0, dec_deg=30.0, in_eclipse=False, moi_cfg=moi
        )

        # High altitude: 800 km
        r_high = 6378e3 + 800e3
        ephem_high = MockEphemeris(position_m=(r_high, 0.0, 0.0))
        model_high = DisturbanceModel(ephem_high, cfg)
        _, comp_high = model_high.compute(
            utime=0.0, ra_deg=45.0, dec_deg=30.0, in_eclipse=False, moi_cfg=moi
        )

        # Lower altitude should have larger GG torque
        assert comp_low["gg"] > comp_high["gg"]


class TestDragTorque:
    """Tests for aerodynamic drag torque calculation."""

    def test_drag_torque_zero_without_cp_offset(self):
        """Drag torque is zero when CoP = CoM."""
        r_mag = 6378e3 + 400e3
        v_mag = 7.67e3  # Orbital velocity at 400 km
        ephem = MockEphemeris(
            position_m=(r_mag, 0.0, 0.0), velocity_m_s=(0.0, v_mag, 0.0)
        )
        cfg = DisturbanceConfig(
            drag_area_m2=1.0,
            drag_coeff=2.2,
            cp_offset_body=(0.0, 0.0, 0.0),  # No offset
        )
        model = DisturbanceModel(ephem, cfg)

        _, components = model.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert components["drag"] == pytest.approx(0.0, abs=1e-15)

    def test_drag_torque_nonzero_with_cp_offset(self):
        """Drag torque is nonzero when CoP != CoM."""
        r_mag = 6378e3 + 400e3
        v_mag = 7.67e3
        ephem = MockEphemeris(
            position_m=(r_mag, 0.0, 0.0), velocity_m_s=(0.0, v_mag, 0.0)
        )
        cfg = DisturbanceConfig(
            drag_area_m2=1.0,
            drag_coeff=2.2,
            # CP offset perpendicular to velocity for maximum torque
            cp_offset_body=(0.0, 0.0, 0.1),  # 10 cm offset along Z
        )
        model = DisturbanceModel(ephem, cfg)

        # Point spacecraft so velocity vector is not along body Z
        _, components = model.compute(
            utime=0.0, ra_deg=90.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert components["drag"] > 0

    def test_drag_torque_zero_without_drag_area(self):
        """Drag torque is zero when drag area is zero."""
        r_mag = 6378e3 + 400e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig(
            drag_area_m2=0.0,
            cp_offset_body=(0.1, 0.0, 0.0),
        )
        model = DisturbanceModel(ephem, cfg)

        _, components = model.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert components["drag"] == pytest.approx(0.0)

    def test_drag_torque_scales_with_area(self):
        """Drag torque scales linearly with drag area."""
        r_mag = 6378e3 + 400e3
        v_mag = 7.67e3
        ephem = MockEphemeris(
            position_m=(r_mag, 0.0, 0.0), velocity_m_s=(0.0, v_mag, 0.0)
        )

        cfg1 = DisturbanceConfig(
            drag_area_m2=1.0,
            drag_coeff=2.2,
            cp_offset_body=(0.1, 0.0, 0.0),
        )
        cfg2 = DisturbanceConfig(
            drag_area_m2=2.0,
            drag_coeff=2.2,
            cp_offset_body=(0.1, 0.0, 0.0),
        )

        model1 = DisturbanceModel(ephem, cfg1)
        model2 = DisturbanceModel(ephem, cfg2)

        _, comp1 = model1.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )
        _, comp2 = model2.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert comp2["drag"] == pytest.approx(2 * comp1["drag"], rel=0.01)


class TestSRPTorque:
    """Tests for solar radiation pressure torque calculation."""

    def test_srp_torque_zero_in_eclipse(self):
        """SRP torque is zero during eclipse."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig(
            solar_area_m2=1.0,
            solar_reflectivity=1.5,
            cp_offset_body=(0.1, 0.0, 0.0),
        )
        model = DisturbanceModel(ephem, cfg)

        _, components = model.compute(
            utime=0.0,
            ra_deg=0.0,
            dec_deg=0.0,
            in_eclipse=True,  # In eclipse
            moi_cfg=(1, 1, 1),
        )

        assert components["srp"] == pytest.approx(0.0)

    def test_srp_torque_nonzero_in_sunlight(self):
        """SRP torque is nonzero in sunlight with offset."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig(
            solar_area_m2=1.0,
            solar_reflectivity=1.5,
            cp_offset_body=(0.1, 0.0, 0.0),
        )
        model = DisturbanceModel(ephem, cfg)

        _, components = model.compute(
            utime=0.0,
            ra_deg=0.0,
            dec_deg=0.0,
            in_eclipse=False,  # Not in eclipse
            moi_cfg=(1, 1, 1),
        )

        assert components["srp"] > 0

    def test_srp_torque_zero_without_area(self):
        """SRP torque is zero when solar area is zero."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig(
            solar_area_m2=0.0,
            cp_offset_body=(0.1, 0.0, 0.0),
        )
        model = DisturbanceModel(ephem, cfg)

        _, components = model.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert components["srp"] == pytest.approx(0.0)

    def test_srp_reflectivity_scaling(self):
        """SRP torque scales with reflectivity factor."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))

        # Absorbing surface (reflectivity = 1.0)
        cfg1 = DisturbanceConfig(
            solar_area_m2=1.0,
            solar_reflectivity=1.0,
            cp_offset_body=(0.1, 0.0, 0.0),
        )
        # Reflective surface (reflectivity = 2.0)
        cfg2 = DisturbanceConfig(
            solar_area_m2=1.0,
            solar_reflectivity=2.0,
            cp_offset_body=(0.1, 0.0, 0.0),
        )

        model1 = DisturbanceModel(ephem, cfg1)
        model2 = DisturbanceModel(ephem, cfg2)

        _, comp1 = model1.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )
        _, comp2 = model2.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert comp2["srp"] == pytest.approx(2 * comp1["srp"], rel=0.01)


class TestMagneticTorque:
    """Tests for residual magnetic dipole torque calculation."""

    def test_magnetic_torque_zero_without_dipole(self):
        """Magnetic torque is zero with no residual dipole."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig(
            residual_magnetic_moment=(0.0, 0.0, 0.0),
        )
        model = DisturbanceModel(ephem, cfg)

        _, components = model.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert components["mag"] == pytest.approx(0.0)

    def test_magnetic_torque_nonzero_with_dipole(self):
        """Magnetic torque is nonzero with residual dipole."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig(
            residual_magnetic_moment=(0.1, 0.1, 0.1),  # 0.1 A*m^2
        )
        model = DisturbanceModel(ephem, cfg)

        _, components = model.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert components["mag"] > 0


class TestInertiaMatrixBuilding:
    """Tests for inertia matrix construction."""

    def test_build_inertia_from_diagonal(self):
        """Diagonal MOI tuple creates diagonal matrix."""
        i_mat = DisturbanceModel._build_inertia((10.0, 5.0, 2.0))

        expected = np.diag([10.0, 5.0, 2.0])
        assert np.allclose(i_mat, expected)

    def test_build_inertia_from_scalar(self):
        """Scalar MOI creates isotropic diagonal matrix."""
        i_mat = DisturbanceModel._build_inertia(5.0)

        expected = np.diag([5.0, 5.0, 5.0])
        assert np.allclose(i_mat, expected)

    def test_build_inertia_from_full_matrix(self):
        """3x3 matrix is returned as-is."""
        full = np.array([[10, 1, 0], [1, 8, 0], [0, 0, 5]], dtype=float)
        i_mat = DisturbanceModel._build_inertia(full.tolist())

        assert np.allclose(i_mat, full)

    def test_build_inertia_fallback_on_error(self):
        """Invalid input falls back to identity matrix."""
        i_mat = DisturbanceModel._build_inertia("invalid")

        expected = np.diag([1.0, 1.0, 1.0])
        assert np.allclose(i_mat, expected)


class TestCoordinateTransforms:
    """Tests for inertial-to-body coordinate transformation."""

    def test_inertial_to_body_identity_at_origin(self):
        """Transform at RA=0, Dec=0 should have Z pointing to +X inertial."""
        vec_inertial = np.array([1.0, 0.0, 0.0])
        vec_body = DisturbanceModel._inertial_to_body(
            vec_inertial, ra_deg=0.0, dec_deg=0.0
        )

        # At RA=0, Dec=0, body Z points to inertial +X
        # So inertial +X should map to body +Z
        assert np.allclose(vec_body, [0.0, 0.0, 1.0], atol=1e-10)

    def test_inertial_to_body_preserves_magnitude(self):
        """Rotation preserves vector magnitude."""
        vec_inertial = np.array([3.0, 4.0, 0.0])
        vec_body = DisturbanceModel._inertial_to_body(
            vec_inertial, ra_deg=45.0, dec_deg=30.0
        )

        assert np.linalg.norm(vec_body) == pytest.approx(5.0, rel=1e-10)

    def test_inertial_to_body_at_pole(self):
        """Transform handles pointing at celestial pole."""
        vec_inertial = np.array([0.0, 0.0, 1.0])
        vec_body = DisturbanceModel._inertial_to_body(
            vec_inertial, ra_deg=0.0, dec_deg=90.0
        )

        # At Dec=90, body Z points to inertial +Z
        assert np.allclose(vec_body, [0.0, 0.0, 1.0], atol=1e-10)


class TestTotalDisturbance:
    """Tests for aggregate disturbance computation."""

    def test_compute_returns_vector_and_components(self):
        """compute() returns torque vector and component breakdown."""
        r_mag = 6378e3 + 500e3
        ephem = MockEphemeris(position_m=(r_mag, 0.0, 0.0))
        cfg = DisturbanceConfig()
        model = DisturbanceModel(ephem, cfg)

        torque, components = model.compute(
            utime=0.0, ra_deg=0.0, dec_deg=0.0, in_eclipse=False, moi_cfg=(1, 1, 1)
        )

        assert torque.shape == (3,)
        assert "total" in components
        assert "gg" in components
        assert "drag" in components
        assert "srp" in components
        assert "mag" in components
        assert "vector" in components

    def test_total_equals_sum_of_components(self):
        """Total torque magnitude equals sum of component magnitudes."""
        r_mag = 6378e3 + 400e3
        v_mag = 7.67e3
        ephem = MockEphemeris(
            position_m=(r_mag, 0.0, 0.0), velocity_m_s=(0.0, v_mag, 0.0)
        )
        cfg = DisturbanceConfig(
            drag_area_m2=1.0,
            drag_coeff=2.2,
            solar_area_m2=1.0,
            solar_reflectivity=1.5,
            cp_offset_body=(0.1, 0.05, 0.0),
            residual_magnetic_moment=(0.1, 0.0, 0.0),
        )
        model = DisturbanceModel(ephem, cfg)

        torque, components = model.compute(
            utime=0.0, ra_deg=45.0, dec_deg=30.0, in_eclipse=False, moi_cfg=(10, 5, 1)
        )

        # Total should be magnitude of the vector sum
        assert components["total"] == pytest.approx(np.linalg.norm(torque), rel=1e-10)
