"""Environmental torque sources: solar radiation pressure on the surfaces,
and the tracker summing sources by name."""

import numpy as np
import pytest

from conops.common.vector import attitude_to_quat
from conops.config import (
    PanelGeometry,
    Payload,
    SolarPanel,
    SolarPanelSet,
    Surface,
    SurfaceModel,
    Telescope,
    TelescopeConfig,
    box_surfaces,
    guess_surfaces,
    panel_surfaces,
    telescope_surfaces,
)
from conops.simulation.geomagnetic import EARTH_RADIUS_KM, TiltedDipoleField
from conops.simulation.momentum import (
    ASTRONOMICAL_UNIT_KM,
    SOLAR_PRESSURE_N_M2,
    GravityGradientTorque,
    MagnetorquerUnloading,
    OrbitSample,
    ResidualDipoleTorque,
    SolarRadiationPressureTorque,
    StoredMomentumTracker,
    solar_radiation_pressure_torque_body,
)

SUN = np.array([ASTRONOMICAL_UNIT_KM, 0.0, 0.0])


def _plate(center, normal, area=2.0, specular=0.2, diffuse=0.3) -> Surface:
    return Surface(
        name="plate",
        center_m=center,
        normal=normal,
        area_m2=area,
        specular_reflectivity=specular,
        diffuse_reflectivity=diffuse,
    )


class TestSolarRadiationPressure:
    def test_a_plate_facing_the_sun_at_an_offset_takes_the_full_force(self) -> None:
        """Normal incidence: the force is along the Sun line, of magnitude
        P A (1 + specular + 2 diffuse / 3), and the torque is the lever arm
        across it."""
        model = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (1.0, 0.0, 0.0))])
        torque = solar_radiation_pressure_torque_body(SUN, model)
        force = -SOLAR_PRESSURE_N_M2 * 2.0 * (1.0 + 0.2 + 2.0 * 0.3 / 3.0)
        expected = np.cross([0.0, 0.5, 0.0], [force, 0.0, 0.0])
        assert torque == pytest.approx(expected, rel=1e-9)
        assert torque[2] == pytest.approx(0.5 * abs(force)) and torque[2] > 0

    def test_a_plate_facing_away_or_in_eclipse_takes_nothing(self) -> None:
        away = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (-1.0, 0.0, 0.0))])
        assert np.all(solar_radiation_pressure_torque_body(SUN, away) == 0.0)
        facing = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (1.0, 0.0, 0.0))])
        assert np.all(
            solar_radiation_pressure_torque_body(SUN, facing, sunlit=False) == 0.0
        )

    def test_oblique_incidence_and_distance(self) -> None:
        """At 60 degrees the absorbed part follows the Sun line and the
        reflected part the normal, both scaled by cos; twice the distance is
        a quarter of the pressure."""
        plate = _plate(
            (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), area=1.0, specular=0.5, diffuse=0.0
        )
        model = SurfaceModel(surfaces=[plate])
        direction = np.array([0.5, np.sqrt(3) / 2, 0.0])
        torque = solar_radiation_pressure_torque_body(SUN[0] * direction, model)
        force = (
            -SOLAR_PRESSURE_N_M2
            * 0.5
            * (0.5 * direction + 2.0 * 0.5 * 0.5 * np.array([1.0, 0.0, 0.0]))
        )
        assert torque == pytest.approx(np.cross([0.0, 0.0, 1.0], force), rel=1e-9)
        far = solar_radiation_pressure_torque_body(2.0 * SUN[0] * direction, model)
        assert far == pytest.approx(torque / 4.0, rel=1e-9)

    def test_a_box_about_its_centre_of_mass_has_no_torque(self) -> None:
        model = SurfaceModel(surfaces=box_surfaces((1.0, 2.0, 3.0)))
        for sun in (SUN, np.array([1.0, 1.0, 1.0]) * SUN[0], -SUN):
            assert solar_radiation_pressure_torque_body(sun, model) == pytest.approx(
                np.zeros(3), abs=1e-20
            )
        offset = SurfaceModel(
            center_of_mass_m=(0.0, 0.0, 0.1), surfaces=box_surfaces((1.0, 2.0, 3.0))
        )
        torque = solar_radiation_pressure_torque_body(SUN, offset)
        # Sunlight along +X pushes the +X face towards -X, 0.1 m below the
        # centre of mass: a torque about +Y of the face's force times the arm.
        face = SOLAR_PRESSURE_N_M2 * 6.0 * (1.0 + 0.2 + 0.2)
        assert torque == pytest.approx([0.0, 0.1 * face, 0.0], rel=1e-9)

    def test_the_source_rotates_the_sun_into_the_body_frame(self) -> None:
        model = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (1.0, 0.0, 0.0))])
        source = SolarRadiationPressureTorque(model)
        sample = OrbitSample(
            utime=0.0,
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            rotation_eci_to_body=np.eye(3),
            sun_eci_km=SUN,
        )
        facing = source.torque_body_n_m(sample)
        assert facing[2] > 0
        # Yawed 180 degrees, the plate faces away from the Sun.
        turned = OrbitSample(
            utime=0.0,
            position_eci_km=sample.position_eci_km,
            rotation_eci_to_body=np.diag([-1.0, -1.0, 1.0]),
            sun_eci_km=SUN,
        )
        assert np.all(source.torque_body_n_m(turned) == 0.0)
        unknown = OrbitSample(
            utime=0.0,
            position_eci_km=sample.position_eci_km,
            rotation_eci_to_body=np.eye(3),
        )
        assert np.all(source.torque_body_n_m(unknown) == 0.0)


class TestSurfaces:
    def test_a_surface_normalises_its_normal_and_bounds_its_reflectivities(
        self,
    ) -> None:
        plate = _plate((0.0, 0.0, 0.0), (0.0, 3.0, 4.0))
        assert plate.normal == pytest.approx((0.0, 0.6, 0.8))
        with pytest.raises(ValueError, match="cannot exceed one"):
            _plate((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), specular=0.7, diffuse=0.4)

    def test_a_box_has_six_faces_with_the_right_areas(self) -> None:
        faces = box_surfaces((1.0, 2.0, 3.0), center_m=(1.0, 0.0, 0.0))
        assert [f.name for f in faces] == [
            f"bus {a}" for a in ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
        ]
        assert faces[0].center_m == (1.5, 0.0, 0.0) and faces[0].area_m2 == 6.0
        assert faces[2].area_m2 == 3.0 and faces[4].area_m2 == 2.0
        assert SurfaceModel(surfaces=faces).total_area_m2 == pytest.approx(22.0)

    def test_panels_and_telescopes_become_plates(self) -> None:
        panels = SolarPanelSet(
            panels=[
                SolarPanel(
                    name="wing",
                    normal=(0.0, 1.0, 0.0),
                    max_power=100.0,
                    geometry=PanelGeometry(
                        center_m=(0.0, 2.0, 0.0),
                        u=(1.0, 0.0, 0.0),
                        v=(0.0, 0.0, 1.0),
                        width_m=3.0,
                        height_m=1.0,
                    ),
                ),
                SolarPanel(name="bare", normal=(0.0, -1.0, 0.0), max_power=100.0),
            ]
        )
        plates = panel_surfaces(panels)
        assert [p.name for p in plates] == ["wing cells", "wing back"]
        assert plates[0].normal == (0.0, 1.0, 0.0) and plates[1].normal == (
            0.0,
            -1.0,
            0.0,
        )
        assert plates[0].area_m2 == 3.0 and plates[0].center_m == (0.0, 2.0, 0.0)
        payload = Payload(
            instruments=[
                Telescope(
                    name="scope",
                    optics=TelescopeConfig(
                        aperture_m=0.5, focal_length_m=3.5, tube_length_m=1.2
                    ),
                )
            ]
        )
        apertures = telescope_surfaces(payload)
        assert len(apertures) == 1
        assert apertures[0].center_m == pytest.approx((1.2, 0.0, 0.0))
        assert apertures[0].area_m2 == pytest.approx(np.pi * 0.25**2)
        assert apertures[0].specular_reflectivity == 0.0
        guessed = guess_surfaces(panels, payload)
        assert len(guessed.surfaces) == 6 + 2 + 1
        assert guessed.center_of_mass_m == (-0.05, 0.0, 0.0)
        assert telescope_surfaces(None) == [] and panel_surfaces(None) == []


class TestTrackerSources:
    def test_the_tracker_sums_its_sources_and_reports_each(self) -> None:
        inertia = np.diag([10.0, 8.0, 6.0])
        model = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (1.0, 0.0, 0.0))])
        tracker = StoredMomentumTracker(
            inertia,
            sources=[
                GravityGradientTorque(inertia),
                SolarRadiationPressureTorque(model),
            ],
        )
        quaternion = attitude_to_quat(0.0, 0.0, 0.0)
        position = np.array([7000.0, 0.0, 0.0])
        first = tracker.update(
            utime=0.0,
            position_eci_km=position,
            attitude_quaternion_eci_to_body=quaternion,
            sun_eci_km=SUN,
        )
        assert set(first.torques_body_n_m) == {
            "gravity_gradient",
            "solar_radiation_pressure",
        }
        total = np.add(
            first.torques_body_n_m["gravity_gradient"],
            first.torques_body_n_m["solar_radiation_pressure"],
        )
        assert first.torque_body_n_m == pytest.approx(total)
        assert (
            first.gravity_gradient_torque_body_n_m
            == first.torques_body_n_m["gravity_gradient"]
        )
        srp = np.array(first.torques_body_n_m["solar_radiation_pressure"])
        assert np.linalg.norm(srp) > 0
        second = tracker.update(
            utime=10.0,
            position_eci_km=position,
            attitude_quaternion_eci_to_body=quaternion,
            sun_eci_km=SUN,
        )
        # Constant torque for ten seconds: the momentum is its impulse.
        assert second.stored_momentum_body_n_m_s == pytest.approx(total * 10.0)
        # In eclipse the pressure stops and only the gravity gradient remains.
        third = tracker.update(
            utime=20.0,
            position_eci_km=position,
            attitude_quaternion_eci_to_body=quaternion,
            sun_eci_km=SUN,
            sunlit=False,
        )
        assert third.torques_body_n_m["solar_radiation_pressure"] == (0.0, 0.0, 0.0)

    def test_the_default_source_is_the_gravity_gradient(self) -> None:
        tracker = StoredMomentumTracker(np.diag([10.0, 8.0, 6.0]))
        assert [source.name for source in tracker.sources] == ["gravity_gradient"]
        sample = tracker.update(
            utime=0.0,
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            attitude_quaternion_eci_to_body=attitude_to_quat(0.0, 45.0, 0.0),
        )
        assert list(sample.torques_body_n_m) == ["gravity_gradient"]


ORBIT = np.array([EARTH_RADIUS_KM + 600.0, 0.0, 0.0])
"""Over the equator of an untilted dipole, where the field is along +Z."""


def _unloading(max_dipole: float = 10.0, time_constant: float = 1800.0):
    return MagnetorquerUnloading(
        TiltedDipoleField(pole_deg=(90.0, 0.0)), max_dipole, time_constant
    )


def _sample(momentum, rotation=None) -> OrbitSample:
    return OrbitSample(
        utime=0.0,
        position_eci_km=ORBIT,
        rotation_eci_to_body=np.eye(3) if rotation is None else rotation,
        position_ecef_km=ORBIT,
        stored_momentum_body_n_m_s=np.asarray(momentum, dtype=np.float64),
    )


class TestMagnetorquerUnloading:
    def test_unloading_opposes_the_momentum_across_the_field(self) -> None:
        """A small momentum across the field is removed over the time
        constant; nothing can be done about the part along the field."""
        source = _unloading(max_dipole=1e6)
        field = _sample([0.0, 0.0, 0.0]).geomagnetic_field_body_t(source.field)
        assert field is not None and field[2] > 0 and abs(field[0]) < 1e-20
        across = source.torque_body_n_m(_sample([0.1, 0.0, 0.0]))
        assert across == pytest.approx([-0.1 / 1800.0, 0.0, 0.0], rel=1e-9)
        along = source.torque_body_n_m(_sample([0.0, 0.0, 0.1]))
        assert along == pytest.approx([0.0, 0.0, 0.0], abs=1e-20)
        mixed = source.torque_body_n_m(_sample([0.1, 0.0, 0.1]))
        assert mixed == pytest.approx([-0.1 / 1800.0, 0.0, 0.0], rel=1e-9)

    def test_the_dipole_saturates(self) -> None:
        """A large momentum asks for more dipole than the torquers have: the
        torque is the dipole cap times the field, against the momentum."""
        source = _unloading(max_dipole=10.0)
        sample = _sample([5.0, 0.0, 0.0])
        field = sample.geomagnetic_field_body_t(source.field)
        assert field is not None
        dipole = source.dipole_body_a_m2(sample.stored_momentum_body_n_m_s, field)
        assert np.linalg.norm(dipole) == pytest.approx(10.0)
        torque = source.torque_body_n_m(sample)
        assert np.linalg.norm(torque) == pytest.approx(10.0 * np.linalg.norm(field))
        assert torque[0] < 0 and abs(torque[1]) < 1e-15 and abs(torque[2]) < 1e-15

    def test_nothing_without_a_field_or_momentum(self) -> None:
        source = _unloading()
        blind = OrbitSample(
            utime=0.0,
            position_eci_km=ORBIT,
            rotation_eci_to_body=np.eye(3),
            stored_momentum_body_n_m_s=np.array([1.0, 0.0, 0.0]),
        )
        assert np.all(source.torque_body_n_m(blind) == 0.0)
        with pytest.raises(ValueError, match="positive"):
            _unloading(max_dipole=0.0)

    def test_the_tracker_unloads_what_it_stores_across_the_field(self) -> None:
        """Starting with momentum across the field and no disturbance, the
        tracker's stored momentum decays towards zero; starting with it along
        the field, nothing changes. The field is reported with the sample."""
        source = _unloading(max_dipole=1e6, time_constant=100.0)
        quaternion = attitude_to_quat(0.0, 0.0, 0.0)
        tracker = StoredMomentumTracker(
            np.eye(3), initial_momentum_body_n_m_s=(1.0, 0.0, 0.0), sources=[source]
        )
        norms = []
        for k in range(0, 601, 10):
            sample = tracker.update(
                utime=float(k),
                position_eci_km=ORBIT,
                attitude_quaternion_eci_to_body=quaternion,
                position_ecef_km=ORBIT,
            )
            norms.append(sample.stored_momentum_norm_n_m_s)
        assert sample.geomagnetic_field_body_t is not None
        assert norms[0] == pytest.approx(1.0) and norms[-1] < 0.01
        assert all(a >= b for a, b in zip(norms, norms[1:]))
        assert set(sample.torques_body_n_m) == {"magnetorquer_unloading"}
        held = StoredMomentumTracker(
            np.eye(3), initial_momentum_body_n_m_s=(0.0, 0.0, 1.0), sources=[source]
        )
        for k in (0.0, 10.0):
            sample = held.update(
                utime=k,
                position_eci_km=ORBIT,
                attitude_quaternion_eci_to_body=quaternion,
                position_ecef_km=ORBIT,
            )
        assert sample.stored_momentum_norm_n_m_s == pytest.approx(1.0)


class TestResidualDipole:
    def test_the_torque_is_the_dipole_across_the_field(self) -> None:
        field = TiltedDipoleField(pole_deg=(90.0, 0.0))
        source = ResidualDipoleTorque(field, (1.0, 0.0, 0.0))
        sample = _sample([0.0, 0.0, 0.0])
        b = sample.geomagnetic_field_body_t(field)
        assert b is not None
        assert source.torque_body_n_m(sample) == pytest.approx(
            np.cross([1.0, 0.0, 0.0], b)
        )
        assert np.all(
            source.torque_body_n_m(
                OrbitSample(
                    utime=0.0, position_eci_km=ORBIT, rotation_eci_to_body=np.eye(3)
                )
            )
            == 0.0
        )
