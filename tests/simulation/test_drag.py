"""Aerodynamic drag as a torque source: the torque on the surfaces, the
relative velocity, the configuration, and the run's telemetry."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import DITL, QueueDITL, create_ditl
from conops.common.geodesy import EARTH_ROTATION_RATE_RAD_S
from conops.common.vector import attitude_to_quat
from conops.config import (
    AtmosphereConfig,
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    StoredMomentumConfig,
    Surface,
    SurfaceModel,
)
from conops.schedulers import PriorityScheduler
from conops.schedulers.momentum import MomentumForecast
from conops.simulation.atmosphere import ExponentialAtmosphere
from conops.simulation.momentum import (
    AerodynamicDragTorque,
    OrbitSample,
    StoredMomentumTracker,
    aerodynamic_drag_torque_body,
    build_torque_sources,
)
from conops.targets import Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
HOURS = 3
END = BEGIN + timedelta(hours=HOURS)
RADIUS_KM = 6371.2 + 600.0


def _plate(center, normal, area=2.0, cd=2.2) -> Surface:  # type: ignore[no-untyped-def]
    return Surface(
        name="plate", center_m=center, normal=normal, area_m2=area, drag_coefficient=cd
    )


class TestDragTorque:
    def test_a_plate_ahead_of_the_centre_of_mass_is_pushed_back(self) -> None:
        """A plate facing the flow, offset from the centre of mass along y, takes
        a force against the flow; the torque is the arm crossed with it."""
        surfaces = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (1.0, 0.0, 0.0))])
        flow = np.array([7.5, 0.0, 0.0])  # km/s along +x: the plate faces it
        density = 1e-12
        torque = aerodynamic_drag_torque_body(flow, density, surfaces)
        force = -0.5 * density * 7500.0**2 * 2.2 * 2.0
        assert torque == pytest.approx(np.cross([0.0, 0.5, 0.0], [force, 0.0, 0.0]))
        # Facing away: nothing. Oblique: the projected area.
        away = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (-1.0, 0.0, 0.0))])
        assert np.allclose(aerodynamic_drag_torque_body(flow, density, away), 0.0)
        oblique = SurfaceModel(
            surfaces=[
                _plate((0.0, 0.5, 0.0), (np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0))
            ]
        )
        assert np.linalg.norm(
            aerodynamic_drag_torque_body(flow, density, oblique)
        ) == pytest.approx(np.linalg.norm(torque) * np.cos(np.pi / 3))

    def test_no_flow_or_no_air_gives_nothing(self) -> None:
        surfaces = SurfaceModel(surfaces=[_plate((0.0, 0.5, 0.0), (1.0, 0.0, 0.0))])
        assert np.allclose(
            aerodynamic_drag_torque_body([0.0, 0.0, 0.0], 1e-12, surfaces), 0.0
        )
        assert np.allclose(
            aerodynamic_drag_torque_body([7.5, 0.0, 0.0], 0.0, surfaces), 0.0
        )
        with pytest.raises(ValueError):
            aerodynamic_drag_torque_body([7.5, 0.0, 0.0], float("nan"), surfaces)

    def test_the_flow_is_relative_to_the_rotating_atmosphere(self) -> None:
        position = np.array([RADIUS_KM, 0.0, 0.0])
        velocity = np.array([0.0, 7.5, 0.0])
        sample = OrbitSample(
            utime=0.0,
            position_eci_km=position,
            rotation_eci_to_body=np.eye(3),
            velocity_eci_km_s=velocity,
            position_ecef_km=position,
        )
        flow = sample.flow_eci_km_s()
        assert flow is not None
        assert flow == pytest.approx(
            velocity - np.cross([0.0, 0.0, EARTH_ROTATION_RATE_RAD_S], position)
        )
        assert flow[1] < 7.5
        no_velocity = OrbitSample(
            utime=0.0, position_eci_km=position, rotation_eci_to_body=np.eye(3)
        )
        assert no_velocity.flow_eci_km_s() is None

    def test_the_source_reports_the_density_it_used(self) -> None:
        # The plate faces the flow along y, ahead of the centre of mass along x.
        surfaces = SurfaceModel(surfaces=[_plate((0.5, 0.0, 0.0), (0.0, 1.0, 0.0))])
        source = AerodynamicDragTorque(surfaces, ExponentialAtmosphere())
        position = np.array([RADIUS_KM, 0.0, 0.0])
        sample = OrbitSample(
            utime=0.0,
            position_eci_km=position,
            rotation_eci_to_body=np.eye(3),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
            position_ecef_km=position,
        )
        torque = source.torque_body_n_m(sample)
        assert source.last_density_kg_m3 is not None and source.last_density_kg_m3 > 0.0
        assert np.linalg.norm(torque) > 0.0
        tracker = StoredMomentumTracker(
            np.diag([1000.0, 1200.0, 800.0]), sources=[source]
        )
        reported = tracker.update(
            utime=0.0,
            position_eci_km=position,
            attitude_quaternion_eci_to_body=attitude_to_quat(0.0, 0.0, 0.0),
            velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
            position_ecef_km=position,
        )
        assert reported.atmosphere_density_kg_m3 == pytest.approx(
            source.last_density_kg_m3
        )
        assert "aerodynamic_drag" in reported.torques_body_n_m


def _config(**momentum: object) -> MissionConfig:
    ephem = TLEEphemeris(tle="examples/example.tle", begin=BEGIN, end=END, step_size=60)
    config = MissionConfig(
        constraint=Constraint(
            sun_constraint=SunConstraint(min_angle=45),
            earth_constraint=EarthLimbConstraint(min_angle=10),
        ),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=np.diag([1000.0, 1200.0, 800.0]),
            attitude_control=AttitudeControlSystem(
                max_slew_rate=0.5,
                slew_acceleration=0.25,
                settle_time=30.0,
                stored_momentum=StoredMomentumConfig(**momentum),  # type: ignore[arg-type]
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = ephem
    return config


def _held(config: MissionConfig) -> Pointing:
    target = Pointing(
        config=config,
        obsid=1,
        name="Held",
        ra=80.0,
        dec=-20.0,
        merit=100.0,
        ss_min=600,
        ss_max=HOURS * 3600,
    )
    target.exptime = HOURS * 3600
    return target


class TestConfiguration:
    def test_drag_enables_tracking_and_builds_its_source(self) -> None:
        config = _config(aerodynamic_drag_enabled=True)
        momentum = config.spacecraft_bus.attitude_control.stored_momentum
        assert momentum.enabled
        sources = build_torque_sources(config)
        assert [s.name for s in sources] == ["aerodynamic_drag"]
        assert isinstance(sources[0], AerodynamicDragTorque)
        assert sources[0].atmosphere.name == "table"
        assert not StoredMomentumConfig().enabled

    def test_the_field_model_is_selected(self) -> None:
        config = _config(residual_dipole_body_a_m2=(1.0, 0.0, 0.0))
        assert build_torque_sources(config)[0].field.name == "dipole"  # type: ignore[attr-defined]
        pytest.importorskip("ppigrf")
        config = _config(
            residual_dipole_body_a_m2=(1.0, 0.0, 0.0), geomagnetic_field="igrf"
        )
        assert build_torque_sources(config)[0].field.name == "igrf"  # type: ignore[attr-defined]
        with pytest.raises(ValueError):
            StoredMomentumConfig(geomagnetic_field="wrong")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            AtmosphereConfig(model="other")  # type: ignore[arg-type]


class TestRuns:
    @pytest.mark.parametrize("model", ["table", "msis"])
    def test_a_run_stores_the_drag_impulse_and_reports_the_density(
        self, model: str
    ) -> None:
        if model == "msis":
            pytest.importorskip("pymsis")
        config = _config(
            aerodynamic_drag_enabled=True, atmosphere=AtmosphereConfig(model=model)
        )
        ditl = create_ditl(config, [_held(config)], BEGIN, END)
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = 60
        assert ditl.calc()
        hk = ditl.telemetry.housekeeping
        norms = np.array(hk.stored_momentum_norm_n_m_s, dtype=float)
        assert norms[-1] > 0.0
        densities = [d for d in hk.atmosphere_density_kg_m3 if d is not None]
        assert densities and all(1e-15 < d < 1e-10 for d in densities)
        torques = [t for t in hk.disturbance_torques_body_n_m if t]
        assert all("aerodynamic_drag" in t for t in torques)
        # Drag at 600 km is small beside the gravity gradient: a thousandth
        # of a newton-metre-second over three hours, not more.
        assert norms[-1] < 0.1

    def test_the_planners_forecast_matches_the_run(self) -> None:
        config = _config(aerodynamic_drag_enabled=True, gravity_gradient_enabled=True)
        target = _held(config)
        scheduler = PriorityScheduler(config, [target], BEGIN, END, step_size=60)
        plan = scheduler.schedule()
        assert scheduler._momentum is not None
        times, norms = scheduler._momentum.profile(scheduler.timeline)
        assert isinstance(scheduler._momentum, MomentumForecast)
        ditl = DITL(
            config=_config(
                aerodynamic_drag_enabled=True, gravity_gradient_enabled=True
            ),
            ephem=config.constraint.ephem,
            plan=plan,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = 60
        assert ditl.calc()
        executed = np.array(
            ditl.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        assert executed[-1] == pytest.approx(norms[-1], rel=0.05, abs=0.01)
