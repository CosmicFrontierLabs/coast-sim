import numpy as np

from conops.simulation.disturbance import DisturbanceConfig, DisturbanceModel


class DummyEphem:
    def __init__(self):
        self.gcrs_pv = type(
            "PV",
            (),
            {
                "position": [np.array([7000e3, 0.0, 0.0])],
                "velocity": [np.array([0.0, 7500.0, 0.0])],
            },
        )()

        class _XYZ:
            def __init__(self):
                self._v = np.array([1.0, 0.0, 0.0])

            def to_value(self):
                return self._v

        self.sun = [type("Sun", (), {"cartesian": type("C", (), {"xyz": _XYZ()})()})()]
        self.lat = [0.0]
        self.long = [0.0]

    def index(self, _time):
        return 0


def test_disturbance_model_compute_returns_components():
    ephem = DummyEphem()
    cfg = DisturbanceConfig(
        cp_offset_body=(0.25, 0.0, 0.0),
        residual_magnetic_moment=(0.05, 0.0, 0.0),
        drag_area_m2=1.0,
        solar_area_m2=1.0,
    )
    model = DisturbanceModel(ephem, cfg)

    torque, components = model.compute(
        utime=0.0,
        ra_deg=0.0,
        dec_deg=0.0,
        in_eclipse=False,
        moi_cfg=(5.0, 5.0, 5.0),
    )

    assert torque.shape == (3,)
    assert "total" in components
    assert "vector" in components
