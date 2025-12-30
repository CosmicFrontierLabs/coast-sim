import numpy as np
from pymsis.msis import Variable  # type: ignore[import-untyped]

from conops.simulation.msis_adapter import MsisAdapter, MsisConfig


def test_msis_adapter_returns_density_within_range(monkeypatch):
    cfg = MsisConfig(use_msis_density=True)
    adapter = MsisAdapter(cfg)

    def fake_calculate(*_args, **_kwargs):
        out = np.zeros((1, 11), dtype=float)
        out[0, int(Variable.MASS_DENSITY)] = 1e-12
        return out

    monkeypatch.setattr(
        "conops.simulation.msis_adapter.calculate",
        fake_calculate,
    )

    rho = adapter.density(
        utime=0.0,
        alt_m=500e3,
        lat_deg=0.0,
        lon_deg=0.0,
        table_density=1e-15,
    )
    assert rho == 1e-12


def test_msis_adapter_out_of_range_returns_none(monkeypatch):
    cfg = MsisConfig(use_msis_density=True)
    adapter = MsisAdapter(cfg)

    def fake_calculate(*_args, **_kwargs):
        out = np.zeros((1, 11), dtype=float)
        out[0, int(Variable.MASS_DENSITY)] = 1e-9
        return out

    monkeypatch.setattr(
        "conops.simulation.msis_adapter.calculate",
        fake_calculate,
    )

    rho = adapter.density(
        utime=0.0,
        alt_m=500e3,
        lat_deg=0.0,
        lon_deg=0.0,
        table_density=1e-15,
    )
    assert rho is None
