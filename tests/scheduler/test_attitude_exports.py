"""Fine executed attitudes remain exportable from an actual planned run."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL
from conops.common.vector import quat_slerp
from conops.config import Payload, SpacecraftFrame, Telescope
from conops.schedulers import PriorityScheduler
from conops.targets import Plan

from .planning_scenario import BEGIN, T0, make_config, make_target


def test_actual_run_exports_declared_payload_frame_and_reloads_fine_sidecar(tmp_path):
    c = make_config(2)
    c.constraint.sun_constraint = None
    c.constraint.earth_constraint = None
    c.spacecraft_bus.frame = SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0))
    c.payload = Payload(instruments=[Telescope(name="Scope")])
    t = make_target(c, 100, 105, 10, minutes=10, snapshot=10, ss_min=10)
    end = BEGIN + timedelta(hours=2)
    plan = PriorityScheduler(c, [t], BEGIN, end, include_passes=False).schedule()
    run = DITL(c, plan=plan, begin=BEGIN, end=end)
    with pytest.raises(ValueError, match="Run the simulation"):
        run.sample_attitude()
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    data = run.sample_attitude(begin=T0, end=T0 + 600, interval_seconds=1)
    assert len(data.samples) >= 601
    assert data.spacecraft_frame == c.spacecraft_bus.frame
    assert run.plan.attitude_timeseries.spacecraft_frame == c.spacecraft_bus.frame

    def q(sample):
        return np.asarray([sample.quat_w, sample.quat_x, sample.quat_y, sample.quat_z])

    exact = next(s for s in data.samples if s.utime == T0 + 10)
    coarse = run.telemetry.housekeeping[:2]
    interpolated = quat_slerp(q(coarse[0]), q(coarse[1]), 10 / 60)
    assert abs(np.dot(q(exact), interpolated)) < 1 - 1e-8
    run.plan.attitude_timeseries = data
    loaded = Plan.load(run.plan.save(tmp_path / "plan.json"), load_timeseries=True)
    assert loaded.attitude_timeseries.samples == data.samples
    loaded.attitude_timeseries.save_stk(tmp_path / "payload.a")
    loaded.attitude_timeseries.save_euler_csv(tmp_path / "payload.csv")
    assert run.sample_attitude().samples[-1].utime == run.acs._executor.state.utime
