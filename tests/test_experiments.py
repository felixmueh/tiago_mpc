"""The experiment tasks, and reach and figure8 end to end on a fake MPC (CPU only)."""
from dataclasses import replace
from functools import partial
import json

import numpy as np
import pinocchio as pin
import pytest

from experiments import figure8 as f8, reach as rc
from tiago_mpc.loops import Fixed, run_simulation
from tiago_mpc.plan import Plan, State
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import ARM_URDF, Q_START, RIGHT_ARM_JOINTS, fk, urdf_friction


def test_figure8_reference_includes_edge_clearance_and_leaves_orientation_free():
    task = f8.Figure8(start_m=(.3, -.2, .1))
    times = 1.3 + np.arange(7)*.002
    phase = 2*np.pi*times/24.
    expected = np.tile(task.start_m, (len(times), 1))
    expected[:, 0] += .02 + .5*.28*(1-np.cos(phase))
    expected[:, 1] += .09*np.sin(2*phase)
    reference = task.reference(1.3, 7, .002)
    np.testing.assert_allclose(reference[:, :3], expected, atol=2e-8)
    assert reference.dtype == np.float32 and not reference[:, 3:].any()
    assert f8.Figure8(start_m=(.5, .1, .2)).reference(6., 2, .01)[0, 0] == pytest.approx(.66)


def test_reach_is_done_within_both_tolerances():
    task = rc.Reach(tuple(fk(Q_START) + [.04, 0., 0.]))
    assert task.done(State(0., Q_START, np.full(7, .1)), 0.)
    assert not task.done(State(0., Q_START, np.full(7, .2)), 0.)
    assert not replace(task, position_tol_m=.03).done(State(0., Q_START, np.zeros(7)), 0.)


class Mpc:
    """Gravity-holding plans from each requested state."""
    def __init__(self, params):
        self.N, self.dt = params.N, params.dt_s
        model = pin.buildModelFromUrdf(str(ARM_URDF))
        self.model, self.data, self.plan = model, model.createData(), None

    def request(self, state, reference, updates=None):
        u = pin.rnea(self.model, self.data, state.q, np.zeros(7), np.zeros(7))
        stats = dict(solve_time_us=1., wall_s=.001, sqp=1, pcg=2, admm=3, admm_linsys=4,
                     zero_pcg_rows=0, admm_linsys_mode="pcg", row_viol=None)
        self.plan = Plan(state.t, self.dt, np.tile(np.r_[state.q, state.qd], (self.N, 1)),
                         np.tile(u, (self.N - 1, 1)), stats)

    def result(self, block=False):
        plan, self.plan = self.plan, None
        return plan

    def close(self):
        pass


@pytest.mark.parametrize("name", ["reach", "figure8"])
def test_experiment_records_summary_trajectory_and_replay(tmp_path, monkeypatch, name):
    module = rc if name == "reach" else f8
    monkeypatch.setattr(module, "MpcWorker", Mpc)
    params = replace(module.TIAGO if name == "reach" else f8.FIGURE8, N=4)
    if name == "reach":
        task = rc.Reach(timeout_s=.05)
        record = rc.reach(SimPlant(substep=.003), task=task, params=params, out=tmp_path / name,
                          run=partial(run_simulation, latency=Fixed(.01)))
    else:
        task = f8.Figure8(period_s=.05)
        record = f8.figure8(SimPlant(substep=1e-4, friction=urdf_friction()), task=task,
                            params=params, out=tmp_path / name)
    summary = json.loads((tmp_path / name / "summary.json").read_text())
    assert summary["status"] == ("timeout" if name == "reach" else "completed")
    assert summary["params"]["N"] == 4 and summary["task"]["type"] == type(task).__name__
    assert summary["plant"]["type"] == "SimPlant" and summary["limits_pass"]
    assert summary["ticks"] == 6 and summary["plans"] == 5
    with np.load(tmp_path / name / "trajectory.npz") as saved:
        np.testing.assert_allclose(saved["time_s"], np.arange(6) * .01, atol=1e-12)
        # Fixed(10 ms) latency: the first plan takes effect at the second tick.
        assert np.isnan(saved["command"][-1]).all()
        assert np.isnan(saved["command"][0]).all() == (name == "reach")
        assert saved["plan_states"].shape[1:] == (4, 14) and saved["plan_sqp"].sum() == summary["plans"]
    with np.load(tmp_path / name / "replay.npz") as replay:
        metadata = json.loads(str(replay["metadata_json"]))
        assert metadata["joint_names"] == list(RIGHT_ARM_JOINTS) and len(metadata["predictions"]) == summary["plans"]
        assert metadata["time_basis"] == "simulation"
    assert record.out == tmp_path / name


def test_reach_commands_every_tick_and_updates_every_third(tmp_path, monkeypatch):
    requests = []

    class Counting(Mpc):
        def request(self, state, reference, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    monkeypatch.setattr(rc, "MpcWorker", Counting)
    params = replace(rc.TIAGO, N=8)  # 10 ms knots, 70 ms plans
    rc.reach(SimPlant(substep=.002), task=rc.Reach(timeout_s=.06), params=params, out=tmp_path / "r",
             command_period=.002, update_period=.006)
    with np.load(tmp_path / "r" / "trajectory.npz") as saved:
        np.testing.assert_allclose(saved["time_s"], np.arange(31) * .002, atol=1e-12)
        assert np.isfinite(saved["command"][:-1]).all()  # a command at every tick but the last
    np.testing.assert_allclose(requests, np.arange(10) * .006, atol=1e-12)  # every third tick


def test_duration_caps_the_tasks_own(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "MpcWorker", Mpc)
    assert rc.arguments("", ["--duration", ".02"]).duration == .02
    record = rc.reach(SimPlant(substep=.003), task=rc.Reach(timeout_s=.06), params=replace(rc.TIAGO, N=4),
                      out=tmp_path / "r", duration=.02)
    assert record.summary["status"] == "timeout" and record.summary["duration_cap_s"] == .02
    assert len(record.ticks) == 3  # 0, 10 and 20 ms
