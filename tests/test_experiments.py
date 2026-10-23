"""The experiment tasks, and reach and figure8 end to end on a fake MPC (CPU only)."""
from dataclasses import replace
from functools import partial
import json

import numpy as np
import pinocchio as pin
import pytest

from experiments import common, figure8 as f8, reach as rc
from tiago_mpc.loops import Fixed, run_simulation
from tiago_mpc.mpc import MpcParams
from tiago_mpc.plan import Plan, State
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import Q_START, RIGHT_ARM_JOINTS, arm_urdf, fk, urdf_friction


def test_figure8_reference_includes_edge_clearance_and_leaves_orientation_free():
    task = f8.Figure8(start_m=(.3, -.2, .1))
    times = 1.3 + np.arange(7)*.002
    phase = 2*np.pi*times/24.
    expected = np.tile(task.start_m, (len(times), 1))
    expected[:, 0] += .02 + .5*.28*(1-np.cos(phase))
    expected[:, 1] += .09*np.sin(2*phase)
    reference, updates = task.at(1.3, 7, .002)
    np.testing.assert_allclose(reference[:, :3], expected, atol=2e-8)
    assert reference.dtype == np.float32 and not reference[:, 3:].any() and updates is None
    assert f8.Figure8(start_m=(.5, .1, .2)).at(6., 2, .01)[0][0, 0] == pytest.approx(.66)


def test_figure8_is_done_at_its_period_despite_summed_command_periods():
    t = 0.
    for _ in range(400):
        t += .01  # 3.9999999999999587
    assert f8.Figure8(period_s=4.).done(None, t) and not f8.Figure8(period_s=4.).done(None, 3.99)


def test_reach_is_done_within_both_tolerances():
    task = rc.Reach(tuple(fk(Q_START) + [.04, 0., 0.]))
    assert task.done(State(0., Q_START, np.full(7, .1)), 0.)
    assert not task.done(State(0., Q_START, np.full(7, .2)), 0.)
    assert not replace(task, position_tol_m=.03).done(State(0., Q_START, np.zeros(7)), 0.)


class Mpc:
    """Gravity-holding plans from each requested state."""
    def __init__(self, params):
        self.params, self.N, self.dt = params, params.N, params.dt_s
        model = pin.buildModelFromUrdf(str(arm_urdf()))
        self.model, self.data, self.plan = model, model.createData(), None

    def warmup(self, state, reference):
        pass

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


@pytest.mark.filterwarnings("ignore:a plan took effect")  # the reach's 10 ms latency
@pytest.mark.parametrize("name", ["reach", "figure8"])
def test_experiment_records_summary_trajectory_and_replay(tmp_path, name):
    if name == "reach":
        task = rc.Reach(timeout_s=.05)
        record = rc.reach(SimPlant(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4)), task=task, out=tmp_path / name,
                          run=partial(run_simulation, latency=Fixed(.01)), command_period=.01)
    else:
        task = f8.Figure8(period_s=.05)
        record = f8.figure8(SimPlant(friction=urdf_friction()), Mpc(replace(f8.FIGURE8, N=4)),
                            task=task, out=tmp_path / name, command_period=.01)
    summary = json.loads((tmp_path / name / "summary.json").read_text())
    assert summary["status"] == ("timeout" if name == "reach" else "completed")
    assert summary["params"]["N"] == 4 and summary["task"]["type"] == type(task).__name__
    assert summary["plant"]["type"] == "SimPlant" and summary["limits_pass"]
    if name == "figure8":
        assert summary["plant"]["substep_s"] == .001
        assert summary["plant"]["coulomb_law"] == "constraint"
        assert summary["params"]["friction"]["width_rad_s"] == .03  # predictor stays smooth
    assert summary["ticks"] == 6 and summary["plans"] == 5
    with np.load(tmp_path / name / "trajectory.npz") as saved:
        np.testing.assert_allclose(saved["time_s"], np.arange(6) * .01, atol=1e-12)
        np.testing.assert_array_equal(saved["state_time_s"], saved["time_s"])  # simulation clock
        # Fixed(10 ms) latency: the first plan takes effect at the second tick.
        assert np.isnan(saved["command"][-1]).all()
        assert np.isnan(saved["command"][0]).all() == (name == "reach")
        assert saved["plan_states"].shape[1:] == (4, 14) and saved["plan_sqp"].sum() == summary["plans"]
    with np.load(tmp_path / name / "replay.npz") as replay:
        metadata = json.loads(str(replay["metadata_json"]))
        assert metadata["joint_names"] == list(RIGHT_ARM_JOINTS) and len(metadata["predictions"]) == summary["plans"]
        assert metadata["time_basis"] == "simulation"
        assert MpcParams.from_dict(metadata["provenance"]["params"]) == record.params
    assert record.out == tmp_path / name


def test_reach_commands_every_tick_and_updates_every_third(tmp_path):
    requests = []

    class Counting(Mpc):
        def request(self, state, reference, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    mpc = Counting(replace(rc.tm.TIAGO, N=8))  # 10 ms knots, 70 ms plans
    rc.reach(SimPlant(substep=.002), mpc, task=rc.Reach(timeout_s=.06), out=tmp_path / "r",
             command_period=.002, update_period=.006)
    with np.load(tmp_path / "r" / "trajectory.npz") as saved:
        np.testing.assert_allclose(saved["time_s"], np.arange(31) * .002, atol=1e-12)
        assert np.isfinite(saved["command"][:-1]).all()  # a command at every tick but the last
    np.testing.assert_allclose(requests, np.arange(10) * .006, atol=1e-12)  # every third tick


def test_duration_caps_the_tasks_own(tmp_path):
    assert common.arguments("", ["--duration", ".02"]).duration == .02
    record = rc.reach(SimPlant(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4)), task=rc.Reach(timeout_s=.06),
                      out=tmp_path / "r", duration=.02)
    assert record.summary["status"] == "timeout" and record.summary["duration_cap_s"] == .02
    assert len(record.ticks) == 11  # 0, 2, ..., 20 ms: the default command period


def test_a_run_without_solves_reports_zero_work(tmp_path):
    """The goal met at the start, or a zero duration: one tick, no solve."""
    params = replace(rc.tm.TIAGO, N=4)
    done = rc.reach(SimPlant(substep=.003), Mpc(params), task=rc.Reach(tuple(fk(Q_START))), out=tmp_path / "d")
    zero = rc.reach(SimPlant(substep=.003), Mpc(params), out=tmp_path / "z", duration=0.)
    for record, status in ((done, "arrived"), (zero, "timeout")):
        summary = json.loads((record.out / "summary.json").read_text())
        assert summary["status"] == status and summary["ticks"] == 1 and summary["solve_calls"] == 0
        assert summary["solve_time_median_ms"] is None and not any(summary["work_totals"].values())
        assert summary["max_abs_torque_nm"] == 0. and summary["limits_pass"]


def test_an_interrupted_run_is_saved_as_interrupted(tmp_path):
    def interrupted(*args, **kw):
        raise KeyboardInterrupt  # Ctrl-C, or SIGTERM through common._interrupt

    with pytest.raises(KeyboardInterrupt):
        rc.reach(SimPlant(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4)), run=interrupted, out=tmp_path / "r")
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["status"] == "interrupted"
