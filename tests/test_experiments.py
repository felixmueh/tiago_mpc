"""The experiment tasks, and reach and figure8 end to end on a fake MPC (CPU only)."""
from dataclasses import replace
import importlib.metadata
import json
from types import SimpleNamespace as NS

import numpy as np
import pinocchio as pin
import pytest

from experiments import common, figure8 as f8, reach as rc
from tiago_mpc.loops import run_stepped
from tiago_mpc import record as rec
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
                          run=run_stepped, solve_latency=.01, command_period=.01)
    else:
        task = f8.Figure8(period_s=.05)
        record = f8.figure8(SimPlant(friction=urdf_friction()), Mpc(replace(f8.FIGURE8, N=4)),
                            task=task, out=tmp_path / name, command_period=.01, solve_latency=0.)
    summary = json.loads((tmp_path / name / "summary.json").read_text())
    assert summary["status"] == ("timeout" if name == "reach" else "completed")
    assert summary["params"]["N"] == 4 and summary["task"]["type"] == type(task).__name__
    assert summary["plant"]["type"] == "SimPlant" and summary["limits_pass"]
    if name == "figure8":
        assert summary["plant"]["substep_s"] == .001
        assert summary["plant"]["friction"]["law"] == dict(name="stick_slip")
        assert summary["params"]["friction"]["law"] == dict(width_rad_s=.03, name="tanh")  # predictor stays smooth
    assert summary["ticks"] == 6 and summary["plans"] == 5
    assert summary["timing"] == dict(loop="run_stepped", command_period_s=.01, update_period_s=.01,
                                     solve_latency=.01 if name == "reach" else 0., late_ticks=0, overruns=0)
    assert summary["plant"]["time"] == "stepped"
    with np.load(tmp_path / name / "trajectory.npz") as saved:
        np.testing.assert_allclose(saved["time_s"], np.arange(6) * .01, atol=1e-12)
        np.testing.assert_array_equal(saved["state_time_s"], saved["time_s"])  # stepped time
        # The reach's 10 ms latency: the first plan takes effect at the second tick.
        assert np.isnan(saved["command"][-1]).all()
        assert np.isnan(saved["command"][0]).all() == (name == "reach")
        assert saved["plan_states"].shape[1:] == (4, 14) and saved["plan_sqp"].sum() == summary["plans"]
    with np.load(tmp_path / name / "replay.npz") as replay:
        metadata = json.loads(str(replay["metadata_json"]))
        assert metadata["joint_names"] == list(RIGHT_ARM_JOINTS) and len(metadata["predictions"]) == summary["plans"]
        assert metadata["time_basis"] == "simulation"
        assert MpcParams.from_dict(metadata["provenance"]["params"]) == record.params
    assert record.out == tmp_path / name
    assert summary["gato"] == rec.gato_source()


def test_gato_source_names_the_installed_gato_and_an_editable_checkouts_commit(monkeypatch):
    import gato

    source = rec.gato_source()
    assert source["version"] == importlib.metadata.version("gato-mpc") and source["file"] == gato.__file__
    assert "commit" not in source  # a PyPI wheel has no direct_url.json; uv.lock pins it
    url = json.dumps(dict(url=rec.ROOT.as_uri(), dir_info=dict(editable=True)))  # this checkout as the fork
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: NS(version="9.9", read_text=lambda f: url))
    source = rec.gato_source()
    assert source["version"] == "9.9" and source["commit"] == rec._git("rev-parse", "HEAD")
    assert source["status"] == rec._git("status", "--short")


def test_reach_commands_every_tick_and_updates_every_third(tmp_path):
    requests = []

    class Counting(Mpc):
        def request(self, state, reference, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    mpc = Counting(replace(rc.tm.TIAGO, N=8))  # 10 ms knots, 70 ms plans
    rc.reach(SimPlant(substep=.002), mpc, task=rc.Reach(timeout_s=.06), out=tmp_path / "r",
             command_period=.002, update_period=.006, solve_latency=0.)
    with np.load(tmp_path / "r" / "trajectory.npz") as saved:
        np.testing.assert_allclose(saved["time_s"], np.arange(31) * .002, atol=1e-12)
        assert np.isfinite(saved["command"][:-1]).all()  # a command at every tick but the last
    np.testing.assert_allclose(requests, np.arange(10) * .006, atol=1e-12)  # every third tick


def test_duration_caps_the_tasks_own(tmp_path):
    assert common.arguments("", ["--duration", ".02"]).duration == .02
    record = rc.reach(SimPlant(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4)), task=rc.Reach(timeout_s=.06),
                      out=tmp_path / "r", duration=.02, solve_latency=0.)
    assert record.summary["status"] == "timeout" and record.summary["duration_cap_s"] == .02
    assert len(record.ticks) == 11  # 0, 2, ..., 20 ms: the default command period


def test_a_run_without_solves_reports_zero_work(tmp_path):
    """The goal met at the start, or a zero duration: one tick, no solve."""
    params = replace(rc.tm.TIAGO, N=4)
    done = rc.reach(SimPlant(substep=.003), Mpc(params), task=rc.Reach(tuple(fk(Q_START))), out=tmp_path / "d",
                    solve_latency=0.)
    zero = rc.reach(SimPlant(substep=.003), Mpc(params), out=tmp_path / "z", duration=0., solve_latency=0.)
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


def test_update_period_0_stays_back_to_back(tmp_path):
    requests = []

    class Counting(Mpc):
        def request(self, state, reference, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    record = rc.reach(SimPlant(substep=.002), Counting(replace(rc.tm.TIAGO, N=8)), task=rc.Reach(timeout_s=.01),
                      out=tmp_path / "r", command_period=.002, update_period=0., solve_latency=0.)
    assert record.summary["timing"]["update_period_s"] == 0.
    np.testing.assert_allclose(requests, np.arange(5) * .002, atol=1e-12)  # a request at every tick


@pytest.mark.parametrize("argv, plant, latency", [([], "sim", 0.), (["--solve-latency", "measured"], "sim", "measured"),
                                                  (["--solve-latency", ".005"], "sim", .005),
                                                  (["--plant", "rehearsal"], "rehearsal", None)])
def test_the_command_line_sets_the_timing(argv, plant, latency):
    args = common.arguments("", argv)
    assert args.plant == plant and args.solve_latency == latency


@pytest.mark.parametrize("argv", [["--plant", "robot", "--solve-latency", "0"], ["--plant", "rehearsal", "--solve-latency", ".01"],
                                  ["--solve-latency", "-1"], ["--solve-latency", "nan"], ["--solve-latency", "real"]])
def test_the_command_line_refuses_a_latency_on_wall_time_or_an_invalid_one(argv):
    with pytest.raises(SystemExit):
        common.arguments("", argv)


@pytest.mark.parametrize("plant, latency, solver, budget", [("sim", 0., "Mpc", None), ("sim", .005, "Mpc", None),
                                                            ("sim", "measured", "Mpc", .01),
                                                            ("rehearsal", None, "MpcProcess", .01)])
def test_launch_picks_the_loop_the_mpc_and_the_budget(monkeypatch, plant, latency, solver, budget):
    from types import SimpleNamespace as NS
    seen = {}

    def compile(name):
        return lambda params: NS(params=params, close=lambda: None, kind=name)

    monkeypatch.setattr(rc.tm, "Mpc", NS(compile=compile("Mpc")))
    monkeypatch.setattr(rc.tm, "MpcProcess", NS(compile=compile("MpcProcess")))

    def experiment(plant, mpc, run, solve_latency, **kw):
        seen.update(plant=plant, mpc=mpc, run=run, solve_latency=solve_latency)
        return NS(summary={}, out=None)

    argv = ["--plant", plant] + ([] if latency in (None, 0.) else ["--solve-latency", str(latency)])
    common.launch(common.arguments("", argv), experiment, lambda: NS(close=lambda: None), rc.tm.TIAGO)
    assert seen["mpc"].kind == solver and seen["mpc"].params.max_sqp_time_s == budget
    assert seen["run"] is (rc.tm.run_stepped if plant == "sim" else rc.tm.run_real_time)
    assert seen["solve_latency"] == latency
