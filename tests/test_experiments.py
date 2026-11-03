"""The experiment tasks, run_experiment with reach's and figure8's evaluate end to end on a
fake MPC, and the mains' command line (CPU only)."""
from dataclasses import dataclass, replace
import importlib.metadata
import json
import signal
from types import SimpleNamespace as NS

import numpy as np
import pinocchio as pin
import pytest

from experiments import common, figure8 as f8, reach as rc
from experiments.latency_compensation import Compensated
from tiago_mpc import record as rec
from tiago_mpc.laws import SampleKnot
from tiago_mpc.loops import STOP, RealTime, VirtualTime
from tiago_mpc.prediction import LatencyCompensation
from tiago_mpc.interfaces import Plan, Prediction, State
from tiago_mpc.record import from_dict, load_run
from tiago_mpc.sim import SimConfig
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
    assert task.done(State(0., Q_START, np.full(7, .01)), 0.)
    assert not task.done(State(0., Q_START, np.full(7, .5)), 0.)
    assert not replace(task, position_tol_m=.03).done(State(0., Q_START, np.zeros(7)), 0.)


class Mpc:
    """Gravity-holding plans from each requested state; extra (Nm) is added to joint 7's
    last knot torque."""
    def __init__(self, params, extra=0.):
        self.params, self.N, self.dt, self.extra = params, params.N, params.dt_s, extra
        model = pin.buildModelFromUrdf(str(arm_urdf()))
        self.model, self.data, self.plan = model, model.createData(), None

    def warmup(self, state, reference):
        pass

    def request(self, state, reference, updates=None):
        u = pin.rnea(self.model, self.data, state.q, np.zeros(7), np.zeros(7))
        stats = dict(solve_time_us=1., wall_s=.001, sqp=1, pcg=2, admm=3, admm_linsys=4,
                     zero_pcg_rows=0, admm_linsys_mode="pcg", row_viol=None)
        torques = np.tile(u, (self.N - 1, 1))
        torques[-1, 6] += self.extra
        measured_t = state.measured_t if isinstance(state, Prediction) else state.t
        self.plan = Plan(state.t, self.dt, np.tile(np.r_[state.q, state.qd], (self.N, 1)), torques, stats,
                         measured_t=measured_t)

    def result(self, block=False):
        plan, self.plan = self.plan, None
        return plan

    def close(self):
        pass


@pytest.mark.parametrize("name", ["reach", "figure8"])
def test_experiment_records_its_run_and_replay(tmp_path, name):
    if name == "reach":
        task = rc.Reach(timeout_s=.05)
        record = common.run_experiment(SimConfig(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4, dt_s=.01)), task,
                                       SampleKnot(), VirtualTime(.01, .01, .01), out=tmp_path / name)
        rc.evaluate(record)
    else:
        task = f8.Figure8(period_s=.05)
        record = common.run_experiment(SimConfig(friction=urdf_friction(), integrator="euler"),
                                       Mpc(replace(f8.FIGURE8, N=4)), task, SampleKnot(), VirtualTime(.01, .01, 0.),
                                       out=tmp_path / name)
        f8.evaluate(record)
    summary = json.loads((tmp_path / name / "summary.json").read_text())
    config, metadata, results = summary["config"], summary["metadata"], summary["metadata"]["results"]
    assert summary["format"] == rec.FORMAT and set(summary) == {"format", "provenance", "config", "metadata"}
    assert metadata["status"] == ("timeout" if name == "reach" else "done") and metadata["reason"] is None
    assert metadata["end_t"] == pytest.approx(.05) and metadata["error"] is None and metadata["plant"] == {}
    assert set(config) == {"task", "params", "plant", "law", "timing", "duration"}
    assert config["params"]["N"] == 4 and config["task"]["type"] == type(task).__name__
    assert config["plant"]["type"] == "SimConfig" and config["law"] == dict(type="SampleKnot")
    assert config["timing"] == dict(type="VirtualTime", command_period=.01, update_period=.01,
                                    solve_latency_s=.01 if name == "reach" else 0., compensation=None)
    assert config["duration"] == .05 and results["limits_pass"]
    if name == "figure8":
        assert config["plant"]["substep"] == .001 and config["plant"]["integrator"] == "euler"
        assert config["plant"]["friction_law"] == dict(type="StickSlip")
        assert config["params"]["friction_law"] == dict(type="Tanh", width_rad_s=.1)  # predictor stays smooth
        assert set(results) >= {"metrics_full_cycle", "metrics_after_initial_1s"}
    else:
        assert set(results) >= {"initial_distance_m", "final_distance_m", "max_distance_m", "final_speed_m_s"}
    assert results["ticks"] == 6 and results["plans"] == 5 and results["late_ticks"] == results["overruns"] == 0
    assert results["late_plans"] == (4 if name == "reach" else 0)  # the reach's 10 ms latency; the last tick sends nothing
    run = load_run(tmp_path / name, classes=[type(task)])
    assert run.config == record.config and run.config["task"] == task and run.metadata == metadata
    np.testing.assert_allclose(run.ticks["t"], np.arange(6) * .01, atol=1e-12)
    np.testing.assert_array_equal(run.ticks["state_t"], run.ticks["t"])  # virtual time
    # The reach's 10 ms latency: the first plan takes effect at the second tick.
    assert np.isnan(run.ticks["command"][-1]).all()
    assert np.isnan(run.ticks["command"][0]).all() == (name == "reach")
    np.testing.assert_array_equal(run.ticks["plan"], [-1, 0, 1, 2, 3, 4] if name == "reach" else [0, 1, 2, 3, 4, 4])
    np.testing.assert_array_equal(run.ticks["update"], [0, 1, 2, 3, 4, -1])
    np.testing.assert_allclose(run.plans["update_t"], np.arange(5) * .01, atol=1e-12)
    assert run.plans["states"].shape[1:] == (4, 14) and run.plans["sqp"].sum() == results["plans"]
    assert ("distance_m" if name == "reach" else "ee_xyz_m") in run.ticks
    with np.load(tmp_path / name / "replay.npz") as replay:
        replay_metadata = json.loads(str(replay["metadata_json"]))
        assert replay_metadata["joint_names"] == list(RIGHT_ARM_JOINTS)
        assert len(replay_metadata["predictions"]) == results["plans"]
        assert replay_metadata["time_basis"] == "simulation"
        assert from_dict(replay_metadata["provenance"]["params"]) == record.config["params"]
    assert record.out == tmp_path / name
    assert summary["provenance"]["gato"] == rec._gato_source() and summary["provenance"]["ros_distro"] is None


def test_provenance_names_the_installed_gato_and_an_editable_checkouts_commit(monkeypatch):
    import gato

    source = rec._gato_source()
    assert source["version"] == importlib.metadata.version("gato-mpc") and source["file"] == gato.__file__
    if source['direct_url'] and source['direct_url'].get('dir_info', {}).get('editable'):
        assert source['commit'] and isinstance(source['status'], str)
    else:
        assert 'commit' not in source  # a PyPI wheel is identified by its locked version
    assert rec.provenance()["gato"] == source and rec.provenance()["repository_commit"] == rec._git("rev-parse", "HEAD")
    url = json.dumps(dict(url=rec.ROOT.as_uri(), dir_info=dict(editable=True)))  # this checkout as the fork
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: NS(version="9.9", read_text=lambda f: url))
    source = rec._gato_source()
    assert source["version"] == "9.9" and source["commit"] == rec._git("rev-parse", "HEAD")
    assert source["status"] == rec._git("status", "--short")


def test_reach_commands_every_tick_and_updates_every_third(tmp_path):
    requests = []

    class Counting(Mpc):
        def request(self, state, reference, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    mpc = Counting(replace(rc.tm.TIAGO, N=8))  # 10 ms knots, 70 ms plans
    record = common.run_experiment(SimConfig(substep=.002), mpc, rc.Reach(timeout_s=.06), SampleKnot(),
                                   VirtualTime(.002, .006, 0.), out=tmp_path / "r")
    run = load_run(record.out)
    np.testing.assert_allclose(run.ticks["t"], np.arange(31) * .002, atol=1e-12)
    assert np.isfinite(run.ticks["command"][:-1]).all()  # a command at every tick but the last
    np.testing.assert_allclose(requests, np.arange(10) * .006, atol=1e-12)  # every third tick


def test_duration_caps_the_tasks_own(tmp_path):
    assert common.arguments("", ["--duration", ".02"]).duration == .02
    task, mpc, timing = rc.Reach(timeout_s=.06), Mpc(replace(rc.tm.TIAGO, N=4)), VirtualTime.ideal()
    record = common.run_experiment(SimConfig(substep=.003), mpc, task, SampleKnot(), timing, out=tmp_path / "r",
                                   duration=.02)
    assert record.metadata["status"] == "timeout" and record.config["duration"] == .02
    assert len(record.ticks["t"]) == 11  # 0, 2, ..., 20 ms: the default command period
    record = common.run_experiment(SimConfig(substep=.003), mpc, task, SampleKnot(), timing, out=tmp_path / "own")
    assert record.config["duration"] == task.duration_s == .06 and len(record.ticks["t"]) == 31  # the task's own
    assert f8.Figure8(period_s=4.).duration_s == 4.


def test_a_run_without_solves_reports_zero_work(tmp_path):
    """The goal met at the start, or a zero duration: one tick, no solve."""
    params, timing = replace(rc.tm.TIAGO, N=4), VirtualTime.ideal()
    done = common.run_experiment(SimConfig(substep=.003), Mpc(params), rc.Reach(tuple(fk(Q_START))), SampleKnot(),
                                 timing, out=tmp_path / "d")
    zero = common.run_experiment(SimConfig(substep=.003), Mpc(params), rc.Reach(), SampleKnot(), timing,
                                 out=tmp_path / "z", duration=0.)
    for record, status in ((done, "arrived"), (zero, "timeout")):
        rc.evaluate(record)
        metadata = json.loads((record.out / "summary.json").read_text())["metadata"]
        results = metadata["results"]
        assert metadata["status"] == status and results["ticks"] == 1 and results["solve_calls"] == 0
        assert results["solve_time_median_ms"] is None and not any(results["work_totals"].values())
        assert results["max_abs_torque_nm"] == 0. and results["limits_pass"]


def test_a_fault_ends_the_run_saved_with_its_reason_and_replay(tmp_path):
    """The first plan's last knot exceeds joint 7's raw effort limit (26 Nm)."""
    record = common.run_experiment(SimConfig(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4), extra=30.), rc.Reach(),
                                   SampleKnot(), VirtualTime(.01, .03, 0.), out=tmp_path / "r")
    run = load_run(record.out)
    assert run.metadata["status"] == "fault" and run.metadata["reason"] == "effort_limit"
    assert run.metadata["end_t"] == 0. and run.metadata["error"] is None
    assert len(run.ticks["t"]) == 1 and np.isnan(run.ticks["command"]).all()
    # The faulting plan is saved, the last row, though no tick's plan index points at it.
    np.testing.assert_array_equal(run.ticks["plan"], [-1])
    assert len(run.plans["t0"]) == 1 and run.plans["update_t"][0] == 0. and run.plans["torques"][0, -1, 6] > 26.
    # a 30 ms plan, 50 ms late
    record = common.run_experiment(SimConfig(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4, dt_s=.01)), rc.Reach(),
                                   SampleKnot(), VirtualTime(.01, .01, .05), out=tmp_path / "late")
    assert record.metadata["status"] == "fault" and record.metadata["reason"] == "lease_expired"
    assert (record.out / "replay.npz").exists() and len(record.ticks["t"]) == 6


def test_an_interrupted_run_is_saved_as_interrupted(tmp_path):
    @dataclass(frozen=True)
    class Abort:  # the hard abort: a second Ctrl-C raises KeyboardInterrupt
        def run(self, *args, **kw):
            raise KeyboardInterrupt

    mpc = Mpc(replace(rc.tm.TIAGO, N=4))
    with pytest.raises(KeyboardInterrupt):
        common.run_experiment(SimConfig(substep=.003), mpc, rc.Reach(), SampleKnot(), Abort(), out=tmp_path / "r")
    metadata = json.loads((tmp_path / "r" / "summary.json").read_text())["metadata"]
    assert metadata["status"] == "interrupted" and "KeyboardInterrupt" in metadata["error"]
    STOP.set()  # the first Ctrl-C: the run ends at its first tick, saved and returned
    record = common.run_experiment(SimConfig(substep=.003), mpc, rc.Reach(), SampleKnot(), VirtualTime.ideal(),
                                   out=tmp_path / "stopped")
    assert record.metadata["status"] == "interrupted" and record.metadata["reason"] == "stop_requested"
    assert load_run(record.out).metadata == record.metadata and len(record.ticks["t"]) == 1


def test_a_failed_build_is_a_saved_error_and_a_failed_compile_leaves_no_run(monkeypatch, tmp_path):
    closed = []
    mpc = Mpc(replace(rc.tm.TIAGO, N=4))
    mpc.close = lambda: closed.append("mpc")
    monkeypatch.setattr(rc.tm, "Mpc", NS(compile=lambda params: mpc))
    for name, given in (("passed", mpc), ("built", rc.tm.TIAGO)):
        with pytest.raises(ValueError, match="StickSlip"):  # a plant with friction names its integrator
            common.run_experiment(SimConfig(friction=urdf_friction()), given, rc.Reach(), SampleKnot(),
                                  VirtualTime.ideal(), out=tmp_path / name)
        metadata = json.loads((tmp_path / name / "summary.json").read_text())["metadata"]
        assert metadata["status"] == "error" and "Euler-only" in metadata["error"]
        assert closed == ([] if name == "passed" else ["mpc"])  # a passed MPC is the caller's

    def fail(params):
        raise RuntimeError("the compile failed")

    monkeypatch.setattr(rc.tm, "Mpc", NS(compile=fail))
    with pytest.raises(RuntimeError, match="compile"):
        common.run_experiment(SimConfig(substep=.003), rc.tm.TIAGO, rc.Reach(), SampleKnot(), VirtualTime.ideal(),
                              out=tmp_path / "compile")
    assert not (tmp_path / "compile").exists()  # a setup error, not a run


def test_update_period_0_stays_back_to_back(tmp_path):
    requests = []

    class Counting(Mpc):
        def request(self, state, reference, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    record = common.run_experiment(SimConfig(substep=.002), Counting(replace(rc.tm.TIAGO, N=8)),
                                   rc.Reach(timeout_s=.01), SampleKnot(), VirtualTime(.002, 0., 0.), out=tmp_path / "r")
    assert record.config["timing"].update_period == 0.
    np.testing.assert_allclose(requests, np.arange(5) * .002, atol=1e-12)  # a request at every tick


@pytest.mark.parametrize("argv, plant, latency", [([], "sim", None), (["--solve-latency", "measured"], "sim", "measured"),
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


def test_the_command_line_refuses_a_plant_the_experiment_does_not_support():
    assert common.arguments("", [], plants=("sim",)).plant == "sim"
    with pytest.raises(SystemExit):
        common.arguments("", ["--plant", "rehearsal"], plants=("sim",))


@pytest.mark.parametrize("argv, timing", [
    ([], VirtualTime(.002, .02, 0.)),
    (["--update-period", ".01"], VirtualTime(.002, .01, 0.)),
    (["--solve-latency", ".005", "--command-period", ".01"], VirtualTime(.01, .02, .005)),
    (["--solve-latency", "measured"], VirtualTime(.002, .02, "measured")),
    (["--plant", "rehearsal", "--update-period", "0"], RealTime(.002, 0.)),
    (["--plant", "rehearsal"], RealTime())])
def test_setup_replaces_only_the_given_options(argv, timing):
    """In the main's timing (here 20 ms updates) or, for the rehearsal, in RealTime()."""
    plant = SimConfig(substep=.003)
    assert common.setup(common.arguments("", argv), plant, VirtualTime(.002, .02, 0.)) == (plant, timing)


def test_setup_replaces_the_options_in_an_experiments_own_timing():
    plant = SimConfig(substep=.003)
    args = common.arguments("", ["--solve-latency", ".03", "--command-period", ".004"])
    assert common.setup(args, plant, Compensated()) == (plant, Compensated(.03, .004))


FORECAST = LatencyCompensation(latency_s=.02)  # the defaults: one 50 Hz control period


@pytest.mark.parametrize("argv, solver, timing, budget", [
    ([], "Mpc", VirtualTime(.002, .02, .02, FORECAST), None),
    (["--solve-latency", ".005", "--command-period", ".01"], "Mpc", VirtualTime(.01, .02, .005, FORECAST), None),
    (["--solve-latency", "measured"], "Mpc", VirtualTime(.002, .02, "measured", FORECAST), .01),
    (["--plant", "rehearsal", "--update-period", "0"], "MpcProcess", RealTime(.002, 0.), .01)])
def test_the_main_runs_on_the_mpc_run_experiment_builds(monkeypatch, tmp_path, argv, solver, timing, budget):
    """reach's main end to end on a fake MPC: run_experiment builds it for the timing, with
    the wall-time budget only where the solve's wall time is the latency, and closes it. The
    record holds the params the MPC resolved, here GATO's warm start."""
    built = []

    def compile(name):
        def build(params):
            assert params.warm_start is None  # the main follows GATO's
            mpc = Mpc(replace(params, warm_start="shift"))  # as MpcParams.resolved
            mpc.kind, mpc.close = name, lambda: built.append("closed")
            built.append(mpc)
            return mpc
        return NS(compile=build)

    monkeypatch.setattr(rc.tm, "Mpc", compile("Mpc"))
    monkeypatch.setattr(rc.tm, "MpcProcess", compile("MpcProcess"))
    monkeypatch.setattr(common.signal, "signal", lambda *args: None)  # the main's stop signals
    assert rc.main([*argv, "--duration", ".02", "--output", str(tmp_path / "r")]) == 1  # the capped reach times out
    run = load_run(tmp_path / "r", classes=[rc.Reach])
    assert built[0].kind == solver and built[1:] == ["closed"]
    assert run.config["params"] == built[0].params == replace(rc.tm.TIAGO, max_sqp_time_s=budget, warm_start="shift")
    assert run.config["timing"] == timing and run.config["plant"] == SimConfig(substep=.003)
    assert run.config["duration"] == .02 and "distance_m" in run.ticks


def test_stop_signals_set_the_flag_and_a_second_one_aborts():
    originals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        STOP.set()
        common.handle_stop_signals()
        assert not STOP.is_set()  # cleared for this launch
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert STOP.is_set()
        with pytest.raises(KeyboardInterrupt):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        for s, handler in originals.items():
            signal.signal(s, handler)


def test_the_random100_study_round_trips_and_load_reads_its_runs(tmp_path):
    from experiments.random100 import run as random100

    config = random100.Random100Config(ids=(3546, 3331), inputs_sha256="0" * 64)
    study = random100.Random100Study(config, random100.cases(config))
    (tmp_path / "study.json").write_text(json.dumps(dict(provenance={}, study=rec.to_dict(study))))
    case = study.cases[0]
    assert case.name == "pair_03546" and len(case.initial_state) == 14
    record = common.run_experiment(replace(config.plant, q=case.initial_state[:7], qd=case.initial_state[7:]),
                                   Mpc(replace(config.params, N=4)), rc.Reach(case.target_m, timeout_s=.01),
                                   SampleKnot(), config.timing, out=tmp_path / "runs" / case.name, case=case.name)
    rc.evaluate(record)
    loaded, records = random100.load(tmp_path)
    assert loaded == study and [r.config["case"] for r in records] == [case.name]  # the second case did not run
    assert records[0].config["task"] == rc.Reach(case.target_m, timeout_s=.01)
    np.testing.assert_array_equal(np.r_[records[0].ticks["q"][0], records[0].ticks["qd"][0]], case.initial_state)
    assert records[0].metadata == record.metadata
