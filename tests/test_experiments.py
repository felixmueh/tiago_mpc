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
from experiments.latency_compensation import Compensated, run_compensated
from test_loops import wall
from tiago_mpc import record as rec
from tiago_mpc.laws import SampleKnot
from tiago_mpc.loops import STOP, RealTimeSettings, VirtualTimeSettings
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
    assert reference.dtype == np.float32 and np.isnan(reference[:, 3:]).all() and updates is None
    assert f8.Figure8(start_m=(.5, .1, .2)).at(6., 2, .01)[0][0, 0] == pytest.approx(.66)


@pytest.mark.parametrize("task", [rc.Reach(orientation_rad=[.1, -.2, 3]), f8.Figure8(orientation_rad=[.1, -.2, 3])])
def test_a_tasks_orientation_target_fills_every_knots_orientation_slots(task):
    assert task.orientation_rad == (.1, -.2, 3.)  # floats in a tuple: a frozen value that records
    reference = task.at(.5, 4, .01)[0]
    np.testing.assert_allclose(reference[:, 3:], np.tile([.1, -.2, 3.], (4, 1)), rtol=1e-6)
    assert np.isfinite(reference[:, :3]).all()


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

    def warmup(self, state, reference=None):
        pass

    def request(self, state, reference=None, updates=None):
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
        task = rc.Reach(timeout_s=.05, orientation_rad=tuple(rc.tm.ee_orientation(Q_START)))
        record = common.run_experiment(SimConfig(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4, dt_s=.01, cost=rc.tm.default_cost(4))), task,
                                       SampleKnot(), VirtualTimeSettings(.01, .01, .01), out=tmp_path / name)
        rc.evaluate(record)
    else:
        task = f8.Figure8(period_s=.05, orientation_rad=(0., 0., 0.))
        record = common.run_experiment(SimConfig(friction=urdf_friction(), integrator="euler"),
                                       Mpc(replace(f8.FIGURE8, N=4, cost=rc.tm.default_cost(4, velocity=.01, torque=1e-6))), task, SampleKnot(), VirtualTimeSettings(.01, .01, 0.),
                                       out=tmp_path / name)
        f8.evaluate(record)
    summary = json.loads((tmp_path / name / "summary.json").read_text())
    config, metadata, results = summary["config"], summary["metadata"], summary["metadata"]["results"]
    assert summary["format"] == rec.FORMAT and set(summary) == {"format", "provenance", "config", "metadata"}
    assert metadata["status"] == ("timeout" if name == "reach" else "done") and metadata["reason"] is None
    assert metadata["end_t"] == pytest.approx(.05) and metadata["error"] is None and metadata["plant"] == {}
    assert set(config) == {"task", "params", "plant", "law", "settings", "loop", "compensation", "safety", "effort_handling", "duration"}
    assert config["params"]["N"] == 4 and config["task"]["type"] == type(task).__name__
    cost = config["params"]["cost"]
    assert cost["type"] == "CostMenu" and cost["frame_pos"]["type"] == "Term"
    np.testing.assert_array_equal(cost["frame_pos"]["weight"], [[4.] * 3] * 3 + [[8.] * 3])
    assert cost["state"]["weight"] == [0.] * 7 + [.01 if name == "figure8" else .05] * 7
    assert cost["input"]["weight"] == (1e-6 if name == "figure8" else 1e-4)
    assert config["params"]["solver"]["type"] == "SolverSettings"
    assert not {"q_cost", "N_cost", "qd_cost", "u_cost"} & config["params"]["solver"].keys()
    assert config["plant"]["type"] == "SimConfig" and config["law"] == dict(type="SampleKnot")
    assert config["settings"] == dict(type="VirtualTimeSettings", command_period=.01, update_period=.01,
                                    solve_latency_s=.01 if name == "reach" else 0.)
    assert config["loop"] == "run_virtual_time" and config["compensation"] is None
    assert config["safety"] == rec.to_dict(rc.tm.ExecutionSafety())
    assert config["effort_handling"] == rec.to_dict(rc.tm.EffortClip())
    assert config["duration"] == .05 and results["limits_pass"]
    if name == "figure8":
        assert config["plant"]["substep"] == .001 and config["plant"]["integrator"] == "euler"
        assert config["plant"]["friction_law"] == dict(type="StickSlip")
        assert config["params"]["friction_law"] == dict(type="Tanh", width_rad_s=.1)  # predictor stays smooth
        assert set(results) >= {"metrics_full_cycle", "metrics_after_initial_1s", "orientation_metrics_full_cycle"}
        assert set(results["orientation_metrics_full_cycle"]) == {"rmse_rad", "median_rad", "p95_rad", "max_rad"}
    else:
        assert set(results) >= {"initial_distance_m", "final_distance_m", "max_distance_m", "final_speed_m_s"}
        assert results["initial_orientation_error_rad"] == pytest.approx(0., abs=1e-9)  # it starts at Q_START
        assert results["max_orientation_error_rad"] >= results["final_orientation_error_rad"]
    assert results["ticks"] == 6 and results["plans"] == 5 and results["late_ticks"] == results["overruns"] == 0
    assert results["late_plans"] == (4 if name == "reach" else 0)  # the reach's 10 ms latency; the last tick sends nothing
    run = load_run(tmp_path / name, classes=[type(task)])
    assert rec.to_dict(run.config) == rec.to_dict(record.config)
    assert run.config["task"] == task and run.metadata == metadata
    np.testing.assert_allclose(run.ticks["t"], np.arange(6) * .01, atol=1e-12)
    np.testing.assert_array_equal(run.ticks["state_t"], run.ticks["t"])  # virtual time
    # The reach's 10 ms latency: the first plan takes effect at the second tick.
    assert np.isnan(run.ticks["command"][-1]).all()
    assert np.isnan(run.ticks["command"][0]).all() == (name == "reach")
    np.testing.assert_array_equal(run.ticks["plan"], [-1, 0, 1, 2, 3, 4] if name == "reach" else [0, 1, 2, 3, 4, 4])
    np.testing.assert_array_equal(run.ticks["update"], [0, 1, 2, 3, 4, -1])
    np.testing.assert_allclose(run.plans["update_t"], np.arange(5) * .01, atol=1e-12)
    assert run.plans["states"].shape[1:] == (4, 14) and run.plans["sqp"].sum() == results["plans"]
    assert ("distance_m" if name == "reach" else "ee_xyz_m") in run.ticks and "orientation_error_rad" in run.ticks
    with np.load(tmp_path / name / "replay.npz") as replay:
        replay_metadata = json.loads(str(replay["metadata_json"]))
        assert replay_metadata["joint_names"] == list(RIGHT_ARM_JOINTS)
        assert len(replay_metadata["predictions"]) == results["plans"]
        assert replay_metadata["time_basis"] == "simulation"
        assert rec.to_dict(from_dict(replay_metadata["provenance"]["params"])) == rec.to_dict(record.config["params"])
        np.testing.assert_allclose(replay["reference_rotvec_rad"],
                                   np.tile(task.orientation_rad, (len(replay["time_s"]), 1)), rtol=1e-6)
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
        def request(self, state, reference=None, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    mpc = Counting(replace(rc.tm.TIAGO, N=8, cost=rc.tm.default_cost(8)))  # 10 ms knots, 70 ms plans
    record = common.run_experiment(SimConfig(substep=.002), mpc, rc.Reach(timeout_s=.06), SampleKnot(),
                                   VirtualTimeSettings(.002, .006, 0.), out=tmp_path / "r")
    run = load_run(record.out)
    np.testing.assert_allclose(run.ticks["t"], np.arange(31) * .002, atol=1e-12)
    assert np.isfinite(run.ticks["command"][:-1]).all()  # a command at every tick but the last
    np.testing.assert_allclose(requests, np.arange(10) * .006, atol=1e-12)  # every third tick


def test_duration_caps_the_tasks_own(tmp_path):
    assert common.arguments("", ["--duration", ".02"]).duration == .02
    task, mpc, timing = rc.Reach(timeout_s=.06), Mpc(replace(rc.tm.TIAGO, N=4, cost=rc.tm.default_cost(4))), VirtualTimeSettings.ideal()
    record = common.run_experiment(SimConfig(substep=.003), mpc, task, SampleKnot(), timing, out=tmp_path / "r",
                                   duration=.02)
    assert record.metadata["status"] == "timeout" and record.config["duration"] == .02
    assert len(record.ticks["t"]) == 11  # 0, 2, ..., 20 ms: the default command period
    record = common.run_experiment(SimConfig(substep=.003), mpc, task, SampleKnot(), timing, out=tmp_path / "own")
    assert record.config["duration"] == task.duration_s == .06 and len(record.ticks["t"]) == 31  # the task's own
    assert f8.Figure8(period_s=4.).duration_s == 4.


@pytest.mark.parametrize('faulted', [False, True])
def test_common_runner_forwards_each_stream_and_drains_handback_once(tmp_path, monkeypatch, capsys, faulted):
    from test_loops import Mpc as FakeMpc, Plant, Task

    class Emitting(Plant):
        def start(self, clock):
            super().start(clock)
            self.diagnostics.append(rc.tm.Diagnostic(clock.now(), 'startup', 'starting'))

        def stop(self):
            super().stop()
            self.diagnostics.append(rc.tm.Diagnostic(self.clock.now(), 'handback', 'handed back'))

    plant, ticks, solves, diagnostics, order = Emitting(), [], [], [], []
    mpc = FakeMpc()
    mpc.params = replace(rc.tm.TIAGO, N=mpc.N, dt_s=mpc.dt, cost=rc.tm.default_cost(mpc.N))
    monkeypatch.setattr(SimConfig, 'build', lambda self: plant)

    def diagnostic(value):
        assert isinstance(value, rc.tm.Diagnostic)
        diagnostics.append(value)
        order.append(value.kind)

    def tick(value):
        assert isinstance(value, rc.tm.Tick)
        ticks.append(value)
        order.append('tick')

    def solved(value):
        assert isinstance(value, rc.tm.SolveCompleted)
        solves.append(value)

    handling = rc.tm.RejectOnEffort(effort=(.01,)*7) if faulted else rc.tm.EffortClip()
    record = common.run_experiment(SimConfig(), mpc, Task(), SampleKnot(), VirtualTimeSettings(.002, .002, 0.),
        effort_handling=handling,
        duration=.004, out=tmp_path / 'r', tick_callbacks=[tick], solve_completed_callbacks=[solved],
        diagnostic_callbacks=[diagnostic])
    assert record.metadata['status'] == ('fault' if faulted else 'timeout')
    assert order[0] == 'startup' and order[-1] == 'handback'
    assert len(ticks) == len(record.ticks['t']) == (1 if faulted else 3)
    assert len(solves) == len(mpc.requests) == len(record.plans['t0']) == (1 if faulted else 2)
    assert record.plans['update_t'] == pytest.approx([value.request_time_s for value in solves])
    np.testing.assert_array_equal(record.ticks['plan'], [0] if faulted else [0, 1, 1])
    assert [value.kind for value in diagnostics].count('startup') == 1
    assert [value.kind for value in diagnostics].count('handback') == 1
    assert record.diagnostics == diagnostics == load_run(record.out).diagnostics
    assert plant.take_diagnostics() == [] and plant.calls == ['start', 'stop']
    printed = capsys.readouterr().err
    assert printed.count('starting (') == printed.count('handed back (') == 1


def test_a_failing_handback_observer_still_closes_and_saves_the_run(tmp_path, monkeypatch):
    from test_loops import Mpc as FakeMpc, Plant, Task

    closed = []

    class Emitting(Plant):
        def stop(self):
            super().stop()
            self.diagnostics.append(rc.tm.Diagnostic(self.clock.now(), 'handback', 'handed back'))

        def close(self):
            closed.append('plant')

    plant, mpc = Emitting(), FakeMpc()
    params = replace(rc.tm.TIAGO, N=mpc.N, dt_s=mpc.dt, cost=rc.tm.default_cost(mpc.N))
    mpc.params, mpc.close = params, lambda: closed.append('mpc')
    monkeypatch.setattr(SimConfig, 'build', lambda self: plant)
    monkeypatch.setattr(rc.tm, 'Mpc', NS(compile=lambda params: mpc))

    def fail(diagnostic):
        assert diagnostic.kind == 'handback'
        raise RuntimeError('handback observer failed')

    out = tmp_path / 'r'
    with pytest.raises(RuntimeError, match='handback observer failed'):
        common.run_experiment(SimConfig(), params, Task(), SampleKnot(), VirtualTimeSettings(.002, .002, 0.),
                              duration=.004, out=out, diagnostic_callbacks=[fail])
    assert closed == ['plant', 'mpc'] and plant.calls == ['start', 'stop']
    run = load_run(out)
    assert [value.kind for value in run.diagnostics] == ['handback']
    assert plant.take_diagnostics() == [] and (out / 'replay.npz').exists()


def test_a_run_without_solves_reports_zero_work(tmp_path):
    """The goal met at the start, or a zero duration: one tick, no solve."""
    params, timing = replace(rc.tm.TIAGO, N=4, cost=rc.tm.default_cost(4)), VirtualTimeSettings.ideal()
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
    """An effort refusal is saved without rejecting unapplied planned knots."""
    record = common.run_experiment(SimConfig(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4, cost=rc.tm.default_cost(4)), extra=30.), rc.Reach(),
                                   SampleKnot(),
                                   VirtualTimeSettings(.01, .03, 0.),
                                   effort_handling=rc.tm.RejectOnEffort(effort=(.01,)*7),
                                   out=tmp_path / "r")
    run = load_run(record.out)
    assert run.metadata["status"] == "fault" and run.metadata["reason"] == "effort_limit"
    assert run.metadata["end_t"] == 0. and run.metadata["error"] is None
    assert len(run.ticks["t"]) == 1 and np.isnan(run.ticks["command"]).all()
    # The selected prediction is retained, although its first command was refused.
    np.testing.assert_array_equal(run.ticks["plan"], [0])
    assert len(run.plans["t0"]) == 1 and run.plans["update_t"][0] == 0. and run.plans["torques"][0, -1, 6] > 26.
    # a 30 ms plan, 50 ms late
    record = common.run_experiment(SimConfig(substep=.003), Mpc(replace(rc.tm.TIAGO, N=4, dt_s=.01, cost=rc.tm.default_cost(4))), rc.Reach(),
                                   SampleKnot(), VirtualTimeSettings(.01, .01, .05), out=tmp_path / "late")
    assert record.metadata["status"] == "fault" and record.metadata["reason"] == "lease_expired"
    assert (record.out / "replay.npz").exists() and len(record.ticks["t"]) == 6


def test_an_interrupted_run_is_saved_as_interrupted(tmp_path):
    def abort(*args, **kw):  # the hard abort: a second Ctrl-C raises KeyboardInterrupt
        raise KeyboardInterrupt

    mpc = Mpc(replace(rc.tm.TIAGO, N=4, cost=rc.tm.default_cost(4)))
    with pytest.raises(KeyboardInterrupt):
        common.run_experiment(SimConfig(substep=.003), mpc, rc.Reach(), SampleKnot(), VirtualTimeSettings.ideal(),
                              loop=abort, out=tmp_path / "r")
    metadata = json.loads((tmp_path / "r" / "summary.json").read_text())["metadata"]
    assert metadata["status"] == "interrupted" and "KeyboardInterrupt" in metadata["error"]
    STOP.set()  # the first Ctrl-C: the run ends at its first tick, saved and returned
    record = common.run_experiment(SimConfig(substep=.003), mpc, rc.Reach(), SampleKnot(), VirtualTimeSettings.ideal(),
                                   out=tmp_path / "stopped")
    assert record.metadata["status"] == "interrupted" and record.metadata["reason"] == "stop_requested"
    assert load_run(record.out).metadata == record.metadata
    assert not np.isfinite(record.ticks["command"]).any()


def test_a_failed_build_is_a_saved_error_and_a_failed_compile_leaves_no_run(monkeypatch, tmp_path):
    closed = []
    mpc = Mpc(replace(rc.tm.TIAGO, N=4, cost=rc.tm.default_cost(4)))
    mpc.close = lambda: closed.append("mpc")
    monkeypatch.setattr(rc.tm, "Mpc", NS(compile=lambda params: mpc))
    for name, given in (("passed", mpc), ("built", rc.tm.TIAGO)):
        with pytest.raises(ValueError, match="StickSlip"):  # a plant with friction names its integrator
            common.run_experiment(SimConfig(friction=urdf_friction()), given, rc.Reach(), SampleKnot(),
                                  VirtualTimeSettings.ideal(), out=tmp_path / name)
        metadata = json.loads((tmp_path / name / "summary.json").read_text())["metadata"]
        assert metadata["status"] == "error" and "Euler-only" in metadata["error"]
        assert closed == ([] if name == "passed" else ["mpc"])  # a passed MPC is the caller's

    def fail(params):
        raise RuntimeError("the compile failed")

    monkeypatch.setattr(rc.tm, "Mpc", NS(compile=fail))
    with pytest.raises(RuntimeError, match="compile"):
        common.run_experiment(SimConfig(substep=.003), rc.tm.TIAGO, rc.Reach(), SampleKnot(), VirtualTimeSettings.ideal(),
                              out=tmp_path / "compile")
    assert not (tmp_path / "compile").exists()  # a setup error, not a run


def test_update_period_0_stays_back_to_back(tmp_path):
    requests = []

    class Counting(Mpc):
        def request(self, state, reference=None, updates=None):
            requests.append(state.t)
            super().request(state, reference, updates)

    record = common.run_experiment(SimConfig(substep=.002), Counting(replace(rc.tm.TIAGO, N=8, cost=rc.tm.default_cost(8))),
                                   rc.Reach(timeout_s=.01), SampleKnot(), VirtualTimeSettings(.002, 0., 0.), out=tmp_path / "r")
    assert record.config["settings"].update_period == 0.
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
    ([], VirtualTimeSettings(.002, .02, 0.)),
    (["--update-period", ".01"], VirtualTimeSettings(.002, .01, 0.)),
    (["--solve-latency", ".005", "--command-period", ".01"], VirtualTimeSettings(.01, .02, .005)),
    (["--solve-latency", "measured"], VirtualTimeSettings(.002, .02, "measured")),
    (["--plant", "rehearsal", "--update-period", "0"], RealTimeSettings(.002, 0.)),
    (["--plant", "rehearsal"], RealTimeSettings())])
def test_setup_replaces_only_the_given_options(argv, timing):
    """In the main's timing (here 20 ms updates) or, for the rehearsal, in RealTimeSettings()."""
    plant = SimConfig(substep=.003)
    loop = rc.tm.run_real_time if isinstance(timing, RealTimeSettings) else rc.tm.run_virtual_time
    assert common.setup(common.arguments("", argv), plant, VirtualTimeSettings(.002, .02, 0.)) == (
        plant, timing, loop)


def test_setup_replaces_the_options_in_an_experiments_own_timing():
    plant = SimConfig(substep=.003)
    args = common.arguments("", ["--solve-latency", ".03", "--command-period", ".004"])
    assert common.setup(args, plant, Compensated(), loop=run_compensated) == (
        plant, Compensated(.03, .004), run_compensated)


@pytest.mark.parametrize("argv, solver, timing, budget, compensation", [
    ([], "Mpc", VirtualTimeSettings(.002, .02, .02), None, LatencyCompensation(latency_s=.02)),
    (["--solve-latency", ".005", "--command-period", ".01"], "Mpc", VirtualTimeSettings(.01, .02, .005),
     None, LatencyCompensation(latency_s=.005)),
    (["--solve-latency", "measured"], "Mpc", VirtualTimeSettings(.002, .02, "measured"),
     .01, LatencyCompensation(initial_latency_s=.02)),
    (["--solve-latency", "measured", "--update-period", ".007"], "Mpc", VirtualTimeSettings(.002, .007, "measured"),
     .01, LatencyCompensation(initial_latency_s=.007)),
    (["--plant", "rehearsal", "--update-period", "0"], "MpcProcess", RealTimeSettings(.002, 0.),
     .01, LatencyCompensation(initial_latency_s=.002)),
])
def test_the_main_runs_on_the_mpc_run_experiment_builds(
    monkeypatch,
    tmp_path,
    argv,
    solver,
    timing,
    budget,
    compensation,
):
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
    expected = replace(rc.tm.TIAGO, max_sqp_time_s=budget, warm_start="shift")
    assert rec.to_dict(run.config["params"]) == rec.to_dict(built[0].params) == rec.to_dict(expected)
    assert run.config["settings"] == timing and run.config["plant"] == SimConfig(substep=.003)
    assert run.config["compensation"] == compensation
    assert isinstance(run.config["effort_handling"],
                      rc.tm.RejectOnEffort if isinstance(timing, RealTimeSettings) else rc.tm.EffortClip)
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
                                   Mpc(replace(config.params, N=4, cost=rc.tm.default_cost(4))), rc.Reach(case.target_m, timeout_s=.01),
                                   SampleKnot(), config.settings, out=tmp_path / "runs" / case.name, case=case.name)
    rc.evaluate(record)
    loaded, records = random100.load(tmp_path)
    assert rec.to_dict(loaded) == rec.to_dict(study)
    assert [r.config["case"] for r in records] == [case.name]  # the second case did not run
    assert records[0].config["task"] == rc.Reach(case.target_m, timeout_s=.01)
    np.testing.assert_array_equal(np.r_[records[0].ticks["q"][0], records[0].ticks["qd"][0]], case.initial_state)
    assert records[0].metadata == record.metadata


@pytest.mark.parametrize('settings, expected', [
    (VirtualTimeSettings(.002, .01, .0055), LatencyCompensation(latency_s=.0055)),
    (VirtualTimeSettings(.002, .007, 'measured'), LatencyCompensation(initial_latency_s=.007)),
    (VirtualTimeSettings(.004, 0., 'measured'), LatencyCompensation(initial_latency_s=.004)),
])
def test_random100_resolves_its_forecast_from_the_chosen_settings_and_preserves_explicit_overrides(settings, expected):
    from experiments.random100 import run as random100

    config = random100.Random100Config(ids=(3546,), inputs_sha256='0'*64, settings=settings)
    assert config.compensation == expected
    assert from_dict(json.loads(json.dumps(rec.to_dict(config))), classes=[random100.Random100Config]) == config
    fixed = LatencyCompensation(latency_s=.013)
    assert replace(config, compensation=fixed).compensation is fixed
    assert replace(config, compensation=None).compensation is None


def test_save_replay_takes_a_reference_orientation_per_sample_nan_for_none(tmp_path):
    from experiments.replay import save_replay
    q, times = np.tile(Q_START, (3, 1)), np.array([0., .1, .2])
    rotation = np.array([[0., 0., .5], [np.nan] * 3, [0., 0., .5]])
    path = save_replay(tmp_path / "r.npz", time_s=times, q=q, joint_names=RIGHT_ARM_JOINTS, time_basis="simulation",
                       end_effector_frame=rc.tm.GRASP_FRAME,
                       reference_xyz_m=np.zeros((3, 3)), reference_time_s=times, reference_frame="torso_lift_link",
                       reference_rotvec_rad=rotation)
    with np.load(path) as replay:
        np.testing.assert_array_equal(replay["reference_rotvec_rad"], rotation)
    for bad in (rotation[:2], np.array([[0., np.nan, .5]] * 3)):  # wrong shape; a partly NaN row
        with pytest.raises(ValueError, match="orientation"):
            save_replay(tmp_path / "bad.npz", time_s=times, q=q, joint_names=RIGHT_ARM_JOINTS, time_basis="simulation",
                        end_effector_frame=rc.tm.GRASP_FRAME,
                        reference_xyz_m=np.zeros((3, 3)), reference_time_s=times, reference_frame="torso_lift_link",
                        reference_rotvec_rad=bad)


@pytest.mark.parametrize('real', [False, True])
def test_runner_resolves_loop_specific_defaults_and_records_the_actual_policies(tmp_path, real, wall):
    loop = rc.tm.run_real_time if real else rc.tm.run_virtual_time
    settings = RealTimeSettings(.002, .006) if real else VirtualTimeSettings.ideal()
    mpc = Mpc(replace(rc.tm.TIAGO, N=8, cost=rc.tm.default_cost(8)))
    @dataclass(frozen=True)
    class Excess:
        def __call__(self, *args):
            return rc.tm.Command(np.full(7, 1000.), np.full(7, 1000.))

    law = Excess()
    record = common.run_experiment(SimConfig(substep=.002), mpc, rc.Reach(), law, settings,
                                   loop=loop, duration=.03 if real else .004, out=tmp_path / 'r')
    assert record.config['loop'] == loop.__name__
    assert record.config['compensation'] == (LatencyCompensation(initial_latency_s=.006) if real else None)
    assert isinstance(record.config['effort_handling'], rc.tm.RejectOnEffort if real else rc.tm.EffortClip)
    assert record.metadata['status'] == ('fault' if real else 'timeout')
    if real:
        assert record.metadata['reason'] == 'effort_limit'
        assert not np.isfinite(record.ticks['command']).any()
    else:
        np.testing.assert_array_equal(record.ticks['command'][0], rc.tm.limits().effort)


def test_runner_accepts_explicit_custom_loop_and_plain_settings(tmp_path):
    settings = Compensated(.01, .01, .01, compensate=False)
    record = common.run_experiment(SimConfig(substep=.003),
                                   Mpc(replace(rc.tm.TIAGO, N=4, cost=rc.tm.default_cost(4))),
                                   rc.Reach(), SampleKnot(), settings, loop=run_compensated,
                                   duration=.03, out=tmp_path / 'r')
    loaded = load_run(record.out, classes=[rc.Reach, Compensated])
    assert loaded.config['settings'] == settings and loaded.config['loop'] == 'run_compensated'
    assert loaded.metadata['status'] == 'timeout' and len(loaded.ticks['t']) == 4
