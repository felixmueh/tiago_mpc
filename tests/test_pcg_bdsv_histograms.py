"""Check paired numerical settings, stats-only storage and truthful incomplete-data analysis."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments import pcg_bdsv_histograms as study
import tiago_mpc as tm


@pytest.mark.parametrize("name", ["figure8", "reach"])
@pytest.mark.parametrize("n", study.KNOTS)
def test_comparison_changes_only_active_solver_and_has_one_sqp(name, n):
    task, plant, params = study.task_configuration(name, n)
    pcg = replace(params, linsys="pcg", solver=replace(params.solver, linsys="pcg"))
    bdsv = replace(params, linsys="bdsv", solver=replace(params.solver, linsys="bdsv"))
    assert replace(pcg, linsys="bdsv", solver=bdsv.solver) == bdsv
    assert study.fingerprint(task, plant, pcg) == study.fingerprint(task, plant, bdsv)
    assert study.fingerprint(task, plant, pcg) != study.fingerprint(task, plant, replace(pcg, dt_s=.02))
    assert params.solver.max_sqp_iters == 1 and params.max_sqp_time_s is None and params.N == n
    assert params.solver.max_pcg_iters == study.PCG_CAP == 1000
    assert len(params.cost.frame_pos.weight) == n
    assert study.SETTINGS.solve_latency_s == 0 and study.SETTINGS.update_period == .05 and params.dt_s == .01
    assert task.duration_s == study.DURATION_S
    assert plant.friction == tm.urdf_friction() and plant.integrator == "euler"
    assert params.friction is None and params.warm_start == "hold" and params.reseed_threshold is None


def test_medium_reach_observes_after_arrival():
    task, _, _ = study.task_configuration("reach")
    q = tm.Q_START + 3 * (tm.Q_REACH - tm.Q_START)
    assert .18 < np.linalg.norm(tm.ee_position(tm.Q_START) - task.reach.target_m) < .22
    state = tm.State(1., q, np.zeros(7))
    assert task.reach.done(state, 1.) and not task.done(state, 1.) and task.done(state, 6.)


def test_default_plan_and_balanced_order_have_correct_case_accounting():
    plan = study.describe(study.KNOTS, 3)
    assert plan["cases"] == 36 and plan["expected_total_solves"] == 2880
    assert plan["batch_size"] == 1 and plan["knots"] == study.KNOTS
    assert plan["expected_solves_per_case"] == 80
    order = study.case_order((32,), 3)
    assert order[:2] == [(32, "figure8", "pcg", 1), (32, "figure8", "bdsv", 1)]
    assert order[2:4] == [(32, "figure8", "bdsv", 2), (32, "figure8", "pcg", 2)]


class CpuSolver:
    def __init__(self, params):
        self.params = NS(linsys=params.linsys)
        self.batch_size = 1
        self.row_mech, self.admm_active = "admm", True
        self.groups = [dict(kind=i, lo=np.array([-i - 1.]), hi=np.array([i + 1.])) for i in range(3)]

    def get_row_groups(self):
        return self.groups

    def set_collect_stats(self, enabled):
        self.collect_stats = enabled

    def enable_limit_barrier(self, **options):
        self.barrier = options
        self.row_mech, self.admm_active = "barrier", False
        self.groups = [dict(kind=i, lo=np.array([-99.]), hi=np.array([99.])) for i in range(3)]

    def set_row_group_bounds(self, index, lower, upper):
        self.groups[index].update(lo=lower.copy(), hi=upper.copy())


def cpu_controller(params):
    return NS(solver=CpuSolver(params), limits="admm")


def cpu_step():
    return NS(limits="barrier", solve=NS(stats=NS(
        stage_times_us=np.array([[100., 20., 400., 200., 10.]]), initial_merit_time_us=50.,
        pcg_times_us=np.array([390.]), step_size=np.array([[1.]]))))


def test_installing_barriers_disables_admm_and_preserves_all_planning_bounds():
    _, _, params = study.task_configuration("figure8")
    mpc = NS(controller=cpu_controller(params))
    result = study.install_barriers(mpc)
    assert result["mechanism"] == mpc.controller.limits == "barrier" and not result["admm_active"]
    assert mpc.controller.solver.barrier == study.BARRIER
    for i, group in enumerate(result["row_groups"]):
        np.testing.assert_array_equal(group["lo"], [-i - 1.])
        np.testing.assert_array_equal(group["hi"], [i + 1.])


@pytest.mark.parametrize("multiplier", [.5, 1., 2.])
def test_histogram_retains_every_tail_sample(multiplier):
    values = np.concatenate((np.linspace(1., 2., 500), [100.]))
    assert np.histogram(values, bins=study.common_bins(values, multiplier))[0].sum() == len(values)


def synthetic_case(tmp_path, monkeypatch):
    binary = tmp_path / "fake.so"
    binary.write_bytes(b"CPU test binary identity only")
    monkeypatch.setattr(study.tm, "Mpc", lambda params, path: NS(
        params=params, controller=cpu_controller(params), step=cpu_step(), close=lambda: None))

    def loop(plant, mpc, task, law, **options):
        assert isinstance(law, tm.SampleKnot) and options["compensation"] is None
        state = tm.State(0., tm.Q_START.copy(), np.zeros(7))
        stats = dict(solve_time_us=900., wall_s=.001, iterations=1, budget_hit=False, sqp=1, pcg=10,
                     zero_pcg_rows=0, admm=0, admm_linsys=0, admm_linsys_mode="", pred_err=0.)
        plan = tm.Plan(0., .01, np.zeros((32, 14)), np.zeros((31, 7)), stats, measured_t=0.)
        for callback in options["solve_completed_callbacks"]:
            callback(tm.SolveCompleted(0., state, np.zeros((32, 6)), None, plan))
        for callback in options["tick_callbacks"]:
            callback(tm.Tick(0., state, np.zeros(7), plan, True, True))
        return tm.Outcome("fault", "retained synthetic fault", 0.)

    monkeypatch.setattr(study.tm, "run_virtual_time", loop)
    directory = tmp_path / "case"
    case = study.run_case(directory, 32, "reach", "pcg", 1, binary)
    manifest = dict(protocol=study.PROTOCOL, knots=[32], repeats=1,
                    cases=[dict(knots=32, task="reach", solver="pcg", repeat=1, path="case", status="fault")])
    study.write_json(tmp_path / "study.json", manifest)
    return directory, case


def test_direct_simulator_recorder_persists_no_trajectory_data_and_preserves_fault(tmp_path, monkeypatch):
    directory, case = synthetic_case(tmp_path, monkeypatch)
    assert case["status"] == "fault" and case["solves"] == 1
    assert {file.name for file in directory.iterdir()} == {"case.json", "stats.npz", "stats.csv", "progress.jsonl"}
    columns = np.load(directory / "stats.npz", allow_pickle=False)
    assert set(columns.files) == {"simulation_s", *study.SCALAR_STATS}
    assert all(columns[key].ndim == 1 for key in columns)
    rows, quality = study.timing_rows(tmp_path)
    assert len(rows) == 1 and not quality[0]["recording_complete"] and quality[0]["status"] == "fault"
    assert quality[0]["unexpected_admm_count"] == 0
    assert case["config"]["batch_size"] == 1
    assert rows[0]["kkt_us"] == 100. and rows[0]["linear_solve_us"] == 390.


def test_batch_size_larger_than_one_is_rejected_before_any_solve(tmp_path, monkeypatch):
    def build(params, binary):
        controller = cpu_controller(params)
        controller.solver.batch_size = 2
        return NS(params=params, controller=controller, close=lambda: None)

    monkeypatch.setattr(study.tm, "Mpc", build)
    binary = tmp_path / "fake.so"
    binary.write_bytes(b"CPU test only")
    case = study.run_case(tmp_path / "case", 128, "figure8", "pcg", 1, binary)
    assert case["status"] == "error" and case["solves"] == 0
    assert "batch size one" in case["error"] and case["config"]["batch_size"] == 2


def test_absent_line_search_is_explicit_in_scalar_diagnostics():
    _, _, params = study.task_configuration("figure8")
    step = cpu_step()
    step.solve.stats.step_size = np.empty((0, 1))
    result = study.scalar_diagnostics(NS(controller=cpu_controller(params), step=step))
    assert result["step_size"] == 0. and result["line_search_ran"] is False
    assert result["batch_size"] == 1


def test_native_worker_exit_preserves_streamed_timings_and_binary_identity(tmp_path, monkeypatch):
    directory, case = synthetic_case(tmp_path, monkeypatch)
    case["status"] = "running"
    study.write_json(directory / "case.json", case)
    process = NS(exitcode=1, start=lambda: None, join=lambda: None)
    monkeypatch.setattr(study.multiprocessing, "get_context", lambda _: NS(Process=lambda **_: process))
    recovered = study.run_isolated_case(directory, 32, "reach", "pcg", 1, None, None)
    assert recovered["status"] == "error" and recovered["solves"] == 1
    assert recovered["binary_sha256"] == case["binary_sha256"]
    assert recovered["quality"]["limits_pass"] is None
    stats = np.load(directory / "stats.npz", allow_pickle=False)
    assert stats["batch_size"].tolist() == [1]
    assert stats["solve_time_us"].tolist() == [900.]


@pytest.mark.parametrize("problem", ["missing", "nan", "mode", "admm", "batch", "cap", "fingerprint", "resolved", "shift", "law", "plant"])
def test_analysis_rejects_misleading_timing_or_pairing(tmp_path, monkeypatch, problem):
    directory, _ = synthetic_case(tmp_path, monkeypatch)
    stats = dict(np.load(directory / "stats.npz", allow_pickle=False))
    case = json.loads((directory / "case.json").read_text())
    if problem == "missing":
        del stats["wall_s"]
    elif problem == "nan":
        stats["solve_time_us"] = np.array([np.nan])
    elif problem == "mode":
        stats["linsys_mode"] = np.array(["bdsv"])
    elif problem == "admm":
        stats["admm"] = np.array([1])
    elif problem == "batch":
        stats["batch_size"] = np.array([2])
    elif problem == "cap":
        case["config"]["params"]["solver"]["max_pcg_iters"] = 120
    elif problem == "fingerprint":
        case["config"]["input_fingerprint"] = "wrong"
    elif problem == "resolved":
        case["config"]["params"]["solver"]["max_sqp_iters"] = 5
    elif problem == "shift":
        case["config"]["params"]["warm_start"] = "shift"
    elif problem == "law":
        case["config"]["law"] = tm.to_dict(tm.FrictionFeedforward())
    else:
        case["config"]["plant"]["friction"] = None
    np.savez(directory / "stats.npz", **stats)
    study.write_json(directory / "case.json", case)
    with pytest.raises(ValueError):
        study.timing_rows(tmp_path)


def test_empty_case_never_implies_limits_pass(tmp_path):
    task, _, _ = study.task_configuration("reach")
    assert study.Quality(task).result()["limits_pass"] is None


def test_actual_virtual_loop_records_80_scalar_solves_without_a_run_record(tmp_path, monkeypatch):
    import pinocchio as pin
    plant = tm.SimConfig().build()
    gravity = pin.computeGeneralizedGravity(plant.model, plant.data, tm.Q_START)
    plant.close()

    class CpuMpc:
        def __init__(self, params, binary):
            self.params, self.N, self.dt, self.plan = params, params.N, params.dt_s, None
            self.controller, self.step = cpu_controller(params), cpu_step()

        def warmup(self, state, reference):
            pass

        def request(self, state, reference, updates=None):
            states = np.tile(np.concatenate((state.q, state.qd)), (self.N, 1))
            stats = dict(solve_time_us=900., wall_s=.001, iterations=1, budget_hit=False, sqp=1, pcg=10,
                         zero_pcg_rows=0, admm=0, admm_linsys=0, admm_linsys_mode="", pred_err=0.)
            self.plan = tm.Plan(state.t, self.dt, states, np.tile(gravity, (self.N - 1, 1)), stats,
                                measured_t=state.t)

        def result(self, block=False):
            plan, self.plan = self.plan, None
            return plan

        def close(self):
            pass

    monkeypatch.setattr(study.tm, "Mpc", CpuMpc)
    binary = tmp_path / "fake.so"
    binary.write_bytes(b"CPU test only")
    case = study.run_case(tmp_path / "case", 32, "reach", "pcg", 1, binary)
    assert case["status"] == "done" and case["solves"] == 80
    assert case["quality"]["observations"] == 2001
    assert {p.name for p in (tmp_path / "case").iterdir()} == {"case.json", "stats.npz", "stats.csv", "progress.jsonl"}
