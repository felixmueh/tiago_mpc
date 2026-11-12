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
    pcg, bdsv = replace(params, admm_linsys="pcg"), replace(params, admm_linsys="bdsv_factor")
    assert replace(pcg, admm_linsys="bdsv_factor") == bdsv
    assert study.fingerprint(task, plant, pcg) == study.fingerprint(task, plant, bdsv)
    assert study.fingerprint(task, plant, pcg) != study.fingerprint(task, plant, replace(pcg, dt_s=.02))
    assert params.solver.max_sqp_iters == 1 and params.max_sqp_time_s is None and params.N == n
    assert len(params.cost.frame_pos.weight) == n
    assert study.SETTINGS.solve_latency_s == 0 and study.SETTINGS.update_period == params.dt_s == .01
    assert task.duration_s == 6


def test_medium_reach_observes_after_arrival():
    task, _, _ = study.task_configuration("reach")
    q = tm.Q_START + 3 * (tm.Q_REACH - tm.Q_START)
    assert .18 < np.linalg.norm(tm.ee_position(tm.Q_START) - task.reach.target_m) < .22
    state = tm.State(1., q, np.zeros(7))
    assert task.reach.done(state, 1.) and not task.done(state, 1.) and task.done(state, 6.)


def test_default_plan_and_balanced_order_have_correct_case_accounting():
    plan = study.describe(study.KNOTS, 3)
    assert plan["cases"] == 72 and plan["expected_total_solves"] == 43200
    assert plan["expected_solves_per_case"] == 600
    order = study.case_order((32,), 3)
    assert order[:2] == [(32, "figure8", "pcg", 1), (32, "figure8", "bdsv_factor", 1)]
    assert order[4:6] == [(32, "figure8", "bdsv_factor", 2), (32, "figure8", "pcg", 2)]


@pytest.mark.parametrize("multiplier", [.5, 1., 2.])
def test_histogram_retains_every_tail_sample(multiplier):
    values = np.concatenate((np.linspace(1., 2., 500), [100.]))
    assert np.histogram(values, bins=study.common_bins(values, multiplier))[0].sum() == len(values)


def synthetic_case(tmp_path, monkeypatch):
    binary = tmp_path / "fake.so"
    binary.write_bytes(b"CPU test binary identity only")
    monkeypatch.setattr(study.tm, "Mpc", lambda params, path: NS(params=params, close=lambda: None))

    def loop(plant, mpc, task, law, **options):
        state = tm.State(0., tm.Q_START.copy(), np.zeros(7))
        stats = dict(solve_time_us=900., wall_s=.001, iterations=1, budget_hit=False, sqp=1, pcg=10,
                     zero_pcg_rows=0, admm=2, admm_linsys=10, admm_linsys_mode="pcg", pred_err=0.)
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
    assert {file.name for file in directory.iterdir()} == {"case.json", "stats.npz", "stats.csv"}
    columns = np.load(directory / "stats.npz", allow_pickle=False)
    assert set(columns.files) == {"simulation_s", *study.SCALAR_STATS}
    assert all(columns[key].ndim == 1 for key in columns)
    rows, quality = study.timing_rows(tmp_path)
    assert len(rows) == 1 and not quality[0]["recording_complete"] and quality[0]["status"] == "fault"
    assert quality[0]["unexpected_admm_count"] == 0


@pytest.mark.parametrize("problem", ["missing", "nan", "mode", "fingerprint", "resolved"])
def test_analysis_rejects_misleading_timing_or_pairing(tmp_path, monkeypatch, problem):
    directory, _ = synthetic_case(tmp_path, monkeypatch)
    stats = dict(np.load(directory / "stats.npz", allow_pickle=False))
    case = json.loads((directory / "case.json").read_text())
    if problem == "missing":
        del stats["wall_s"]
    elif problem == "nan":
        stats["solve_time_us"] = np.array([np.nan])
    elif problem == "mode":
        stats["admm_linsys_mode"] = np.array(["bdsv_factor"])
    elif problem == "fingerprint":
        case["config"]["input_fingerprint"] = "wrong"
    else:
        case["config"]["params"]["solver"]["max_sqp_iters"] = 5
    np.savez(directory / "stats.npz", **stats)
    study.write_json(directory / "case.json", case)
    with pytest.raises(ValueError):
        study.timing_rows(tmp_path)


def test_empty_case_never_implies_limits_pass(tmp_path):
    task, _, _ = study.task_configuration("reach")
    assert study.Quality(task).result()["limits_pass"] is None


def test_actual_virtual_loop_records_600_scalar_solves_without_a_run_record(tmp_path, monkeypatch):
    import pinocchio as pin
    plant = tm.SimConfig().build()
    gravity = pin.computeGeneralizedGravity(plant.model, plant.data, tm.Q_START)
    plant.close()

    class CpuMpc:
        def __init__(self, params, binary):
            self.params, self.N, self.dt, self.plan = params, params.N, params.dt_s, None

        def warmup(self, state, reference):
            pass

        def request(self, state, reference, updates=None):
            states = np.tile(np.concatenate((state.q, state.qd)), (self.N, 1))
            stats = dict(solve_time_us=900., wall_s=.001, iterations=1, budget_hit=False, sqp=1, pcg=10,
                         zero_pcg_rows=0, admm=2, admm_linsys=10, admm_linsys_mode="pcg", pred_err=0.)
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
    assert case["status"] == "done" and case["solves"] == 600
    assert case["quality"]["observations"] == 3001
    assert {p.name for p in (tmp_path / "case").iterdir()} == {"case.json", "stats.npz", "stats.csv"}
