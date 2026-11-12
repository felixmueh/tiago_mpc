"""Check the recovered protocol, scalar-only recording and failure accounting without CUDA."""
from dataclasses import asdict
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import jetson_nb_matrix as matrix


def test_original_grid_and_scaled_cost_protocol():
    protocol = matrix.describe(matrix.Matrix())
    assert len(protocol["cells"]) == 30
    assert protocol["measured_calls"] == 18000
    assert protocol["excluded_initialization_calls"] == 60
    params = protocol["solver"]
    assert params["max_sqp_iters"] == 1 and params["max_pcg_iters"] == 120
    assert params["q_cost"] == 15 and params["N_cost"] == 80
    assert params["qd_cost"] == .0375 and params["u_cost"] == .000375
    assert params["q_lim_cost"] == .0375 and params["vel_lim_cost"] == .00375
    assert params["ctrl_lim_cost"] == .01125 and params["linsys"] == "pcg"
    assert protocol["cells"][-1]["horizon_s"] == 511 * .03


def test_reference_uses_fixed_spacing_and_periodic_interpolation():
    start = np.array([.4, -.2, -.3])
    samples = matrix.reference_samples(start)
    assert samples.shape == (2400, 6)
    np.testing.assert_allclose(samples[0, :3], start + [.02, 0, 0])
    np.testing.assert_allclose(matrix.reference_at(samples, .005, 16)[0], (samples[0] + samples[1]) / 2)
    np.testing.assert_allclose(matrix.reference_at(samples, 24., 16), matrix.reference_at(samples, 0., 16), atol=1e-12)
    np.testing.assert_allclose(matrix.reference_at(samples, .01, 512)[:16], matrix.reference_at(samples, .01, 16))


class Solver:
    N = 16
    batch_size = 2
    params = matrix.solver_parameters()

    def __init__(self):
        self.inputs = []
        self.results = []
        self.resets = 0

    def reset_dual(self):
        pass

    def reset_rho(self):
        self.resets += 1

    def solve(self, states, references, seed, *, budget_s):
        assert budget_s is None
        self.inputs.append((states.copy(), references.copy(), seed.copy()))
        xu = seed.copy()
        xu[0, 14:21] = np.arange(7) + len(self.inputs)
        xu[1, 14:21] = -10
        stats = SimpleNamespace(pcg_iters=np.array([[3, 4]]), sqp_iters=np.array([1, 1]),
                                initial_merit=np.array([4., 5.]), final_merit=np.array([2., 3.]))
        result = SimpleNamespace(xu=xu, solve_time_us=100, stats=stats)
        self.results.append(result)
        return result


def test_initialization_is_excluded_and_lane_zero_is_unshifted(monkeypatch):
    matrix.tm.STOP.clear()
    monkeypatch.setattr(matrix, "quality", lambda *args: dict(tracking_error_m=.01))
    solver, stream, commands = Solver(), io.StringIO(), []
    def advance(state, torque, duration, step):
        commands.append(torque.copy())
        assert duration == .01 and step == .001
        return state + .001
    result = matrix.run_updates(solver, advance, stream, matrix.Matrix(updates=3))
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert result["status"] == "completed" and result["samples"] == 3
    assert solver.resets == 4 and len(solver.inputs) == 4
    assert [row["simulation_s"] for row in rows] == [0., .01, .02, .03]
    assert [row["initialization"] for row in rows] == [True, False, False, False]
    assert all(not isinstance(value, (dict, list)) for row in rows for value in row.values())
    for index in range(1, 4):
        states, _, seed = solver.inputs[index]
        expected = solver.results[index - 1].xu[0].copy()
        expected[:14] = states[0]
        np.testing.assert_array_equal(seed, np.tile(expected, (2, 1)))
        np.testing.assert_array_equal(commands[index - 1], solver.results[index - 1].xu[0, 14:21])


def test_nonfinite_plan_retains_bad_sample_without_another_update(monkeypatch):
    matrix.tm.STOP.clear()
    monkeypatch.setattr(matrix, "quality", lambda *args: dict(tracking_error_m=.01))
    solver, stream = Solver(), io.StringIO()
    original = solver.solve
    def solve(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(solver.inputs) == 2:
            result.xu[0, -1] = np.nan
        return result
    solver.solve = solve
    result = matrix.run_updates(solver, lambda state, *args: state, stream, matrix.Matrix(updates=3))
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert result["status"] == "nonfinite_plan" and result["samples"] == 1
    assert len(rows) == 2 and rows[-1]["finite_plan"] is False


def test_partial_rows_stay_available_but_headline_requires_both_repeats(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "plot_heatmaps", lambda *args: None)
    config = matrix.Matrix(knots=(16,), batches=(1,), updates=2)
    study = dict(config=asdict(config), attempts=[dict(N=16, batch_size=1, repeat=1, status="completed"),
                                                dict(N=16, batch_size=1, repeat=2, status="timeout")])
    (tmp_path / "study.json").write_text(json.dumps(study))
    base = dict(initialization=False, native_ms=1., wrapper_ms=1.1, finite_plan=True,
                pcg_slots=1, pcg_cap_count=0, sqp_mean=1., tracking_error_m=.01)
    for repeat, count in ((1, 2), (2, 1)):
        folder = tmp_path / f"N16_B1_R{repeat}"
        folder.mkdir()
        (folder / "solves.jsonl").write_text("".join(json.dumps(dict(base, update=i)) + "\n" for i in range(count)))
    matrix.analyze(tmp_path)
    cells = json.loads((tmp_path / "cells.json").read_text())
    assert cells[0]["measured_samples"] == 3 and cells[0]["completed_repeats"] == 1
    assert cells[0]["native_mean_ms"] is None and cells[0]["wrapper_mean_ms"] is None
    assert len((tmp_path / "solve-times.csv").read_text().splitlines()) == 4
    assert json.loads((tmp_path / "completion.json").read_text())["all_completed"] is False
    assert not list(tmp_path.rglob("*.npz"))


@pytest.mark.parametrize("complete,interrupted,expected", [(True, False, 0), (False, False, 1), (False, True, 130)])
def test_matrix_cli_reports_partial_and_interrupted_work(tmp_path, monkeypatch, complete, interrupted, expected):
    monkeypatch.setattr(matrix, "handle_stop_signals", lambda: None)
    monkeypatch.setattr(matrix.tm, "STOP", SimpleNamespace(is_set=lambda: interrupted))
    def run(out, config):
        (out / "completion.json").write_text(json.dumps(dict(all_completed=complete)))
    monkeypatch.setattr(matrix, "run_matrix", run)
    assert matrix.main(["--output", str(tmp_path)]) == expected
