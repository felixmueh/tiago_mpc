"""CPU checks for process protocol and the native/GPU coordinate boundary."""
from dataclasses import replace
import json
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from tiago_mpc.mpc_worker import MpcEngine, WorkerConfig, worker_loop
from tiago_mpc.configuration import load_configuration
from tiago_mpc.controller import set_bounds
from tiago_mpc.protocol import SolveReply, SolveRequest, TaskReference, WorkerReady
from tiago_mpc.ros2 import ArmState

SIGNS = np.array([1., 1., 1., 1., 1., -1., 1.])
ROOT = Path(__file__).resolve().parents[1]


def test_tracking_configuration_keeps_weights_limits_and_explicit_smooth_losses():
    selected, settings, effective = load_configuration(ROOT / "configs/tracking.json")
    assert selected == ROOT / "configs/tracking.json"
    assert (effective["N"], effective["dt_s"]) == (32, .002)
    assert effective["solver_params"]["q_cost"] == 4
    assert effective["solver_params"]["N_cost"] == 8
    assert effective["solver_params"]["u_cost"] == 1e-6
    assert all(effective["solver_params"][key] == 0
               for key in ("q_lim_cost", "vel_lim_cost", "ctrl_lim_cost"))
    assert effective["controller"]["limits_kw"]["iters"] == 100
    losses = effective["prediction_joint_dynamics"]
    assert losses == {**settings["joint_dynamics"], "model": "coulomb_tanh_viscous",
                      "smoothing_rad_s": .03}
    _, _, reaching = load_configuration(None)
    assert reaching["prediction_joint_dynamics"] is None
    assert reaching["dt_s"] == .01


def test_tracking_reference_matches_simulation_including_edge_clearance():
    shape = json.loads((ROOT / "configs/tracking.json").read_text())["figure8"]
    task = TaskReference((.3, -.2, .1), kind="figure8", **shape)
    times = 1.3 + np.arange(7)*.002
    phase = 2*np.pi*times/shape["period_s"]
    expected = np.tile(task.target_xyz_m, (len(times), 1))
    expected[:, 0] += shape["edge_clearance_m"] + .5*shape["x_span_m"]*(1-np.cos(phase))
    expected[:, 1] += shape["y_amplitude_m"]*np.sin(2*phase)
    np.testing.assert_allclose(task.horizon(1.3, 7, .002)[:, :3], expected, atol=2e-8)
    assert task.horizon(0., 2, .002)[0, 0] == pytest.approx(.32)


def request(*, generation=1, elapsed=0.0, deadline=100.0):
    return SolveRequest(
        7, ArmState(np.arange(7) * .1, np.arange(7) * -.02, 1.0, 10.0, seq=11),
        TaskReference((.3, -.2, .1), generation=generation, kind="figure8"),
        elapsed, deadline, dt_elapsed_s=.017)


class Controller:
    def __init__(self):
        self.resets = []
        self.calls = []
        self.torque = np.arange(7, dtype=float)
        self.bad_plan = False

    def reset(self, x):
        self.resets.append(x.copy())

    def step(self, x, reference, *, dt_elapsed, budget_s):
        self.calls.append((x.copy(), reference.copy(), dt_elapsed, budget_s))
        # Final state has signed q=2*x_q so native FK must receive native 2*q.
        plan = np.r_[x, self.torque, x * 2]
        if self.bad_plan:
            plan[-1] = np.nan
        raw = SimpleNamespace(sqp_iters=np.array([2]), pcg_iters=np.array([[3], [0]]),
                              admm_iters=np.array([5]), admm_linsys_iters=np.array([20]))
        return SimpleNamespace(
            u=self.torque.copy(), xu_best=plan, solve=SimpleNamespace(stats=raw, solve_time_us=1234),
            wall_s=.002, iterations=2, budget_hit=False, row_viol=np.array([.01, .02, .03]),
            admm_linsys="pcg")


def engine(controller=None):
    calls = []

    def fk(q):
        calls.append(q.copy())
        return q[:3]

    controller = controller or Controller()
    result = MpcEngine(controller, signs=SIGNS, n=2, dt_s=.01,
                       ee_position=fk, budget_s=.004, details={"N": 2})
    return result, controller, calls


def test_native_order_and_signed_joint_cross_only_gpu_boundary():
    item = request(elapsed=1.5)
    worker, controller, fk_calls = engine()
    u, stats = worker.solve(item)
    expected_x = np.r_[item.state.q, item.state.qd] * np.tile(SIGNS, 2)
    np.testing.assert_allclose(controller.resets[0], expected_x)
    np.testing.assert_allclose(controller.calls[0][0], expected_x)
    np.testing.assert_allclose(u, np.arange(7) * SIGNS)
    np.testing.assert_allclose(fk_calls[0], item.state.q)
    np.testing.assert_allclose(fk_calls[1], item.state.q * 2)
    assert stats["sqp"] == 2 and stats["pcg"] == 3 and stats["zero_pcg_rows"] == 1
    assert stats["row_viol"] == [.01, .02, .03]
    assert stats["position_error_m"] == pytest.approx(
        np.linalg.norm(item.state.q[:3] - item.task.horizon(1.5, 2, .01)[0, :3]))


def test_public_elapsed_and_task_phase_pass_through_without_private_shift():
    item = request(elapsed=3.)
    worker, controller, _ = engine()
    worker.solve(item)
    _, horizon, elapsed, budget = controller.calls[0]
    np.testing.assert_array_equal(horizon, item.task.horizon(3., 2, .01))
    assert elapsed == .017 and budget == .004
    # The future reference advances by the configured knot dt, not dt_elapsed.
    assert not np.array_equal(horizon[0], horizon[1])
    assert not hasattr(controller, "_XU")


def test_only_new_generation_resets_from_current_measured_state():
    worker, controller, _ = engine()
    worker.solve(request())
    worker.solve(request(elapsed=.02))
    newer = request(generation=2, elapsed=0.)
    newer = replace(newer, state=replace(newer.state, q=np.ones(7) * .2))
    worker.solve(newer)
    assert len(controller.resets) == 2
    np.testing.assert_allclose(controller.resets[-1][:7], newer.state.q * SIGNS)


@pytest.mark.parametrize("field", ["torque", "plan", "state"])
def test_nonfinite_results_or_measurements_are_rejected(field):
    worker, controller, _ = engine()
    item = request()
    if field == "torque":
        controller.torque[5] = np.nan
    elif field == "plan":
        controller.bad_plan = True
    else:
        item = replace(item, state=replace(item.state, q=np.full(7, np.nan)))
    with pytest.raises((FloatingPointError, ValueError)):
        worker.solve(item)


def run_loop(items, factory, times):
    requests, replies = Queue(), Queue()
    for item in items:
        requests.put(item)
    requests.put(None)
    clock_values = iter(times)
    worker_loop(WorkerConfig(), requests, replies, Event(),
                engine_factory=factory, clock=lambda: next(clock_values))
    return list(replies.queue), requests


def test_expiry_is_original_absolute_deadline_even_if_solve_finishes_late():
    worker, _, _ = engine()
    replies, _ = run_loop([request(deadline=12.)], lambda _: worker, [11., 13.])
    assert replies[0] == WorkerReady(True, details={"N": 2})
    reply = replies[1]
    assert isinstance(reply, SolveReply)
    assert reply.finished_monotonic_sec == 13.
    assert reply.expires_monotonic_sec == 12.
    assert reply.source_received_monotonic_sec == 10. and reply.source_seq == 11
    assert reply.generation == 1 and reply.request_id == 7


def test_already_expired_request_avoids_solve_and_preserves_deadline():
    worker, controller, _ = engine()
    replies, _ = run_loop([request(deadline=12.)], lambda _: worker, [12.])
    assert not controller.calls
    assert replies[1].error == "request expired before solve"
    assert replies[1].expires_monotonic_sec == 12.
    assert replies[1].u_native is None


def test_initialization_failure_reports_not_ready():
    def fail(_):
        raise RuntimeError("model mismatch")

    replies, _ = run_loop([], fail, [])
    assert replies == [WorkerReady(False, error="RuntimeError: model mismatch")]


def test_solve_failure_reports_error_and_exits_before_next_request():
    worker, controller, _ = engine()
    controller.torque[0] = np.nan
    replies, pending = run_loop([request(), request(generation=2)], lambda _: worker, [11., 12.])
    assert len(replies) == 2
    assert "FloatingPointError" in replies[1].error
    assert replies[1].u_native is None
    assert pending.qsize() == 2  # next request and shutdown marker remain unconsumed


def test_selected_current_bounds_map_native_joint_box_and_limit_fractions():
    rows = {}
    solver = SimpleNamespace(set_row_group_bounds=lambda group, lo, hi: rows.update({group: (lo, hi)}))
    model = SimpleNamespace(lowerPositionLimit=np.full(7, -2.), upperPositionLimit=np.full(7, 3.),
                            velocityLimit=np.arange(7) + 1., effortLimit=np.arange(7) + 10.)
    set_bounds(solver, model, SIGNS, dict(joint_margin_rad=.1, velocity_fraction=.8, effort_fraction=.85))
    assert rows[0][0][5] == -2.9 and rows[0][1][5] == 1.9
    assert rows[0][0][0] == -1.9 and rows[0][1][0] == 2.9
    np.testing.assert_allclose(rows[1][1], .8 * model.velocityLimit)
    np.testing.assert_allclose(rows[2][1], .85 * model.effortLimit)
    rows.clear()
    set_bounds(solver, model, SIGNS, dict(joint_margin_rad=.1, velocity_fraction=None, effort_fraction=None))
    assert set(rows) == {0}  # explicit legacy config retains its compiled speed/effort offsets


@pytest.mark.parametrize("budget", [0., -1., np.nan, np.inf])
def test_budget_must_be_finite_and_positive(budget):
    with pytest.raises(ValueError):
        WorkerConfig(budget_s=budget)
