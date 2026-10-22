"""ROS-free persistent MPC child process; imports no CUDA or Pinocchio on load."""
from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import time

import numpy as np

from .playback import knot_torques
from .protocol import SolveReply, SolveRequest, WorkerReady



@dataclass(frozen=True)
class WorkerConfig:
    config_path: str | None = None
    budget_s: float | None = None

    def __post_init__(self):
        if self.budget_s is not None and (not math.isfinite(self.budget_s) or self.budget_s <= 0):
            raise ValueError("budget_s must be finite and positive")


class MpcEngine:
    """Small testable boundary around the public GATO MPCController API."""

    def __init__(self, controller, *, signs, n, dt_s, ee_position, budget_s=None, details=None):
        self.controller = controller
        self.signs = np.asarray(signs, dtype=float)
        if self.signs.shape != (7,) or not np.isin(self.signs, [-1., 1.]).all():
            raise ValueError("Expected seven GPU coordinate signs")
        self.state_signs = np.tile(self.signs, 2)
        self.n = n
        self.dt_s = dt_s
        self.ee_position = ee_position
        self.budget_s = budget_s
        self.details = details or {}
        self._generation = None

    @classmethod
    def build(cls, config):
        """Initialize and check the model in the worker, before WorkerReady."""
        from .configuration import load_configuration
        from .controller import build_controller
        from .model_runtime import ee_position

        selected, _, effective = load_configuration(config.config_path)
        setup = build_controller(effective)
        details = dict(setup.details, configuration_file=str(selected.resolve()))
        return cls(setup.controller, signs=setup.signs,
                   n=effective["N"], dt_s=effective["dt_s"],
                   ee_position=lambda q: ee_position(setup.model, setup.data, q, setup.frame_id),
                   budget_s=config.budget_s, details=details)

    def solve(self, request):
        q, qd = np.asarray(request.state.q, dtype=float), np.asarray(request.state.qd, dtype=float)
        if q.shape != (7,) or qd.shape != (7,) or not np.isfinite(np.r_[q, qd]).all():
            raise ValueError("Measured state must contain seven finite native positions/velocities")
        if request.dt_elapsed_s is not None and (
                not math.isfinite(request.dt_elapsed_s) or request.dt_elapsed_s < 0):
            raise ValueError("dt_elapsed_s must be finite and nonnegative")
        reference = request.task.horizon(request.task_elapsed_s, self.n, self.dt_s)
        x_gpu = np.r_[q, qd] * self.state_signs
        reset = self._generation != request.task.generation
        if reset:
            self.controller.reset(x_gpu)
            self._generation = request.task.generation
        step = self.controller.step(x_gpu, reference,
                                    dt_elapsed=request.dt_elapsed_s, budget_s=self.budget_s)
        # The whole plan leaves the worker; the runner plays it back by knot.
        # TODO(torque command): what remains deferred is noted in tiago_mpc/runner.py.
        plan = np.asarray(step.xu_best, dtype=float).reshape(-1)
        if plan.size != self.n * 21 - 7 or not np.isfinite(plan).all():
            raise FloatingPointError("Nonfinite or malformed MPC trajectory")
        torques = knot_torques(plan, self.signs)
        measured_ee = np.asarray(self.ee_position(q), dtype=float)
        terminal_q_native = plan[-14:-7] * self.signs
        predicted_ee = np.asarray(self.ee_position(terminal_q_native), dtype=float)
        if not np.isfinite(np.r_[measured_ee, predicted_ee]).all():
            raise FloatingPointError("Nonfinite native end-effector position")
        raw = step.solve.stats
        stats = dict(
            solve_time_us=float(step.solve.solve_time_us), wall_s=float(step.wall_s),
            iterations=int(step.iterations), budget_hit=bool(step.budget_hit),
            sqp=int(np.sum(raw.sqp_iters)), pcg=int(np.sum(raw.pcg_iters)),
            zero_pcg_rows=int(np.sum(np.asarray(raw.pcg_iters) == 0)),
            admm=int(np.sum(raw.admm_iters)) if raw.admm_iters is not None else 0,
            admm_linsys=int(np.sum(raw.admm_linsys_iters)) if raw.admm_linsys_iters is not None else 0,
            admm_linsys_mode=step.admm_linsys,
            row_viol=None if step.row_viol is None else np.asarray(step.row_viol).tolist(),
            position_error_m=float(np.linalg.norm(measured_ee - reference[0, :3])),
            predicted_endpoint_error_m=float(np.linalg.norm(predicted_ee - reference[-1, :3])),
            ee_xyz_m=measured_ee.tolist(), controller_reset=reset,
            task_elapsed_s=float(request.task_elapsed_s))
        return torques, stats


def _reply(request, now, *, torques=None, dt_s=math.nan, stats=None, error=""):
    return SolveReply(
        request_id=request.request_id, generation=request.task.generation,
        source_seq=request.state.seq,
        source_received_monotonic_sec=request.state.received_monotonic_sec,
        finished_monotonic_sec=now, expires_monotonic_sec=request.deadline_monotonic_sec,
        torques_native=torques, dt_s=dt_s, stats={} if stats is None else stats, error=error)


def worker_loop(config, request_q, response_q, stop_event, *, engine_factory=None, clock=None):
    """CPU-testable process body. Fatal solve errors produce a reply then exit."""
    engine_factory = MpcEngine.build if engine_factory is None else engine_factory
    clock = time.monotonic if clock is None else clock
    try:
        engine = engine_factory(config)
    except Exception as exc:
        response_q.put(WorkerReady(False, error=f"{type(exc).__name__}: {exc}"))
        return
    response_q.put(WorkerReady(True, details=engine.details))
    while not stop_event.is_set():
        try:
            request = request_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if request is None:
            return
        if not isinstance(request, SolveRequest):
            return  # Invalid process protocol; parent observes worker death.
        now = clock()
        if not math.isfinite(request.deadline_monotonic_sec) or now >= request.deadline_monotonic_sec:
            response_q.put(_reply(request, now, error="request expired before solve"))
            continue
        try:
            torques, stats = engine.solve(request)
        except Exception as exc:
            response_q.put(_reply(request, clock(), error=f"{type(exc).__name__}: {exc}"))
            return
        response_q.put(_reply(request, clock(), torques=torques, dt_s=engine.dt_s, stats=stats))


def worker_main(config, request_q, response_q, stop_event):
    """Spawn target: keep construction of CUDA/model objects inside this call."""
    worker_loop(config, request_q, response_q, stop_event)
