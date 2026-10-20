"""ROS-free persistent MPC child process; imports no CUDA or Pinocchio on load."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import queue
import sys
import time

import numpy as np

from .protocol import SolveReply, SolveRequest, WorkerReady
from .ros2 import RIGHT_ARM_JOINTS

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkerConfig:
    config_path: str | None = None
    budget_s: float | None = None
    cache_dir: str | None = None

    def __post_init__(self):
        if self.budget_s is not None and (not math.isfinite(self.budget_s) or self.budget_s <= 0):
            raise ValueError("budget_s must be finite and positive")


def _configuration(path):
    # Resolve this checkout explicitly; avoid importing a different examples
    # package from a user's environment. Called only by the child factory.
    spec = importlib.util.spec_from_file_location(
        "_tiago_mpc_reaching_config", ROOT / "examples/reaching_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_configuration(path)


def _set_bounds(solver, model, signs, bounds):
    """Install explicit selected bounds; preserve explicit legacy omissions."""
    margin = float(bounds["joint_margin_rad"])
    if (not math.isfinite(margin) or margin < 0
            or margin >= .5 * np.min(model.upperPositionLimit - model.lowerPositionLimit)):
        raise ValueError("Invalid joint margin")
    lo = (model.lowerPositionLimit + margin) * signs
    hi = (model.upperPositionLimit - margin) * signs
    solver.set_row_group_bounds(0, np.minimum(lo, hi), np.maximum(lo, hi))
    for group, key, raw in ((1, "velocity_fraction", model.velocityLimit),
                             (2, "effort_fraction", model.effortLimit)):
        fraction = bounds[key]
        if fraction is None:
            continue  # The explicit historical profile retains compiled offsets.
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Limit fractions must be finite and in (0, 1]")
        solver.set_row_group_bounds(group, -fraction * raw, fraction * raw)


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
        cache_dir = Path(config.cache_dir or ROOT / "_expr_data/ros2-runner-20261020/cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GATO_CACHE_DIR"] = str(cache_dir / "gato")
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
        import gato
        import pinocchio as pin
        from .model_runtime import EE_FRAME, PLANT, check_model, ee_position, runtime_urdf

        selected, _, effective = _configuration(config.config_path)
        path, gpu_path, signs, names, reference_frame = runtime_urdf(cache_dir)
        model = pin.buildModelFromUrdf(str(path))
        if (model.nq, model.nv) != (7, 7) or tuple(model.names)[1:] != RIGHT_ARM_JOINTS or names != RIGHT_ARM_JOINTS:
            raise ValueError("Native model joint dimensions/order differ from ROS right arm")
        data = model.createData()
        frame_id = model.getFrameId(EE_FRAME)
        if frame_id >= model.nframes:
            raise ValueError(f"Missing end-effector frame {EE_FRAME}")
        n, dt_s = effective["N"], effective["dt_s"]
        gato.register_robot(PLANT, gpu_path, N=[n], ee_frame=EE_FRAME, jobs=1)
        solver = gato.BSQP(str(gpu_path), batch_size=1, N=n, dt=dt_s,
                           plant_type=PLANT, params=gato.SolverParams(**effective["solver_params"]))
        model_errors = check_model(solver, model, data, frame_id, signs)
        controller = gato.MPCController(solver, limits="admm", linsys="pcg", **effective["controller"])
        _set_bounds(solver, model, signs, effective["limit_bounds"])
        details = dict(
            N=n, dt_s=dt_s, configuration_file=str(selected.resolve()),
            effective_configuration=effective, reference_frame=reference_frame,
            ee_frame=EE_FRAME, joint_names=list(names), gpu_joint_coordinate_signs=signs.tolist(),
            q_lower=model.lowerPositionLimit.tolist(), q_upper=model.upperPositionLimit.tolist(),
            velocity_limits=model.velocityLimit.tolist(), effort_limits=model.effortLimit.tolist(),
            model_step_max_abs_errors=model_errors, cache_dir=str(cache_dir))
        return cls(controller, signs=signs, n=n, dt_s=dt_s,
                   ee_position=lambda q: ee_position(model, data, q, frame_id),
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
        u = np.asarray(step.u, dtype=float)
        plan = np.asarray(step.xu_best, dtype=float).reshape(-1)
        if u.shape != (7,) or not np.isfinite(u).all():
            raise FloatingPointError("Nonfinite or malformed MPC torque")
        if plan.size != self.n * 21 - 7 or not np.isfinite(plan).all():
            raise FloatingPointError("Nonfinite or malformed MPC trajectory")
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
        return u * self.signs, stats


def _reply(request, now, *, u=None, stats=None, error=""):
    return SolveReply(
        request_id=request.request_id, generation=request.task.generation,
        source_seq=request.state.seq,
        source_received_monotonic_sec=request.state.received_monotonic_sec,
        finished_monotonic_sec=now, expires_monotonic_sec=request.deadline_monotonic_sec,
        u_native=u, stats={} if stats is None else stats, error=error)


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
            u, stats = engine.solve(request)
        except Exception as exc:
            response_q.put(_reply(request, clock(), error=f"{type(exc).__name__}: {exc}"))
            return
        response_q.put(_reply(request, clock(), u=u, stats=stats))


def worker_main(config, request_q, response_q, stop_event):
    """Spawn target: keep construction of CUDA/model objects inside this call."""
    worker_loop(config, request_q, response_q, stop_event)
