"""MPC parameters with the TIAGo defaults, and the MPC built from them.

Experiments override the defaults with dataclasses.replace, for example
replace(TIAGO, solver=TIAGO.solver.replace(mu=1.)). MpcInProcess and MpcWorker
share one interface: request(state, reference, updates=None), result(block=False)
-> Plan or None, N, dt and close().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys

from gato import SolverParams
import numpy as np

from .joint_dynamics import Friction, register_robot, urdf_friction
from .model_runtime import EE_FRAME, check_model, ee_position, runtime_urdf
from .plan import Plan
from .plant_cache import cache_root, plant_name


@dataclass(frozen=True)
class AdmmLimits:
    """MPCController limits_kw for the ADMM limit rows."""
    rho: float = .01
    iters: int = 100
    eps_rel: float = 1e-3
    rho_q: float = 1.
    rho_qd: float = .3
    rho_u: float = 5e-4


@dataclass(frozen=True)
class MpcParams:
    """Everything the solver needs. The defaults are TIAGo reaching, loss-free."""
    N: int = 32
    dt_s: float = .01
    solver: SolverParams = SolverParams(
        max_sqp_iters=5, max_pcg_iters=120, pcg_tol=1e-3, mu=10., rho=.01,
        q_cost=4., qd_cost=.05, u_cost=1e-4, N_cost=8.,
        q_lim_cost=0., vel_lim_cost=0., ctrl_lim_cost=0.)
    ee_frame: str = EE_FRAME
    warm_start: str = "shift"
    reset_rho_each_step: bool = True
    admm_linsys: str = "pcg"
    admm: AdmmLimits = AdmmLimits()
    # Planning limits: joint box shrunk by the margin; speed and effort fractions of the raw limits.
    joint_margin_rad: float = .1
    velocity_fraction: float = .8
    effort_fraction: float = .85
    friction: Friction | None = None  # predictor friction; None is loss-free
    warmup: int = 0  # warm-up solves from the first request's state, before its plan
    budget_s: float | None = None  # SQP wall budget per solve (MPCController.step); None: max_sqp_iters

    def __post_init__(self):
        if self.N < 2 or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("N must be >=2 and dt_s finite and positive")
        if self.budget_s is not None and (not math.isfinite(self.budget_s) or self.budget_s <= 0):
            raise ValueError("budget_s must be finite and positive")

    @classmethod
    def from_dict(cls, values):
        """Inverse of dataclasses.asdict, also after a JSON round trip."""
        values = dict(values, solver=SolverParams(**values["solver"]),
                      admm=AdmmLimits(**values["admm"]))
        if values["friction"] is not None:
            values["friction"] = Friction(**values["friction"])
        return cls(**values)


TIAGO = MpcParams()
# The figure-eight protocol; moves to experiments/figure8.py in step 3.
FIGURE8 = replace(TIAGO, dt_s=.01, solver=TIAGO.solver.replace(mu=1., qd_cost=.01, u_cost=1e-6),
                  admm=replace(TIAGO.admm, rho_q=.1, rho_qd=.03), friction=urdf_friction(.03))


def set_bounds(solver, model, signs, params):
    """Install the planning limits in GPU coordinates."""
    margin = float(params.joint_margin_rad)
    if (not math.isfinite(margin) or margin < 0
            or margin >= .5 * np.min(model.upperPositionLimit - model.lowerPositionLimit)):
        raise ValueError("Invalid joint margin")
    lo = (model.lowerPositionLimit + margin) * signs
    hi = (model.upperPositionLimit - margin) * signs
    solver.set_row_group_bounds(0, np.minimum(lo, hi), np.maximum(lo, hi))
    for group, fraction, raw in ((1, params.velocity_fraction, model.velocityLimit),
                                 (2, params.effort_fraction, model.effortLimit)):
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Limit fractions must be finite and in (0, 1]")
        solver.set_row_group_bounds(group, -fraction * raw, fraction * raw)


@dataclass
class MpcSetup:
    controller: object
    model: object
    data: object
    frame_id: int
    signs: np.ndarray
    urdf_path: Path
    gpu_urdf_path: Path
    details: dict


def build_controller(params: MpcParams):
    """Build/check once; callers retain ownership of timing, state and I/O."""
    cache_dir = cache_root()  # exports GATO_CACHE_DIR before GATO compiles or loads
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    import gato
    import pinocchio as pin

    friction, ee_frame, n = params.friction, params.ee_frame, params.N
    path, gpu_path, signs, names, reference_frame = runtime_urdf(cache_dir, friction)
    model = pin.buildModelFromUrdf(str(path))
    expected_names = tuple(f"arm_right_{i}_joint" for i in range(1, 8))
    if (model.nq, model.nv) != (7, 7) or tuple(model.names)[1:] != expected_names or names != expected_names:
        raise ValueError("Expected seven native right-arm joints in order")
    data = model.createData()
    frame_id = model.getFrameId(ee_frame)
    if frame_id >= model.nframes:
        raise ValueError(f"Missing end-effector frame {ee_frame}")
    plant = plant_name(gpu_path, ee_frame, friction)
    adapter = register_robot(plant, gpu_path, n_points=n, ee_frame=ee_frame, friction=friction)
    solver = gato.BSQP(str(gpu_path), batch_size=1, N=n, dt=params.dt_s,
                       plant_type=plant, params=params.solver)
    errors = check_model(solver, model, data, frame_id, signs, friction)
    controller = gato.MPCController(
        solver, limits="admm", linsys="pcg", warm_start=params.warm_start,
        reset_rho_each_step=params.reset_rho_each_step, admm_linsys=params.admm_linsys,
        limits_kw=asdict(params.admm))
    set_bounds(solver, model, signs, params)
    details = dict(N=n, dt_s=params.dt_s, params=asdict(params),
        reference_frame=reference_frame, end_effector_frame=ee_frame,
        joint_names=list(names), gpu_joint_coordinate_signs=signs.tolist(), generator_adapter=adapter,
        joint_friction_smoothing_rad_s=(None if solver.joint_friction_smoothing is None
                                       else solver.joint_friction_smoothing.tolist()),
        q_lower=model.lowerPositionLimit.tolist(), q_upper=model.upperPositionLimit.tolist(),
        velocity_limits=model.velocityLimit.tolist(), effort_limits=model.effortLimit.tolist(),
        model_step_max_abs_errors=errors, plant_name=plant, cache_dir=str(cache_dir))
    return MpcSetup(controller, model, data, frame_id, signs, path, gpu_path, details)


# Named solver changes, applied before a solve in either Mpc. Each persists in the
# solver until changed again. Add a name when an experiment needs it.
UPDATES = {
    # dict of scalar weights: q_cost, qd_cost, u_cost, N_cost, *_lim_cost
    "cost_weights": lambda mpc, weights: mpc.controller.solver.set_cost_weights(**weights),
    # (N, 3) [ee, qd, u] weights per knot; None returns to the scalar weights
    "knot_weights": lambda mpc, weights: mpc.controller.solver.set_cost_weights_per_knot(weights),
    # (weight, q_nom): running posture cost toward the native posture q_nom
    "posture": lambda mpc, value: (mpc.controller.solver.set_q_pos_cost(value[0]),
                                   mpc.controller.solver.set_q_nom(np.asarray(value[1]) * mpc.signs)),
    # external wrench in GATO's layout (BSQP.set_f_ext_B)
    "external_force": lambda mpc, wrench: mpc.controller.solver.set_f_ext_B(wrench),
}


def _known(updates):
    unknown = set(updates or ()) - UPDATES.keys()
    if unknown:
        raise KeyError(f"unknown MPC updates: {sorted(unknown)}")


def _checked(mpc, plan, t):
    """Any knot may be commanded before the plan's lease ends, so every knot is checked."""
    if plan.t0 != t:
        raise RuntimeError("MPC result does not match the request")
    if plan.dt != mpc.dt:
        raise RuntimeError("MPC plan has an unexpected knot interval")
    if (plan.states.shape != (mpc.N, 14) or plan.torques.shape != (mpc.N - 1, 7)
            or not np.isfinite(plan.states).all() or not np.isfinite(plan.torques).all()
            or np.any(np.abs(plan.torques) > mpc.details["effort_limits"])):
        raise RuntimeError("invalid MPC plan or raw effort limit exceeded")
    return plan


class MpcInProcess:
    """The GATO controller in this process: request solves, result returns the plan.

    The first request resets the controller from its state (and runs the warm-up
    solves); later requests never reset it. A fresh controller needs a new Mpc,
    or controller.reset. ``controller`` is the raw GATO MPCController.
    """

    def __init__(self, params):
        setup = build_controller(params)
        self.params, self.N, self.dt = params, params.N, params.dt_s
        self.controller, self.signs, self.details = setup.controller, setup.signs, setup.details
        self._fk = lambda q: ee_position(setup.model, setup.data, q, setup.frame_id)
        self._previous_t = self._plan = None

    def request(self, state, reference, updates=None):
        _known(updates)
        self._plan = self._solve(state, reference, updates)

    def result(self, block=False):
        plan, self._plan = self._plan, None
        return None if plan is None else _checked(self, plan, plan.t0)

    def close(self):
        pass

    def _solve(self, state, reference, updates):
        q, qd = np.asarray(state.q, dtype=float), np.asarray(state.qd, dtype=float)
        if q.shape != (7,) or qd.shape != (7,) or not np.isfinite(np.r_[q, qd]).all():
            raise ValueError("Measured state must contain seven finite native positions/velocities")
        state_signs = np.tile(self.signs, 2)
        x = np.r_[q, qd] * state_signs
        for name, value in (updates or {}).items():
            UPDATES[name](self, value)
        if self._previous_t is None:
            self.controller.reset(x)
            for _ in range(self.params.warmup):
                self.controller.warmup(x, reference)
        # GATO uses dt_elapsed only to score hypotheses (none here).
        dt_elapsed = None if self._previous_t is None else state.t - self._previous_t
        self._previous_t = state.t
        step = self.controller.step(x, reference, dt_elapsed=dt_elapsed, budget_s=self.params.budget_s)
        xu = np.asarray(step.xu_best, dtype=float).reshape(-1)
        if xu.size != self.N * 21 - 7:
            raise ValueError("Malformed GATO trajectory")
        knots = np.append(xu, np.zeros(7)).reshape(self.N, 21)  # [x_k u_k], no u after the last state
        raw = step.solve.stats
        stats = dict(
            solve_time_us=float(step.solve.solve_time_us), wall_s=float(step.wall_s),
            iterations=int(step.iterations), budget_hit=bool(step.budget_hit),
            sqp=int(np.sum(raw.sqp_iters)), pcg=int(np.sum(raw.pcg_iters)),
            zero_pcg_rows=int(np.sum(np.asarray(raw.pcg_iters) == 0)),
            admm=int(np.sum(raw.admm_iters)) if raw.admm_iters is not None else 0,
            admm_linsys=int(np.sum(raw.admm_linsys_iters)) if raw.admm_linsys_iters is not None else 0,
            admm_linsys_mode=step.admm_linsys,
            row_viol=None if step.row_viol is None else np.array(step.row_viol),
            # The ROS runner's arrival check; moves to the task with the loop.
            position_error_m=float(np.linalg.norm(
                self._fk(q) - np.asarray(reference).reshape(self.N, 6)[0, :3])))
        return Plan(state.t, self.dt, knots[:, :14] * state_signs, knots[:-1, 14:] * self.signs, stats)


_SPAWN = mp.get_context("spawn")


class MpcWorker:
    """MpcInProcess in a spawned worker process, behind the same interface.

    __init__ returns once the worker has built and checked its controller. A
    failed solve replies with its error and ends the worker; result() raises on
    that reply, and on a dead worker, at the next call.
    """

    def __init__(self, params):
        self.N, self.dt = params.N, params.dt_s
        self._requests, self._replies = _SPAWN.Queue(), _SPAWN.Queue()
        self._process = _SPAWN.Process(target=_serve, args=(params, self._requests, self._replies),
                                       daemon=True)
        self._process.start()
        self._pending = None  # the outstanding request's state time
        try:
            self.details = self._reply(block=True)
        except BaseException:
            self.close()
            raise

    def request(self, state, reference, updates=None):
        _known(updates)
        if self._pending is not None:
            raise RuntimeError("an MPC request is pending")
        self._requests.put((state, reference, updates))
        self._pending = state.t

    def result(self, block=False):
        reply = self._reply(block and self._pending is not None)
        if reply is None:
            return None
        if self._pending is None:
            raise RuntimeError("unexpected MPC reply")
        t, self._pending = self._pending, None
        return _checked(self, reply, t)

    def close(self):
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(1.)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1.)
        for channel in (self._requests, self._replies):
            channel.cancel_join_thread()
            channel.close()

    def _reply(self, block):
        while True:
            try:
                reply = self._replies.get(timeout=.05) if block else self._replies.get_nowait()
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError(f"MPC worker exited unexpectedly: {self._process.exitcode}")
                if block:
                    continue
                return None
            if isinstance(reply, str):
                raise RuntimeError(f"MPC worker: {reply}")
            return reply


def _serve(params, requests, replies):
    """The worker: build, reply with the details (or the error), then one plan per request."""
    try:
        mpc = MpcInProcess(params)
        replies.put(mpc.details)
        while (item := requests.get()) is not None:
            replies.put(mpc._solve(*item))
    except Exception as exc:
        replies.put(f"{type(exc).__name__}: {exc}")
