"""MPC parameters with the TIAGo defaults, GATO's solver module for them, and the MPC.

Experiments override the defaults with dataclasses.replace, for example
replace(TIAGO, solver=TIAGO.solver.replace(mu=1.)). compile_binary compiles GATO's
module for the params (minutes, once per variant), find_binary finds it, Mpc and
MpcProcess load it; Mpc.compile(params) does both. They share one interface:
warmup(state, reference), request(state, reference, updates=None), which raises while
the previous request's plan is unread, result(block=False) -> Plan or None, N, dt,
params and close().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
from typing import Any

from gato import SolverParams
import numpy as np

from .dynamics import Friction, passive_torque
from .plan import Plan, State
from .tiago import ARM_URDF, GRASP_FRAME, Q_REACH, Q_START, ROOT, digest, fk, limits, runtime_urdf


@dataclass(frozen=True)
class AdmmLimits:
    """MPCController limits_kw for the ADMM limit rows; the defaults are GATO's
    (MPCController.LIMIT_DEFAULTS): at most iters inner iterations per SQP iteration."""
    rho: float = .01
    iters: int = 10
    eps_rel: float = 1e-2
    rho_q: float = .1
    rho_qd: float = .03
    rho_u: float = 1e-5


@dataclass(frozen=True)
class MpcParams:
    """Everything the solver needs. The defaults are TIAGo reaching, loss-free."""
    N: int = 32
    dt_s: float = .01
    solver: SolverParams = SolverParams(
        max_sqp_iters=5, max_pcg_iters=120, pcg_tol=1e-3, mu=10., rho=.01,
        q_cost=4., qd_cost=.05, u_cost=1e-4, N_cost=8.,
        q_lim_cost=0., vel_lim_cost=0., ctrl_lim_cost=0.)
    ee_frame: str = GRASP_FRAME
    warm_start: str = "shift"
    reset_rho_each_step: bool = True
    admm_linsys: str = "pcg"
    admm: AdmmLimits = AdmmLimits()
    # Planning limits: joint box shrunk by the margin; speed and effort fractions of the raw limits.
    joint_margin_rad: float = .1
    velocity_fraction: float = .8
    effort_fraction: float = .85
    friction: Friction | None = None  # predictor friction; None is loss-free
    # Wall time for a solve's SQP iterations (MPCController.step's budget_s), within
    # solver.max_sqp_iters; None: no time limit. Wall clock only: simulations set None.
    max_sqp_time_s: float | None = .01
    warmup: bool = True  # Mpc.warmup runs GATO's warm-up solve before the first plan

    def __post_init__(self):
        if self.N < 2 or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("N must be >=2 and dt_s finite and positive")
        if not isinstance(self.solver.max_sqp_iters, int) or self.solver.max_sqp_iters < 1:
            raise ValueError("solver.max_sqp_iters must be a positive integer")
        time = self.max_sqp_time_s
        if time is not None and (not math.isfinite(time) or time <= 0):
            raise ValueError("max_sqp_time_s must be finite and positive, or None")
        if self.friction is not None and self.friction.width_rad_s is None:
            raise ValueError("GATO predicts smooth Coulomb friction only; hard sign is plant-only")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MpcParams:
        """Inverse of dataclasses.asdict, also after a JSON round trip."""
        values = dict(values, solver=SolverParams(**values["solver"]),
                      admm=AdmmLimits(**values["admm"]))
        if values["friction"] is not None:
            values["friction"] = Friction(**values["friction"])
        return cls(**values)


TIAGO = MpcParams()


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


def cache_root() -> Path:
    """GATO_CACHE_DIR if set (host runs), else <repo>/.cache/gato; exported for GATO."""
    root = Path(os.environ.get("GATO_CACHE_DIR") or ROOT / ".cache/gato").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GATO_CACHE_DIR"] = str(root)
    return root


# GATO compiles one CUDA module per registered plant name and N (minutes). A fixed name
# per variant would let two variants overwrite each other in a shared cache, so the
# name hashes everything compiled into the module. The hash is a cache key, not provenance.
def plant_name(gpu_urdf_path: Path, ee_frame: str, friction: Friction | None = None) -> str:
    """tiago_right_<hash> over the GPU URDF bytes, end-effector frame and predictor
    friction as passed to registration (coefficients and width)."""
    config = json.dumps(friction and asdict(friction), sort_keys=True).encode()
    return "tiago_right_" + digest(Path(gpu_urdf_path).read_bytes(), ee_frame.encode(), config)


def _plant(params: MpcParams):
    """The Pinocchio arm, GPU URDF, GPU coordinate signs and GATO plant name for params.
    Exports GATO_CACHE_DIR and puts this environment's cmake first on PATH: GATO's
    module key includes the cmake it finds."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(ARM_URDF))
    if not model.existFrame(params.ee_frame):
        raise ValueError(f"Missing end-effector frame {params.ee_frame}")
    bin_dir = str(Path(sys.executable).parent)
    if os.environ["PATH"].split(os.pathsep)[0] != bin_dir:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
    gpu_path, signs = runtime_urdf(cache_root(), params.friction)
    return model, gpu_path, signs, plant_name(gpu_path, params.ee_frame, params.friction)


def _register(params: MpcParams, gpu_path: Path, plant: str) -> None:
    """GATO's code generation and manifest entry for the plant (seconds when new, never a
    compile). BSQP reads the end-effector frame from it."""
    import gato

    friction = params.friction
    gato.register_robot(plant, gpu_path, N=[params.N], ee_frame=params.ee_frame,
                        **({} if friction is None else dict(joint_friction_smoothing=friction.width_rad_s)))


def find_binary(params: MpcParams) -> Path | None:
    """The cached GATO module for params, or None when it is not built or the plant not
    registered. Never registers or compiles."""
    from gato import _cache

    plant = _plant(params)[3]
    return _cache.find_module(plant, params.N)


def compile_binary(params: MpcParams, use_cache: bool = True, jobs: int = 2) -> Path:
    """Register the plant and compile its GATO module (minutes) unless cached; returns its
    path. CPU only: no CUDA context. use_cache=False rebuilds: the old file is unlinked,
    not overwritten, so processes that loaded it keep their mapping. jobs: parallel
    compile jobs."""
    from gato import _cache

    _, gpu_path, _, plant = _plant(params)
    _register(params, gpu_path, plant)
    if not use_cache:
        with _cache._locked(_cache.cache_dir()):  # not while another process compiles it
            old = _cache.find_module(plant, params.N)
            if old is not None:
                old.unlink()
    return _cache.ensure_module(plant, params.N, jobs=jobs)


def load_controller(params: MpcParams, binary: Path | None):
    """Load exactly binary as GATO's module for params (never compiles), then build and
    check the controller; returns it and the GPU coordinate signs."""
    import importlib.util

    import gato

    model, gpu_path, signs, plant = _plant(params)
    name = f"bsqpN{params.N}_{plant}"
    if binary is None or not Path(binary).is_file():
        raise FileNotFoundError(f"No GATO module for these params{f' at {binary}' if binary else ''}; "
                                "compile_binary(params) builds it")
    binary = Path(binary).resolve()
    if binary.name.split(".")[0] != name:
        raise ValueError(f"{binary.name} is not the GATO module {name} for these params")
    # BSQP imports gato.<name> from sys.modules, so it never reaches its compile fallback.
    module = sys.modules.get("gato." + name)
    if module is None:
        spec = importlib.util.spec_from_file_location("gato." + name, binary)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {binary}")
        try:
            module = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(spec.name, None)
            raise ImportError(f"Cannot import {binary}: {exc}") from exc
    elif Path(module.__file__ or "").resolve() != binary:
        raise RuntimeError(f"gato.{name} is already loaded from {module.__file__}, not {binary}")
    _register(params, gpu_path, plant)
    friction = params.friction
    # Explicit friction widths: BSQP then never looks up the registration that built the module.
    solver = gato.BSQP(str(gpu_path), batch_size=1, N=params.N, dt=params.dt_s, plant_type=plant,
                       params=params.solver, joint_friction_smoothing=friction and friction.width_rad_s)
    check_model(solver, model, model.createData(), params.ee_frame, signs, friction)
    controller = gato.MPCController(
        solver, limits="admm", linsys="pcg", warm_start=params.warm_start,
        reset_rho_each_step=params.reset_rho_each_step, admm_linsys=params.admm_linsys,
        limits_kw=asdict(params.admm))
    set_bounds(solver, model, signs, params)
    return controller, signs


def check_model(solver, model, data, ee_frame, signs, friction=None):
    """The GPU model against Pinocchio's native one at two poses: FK; moving and at rest
    (the steepest tanh slope), one step of 1 ms and of dt against GATO's constant-
    acceleration (trapezoidal) step, and the first knot's A-block against finite
    differences of that step. Without friction the passive torque is zero."""
    import pinocchio as pin

    def step(x, u, h):
        q, v = x[:7], x[7:]
        a = pin.aba(model, data, q, v, u - passive_torque(v, friction)).copy()
        return np.r_[q + v*h + .5*a*h*h, v + a*h]

    ss = np.tile(signs, 2)
    for q in (Q_START, Q_REACH):
        np.testing.assert_allclose(solver.ee_pos(q*signs), fk(q, ee_frame), atol=1e-5, rtol=1e-4)
        for v in (np.array([.3, -.35, .4, -.45, .5, -.55, .6]), np.zeros(7)):
            x, u = np.r_[q, v], pin.rnea(model, data, q, v, np.zeros(7)) + passive_torque(v, friction)
            for h in (.001, solver.dt):
                gpu = np.asarray(solver.sim_forward((x*ss)[None], (u*signs)[None], h))[0]*ss
                np.testing.assert_allclose(gpu, step(x, u, h), atol=2e-3, rtol=1e-3)
            xu = np.tile(np.r_[x*ss, u*signs], solver.N)[:-7]
            kkt = solver.debug_setup_kkt(xu[None], (x*ss)[None], np.zeros((1, solver.N*6)))
            a_block = ss[:, None]*np.asarray(kkt["A"])[0].reshape(solver.N, 14, 14)[0].T*ss
            fd = np.array([step(x + d, u, solver.dt) - step(x - d, u, solver.dt)
                           for d in np.eye(14)*1e-5]).T / 2e-5
            np.testing.assert_allclose(a_block, fd, atol=2e-3, rtol=1e-2)
    solver.reset()


# Named solver changes, applied before a solve by Mpc (also in MpcProcess). Each persists
# in the solver until changed again. Add a name when an experiment needs it.
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
            or np.any(np.abs(plan.torques) > limits().effort)):
        raise RuntimeError("invalid MPC plan or raw effort limit exceeded")
    return plan


class Mpc:
    """The GATO controller in this process: request solves, result returns the plan.

    binary is the GATO module for params (find_binary, compile_binary); Mpc loads exactly
    it and never compiles. warmup resets the controller from its state and runs the
    warm-up solve; without one, the first request resets it from its state. Later
    requests never reset it. A fresh controller needs a new Mpc, or controller.reset.
    ``controller`` is the raw GATO MPCController, ``step`` the raw GATO result of the
    latest solve (MPCController.step; None before the first).
    """

    def __init__(self, params: MpcParams, binary: Path | None):
        self.controller, self.signs = load_controller(params, binary)
        self.params, self.N, self.dt = params, params.N, params.dt_s
        self.step: Any = None
        self._reset = False  # the controller has been reset from a state
        self._previous_t: float | None = None
        self._plan: Plan | None = None

    @classmethod
    def compile(cls, params: MpcParams, use_cache: bool = True) -> Mpc:
        """Mpc on compile_binary(params, use_cache)."""
        return cls(params, compile_binary(params, use_cache))

    def warmup(self, state: State, reference: np.ndarray) -> None:
        """With params.warmup, reset the controller from state and run GATO's warm-up solve
        (MPCController.warmup, the solver's own limits, no time limit); else nothing."""
        if self.params.warmup:
            x = self._x(state)
            self.controller.reset(x)
            self.controller.warmup(x, reference)
            self._reset = True

    def request(self, state: State, reference: np.ndarray, updates: dict[str, Any] | None = None) -> None:
        _known(updates)
        if self._plan is not None:
            raise RuntimeError("an MPC request is pending")
        self._plan = self._solve(state, reference, updates)

    def result(self, block: bool = False) -> Plan | None:
        plan, self._plan = self._plan, None
        return None if plan is None else _checked(self, plan, plan.t0)

    def close(self) -> None:
        pass

    def _x(self, state):
        """The native state in GPU coordinates."""
        q, qd = np.asarray(state.q, dtype=float), np.asarray(state.qd, dtype=float)
        if q.shape != (7,) or qd.shape != (7,) or not np.isfinite(np.r_[q, qd]).all():
            raise ValueError("Measured state must contain seven finite native positions/velocities")
        return np.r_[q, qd] * np.tile(self.signs, 2)

    def _solve(self, state, reference, updates):
        x = self._x(state)
        for name, value in (updates or {}).items():
            UPDATES[name](self, value)
        if not self._reset:
            self.controller.reset(x)
            self._reset = True
        # GATO uses dt_elapsed only to score hypotheses (none here).
        dt_elapsed = None if self._previous_t is None else state.t - self._previous_t
        self._previous_t = state.t
        self.step = step = self.controller.step(x, reference, dt_elapsed=dt_elapsed,
                                                budget_s=self.params.max_sqp_time_s)
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
            row_viol=None if step.row_viol is None else np.array(step.row_viol))
        states = knots[:, :14] * np.tile(self.signs, 2)
        return Plan(state.t, self.dt, states, knots[:-1, 14:] * self.signs, stats)


_SPAWN = mp.get_context("spawn")


class MpcProcess:
    """Mpc in a spawned process, behind the same interface.

    __init__ returns once the process has loaded binary and checked its controller;
    it never compiles. A failed solve replies with its error and ends the process;
    result() raises on that reply, and on a dead process, at the next call.
    """

    def __init__(self, params: MpcParams, binary: Path | None):
        self.params, self.N, self.dt = params, params.N, params.dt_s
        self._requests, self._replies = _SPAWN.Queue(), _SPAWN.Queue()
        self._process = _SPAWN.Process(target=_serve, args=(params, binary, self._requests, self._replies),
                                       daemon=True)
        self._process.start()
        self._pending: float | None = None  # the outstanding request's state time
        try:
            self._reply(block=True)  # ready
        except BaseException:
            self.close()
            raise

    @classmethod
    def compile(cls, params: MpcParams, use_cache: bool = True) -> MpcProcess:
        """This process compiles (compile_binary), the MPC process loads the module."""
        return cls(params, compile_binary(params, use_cache))

    def warmup(self, state: State, reference: np.ndarray) -> None:
        """Mpc.warmup in the MPC process; returns when it is done."""
        if self._pending is not None:
            raise RuntimeError("an MPC request is pending")
        if self.params.warmup:
            self._requests.put(("warmup", state, reference))
            self._reply(block=True)

    def request(self, state: State, reference: np.ndarray, updates: dict[str, Any] | None = None) -> None:
        _known(updates)
        if self._pending is not None:
            raise RuntimeError("an MPC request is pending")
        self._requests.put(("_solve", state, reference, updates))
        self._pending = state.t

    def result(self, block: bool = False) -> Plan | None:
        reply = self._reply(block and self._pending is not None)
        if reply is None:
            return None
        if self._pending is None:
            raise RuntimeError("unexpected MPC reply")
        t, self._pending = self._pending, None
        return _checked(self, reply, t)

    def close(self) -> None:
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
                    raise RuntimeError(f"MPC process exited unexpectedly: {self._process.exitcode}")
                if block:
                    continue
                return None
            if isinstance(reply, str):
                raise RuntimeError(f"MPC process: {reply}")
            return reply


def _serve(params, binary, requests, replies):
    """The process: load, reply None when ready (or the error), then one reply per
    request: a plan, or None after the warm-up."""
    try:
        mpc = Mpc(params, binary)
        replies.put(None)
        while (item := requests.get()) is not None:
            method, *args = item
            replies.put(getattr(mpc, method)(*args))
    except Exception as exc:
        replies.put(f"{type(exc).__name__}: {exc}")
