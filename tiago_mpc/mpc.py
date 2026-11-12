"""MPC parameters with the TIAGo defaults, GATO's solver module for them, and the MPC.

Experiments override the defaults with dataclasses.replace, for example
replace(TIAGO, solver=replace(TIAGO.solver, mu=1.)). compile_binary compiles GATO's
module for the params (minutes, once per variant), find_binary finds it, Mpc and
MpcProcess load it; Mpc.compile(params) does both. They share one interface:
warmup(state, reference), request(state, reference, updates=None), which raises while
the previous request's plan is unread, result(block=False) -> Plan or None, N, dt,
params and close().
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
from typing import Any

from gato import SolverParams
from gato.cost_menu import CostMenu
import numpy as np

from tiago_mpc.collision import Collision, arm_cover
from tiago_mpc.costs import default_cost, with_reference
from tiago_mpc.dynamics import Friction, Tanh, passive_torque
from tiago_mpc.interfaces import Plan, Prediction, State
from tiago_mpc.tiago import ARMATURE_KG_M2, GRASP_FRAME, Q_REACH, Q_START, arm_urdf, fk, runtime_urdf
from tiago_mpc.util import cache_root, digest


@dataclass(frozen=True)
class SolverSettings:
    """GATO's solver settings, with tracking weights owned by the cost menu."""
    max_sqp_iters: int = 1
    max_pcg_iters: int = 200
    pcg_tol: float = 1e-4
    solve_ratio: float = 1.
    mu: float = 10.
    rho: float = .01
    adapt_rho: bool = True
    linsys: str | None = None
    q_lim_cost: float = .01
    vel_lim_cost: float = 0.
    ctrl_lim_cost: float = 0.
    exact_hessian: bool = False
    cost_curvature: bool = False
    defect_telemetry: bool = False
    defect_exit: float = 0.

    def native(self) -> SolverParams:
        """Return the native solver parameters accepted by GATO."""
        return SolverParams(**asdict(self))

    @classmethod
    def resolved(cls, params: SolverParams) -> SolverSettings:
        """Read the exposed settings back from GATO's native parameters."""
        return cls(**{field.name: getattr(params, field.name) for field in fields(cls)})


@dataclass(frozen=True)
class AdmmLimits:
    """Settings of GATO's ADMM limit rows, which keep the plan within the joint limits.

    None follows GATO's default. GATO's shared rho is left out because every box sets its own,
    and its torque-box options u_scale and u_clip because effort_fraction sets the torque box."""
    iters: int | None = None          # inner iterations per SQP iteration, at most
    eps_abs: float | None = None      # residual exit: absolute tolerance
    eps_rel: float | None = None      # residual exit: relative tolerance
    check_every: int | None = None    # inner iterations between exit checks
    rho_q: float | None = None        # penalty of the joint position rows
    rho_qd: float | None = None       # penalty of the joint velocity rows
    rho_u: float | None = None        # penalty of the torque rows

    @classmethod
    def resolved(cls, solver) -> AdmmLimits:
        """The settings GATO's solver runs, read from the solver itself.

        The penalties come back in float32. Each is written as its shortest decimal, which
        gives the same float32 again when a record is replayed."""
        from gato.rowkinds import KIND_BOX_Q, KIND_BOX_QD, KIND_BOX_U

        rho = {int(group["kind"]): group["rho"] for group in solver.get_row_groups()}
        boxes = dict(rho_q=KIND_BOX_Q, rho_qd=KIND_BOX_QD, rho_u=KIND_BOX_U)
        if not set(boxes.values()) <= set(rho):
            raise RuntimeError("the solver has no joint limit rows")
        return cls(iters=solver.admm_iters, eps_abs=solver.admm_eps_abs, eps_rel=solver.admm_eps_rel,
                   check_every=solver.admm_check_every,
                   **{name: float(str(np.float32(rho[kind]))) for name, kind in boxes.items()})


# The MpcParams fields that are MPCController settings: passed only when set, and read back
# from the controller once it is built.
CONTROLLER_SETTINGS = ("warm_start", "reset_rho_each_step", "admm_linsys", "admm_threshold", "linsys",
                       "bdsv_threshold", "reseed_threshold")


@dataclass(frozen=True)
class MpcParams:
    """Everything the solver needs. The defaults are TIAGo reaching, loss-free, with the fitted armature.

    A setting that GATO has a default for follows it unless a value is set here: a
    SolverSettings field that is not given, or None in AdmmLimits or a controller setting. A
    set value deviates from GATO, and tests/test_gato_defaults.py lists the deviations of
    TIAGO and FIGURE8. An MPC's params, and so a record's, are the values GATO resolved and
    ran (see resolved)."""
    N: int = 32
    dt_s: float = .03
    solver: SolverSettings = SolverSettings(
        max_sqp_iters=5, max_pcg_iters=120, pcg_tol=1e-3,
        q_lim_cost=0.)  # limits through ADMM only
    # All execution uses GATO's native menu. Its direct tracking objective may be faster
    # for position-only tasks, but cannot supply the orientation cost.
    # A reference passed to request or warmup explicitly replaces its grasp-frame targets.
    cost: CostMenu = field(default_factory=default_cost)
    # Controller settings (CONTROLLER_SETTINGS); None follows GATO.
    warm_start: str | None = None             # "shift" or "hold"
    reset_rho_each_step: bool | None = None
    # The ADMM inner loop's linear solver: "pcg", "bdsv_factor" or "auto". Under "auto", a solve
    # uses bdsv_factor when its pred_err (measured minus predicted state) exceeds admm_threshold
    # or the previous solve hit the PCG cap, and pcg otherwise. GATO's default picks by N.
    admm_linsys: str | None = None
    admm_threshold: float | None = None
    # The SQP solve's linear solver and its threshold under "auto". The solve does not use them
    # while ADMM limit rows are active; they are recorded for completeness.
    linsys: str | None = None
    bdsv_threshold: float | None = None
    reseed_threshold: float | None = None     # pred_err above which GATO re-seeds the warm start
    # GATO's rho_u suits its own control cost; 5e-4 is five times TIAGO's torque weight of 1e-4.
    admm: AdmmLimits = AdmmLimits(rho_u=5e-4)
    # Planning limits: joint box shrunk by the margin; speed and effort fractions of the raw limits.
    joint_margin_rad: float = .1
    velocity_fraction: float = .6
    effort_fraction: float = .85
    friction: Friction | None = None  # predictor friction; None is loss-free
    # GATO's differentiable Coulomb evaluation. Its explicit velocity update needs
    # dt < 2/lambda_max(M^-1 diag(b + c/w)): with the fitted coefficients and armature,
    # 17 ms at w = 0.1 (5.8 ms at 0.03), over 500 poses; w = 0.2 gives 30 ms.
    friction_law: Tanh = Tanh(.1)
    armature: tuple[float, ...] = ARMATURE_KG_M2  # reflected joint inertia, kg m²; never link inertia
    # Wall time for a solve's SQP iterations (MPCController.step's budget_s), within
    # solver.max_sqp_iters; None: no time limit. run_experiment, when it builds the MPC, keeps
    # it only where the solve's wall time is the latency (RealTimeSettings, VirtualTimeSettings "measured").
    max_sqp_time_s: float | None = .01
    warmup: bool = True  # Mpc.warmup runs GATO's warm-up solve before the first plan
    # GATO's collision rows against the body and obstacles (collision.py); None: no rows, and
    # the GPU URDF keeps the arm's meshes (GATO's own spheres, unused).
    collision: Collision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.solver, SolverSettings):
            raise TypeError("solver must be SolverSettings; tracking weights belong in cost")
        if not isinstance(self.cost, CostMenu):
            raise TypeError("cost must be a native gato.cost_menu.CostMenu")
        armature = tuple(float(value) for value in self.armature)
        if len(armature) != 7 or not all(math.isfinite(value) and value >= 0 for value in armature):
            raise ValueError('armature must contain seven finite nonnegative values')
        object.__setattr__(self, 'armature', armature)
        if self.N < 2 or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("N must be >=2 and dt_s finite and positive")
        if not isinstance(self.solver.max_sqp_iters, int) or self.solver.max_sqp_iters < 1:
            raise ValueError("solver.max_sqp_iters must be a positive integer")
        time = self.max_sqp_time_s
        if time is not None and (not math.isfinite(time) or time <= 0):
            raise ValueError("max_sqp_time_s must be finite and positive, or None")
        if not isinstance(self.friction_law, Tanh):
            raise ValueError("GATO predicts Tanh Coulomb friction only")

    def resolved(self, controller) -> MpcParams:
        """These params with the values GATO resolved and runs, read from controller.

        The fields GATO does not know, such as N and the friction, stay as they are. The
        controller is load_controller's."""
        return replace(self, solver=SolverSettings.resolved(controller.solver.params),
                       admm=AdmmLimits.resolved(controller.solver),
                       **{name: getattr(controller, name) for name in CONTROLLER_SETTINGS})


TIAGO = MpcParams()


def set_bounds(solver, model, params):
    """Install the planning limits."""
    margin = float(params.joint_margin_rad)
    if (not math.isfinite(margin) or margin < 0
            or margin >= .5 * np.min(model.upperPositionLimit - model.lowerPositionLimit)):
        raise ValueError("Invalid joint margin")
    solver.set_row_group_bounds(0, model.lowerPositionLimit + margin, model.upperPositionLimit - margin)
    for group, fraction, raw in ((1, params.velocity_fraction, model.velocityLimit),
                                 (2, params.effort_fraction, model.effortLimit)):
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Limit fractions must be finite and in (0, 1]")
        solver.set_row_group_bounds(group, -fraction * raw, fraction * raw)


# GATO compiles one CUDA module per registered plant name and N (minutes). A fixed name
# per variant would let two variants overwrite each other in a shared cache, so the
# name hashes everything compiled into the module, and GATO's version: each version's
# registration (its manifest entry) then has its own name. The hash is a cache key, not provenance.
def plant_name(
    gpu_urdf_path: Path,
    friction: Friction | None = None,
    armature: tuple[float, ...] = (0.,) * 7,
    friction_law: Tanh = Tanh(.1),
) -> str:
    """tiago_right_<hash> over the installed GATO version, the GPU URDF bytes,
    grasp frame, predictor friction (coefficients and Tanh width), and nonzero
    reflected armature as passed to registration."""
    config = json.dumps(friction and dict(
        coulomb_nm=friction.coulomb_nm, viscous_nm_s_per_rad=friction.viscous_nm_s_per_rad,
        width_rad_s=friction_law.width_rad_s), sort_keys=True).encode()
    if any(armature):
        config += json.dumps(dict(armature=armature), sort_keys=True).encode()
    return "tiago_right_" + digest(importlib.metadata.version("gato-mpc").encode(),
                                   Path(gpu_urdf_path).read_bytes(), GRASP_FRAME.encode(), config)


def _plant(params: MpcParams):
    """The Pinocchio arm, GPU URDF and GATO plant name for params.
    Exports GATO_CACHE_DIR (cache_root) and puts this environment's cmake first on PATH:
    GATO's module key includes the cmake it finds."""
    import pinocchio as pin

    cache_root()
    model = pin.buildModelFromUrdf(str(arm_urdf()))
    model.armature = np.asarray(params.armature)
    if not model.existFrame(GRASP_FRAME):
        raise ValueError(f"Missing end-effector frame {GRASP_FRAME}")
    bin_dir = str(Path(sys.executable).parent)
    if os.environ["PATH"].split(os.pathsep)[0] != bin_dir:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
    spheres = None if params.collision is None else arm_cover(params.collision.arm)
    gpu_path = runtime_urdf(params.friction, spheres)
    return model, gpu_path, plant_name(gpu_path, params.friction, params.armature, params.friction_law)


def _register(params: MpcParams, gpu_path: Path, plant: str) -> None:
    """GATO's code generation and manifest entry for the plant (seconds when new, never a
    compile). BSQP reads the end-effector frame from it."""
    import gato

    friction = params.friction
    armature = dict(joint_armature=params.armature) if any(params.armature) else {}
    # With joint_dynamics, GRiD generates the GPU URDF's damping and tanh Coulomb friction
    # into the dynamics and derivatives, with the width compiled in.
    gato.register_robot(plant, gpu_path, N=[params.N], ee_frame=GRASP_FRAME, **armature,
                        **({} if friction is None
                           else dict(joint_dynamics=True, friction_width=params.friction_law.width_rad_s)))


def find_binary(params: MpcParams) -> Path | None:
    """The cached GATO module for params, or None when it is not built or the plant not
    registered. Never registers or compiles."""
    from gato import _cache

    plant = _plant(params)[2]
    return _cache.find_module(plant, params.N)


def compile_binary(params: MpcParams, use_cache: bool = True, jobs: int = 2) -> Path:
    """Register the plant and compile its GATO module (minutes) unless cached; returns its
    path. CPU only: no CUDA context. use_cache=False rebuilds: the old file is unlinked,
    not overwritten, so processes that loaded it keep their mapping. jobs: parallel
    compile jobs."""
    from gato import _cache

    _, gpu_path, plant = _plant(params)
    _register(params, gpu_path, plant)
    if not use_cache:
        with _cache._locked(_cache.cache_dir()):  # not while another process compiles it
            old = _cache.find_module(plant, params.N)
            if old is not None:
                old.unlink()
    return _cache.ensure_module(plant, params.N, jobs=jobs)


def load_controller(params: MpcParams, binary: Path | None):
    """Load exactly binary as GATO's module for params (never compiles), then build and
    check the controller; returns it."""
    import importlib.util

    import gato
    from gato import _cache

    model, gpu_path, plant = _plant(params)
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
    # Register only a plant new to this cache (e.g. a binary from another): registering
    # takes the cache lock, which a compile holds for minutes, and rewrites the manifest.
    # Skipping is safe because the plant name hashes every code-generation input.
    found = _cache.resolve(plant)
    if found is None or not all((found[1] / plant / f).exists() for f in ("grid.cuh", "limits.cuh")):
        _register(params, gpu_path, plant)
    solver = gato.BSQP(str(gpu_path), batch_size=1, N=params.N, dt=params.dt_s, plant_type=plant,
                       params=params.solver.native())
    check_model(solver, model, model.createData(), params.friction, params.friction_law)
    given = {name: getattr(params, name) for name in CONTROLLER_SETTINGS if getattr(params, name) is not None}
    controller = gato.MPCController(
        solver, limits="admm", limits_kw={key: value for key, value in asdict(params.admm).items()
                                          if value is not None}, **given)
    if controller.limits != "admm":
        raise RuntimeError(f"GATO installed the {controller.limits!r} limit rows, not ADMM")
    set_bounds(solver, model, params)
    if params.collision is not None:  # after the limit groups: enabling those drops appended groups
        solver.set_collision_environment(**params.collision.environment())
        solver.enable_collision(params.collision.mechanism, margin=params.collision.margin_m,
                                rho=params.collision.rho)
    return controller


def check_model(
    solver,
    model,
    data,
    friction: Friction | None = None,
    friction_law: Tanh = Tanh(.1),
):
    """The GPU model against Pinocchio's native one at two poses: FK; moving and at rest
    (the steepest tanh slope), one step of 1 ms and of dt against GATO's constant-
    acceleration (trapezoidal) step, and the first knot's A-block against finite
    differences of that step. Without friction the passive torque is zero."""
    import pinocchio as pin

    def step(x, u, h):
        q, v = x[:7], x[7:]
        a = pin.aba(model, data, q, v, u - passive_torque(v, friction, friction_law)).copy()
        return np.r_[q + v*h + .5*a*h*h, v + a*h]

    for q in (Q_START, Q_REACH):
        np.testing.assert_allclose(solver.ee_pos(q), fk(q, GRASP_FRAME), atol=1e-5, rtol=1e-4)
        for v in (np.array([.3, -.35, .4, -.45, .5, -.55, .6]), np.zeros(7)):
            x, u = np.r_[q, v], pin.rnea(model, data, q, v, np.zeros(7)) + passive_torque(v, friction, friction_law)
            for h in (.001, solver.dt):
                gpu = np.asarray(solver.sim_forward(x[None], u[None], h))[0]
                np.testing.assert_allclose(gpu, step(x, u, h), atol=2e-3, rtol=1e-3)
            xu = np.tile(np.r_[x, u], solver.N)[:-7]
            kkt = solver.debug_setup_kkt(xu[None], x[None], np.zeros((1, solver.N*6)))
            a_block = np.asarray(kkt["A"])[0].reshape(solver.N, 14, 14)[0].T
            fd = np.array([step(x + d, u, solver.dt) - step(x - d, u, solver.dt)
                           for d in np.eye(14)*1e-5]).T / 2e-5
            np.testing.assert_allclose(a_block, fd, atol=2e-3, rtol=1e-2)
    solver.reset()


# Named solver changes, applied before a solve by Mpc (also in MpcProcess). Each persists
# in the solver until changed again. Add a name when an experiment needs it.
UPDATES = {
    # External wrench in GATO's layout (BSQP.set_f_ext_B).
    "external_force": lambda mpc, wrench: mpc.controller.solver.set_f_ext_B(wrench),
}


def _known(updates):
    unknown = set(updates or ()) - UPDATES.keys()
    if unknown:
        raise KeyError(f"unknown MPC updates: {sorted(unknown)}")


def _position(reference: np.ndarray | None, n: int) -> np.ndarray:
    """Supply GATO's unused reference argument with finite orientation slots."""
    if reference is None:
        return np.zeros((n, 6), dtype=np.float32)
    reference = np.array(reference, dtype=np.float32)
    reference[:, 3:] = 0.
    return reference


def _checked(mpc, plan, t, measured_t):
    """Check the returned trajectory and native solve duration against the request."""
    if plan.t0 != t:
        raise RuntimeError("MPC result does not match the request")
    if plan.measured_t != measured_t:
        raise RuntimeError("MPC result does not match the measured state")
    if plan.dt != mpc.dt:
        raise RuntimeError("MPC plan has an unexpected knot interval")
    if (plan.states.shape != (mpc.N, 14) or plan.torques.shape != (mpc.N - 1, 7)
            or not np.isfinite(plan.states).all() or not np.isfinite(plan.torques).all()):
        raise RuntimeError("invalid MPC plan")
    took = plan.stats["wall_s"]
    if not isinstance(took, (float, int)) or not math.isfinite(took) or took < 0:
        raise ValueError("MPC wall_s must be finite nonnegative seconds")
    return plan


def _same_menu(left: CostMenu, right: CostMenu | None) -> bool:
    """Compare native menu fields with an owned snapshot, including mutable arrays."""
    if right is None:
        return False
    for item in fields(CostMenu):
        a, b = getattr(left, item.name), getattr(right, item.name)
        if item.name == "q_ref":
            if a is not b and not np.array_equal(a, b):
                return False
        elif a is None or b is None:
            if a is not b:
                return False
        elif type(a) is not type(b) or any(
            getattr(a, f.name) is not getattr(b, f.name)
            and not np.array_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a)
        ):
            return False
    return True


class Mpc:
    """The GATO controller in this process: request solves, result returns the plan.

    binary is the GATO module for params (find_binary, compile_binary); Mpc loads exactly
    it and never compiles. warmup resets the controller from its state and runs the
    warm-up solve; without one, the first request resets it from its state. Later
    requests never reset it. A fresh controller needs a new Mpc, or controller.reset.
    ``controller`` is the raw GATO MPCController, ``step`` the raw GATO result of the
    latest solve (MPCController.step; None before the first). A reference replaces
    the menu's grasp-frame targets; None uses its native targets unchanged.
    """

    def __init__(self, params: MpcParams, binary: Path | None):
        self.controller = load_controller(params, binary)
        self.params = params.resolved(self.controller)
        self.N, self.dt = params.N, params.dt_s
        self.step: Any = None
        self._reset = False  # the controller has been reset from a state
        self._previous_t: float | None = None
        self._plan: Plan | None = None
        self._cost_template: CostMenu | None = None
        self._cost_reference: np.ndarray | None = None

    @classmethod
    def compile(cls, params: MpcParams, use_cache: bool = True) -> Mpc:
        """Mpc on compile_binary(params, use_cache)."""
        return cls(params, compile_binary(params, use_cache))

    def warmup(self, state: State, reference: np.ndarray | None = None) -> None:
        """With params.warmup, reset the controller from state and run GATO's warm-up solve
        (MPCController.warmup, the solver's own limits, no time limit); else nothing."""
        if self.params.warmup:
            x = self._x(state)
            self.controller.reset(x)
            self._objective(reference)
            self.controller.warmup(x, _position(reference, self.N))
            self._reset = True

    def request(
        self,
        state: State | Prediction,
        reference: np.ndarray | None = None,
        updates: dict[str, Any] | None = None,
    ) -> None:
        _known(updates)
        if self._plan is not None:
            raise RuntimeError("an MPC request is pending")
        self._plan = self._solve(state, reference, updates)

    def result(self, block: bool = False) -> Plan | None:
        plan, self._plan = self._plan, None
        return None if plan is None else _checked(self, plan, plan.t0, plan.measured_t)

    def close(self) -> None:
        """Release controller/native buffers even when a traceback retains this Mpc."""
        self.controller = self.step = self._plan = None

    def _objective(self, reference: np.ndarray | None) -> None:
        """Apply changed menu settings or explicitly supplied grasp-frame targets."""
        if reference is not None:
            reference = np.asarray(reference)
            if reference.shape != (self.N, 6):
                raise ValueError(f"the reference must be ({self.N}, 6)")
        same_template = _same_menu(self.params.cost, self._cost_template)
        same_reference = (reference is None and self._cost_reference is None
                          or reference is not None and self._cost_reference is not None
                          and np.array_equal(reference, self._cost_reference, equal_nan=True))
        if same_template and same_reference:
            return
        template = self._cost_template if same_template else deepcopy(self.params.cost)
        assert template is not None
        menu = template if reference is None else with_reference(template, reference)
        # GATO may change device fields before an upload raises, so no previous menu stays cached.
        self._cost_template = None
        self._cost_reference = None
        self.controller.solver.set_cost_menu(menu)
        self._cost_template = template
        self._cost_reference = None if reference is None else reference.copy()

    def _x(self, state):
        """The solver's state vector."""
        q, qd = np.asarray(state.q, dtype=float), np.asarray(state.qd, dtype=float)
        if q.shape != (7,) or qd.shape != (7,):
            raise ValueError("Measured state must contain seven finite native positions/velocities")
        x = np.concatenate((q, qd))
        if not np.isfinite(x).all():
            raise ValueError("Measured state must contain seven finite native positions/velocities")
        return x

    def _solve(self, state: State | Prediction, reference: np.ndarray | None, updates):
        x = self._x(state)
        for name, value in (updates or {}).items():
            UPDATES[name](self, value)
        if not self._reset:
            self.controller.reset(x)
            self._reset = True
        # GATO uses dt_elapsed only to score hypotheses (none here).
        dt_elapsed = None if self._previous_t is None else state.t - self._previous_t
        self._previous_t = state.t
        self._objective(reference)
        self.step = step = self.controller.step(x, _position(reference, self.N), dt_elapsed=dt_elapsed,
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
            admm_linsys_mode=step.admm_linsys, pred_err=float(step.pred_err),
            row_viol=None if step.row_viol is None else np.array(step.row_viol))
        measured_t = state.measured_t if isinstance(state, Prediction) else state.t
        return Plan(state.t, self.dt, knots[:, :14], knots[:-1, 14:], stats, measured_t=measured_t)


_SPAWN = mp.get_context("spawn")


class MpcProcess:
    """Mpc in a spawned process, behind the same interface.

    __init__ returns once the process has loaded binary and checked its controller;
    it never compiles. A failed solve replies with its error and ends the process;
    result() raises on that reply, and on a dead process, at the next call.
    """

    def __init__(self, params: MpcParams, binary: Path | None):
        self.params, self.N, self.dt = params, params.N, params.dt_s  # resolved once loaded
        self._requests, self._replies = _SPAWN.Queue(), _SPAWN.Queue()
        self._process = _SPAWN.Process(target=_serve, args=(params, binary, self._requests, self._replies),
                                       daemon=True)
        self._process.start()
        self._pending: tuple[float, float] | None = None  # predicted origin and measured time
        try:
            self.params = self._reply(block=True)  # ready: the resolved params
        except BaseException:
            self.close()
            raise

    @classmethod
    def compile(cls, params: MpcParams, use_cache: bool = True) -> MpcProcess:
        """This process compiles (compile_binary), the MPC process loads the module."""
        return cls(params, compile_binary(params, use_cache))

    def warmup(self, state: State, reference: np.ndarray | None = None) -> None:
        """Mpc.warmup in the MPC process; returns when it is done."""
        if self._pending is not None:
            raise RuntimeError("an MPC request is pending")
        if self.params.warmup:
            self._requests.put(("warmup", state, reference))
            self._reply(block=True)

    def request(
        self,
        state: State | Prediction,
        reference: np.ndarray | None = None,
        updates: dict[str, Any] | None = None,
    ) -> None:
        _known(updates)
        if self._pending is not None:
            raise RuntimeError("an MPC request is pending")
        self._requests.put(("_solve", state, reference, updates))
        self._pending = (state.t, state.measured_t if isinstance(state, Prediction) else state.t)

    def result(self, block: bool = False) -> Plan | None:
        reply = self._reply(block and self._pending is not None)
        if reply is None:
            return None
        if self._pending is None:
            raise RuntimeError("unexpected MPC reply")
        t, self._pending = self._pending, None
        return _checked(self, reply, *t)

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
    """The process: load, reply the resolved params when ready (or the error), then one reply per
    request: a plan, or None after the warm-up. It ignores SIGINT: Ctrl-C reaches the whole
    foreground process group, and the parent stops the run and closes this process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        mpc = Mpc(params, binary)
        replies.put(mpc.params)
        while (item := requests.get()) is not None:
            method, *args = item
            replies.put(getattr(mpc, method)(*args))
    except Exception as exc:
        replies.put(f"{type(exc).__name__}: {exc}")
