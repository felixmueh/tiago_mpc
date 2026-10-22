"""MPC parameters with the TIAGo defaults, and the GATO controller built from them.

Experiments override the defaults with dataclasses.replace, for example
replace(TIAGO, solver=TIAGO.solver.replace(mu=1.)).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import os
from pathlib import Path
import sys

from gato import SolverParams
import numpy as np

from .joint_dynamics import Friction, register_robot, urdf_friction
from .model_runtime import EE_FRAME, check_model, runtime_urdf
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
    warmup: int = 0  # solves before the first plan is used (step 2)

    def __post_init__(self):
        if self.N < 2 or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("N must be >=2 and dt_s finite and positive")

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
