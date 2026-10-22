"""Shared model checks and public GATO controller construction for simulation/ROS."""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import numpy as np

from .model_runtime import EE_FRAME, PLANT, check_model, runtime_urdf


def set_bounds(solver, model, signs, bounds):
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


def build_controller(effective, *, cache_dir, ee_frame=EE_FRAME):
    """Build/check once; callers retain ownership of timing, state and I/O."""
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GATO_CACHE_DIR", str(cache_dir / "gato"))
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    import gato
    import pinocchio as pin
    from .joint_dynamics import register_robot

    losses = effective["prediction_joint_dynamics"]
    path, gpu_path, signs, names, reference_frame = runtime_urdf(cache_dir, losses)
    model = pin.buildModelFromUrdf(str(path))
    expected_names = tuple(f"arm_right_{i}_joint" for i in range(1, 8))
    if (model.nq, model.nv) != (7, 7) or tuple(model.names)[1:] != expected_names or names != expected_names:
        raise ValueError("Expected seven native right-arm joints in order")
    data = model.createData()
    frame_id = model.getFrameId(ee_frame)
    if frame_id >= model.nframes:
        raise ValueError(f"Missing end-effector frame {ee_frame}")
    n, dt_s = effective["N"], effective["dt_s"]
    plant = PLANT + "_smooth" if losses else PLANT
    adapter = register_robot(plant, gpu_path, n_points=n, ee_frame=ee_frame, config=losses)
    solver = gato.BSQP(str(gpu_path), batch_size=1, N=n, dt=dt_s,
                       plant_type=plant, params=gato.SolverParams(**effective["solver_params"]))
    errors = check_model(solver, model, data, frame_id, signs, losses)
    controller = gato.MPCController(solver, limits="admm", linsys="pcg", **effective["controller"])
    set_bounds(solver, model, signs, effective["limit_bounds"])
    details = dict(N=n, dt_s=dt_s, effective_configuration=effective,
        reference_frame=reference_frame, ee_frame=ee_frame, end_effector_frame=ee_frame,
        joint_names=list(names), gpu_joint_coordinate_signs=signs.tolist(),
        prediction_joint_dynamics=losses, generator_adapter=adapter,
        joint_friction_smoothing_rad_s=(None if solver.joint_friction_smoothing is None
                                       else solver.joint_friction_smoothing.tolist()),
        q_lower=model.lowerPositionLimit.tolist(), q_upper=model.upperPositionLimit.tolist(),
        velocity_limits=model.velocityLimit.tolist(), effort_limits=model.effortLimit.tolist(),
        model_step_max_abs_errors=errors, plant_name=plant, cache_dir=str(cache_dir))
    return MpcSetup(controller, model, data, frame_id, signs, path, gpu_path, details)
