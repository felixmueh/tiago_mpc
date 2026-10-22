"""Explicit hard-sign plant and smooth GATO predictor passive joint laws."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"arm_right_{i}_joint" for i in range(1, 8)]


@dataclass(frozen=True)
class Friction:
    """Joint losses: Coulomb plus viscous, per native arm joint.

    The Coulomb term is tanh(v/width_rad_s), the smooth law GATO predicts, or
    sign(v) without a width (hard sign, simulated plant only).
    """
    coulomb_nm: tuple[float, ...]
    viscous_nm_s_per_rad: tuple[float, ...]
    width_rad_s: float | None = None

    def __post_init__(self):
        for key in ("coulomb_nm", "viscous_nm_s_per_rad"):
            values = tuple(float(v) for v in getattr(self, key))
            if len(values) != 7 or not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"{key} must contain seven finite nonnegative coefficients")
            object.__setattr__(self, key, values)
        if self.width_rad_s is not None and not (math.isfinite(self.width_rad_s) and self.width_rad_s > 0):
            raise ValueError("width_rad_s must be finite and positive")


def urdf_friction(width_rad_s=None):
    """The reduced arm URDF's friction and damping coefficients (the only default)."""
    import pinocchio as pin
    from .model_runtime import URDF
    model = pin.buildModelFromUrdf(str(URDF))
    return Friction(tuple(model.friction), tuple(model.damping), width_rad_s)


def passive_torque(velocity, friction):
    """Positive resisting bias: subtract from actuator torque before ABA."""
    v = np.asarray(velocity, dtype=float)
    if friction is None:
        return np.zeros_like(v)
    coulomb = np.sign(v) if friction.width_rad_s is None else np.tanh(v / friction.width_rad_s)
    return np.asarray(friction.viscous_nm_s_per_rad)*v + np.asarray(friction.coulomb_nm)*coulomb


def plant_rk4(model, data, q, velocity, actuator_torque, dt, friction):
    """Native-coordinate RK4 with passive torque recomputed at every stage."""
    import pinocchio as pin

    def acceleration(q_stage, v_stage):
        return pin.aba(model, data, q_stage, v_stage,
                       actuator_torque-passive_torque(v_stage, friction)).copy()
    k1q = velocity
    k1v = acceleration(q, velocity)
    k2q = velocity + dt*k1v/2
    k2v = acceleration(pin.integrate(model, q, dt*k1q/2), k2q)
    k3q = velocity + dt*k2v/2
    k3v = acceleration(pin.integrate(model, q, dt*k2q/2), k3q)
    k4q = velocity + dt*k3v
    k4v = acceleration(pin.integrate(model, q, dt*k3q), k4q)
    return (pin.integrate(model, q, dt*(k1q+2*k2q+2*k3q+k4q)/6),
            velocity + dt*(k1v+2*k2v+2*k3v+k4v)/6)


def configure_urdf(tree, friction):
    """Write only generated copies. Dissipation coefficients survive axis flips."""
    if friction is None:
        return
    import xml.etree.ElementTree as ET
    joints = [j for j in tree.iter("joint") if j.get("type") != "fixed"]
    if [j.get("name") for j in joints] != JOINT_NAMES:
        raise ValueError("Passive coefficients require the native seven-joint TIAGo ordering")
    for joint, c, b in zip(joints, friction.coulomb_nm, friction.viscous_nm_s_per_rad):
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("friction", repr(float(c)))
        dynamics.set("damping", repr(float(b)))


def register_robot(name, urdf_path, *, n_points, ee_frame, friction):
    """Use public GATO registration; passive prediction is explicitly opt-in."""
    import gato
    if friction is None:
        gato.register_robot(name, urdf_path, N=[n_points], ee_frame=ee_frame, jobs=1)
        return None
    if friction.width_rad_s is None:
        raise ValueError("GATO prediction supports smooth Coulomb friction; hard sign is plant-only")
    return gato.register_robot(name, urdf_path, N=[n_points], ee_frame=ee_frame,
                               jobs=1, joint_friction_smoothing=friction.width_rad_s)


def model_receipt(solver, name, gpu_path):
    """Model identity and effective runtime widths for a saved cohort."""
    from gato import _cache
    entry, plant_root = _cache.resolve(name)
    directory = plant_root/name
    binary = Path(solver.lib.__file__)
    return dict(plant_name=name, codegen_key=entry["codegen_key"],
        joint_friction_smoothing_rad_s=(None if solver.joint_friction_smoothing is None
                                       else solver.joint_friction_smoothing.tolist()),
        prediction_urdf=str(gpu_path), header_path=str(directory/"grid.cuh"),
        native_source_id=solver.lib.NATIVE_SOURCE_ID, module_path=str(binary))
