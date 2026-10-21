"""Explicit hard-sign plant and smooth GATO predictor passive joint laws."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"arm_right_{i}_joint" for i in range(1, 8)]


def validate(config):
    if config is None:
        return None
    if set(config) - {"smoothing_rad_s"} != {"model", "coulomb_nm", "viscous_nm_s_per_rad", "provenance"}:
        raise ValueError("joint_dynamics requires model, coulomb_nm, viscous_nm_s_per_rad, provenance")
    if config["model"] not in ("coulomb_sign_viscous", "coulomb_tanh_viscous"):
        raise ValueError("Expected coulomb_sign_viscous or coulomb_tanh_viscous")
    for key in ("coulomb_nm", "viscous_nm_s_per_rad"):
        values = np.asarray(config[key], dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{key} must contain seven finite nonnegative coefficients")
    if not isinstance(config["provenance"], str) or not config["provenance"].strip():
        raise ValueError("joint_dynamics provenance must describe the coefficient assumptions")
    if config["model"] == "coulomb_tanh_viscous":
        widths = np.broadcast_to(np.asarray(config.get("smoothing_rad_s"), dtype=float), (7,))
        if not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("smoothing_rad_s must be positive finite widths")
    return config


def passive_torque(velocity, config):
    """Positive resisting bias: subtract from actuator torque before ABA."""
    v = np.asarray(velocity, dtype=float)
    if config is None:
        return np.zeros_like(v)
    friction = (np.tanh(v / np.asarray(config["smoothing_rad_s"]))
                if config["model"] == "coulomb_tanh_viscous" else np.sign(v))
    return np.asarray(config["viscous_nm_s_per_rad"])*v + np.asarray(config["coulomb_nm"])*friction


def plant_rk4(model, data, q, velocity, actuator_torque, dt, config):
    """Native-coordinate RK4 with passive torque recomputed at every stage."""
    import pinocchio as pin

    def acceleration(q_stage, v_stage):
        return pin.aba(model, data, q_stage, v_stage,
                       actuator_torque-passive_torque(v_stage, config)).copy()
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


def configure_urdf(tree, config):
    """Write only generated copies. Dissipation coefficients survive axis flips."""
    if config is None:
        return
    import xml.etree.ElementTree as ET
    joints = [j for j in tree.iter("joint") if j.get("type") != "fixed"]
    if [j.get("name") for j in joints] != JOINT_NAMES:
        raise ValueError("Passive coefficients require the native seven-joint TIAGo ordering")
    for joint, c, b in zip(joints, config["coulomb_nm"], config["viscous_nm_s_per_rad"]):
        joint.append(ET.Comment(" " + config["provenance"] +
                                "; Nm / Nm s per rad; coefficients only, runtime force law is explicit. "))
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("friction", repr(float(c)))
        dynamics.set("damping", repr(float(b)))


def register_robot(name, urdf_path, *, n_points, ee_frame, config):
    """Use public GATO registration; passive prediction is explicitly opt-in."""
    import gato
    if config is None:
        gato.register_robot(name, urdf_path, N=[n_points], ee_frame=ee_frame, jobs=1)
        return None
    if config["model"] != "coulomb_tanh_viscous":
        raise ValueError("GATO prediction supports smooth Coulomb friction; hard sign is plant-only")
    return gato.register_robot(name, urdf_path, N=[n_points], ee_frame=ee_frame,
                               jobs=1, joint_friction_smoothing=config["smoothing_rad_s"])


def model_receipt(solver, name, gpu_path):
    """Exact generated-header and loaded binary identity for a saved cohort."""
    from gato import _cache
    entry, plant_root = _cache.resolve(name)
    directory = plant_root/name
    binary = Path(solver.lib.__file__)
    return dict(plant_name=name, codegen_key=entry["codegen_key"],
        prediction_urdf_sha256=hashlib.sha256(Path(gpu_path).read_bytes()).hexdigest(),
        header_path=str(directory/"grid.cuh"),
        header_sha256=hashlib.sha256((directory/"grid.cuh").read_bytes()).hexdigest(),
        limits_header_sha256=hashlib.sha256((directory/"limits.cuh").read_bytes()).hexdigest(),
        native_source_id=solver.lib.NATIVE_SOURCE_ID,
        module_path=str(binary), module_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
