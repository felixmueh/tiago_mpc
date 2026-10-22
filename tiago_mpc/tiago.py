"""TIAGo facts: the right arm's joints, postures, frames, models and raw limits,
and the URDF preparation for the GPU.

The URDFs are the source of the coefficients and limits (scripts/prepare_models.py
patches them). Importing this module loads only NumPy; Pinocchio loads on first use.
The native URDF stays in ROS coordinates. Only the generated GPU URDF and the
solver's state/control boundary normalize negative joint axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .dynamics import Friction

ROOT = Path(__file__).resolve().parents[1]  # the repository
ARM_URDF = ROOT / "models/tiago_right/tiago_right_arm.urdf"  # the reduced right arm
FULL_URDF = ROOT / "models/tiago_pro/tiago_pro.urdf"  # the whole robot (safety, replays)
RIGHT_ARM_JOINTS = tuple(f"arm_right_{i}_joint" for i in range(1, 8))  # native order
ROOT_FRAME = "torso_lift_link"  # the reduced arm's fixed root; positions are in this frame
GRASP_FRAME = "arm_right_grasp_joint"  # nominal grasp center, the default end effector
FLANGE_FRAME = "arm_right_tool_joint"
# Runs start here at rest (checked on the robot): PAL's comfortable_high_clearance with
# joint 4 at -2.33 instead of -2.35, inside the MPC's planning box (its lower limit -2.4435
# plus MpcParams.joint_margin_rad 0.1).
Q_START = np.array([-0.39, -1.73, -0.38, -2.33, 0.0, -1.21, 0.04])
# A nearby posture: the default reach target's, and the model check's second pose.
Q_REACH = Q_START + np.array([0.18, 0.12, -0.10, 0.18, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class Limits:
    """Raw joint limits in native order: position (rad), velocity (rad/s), effort (Nm)."""
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray


@cache
def _arm():
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(ARM_URDF))
    return model, model.createData()


@cache
def limits() -> Limits:
    """The reduced arm URDF's raw limits (read-only)."""
    model = _arm()[0]
    arrays = [np.array(a) for a in (model.lowerPositionLimit, model.upperPositionLimit,
                                    model.velocityLimit, model.effortLimit)]
    for a in arrays:
        a.flags.writeable = False
    return Limits(*arrays)


def urdf_friction(width_rad_s: float | None = None) -> Friction:
    """The reduced arm URDF's friction and damping coefficients (the only default)."""
    model = _arm()[0]
    return Friction(tuple(model.friction), tuple(model.damping), width_rad_s)


def fk(q: np.ndarray, frame: str = GRASP_FRAME) -> np.ndarray:
    """Position of frame in ROOT_FRAME, metres."""
    import pinocchio as pin
    model, data = _arm()
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(frame)].translation.copy()


def digest(*parts: bytes) -> str:
    """Eight hex characters over the parts (length-prefixed, so order and splits matter)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little") + part)
    return h.hexdigest()[:8]


def runtime_urdf(cache_dir, friction=None):
    """The GPU URDF and the GPU coordinate signs.

    A copy of the reduced arm with resolved mesh paths and friction's coefficients, if
    given, whose negative joint axes are turned positive, with their limits. It goes to
    <cache_dir>/models/<hash>/, so concurrent processes with different coefficients
    never share a path."""
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    for relative, info in manifest.items():
        path = ROOT / "models" / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Model asset differs from manifest (or is an LFS pointer): {path}")
    tree = ET.parse(ARM_URDF)
    joints = [joint for joint in tree.iter("joint") if joint.get("type") != "fixed"]
    if tuple(joint.get("name") for joint in joints) != RIGHT_ARM_JOINTS:
        raise ValueError("Expected the seven native right-arm joints in order")
    if (set(link.get("name") for link in tree.iter("link"))
            - {joint.find("child").get("link") for joint in tree.iter("joint")}) != {ROOT_FRAME}:
        raise ValueError(f"Expected the arm's root frame {ROOT_FRAME}")
    if friction is not None:  # the coefficients survive the axis flips below
        for joint, c, b in zip(joints, friction.coulomb_nm, friction.viscous_nm_s_per_rad):
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("friction", repr(float(c)))
            dynamics.set("damping", repr(float(b)))
    for mesh in tree.iter("mesh"):
        path = ROOT / "models/tiago_pro/meshes" / Path(mesh.attrib["filename"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    signs = []
    for joint in joints:
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if not (np.array_equal(axis, [0, 0, 1]) or np.array_equal(axis, [0, 0, -1])):
            raise ValueError(f"Unsupported arm axis: {axis}")
        sign = float(axis[2])
        signs.append(sign)
        if sign < 0:
            joint.find("axis").set("xyz", "0 0 1")
            limit = joint.find("limit")
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
            limit.set("lower", repr(-hi))
            limit.set("upper", repr(-lo))
            safety = joint.find("safety_controller")
            if safety is not None:
                lo, hi = float(safety.get("soft_lower_limit")), float(safety.get("soft_upper_limit"))
                safety.set("soft_lower_limit", repr(-hi))
                safety.set("soft_upper_limit", repr(-lo))
    payload = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    path = Path(cache_dir) / "models" / digest(payload) / "tiago_right_grid.urdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, payload)
    return path, np.asarray(signs)


def _write_atomic(path, payload):
    if path.exists() and path.read_bytes() == payload:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)
