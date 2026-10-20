"""Child-side model preparation and the bounded reaching model convention check.

The native URDF stays in ROS coordinates. Only the generated GPU URDF and the
solver's state/control boundary normalize negative joint axes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EE_FRAME = "arm_right_tool_joint"
PLANT = "tiago_mpc_right"
Q_CHECK = np.array([-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04])
Q_DELTA_CHECK = np.array([0.18, 0.12, -0.10, 0.18, 0.0, 0.0, 0.0])


def runtime_urdf(cache_dir):
    """Adapt examples/reach_sim.py's manifest check and signed-axis conversion."""
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    for relative, info in manifest.items():
        path = ROOT / "models" / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Model asset differs from manifest (or is an LFS pointer): {path}")
    tree = ET.parse(ROOT / "models/tiago_right/tiago_right_arm.urdf")
    for mesh in tree.iter("mesh"):
        path = ROOT / "models/tiago_pro/meshes" / Path(mesh.attrib["filename"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    root_link = set(link.get("name") for link in tree.iter("link")) - {
        joint.find("child").get("link") for joint in tree.iter("joint")}
    if len(root_link) != 1:
        raise ValueError("Expected one native arm root frame")
    out = Path(cache_dir) / "models/tiago_right_arm.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True))
    signs, names = [], []
    for joint in tree.iter("joint"):
        if joint.get("type") == "fixed":
            continue
        names.append(joint.get("name"))
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
    gpu_path = out.with_name("tiago_right_grid.urdf")
    gpu_path.write_bytes(ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True))
    return out, gpu_path, np.asarray(signs), tuple(names), root_link.pop()


def ee_position(model, data, q, frame_id):
    import pinocchio as pin

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id].translation.copy()


def check_model(solver, model, data, frame_id, signs):
    """Same two-pose FK/dynamics check as the existing reaching simulation."""
    import pinocchio as pin
    from gato.common import rk4

    errors = []
    for q in (Q_CHECK, Q_CHECK + Q_DELTA_CHECK):
        expected = ee_position(model, data, q, frame_id)
        if not np.allclose(solver.ee_pos(q * signs), expected, atol=1e-5, rtol=1e-4):
            raise ValueError("GATO/Pinocchio end-effector frames disagree")
        v = np.array([0.10, -0.08, 0.06, -0.04, 0.02, -0.07, 0.03])
        u = pin.rnea(model, data, q, v, np.zeros(7)) + np.linspace(-0.3, 0.3, 7)
        state_signs = np.tile(signs, 2)
        gpu = np.asarray(solver.sim_forward((np.r_[q, v] * state_signs)[None],
                                            (u * signs)[None], 0.003))[0] * state_signs
        q_next, v_next = rk4(model, data, q, v, u, 0.003)
        error = float(np.max(np.abs(gpu - np.r_[q_next, v_next])))
        errors.append(error)
        if not np.isfinite(gpu).all() or error > 2e-3:
            raise ValueError(f"GATO/Pinocchio one-step disagreement: {error:.6g}")
    solver.reset()
    return errors
