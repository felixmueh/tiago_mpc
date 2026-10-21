"""Shared model preparation and bounded predictor convention checks.

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
URDF = ROOT / "models/tiago_right/tiago_right_arm.urdf"
EE_FRAME = "arm_right_grasp_joint"
PLANT = "tiago_mpc_right"
Q_CHECK = np.array([-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04])
Q_DELTA_CHECK = np.array([0.18, 0.12, -0.10, 0.18, 0.0, 0.0, 0.0])


def runtime_urdf(cache_dir, joint_dynamics=None):
    """Resolve assets and normalize negative axes only in the generated GPU copy."""
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    for relative, info in manifest.items():
        path = ROOT / "models" / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Model asset differs from manifest (or is an LFS pointer): {path}")
    tree = ET.parse(URDF)
    if joint_dynamics is not None:
        from .joint_dynamics import configure_urdf
        configure_urdf(tree, joint_dynamics)
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
    _write_urdf(out, tree)
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
    _write_urdf(gpu_path, tree)
    return out, gpu_path, np.asarray(signs), tuple(names), root_link.pop()


def _write_urdf(path, tree):
    payload = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)


def ee_position(model, data, q, frame_id):
    import pinocchio as pin

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id].translation.copy()


def check_model(solver, model, data, frame_id, signs, prediction_losses=None, *, q_goal=None):
    """Same two-pose FK/dynamics check as the existing reaching simulation."""
    import pinocchio as pin
    if prediction_losses is not None:
        return _check_dissipative_model(solver, model, data, frame_id, signs, prediction_losses)

    from gato.common import rk4

    errors = []
    for q in (Q_CHECK, Q_CHECK + Q_DELTA_CHECK if q_goal is None else q_goal):
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


def _check_dissipative_model(solver, model, data, frame_id, signs, prediction_losses):
    """Value/A-block checks against the same constant-acceleration predictor step."""
    import pinocchio as pin
    from .joint_dynamics import passive_torque

    q = Q_CHECK.copy()
    v = np.array([.3, -.35, .4, -.45, .5, -.55, .6])
    u = pin.rnea(model, data, q, v, np.zeros(7)) + passive_torque(v, prediction_losses)
    x = np.r_[q, v]; ss = np.tile(signs, 2)
    def native_predictor(state, h):
        qn, vn = state[:7], state[7:]
        acceleration = pin.aba(model, data, qn, vn, u-passive_torque(vn, prediction_losses)).copy()
        return np.r_[qn+vn*h+.5*acceleration*h*h, vn+acceleration*h]
    checks = {}
    expected_ee = ee_position(model, data, q, frame_id)
    actual_ee = np.asarray(solver.ee_pos(q*signs))
    np.testing.assert_allclose(actual_ee, expected_ee, atol=1e-5, rtol=1e-4)
    checks["ee_position_max_abs_error_m"] = float(np.max(np.abs(actual_ee-expected_ee)))
    for h in (.001, solver.dt):
        expected = native_predictor(x, h)
        actual = np.asarray(solver.sim_forward((x*ss)[None], (u*signs)[None], h))[0]*ss
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-3)
        checks[f"step_{h:g}s_max_abs_error"] = float(np.max(np.abs(actual-expected)))
    xu = np.tile(np.r_[x*ss, u*signs], solver.N)[:-7]
    kkt = solver.debug_setup_kkt(xu[None], (x*ss)[None], np.zeros((1, solver.N*6)))
    analytic_gpu = np.asarray(kkt["A"])[0].reshape(solver.N, 14, 14)[0].T
    analytic_native = ss[:, None]*analytic_gpu*ss[None, :]
    fd = np.empty((14, 14)); epsilon = 1e-5
    for j in range(14):
        delta = np.eye(14)[j]*epsilon
        plus, minus = x+delta, x-delta
        fp, fm = native_predictor(plus, solver.dt), native_predictor(minus, solver.dt)
        fd[:, j] = (fp-fm)/(2*epsilon)
    np.testing.assert_allclose(analytic_native, fd, atol=2e-3, rtol=1e-2)
    checks["analytic_A_native_FD_max_abs_error"] = float(np.max(np.abs(analytic_native-fd)))
    checks["derivative_scope"] = "one signed nonzero-velocity state; no zero crossing; native ABA + constant-acceleration predictor integration"
    if prediction_losses and prediction_losses["model"] == "coulomb_tanh_viscous":
        # A cheap extra preflight for new widths: the maximum tanh slope is at rest.
        v = np.zeros(7)
        u = pin.rnea(model, data, q, v, np.zeros(7)).copy()
        x = np.r_[q, v]
        actual = np.asarray(solver.sim_forward((x*ss)[None], (u*signs)[None], solver.dt))[0]*ss
        expected = native_predictor(x, solver.dt)
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-3)
        xu = np.tile(np.r_[x*ss, u*signs], solver.N)[:-7]
        kkt = solver.debug_setup_kkt(xu[None], (x*ss)[None], np.zeros((1, solver.N*6)))
        analytic_gpu = np.asarray(kkt["A"])[0].reshape(solver.N, 14, 14)[0].T
        analytic_native = ss[:, None]*analytic_gpu*ss[None, :]
        for j in range(14):
            delta = np.eye(14)[j]*epsilon
            fd[:, j] = (native_predictor(x+delta, solver.dt)-native_predictor(x-delta, solver.dt))/(2*epsilon)
        np.testing.assert_allclose(analytic_native, fd, atol=2e-3, rtol=1e-2)
        checks["zero_velocity_step_max_abs_error"] = float(np.max(np.abs(actual-expected)))
        checks["zero_velocity_A_native_FD_max_abs_error"] = float(np.max(np.abs(analytic_native-fd)))
    solver.reset()
    return checks
