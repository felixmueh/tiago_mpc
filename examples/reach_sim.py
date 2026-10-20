#!/usr/bin/env python3
"""One position reach: upstream GATO MPC and an independent Pinocchio RK4 plant."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
# Keep generated headers, CUDA builds and the runtime URDF inside this checkout.
os.environ.setdefault("GATO_CACHE_DIR", str(ROOT / ".cache/gato"))
# Let GATO's CMake subprocess use tools installed in this Python environment.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]

import gato
from gato.common import rk4

URDF = ROOT / "models/tiago_right/tiago_right_arm.urdf"
MESHES = ROOT / "models/tiago_pro/meshes"
PLANT = "tiago_mpc_right"
EE_FRAME = "arm_right_tool_joint"
N, DT_S, SIM_DT_S = 32, 0.030, 0.003
TIMEOUT_S, POSITION_TOL_M, SPEED_L1_TOL_RAD_S = 8.0, 0.05, 1.0
Q_START_RAD = np.array([-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04])
# A reachable target is FK(q_start + delta); q_goal is NOT a posture objective.
Q_GOAL_DELTA_RAD = np.array([0.18, 0.12, -0.10, 0.18, 0.0, 0.0, 0.0])
# Current fork reaching costs as an explicit initial port configuration.
# Row-group ADMM limits are additional upstream constraints, not safety bounds.
PARAMS = gato.SolverParams(
    max_sqp_iters=5, max_pcg_iters=120, pcg_tol=1e-3,
    mu=10.0, rho=0.01, q_cost=4.0, N_cost=80.0,
    qd_cost=0.05, u_cost=0.0001,
    q_lim_cost=0.02125, vel_lim_cost=1.275, ctrl_lim_cost=0.02,
)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def runtime_urdf():
    """Resolve meshes; normalize signed axes only in the GPU coordinate model."""
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    for relative, info in manifest.items():
        p = ROOT / "models" / relative
        if hashlib.sha256(p.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Model asset differs from manifest (or is an LFS pointer): {p}")
    tree = ET.parse(URDF)
    for mesh in tree.iter("mesh"):
        path = MESHES / Path(mesh.attrib["filename"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    out = ROOT / ".cache/models/tiago_right_arm.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if not out.exists() or out.read_bytes() != payload:
        out.write_bytes(payload)
    signs = []
    for joint in tree.iter("joint"):
        if joint.get("type") == "fixed":
            continue
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if not (np.array_equal(axis, [0, 0, 1]) or np.array_equal(axis, [0, 0, -1])):
            raise ValueError(f"Unsupported arm axis: {axis}")
        sign = float(axis[2]); signs.append(sign)
        if sign < 0:
            joint.find("axis").set("xyz", "0 0 1")
            limit = joint.find("limit")
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
            limit.set("lower", repr(-hi)); limit.set("upper", repr(-lo))
            safety = joint.find("safety_controller")
            if safety is not None:
                lo, hi = float(safety.get("soft_lower_limit")), float(safety.get("soft_upper_limit"))
                safety.set("soft_lower_limit", repr(-hi)); safety.set("soft_upper_limit", repr(-lo))
    gpu_path = out.with_name("tiago_right_grid.urdf")
    payload = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if not gpu_path.exists() or gpu_path.read_bytes() != payload:
        gpu_path.write_bytes(payload)
    return out, gpu_path, np.asarray(signs)


def ee_position(model, data, q, fid):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[fid].translation.copy()


def check_model(solver, model, data, fid, q_goal, signs):
    """Bounded convention check at both task poses, including the signed axis."""
    errors = []
    for q in (Q_START_RAD, q_goal):
        expected = ee_position(model, data, q, fid)
        if not np.allclose(solver.ee_pos(q*signs), expected, atol=1e-5, rtol=1e-4):
            raise ValueError("GATO/Pinocchio end-effector frames disagree")
        v = np.array([0.10, -0.08, 0.06, -0.04, 0.02, -0.07, 0.03])
        u = pin.rnea(model, data, q, v, np.zeros(7)) + np.linspace(-0.3, 0.3, 7)
        x = np.r_[q, v]
        gpu = np.asarray(solver.sim_forward((x*np.tile(signs, 2))[None],
                                            (u*signs)[None], SIM_DT_S))[0]*np.tile(signs, 2)
        q_next, v_next = rk4(model, data, q, v, u, SIM_DT_S)
        error = float(np.max(np.abs(gpu - np.r_[q_next, v_next])))
        errors.append(error)
        if not np.isfinite(gpu).all() or error > 2e-3:
            raise ValueError(f"GATO/Pinocchio one-step disagreement: {error:.6g}")
    solver.reset()
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold", action="store_true", help="target the initial tool position")
    p.add_argument("--timeout", type=float, default=TIMEOUT_S, help="simulated seconds")
    p.add_argument("--output", type=Path, help="fresh output directory")
    p.add_argument("--build-only", action="store_true", help="build and check model conventions")
    p.add_argument("--input", type=Path, help="frozen experiment NPZ with initial_state/target_xyz/reference")
    args = p.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        p.error("--timeout must be finite and positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = args.output or ROOT / "results" / stamp
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "reach_sim.py")
    result = dict(status="started", execution_stage="setup", mode="hold" if args.hold else "reach",
                  repository_commit=git("rev-parse", "HEAD") if (ROOT / ".git").exists()
                  and git("rev-list", "--all", "--count") != "0" else None,
                  repository_status=git("status", "--short"),
                  gato_commit=git("rev-parse", "HEAD", cwd=ROOT / "external/gato"),
                  gato_status=git("status", "--short", cwd=ROOT / "external/gato"),
                  versions={k: importlib.metadata.version(k) for k in
                            ("gato-mpc", "grid-rbd", "pin", "numpy")},
                  N=N, dt_s=DT_S, sim_dt_s=SIM_DT_S, timeout_s=args.timeout,
                  position_tol_m=POSITION_TOL_M, speed_l1_tol_rad_s=SPEED_L1_TOL_RAD_S,
                  solver_params=PARAMS.asdict(), limits="admm", linsys="pcg",
                  timing="fixed simulation time; solve latency does not advance plant",
                  orientation="unconstrained", collisions="not checked",
                  urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest())
    times, states, controls, errors, solve_us, wall_s, row_viol = [], [], [], [], [], [], []
    work = []
    exit_code = 1
    try:
        path, gpu_path, signs = runtime_urdf()
        state_signs = np.tile(signs, 2)
        result["gpu_joint_coordinate_signs"] = signs.tolist()
        result["gpu_urdf_sha256"] = hashlib.sha256(gpu_path.read_bytes()).hexdigest()
        model = pin.buildModelFromUrdf(str(path))
        data = model.createData()
        if (model.nq, model.nv) != (7, 7):
            raise ValueError(f"Expected seven-joint arm; got {model.nq}/{model.nv}")
        result["joint_names"] = list(model.names)[1:]
        fid = model.getFrameId(EE_FRAME)
        if fid >= model.nframes:
            raise ValueError(f"Missing end-effector frame {EE_FRAME}")
        x_start = np.r_[Q_START_RAD, np.zeros(7)]
        q_goal = Q_START_RAD + (0.0 if args.hold else Q_GOAL_DELTA_RAD)
        reference = None
        if args.input:
            if args.hold or args.build_only:
                raise ValueError("--input cannot be combined with --hold/--build-only")
            with np.load(args.input, allow_pickle=False) as saved:
                x_start = saved["initial_state"].copy()
                q_goal = saved["source_goal_q"].copy()
                target = saved["target_xyz"].copy()
                reference = saved["reference"].astype(np.float32).reshape(N, 6)
                historical_seed = saved["first_seed"]
                expected_seed = np.tile(np.r_[x_start, np.zeros(7)], N)[:-7]
                if not np.allclose(historical_seed.ravel(), expected_seed, atol=1e-6):
                    raise ValueError("Frozen task seed differs from state-hold/zero-control policy")
            if (x_start.shape != (14,) or q_goal.shape != (7,) or target.shape != (3,)
                    or not all(np.isfinite(a).all() for a in (x_start, q_goal, target, reference))):
                raise ValueError("Invalid frozen input")
            result.update(mode="frozen_random_reach", input_file=str(args.input.resolve()),
                          input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
        for q in (x_start[:7], q_goal):
            if np.any(q < model.lowerPositionLimit) or np.any(q > model.upperPositionLimit):
                raise ValueError("Start/target witness outside joint limits")
        witness_target = ee_position(model, data, q_goal, fid)
        if reference is None:
            target = witness_target
            reference = np.zeros((N, 6), dtype=np.float32)
            reference[:, :3] = target
        elif not np.allclose(target, witness_target, atol=1e-5, rtol=1e-4):
            raise ValueError("Frozen target frame differs from the native TIAGo model")
        initial = ee_position(model, data, x_start[:7], fid)
        result.update(q_start_rad=x_start[:7].tolist(), q_goal_witness_rad=q_goal.tolist(),
                      target_m=target.tolist(), initial_distance_m=float(np.linalg.norm(target-initial)))
        print(f"Target displacement: {result['initial_distance_m']*1000:.1f} mm", flush=True)
        result["status"] = "building_model"
        (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        gato.register_robot(PLANT, gpu_path, N=[N], ee_frame=EE_FRAME, jobs=1)
        solver = gato.BSQP(str(gpu_path), batch_size=1, N=N, dt=DT_S,
                           plant_type=PLANT, params=PARAMS)
        result["status"] = "checking_model"
        (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        # Reuse the already qualified bounded convention check; random task poses
        # are task inputs, not an expanded numerical qualification campaign.
        result["model_step_max_abs_errors"] = check_model(
            solver, model, data, fid, Q_START_RAD+Q_GOAL_DELTA_RAD, signs)
        print("GATO/Pinocchio model checks passed", flush=True)
        if args.build_only:
            result["status"] = "model_checked"
            return 0
        controller = gato.MPCController(solver, limits="admm", linsys="pcg")
        result.update(reset_rho_each_step=controller.reset_rho_each_step,
                      admm_defaults=controller.LIMIT_DEFAULTS["admm"],
                      admm_linsys_policy=controller.admm_linsys)
        result.update(status="running", execution_stage="episode")
        (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        x = x_start.copy()
        controller.reset(x*state_signs)
        times.append(0.0); states.append(x.copy()); errors.append(float(np.linalg.norm(initial-target)))
        t, reached = 0.0, False
        # Hold mode runs the full interval to check stability. Reach stops at arrival.
        while t < args.timeout - 1e-12:
            step = controller.step(x*state_signs, reference.ravel())
            u = np.asarray(step.u, dtype=float)*signs
            stats = step.solve.stats
            work.append(dict(sqp=int(np.sum(stats.sqp_iters)),
                             pcg=int(np.sum(stats.pcg_iters)),
                             admm=int(np.sum(stats.admm_iters)),
                             admm_linsys=int(np.sum(stats.admm_linsys_iters)),
                             admm_linsys_mode=step.admm_linsys,
                             zero_pcg_rows=int(np.sum(np.asarray(stats.pcg_iters) == 0))))
            solve_us.append(float(step.solve.solve_time_us)); wall_s.append(float(step.wall_s))
            row_viol.append(np.asarray(step.row_viol).copy())
            if not np.isfinite(u).all() or not np.isfinite(step.solve.xu).all():
                np.savez_compressed(out / "failure_solve.npz", state=x, reference=reference,
                                    torque_nm=u, xu=step.solve.xu)
                raise FloatingPointError("Nonfinite torque or predicted trajectory")
            interval_end = min(t + DT_S, args.timeout)
            while t < interval_end - 1e-12:
                h = min(SIM_DT_S, interval_end-t)
                q, v = rk4(model, data, x[:7], x[7:], u, h)
                x = np.r_[q, v]
                if not np.isfinite(x).all():
                    np.savez_compressed(out / "failure_plant.npz", state=x, torque_nm=u, time_s=t+h)
                    raise FloatingPointError("Nonfinite simulated state")
                t += h
                controls.append(u.copy()); times.append(t); states.append(x.copy())
                errors.append(float(np.linalg.norm(ee_position(model, data, q, fid)-target)))
            reached = errors[-1] <= POSITION_TOL_M and np.abs(x[7:]).sum() <= SPEED_L1_TOL_RAD_S
            if reached and not args.hold:
                break
        state_array, u_array = np.asarray(states), np.asarray(controls)
        q, v = state_array[:, :7], state_array[:, 7:]
        excess = dict(position_rad=float(max(0, np.max(model.lowerPositionLimit-q),
                                                   np.max(q-model.upperPositionLimit))),
                      speed_rad_s=float(max(0, np.max(np.abs(v)-model.velocityLimit))),
                      torque_nm=float(max(0, np.max(np.abs(u_array)-model.effortLimit))))
        limits_pass = max(excess.values()) <= 1e-5
        result.update(status="arrived" if reached else "timeout", reached=bool(reached),
                      simulated_time_s=t, final_distance_m=errors[-1],
                      final_speed_l1_rad_s=float(np.abs(x[7:]).sum()),
                      max_distance_m=max(errors), max_limit_excess=excess,
                      limits_pass=limits_pass, solve_calls=len(solve_us),
                      max_abs_torque_nm=float(np.max(np.abs(u_array))),
                      work_totals={k:sum(w[k] for w in work) for k in
                                   ("sqp", "pcg", "admm", "admm_linsys", "zero_pcg_rows")},
                      solve_time_median_ms=float(np.median(solve_us)/1000))
        exit_code = 0 if reached and limits_pass else 1
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        (out / "solve_work.json").write_text(json.dumps(work)+"\n")
        np.savez_compressed(out / "trajectory.npz", time_s=times, state=states,
                            applied_torque_nm=controls, distance_m=errors,
                            solve_time_us=solve_us, controller_wall_s=wall_s,
                            planned_row_violation=row_viol)
        (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps(result, indent=2), flush=True)
        print(f"Saved {out}", flush=True)
        if times:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            axes[0].plot(times, 1000*np.asarray(errors))
            axes[0].axhline(1000*POSITION_TOL_M, color="gray", ls="--")
            axes[0].set_ylabel("Position error (mm)")
            axes[1].plot(times, np.asarray(states)[:, 7:])
            axes[1].set_ylabel("Joint speed (rad/s)"); axes[1].set_xlabel("Simulation time (s)")
            fig.tight_layout(); fig.savefig(out / "reach.png", dpi=150); plt.close(fig)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
