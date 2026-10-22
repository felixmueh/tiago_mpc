#!/usr/bin/env python3
"""One position reach: upstream GATO MPC and an independent Pinocchio RK4 plant."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import shutil
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gato
import pinocchio as pin
from gato.common import rk4
from tiago_mpc.configuration import load_configuration
from tiago_mpc.controller import build_controller
from tiago_mpc.model_runtime import (URDF, EE_FRAME, Q_CHECK, Q_DELTA_CHECK,
                                     ee_position, runtime_urdf as prepare_urdf,
                                     check_model as check_predictor)
from tiago_mpc.plant_cache import cache_root
from tiago_mpc.simulation import control_intervals
from tiago_mpc.joint_dynamics import plant_rk4

SIM_DT_S = 0.003
TIMEOUT_S, POSITION_TOL_M, SPEED_L1_TOL_RAD_S = 8.0, 0.05, 1.0
Q_START_RAD = Q_CHECK.copy()
# A reachable target is FK(q_start + delta); q_goal is NOT a posture objective.
Q_GOAL_DELTA_RAD = Q_DELTA_CHECK.copy()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


# Keep the viewer's existing example API while sharing the implementation.
def runtime_urdf(joint_dynamics=None):
    native, gpu, signs, _, _ = prepare_urdf(cache_root(), joint_dynamics)
    return native, gpu, signs


def check_model(solver, model, data, fid, q_goal, signs):
    return check_predictor(solver, model, data, fid, signs, q_goal=q_goal)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold", action="store_true", help="target the initial tool position")
    p.add_argument("--timeout", type=float, default=TIMEOUT_S, help="simulated seconds")
    p.add_argument("--output", type=Path, help="fresh output directory")
    p.add_argument("--build-only", action="store_true", help="build and check model conventions")
    p.add_argument("--ee-frame", choices=(EE_FRAME, "arm_right_tool_joint"), default=EE_FRAME,
                   help="tracked point; default nominal grasp center, flange for historical tasks")
    p.add_argument("--input", type=Path, help="frozen experiment NPZ with initial_state/target_xyz/reference")
    p.add_argument("--config", type=Path, help="JSON overrides against the historical base; default configs/reaching.json")
    p.add_argument("--solve-time-mode", choices=("fixed", "measured"), default="fixed",
                   help="charge fixed latency or measured controller wall time to the plant")
    p.add_argument("--solve-time-s", type=float, default=0., help="fixed solve latency in seconds (default zero)")
    args = p.parse_args()
    try:
        control_intervals(1., 0., mode=args.solve_time_mode, fixed_s=args.solve_time_s)
    except ValueError as exc:
        p.error(str(exc))
    ee_frame = args.ee_frame
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        p.error("--timeout must be finite and positive")
    try:
        config_path, settings, effective = load_configuration(args.config)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    n_points, knot_dt = effective["N"], effective["dt_s"]
    plant_losses = settings.get("joint_dynamics")
    controller_options = effective["controller"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = args.output or ROOT / "_expr_data" / "reaching" / stamp
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "reach_sim.py")
    shutil.copyfile(config_path, out / "config.json")
    for name in ("configuration.py", "controller.py", "model_runtime.py", "joint_dynamics.py", "simulation.py"):
        shutil.copyfile(ROOT / "tiago_mpc" / name, out / name)
    result = dict(status="started", execution_stage="setup", mode="hold" if args.hold else "reach",
                  repository_commit=git("rev-parse", "HEAD") if (ROOT / ".git").exists()
                  and git("rev-list", "--all", "--count") != "0" else None,
                  repository_status=git("status", "--short"),
                  gato_commit=git("rev-parse", "HEAD", cwd=ROOT / "external/gato"),
                  gato_status=git("status", "--short", cwd=ROOT / "external/gato"),
                  versions={k: importlib.metadata.version(k) for k in
                            ("gato-mpc", "grid-rbd", "pin", "numpy")},
                  N=n_points, dt_s=knot_dt, prediction_horizon_s=(n_points-1)*knot_dt,
                  sim_dt_s=SIM_DT_S, timeout_s=args.timeout,
                  position_tol_m=POSITION_TOL_M, speed_l1_tol_rad_s=SPEED_L1_TOL_RAD_S,
                  solver_params=effective["solver_params"], configuration=settings,
                  configuration_file=str(config_path.resolve()),
                  effective_configuration=effective, limits="admm", linsys="pcg",
                  simulator_joint_dynamics=plant_losses,
                  prediction_joint_dynamics=effective["prediction_joint_dynamics"],
                  timing=dict(mode=args.solve_time_mode, fixed_s=args.solve_time_s,
                              measured_source="MPCController.step wall_s",
                              previous_control_during_solve=True, initial_control="zero torque"),
                  orientation="unconstrained", collisions="not checked",
                  end_effector_frame=ee_frame)
    times, states, controls, errors, solve_us, wall_s, row_viol = [], [], [], [], [], [], []
    work, charged_s, update_s, solve_start_s = [], [], [], []
    predicted_states, prediction_errors = [], []
    exit_code = 1
    try:
        setup = build_controller(effective, ee_frame=ee_frame)
        controller = setup.controller
        model, data, fid, signs = setup.model, setup.data, setup.frame_id, setup.signs
        gpu_path = setup.gpu_urdf_path
        state_signs = np.tile(signs, 2)
        result["gpu_joint_coordinate_signs"] = signs.tolist()
        result["joint_names"] = list(model.names)[1:]
        result["model_step_max_abs_errors"] = setup.details["model_step_max_abs_errors"]
        print("GATO/Pinocchio model checks passed", flush=True)
        if args.build_only:
            result["status"] = "model_checked"
            return 0
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
                reference = saved["reference"].astype(np.float32).reshape(-1, 6)
                historical_seed = saved["first_seed"]
                expected_seed = np.tile(np.r_[x_start, np.zeros(7)], len(reference))[:-7]
                if not np.allclose(historical_seed.ravel(), expected_seed, atol=1e-6):
                    raise ValueError("Frozen task seed differs from state-hold/zero-control policy")
                if len(reference) != n_points:
                    if not np.allclose(reference, reference[:1], atol=1e-6):
                        raise ValueError("Only constant goals can use a different knot count")
                    reference = np.repeat(reference[:1], n_points, axis=0)
            if (x_start.shape != (14,) or q_goal.shape != (7,) or target.shape != (3,)
                    or not all(np.isfinite(a).all() for a in (x_start, q_goal, target, reference))):
                raise ValueError("Invalid frozen input")
            result.update(mode="frozen_random_reach", input_file=str(args.input.resolve()))
        for q in (x_start[:7], q_goal):
            if np.any(q < model.lowerPositionLimit) or np.any(q > model.upperPositionLimit):
                raise ValueError("Start/target witness outside joint limits")
        witness_target = ee_position(model, data, q_goal, fid)
        if reference is None:
            target = witness_target
            reference = np.zeros((n_points, 6), dtype=np.float32)
            reference[:, :3] = target
        elif not np.allclose(target, witness_target, atol=1e-5, rtol=1e-4):
            raise ValueError("Frozen target frame differs from the native TIAGo model")
        initial = ee_position(model, data, x_start[:7], fid)
        result.update(q_start_rad=x_start[:7].tolist(), q_goal_witness_rad=q_goal.tolist(),
                      target_m=target.tolist(), initial_distance_m=float(np.linalg.norm(target-initial)))
        print(f"Target displacement: {result['initial_distance_m']*1000:.1f} mm", flush=True)
        result["status"] = "building_model"
        (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        result.update(reset_rho_each_step=controller.reset_rho_each_step,
                      admm_defaults=controller.LIMIT_DEFAULTS["admm"],
                      admm_effective={**controller.LIMIT_DEFAULTS["admm"], **controller_options.get("limits_kw", {})},
                      admm_linsys_policy=controller.admm_linsys)
        result.update(status="running", execution_stage="episode")
        (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        x = x_start.copy()
        controller.reset(x*state_signs)
        times.append(0.0); states.append(x.copy()); errors.append(float(np.linalg.norm(initial-target)))
        t, reached = 0.0, False
        previous_u = np.zeros(7)
        elapsed = None
        # Hold mode runs the full interval to check stability. Reach stops at arrival.
        while t < args.timeout - 1e-9:
            solve_start_s.append(t)
            step = controller.step(x*state_signs, reference.ravel(), dt_elapsed=elapsed)
            u = np.asarray(step.u, dtype=float)*signs
            stats = step.solve.stats
            predicted_next = np.asarray(step.solve.xu)[0, 21:35]*state_signs
            predicted_states.append(predicted_next.copy())
            if len(work) < int(settings.get("save_plan_steps", 0)):
                np.savez_compressed(out / f"plan_{len(work):04d}.npz", state=x,
                                    reference=reference, xu=step.solve.xu, signs=signs)
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
            intervals = control_intervals(knot_dt, step.wall_s, mode=args.solve_time_mode,
                                          fixed_s=args.solve_time_s, remaining_s=args.timeout-t)
            charged_s.append(intervals.charged_s); update_s.append(intervals.elapsed_s)
            interval_start = t
            for applied_u, duration in ((previous_u, intervals.previous_s), (u, intervals.new_s)):
                interval_end = t + duration
                while t < interval_end - 1e-12:
                    h = min(SIM_DT_S, interval_end-t)
                    if plant_losses is None:
                        q, v = rk4(model, data, x[:7], x[7:], applied_u, h)
                    else:
                        q, v = plant_rk4(model, data, x[:7], x[7:], applied_u, h, plant_losses)
                    x = np.r_[q, v]
                    if not np.isfinite(x).all():
                        np.savez_compressed(out / "failure_plant.npz", state=x, torque_nm=applied_u, time_s=t+h)
                        raise FloatingPointError("Nonfinite simulated state")
                    t += h
                    controls.append(applied_u.copy()); times.append(t); states.append(x.copy())
                    errors.append(float(np.linalg.norm(ee_position(model, data, q, fid)-target)))
            # Snap to the update boundary instead of accumulating RK4 rounding.
            t = min(interval_start + intervals.elapsed_s, args.timeout)
            times[-1] = t
            previous_u, elapsed = u, intervals.elapsed_s
            # A knot prediction and delayed execution cover different controls/times.
            if intervals.charged_s == 0 and abs(elapsed-knot_dt) < 1e-12:
                prediction_errors.append(np.array([np.max(np.abs(x[:7]-predicted_next[:7])),
                                                   np.max(np.abs(x[7:]-predicted_next[7:]))]))
            else:
                prediction_errors.append(np.full(2, np.nan))
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
                            planned_row_violation=row_viol, predicted_next_state=predicted_states,
                            solve_start_s=solve_start_s, charged_solve_time_s=charged_s,
                            control_interval_s=update_s,
                            prediction_error_qv=prediction_errors)
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
