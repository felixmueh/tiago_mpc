#!/usr/bin/env python3
"""Horizontal figure-eight, with explicit plant/predictor losses."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiago_mpc.joint_dynamics import passive_torque, plant_rk4, model_receipt, urdf_friction
from tiago_mpc.model_runtime import Q_CHECK, ee_position, runtime_urdf
import pinocchio as pin
from tiago_mpc.mpc import FIGURE8, MpcInProcess
from tiago_mpc.plant_cache import cache_root
from tiago_mpc.protocol import TaskReference, figure8_reference as reference
from tiago_mpc.plan import State
from tiago_mpc.simulation import control_intervals, control_segments

# Its defaults: 0.28 m span, 0.09 m amplitude, 0.02 m clearance, 24 s period. The run
# lasts one period and starts at the arm's start position.
TASK = TaskReference((0., 0., 0.), kind="figure8")
SIM_DT_S = 1e-4  # plant RK4 substep
SECONDARY_METRICS_START_S = 1.0
PLANT_FRICTION = urdf_friction()  # hard sign; the predictor's may differ


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def metrics(error):
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), median_m=float(np.median(error)),
                p95_m=float(np.percentile(error, 95)), max_m=float(np.max(error)))


def main(argv=None, params=FIGURE8, plant_friction=PLANT_FRICTION):
    """params: the MPC; plant_friction: the simulated plant's joint losses.
    --losses selects which of the two frictions are used."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", choices=("neither", "plant-only", "smooth-predictor"),
                        default="smooth-predictor", help="Default: hard-sign plant and smooth GATO predictor")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--stop-on-raw-limit", action="store_true",
                        help="Retain and stop an episode on a sampled raw position/speed limit violation")
    parser.add_argument("--solve-time-mode", choices=("fixed", "measured"), default="fixed",
                        help="charge fixed latency or measured controller wall time to the plant")
    parser.add_argument("--solve-time-s", type=float, default=0., help="fixed solve latency in seconds (default zero)")
    parser.add_argument("--control-update-period-s", type=float, help="nominal seconds between control updates (default: the knot interval dt_s)")
    args = parser.parse_args(argv)
    try:
        control_intervals(1. if args.control_update_period_s is None else args.control_update_period_s, 0.,
                          mode=args.solve_time_mode, fixed_s=args.solve_time_s)
    except ValueError as exc:
        parser.error(str(exc))
    plant_losses = plant_friction if args.losses != "neither" else None
    if args.losses != "smooth-predictor":
        params = replace(params, friction=None)
    n, dt, sim_dt, duration = params.N, params.dt_s, SIM_DT_S, TASK.period_s
    update_period_s = dt if args.control_update_period_s is None else args.control_update_period_s
    if (int(n) != n or n < 2 or not np.isfinite([dt, sim_dt, duration]).all()
            or min(dt, sim_dt, duration) <= 0 or sim_dt > dt):
        raise ValueError("Invalid tracking dimensions/times")
    if abs(round(duration/update_period_s)*update_period_s-duration) > 1e-10:
        raise ValueError("Duration must contain complete controller intervals")
    args.output.mkdir(parents=True, exist_ok=False); out = args.output
    for source in (Path(__file__), *(ROOT/"tiago_mpc"/name for name in
            ("mpc.py", "model_runtime.py", "joint_dynamics.py", "simulation.py",
             "protocol.py", "plan.py", "laws.py"))):
        shutil.copyfile(source, out/source.name)
    result = dict(status="setup", losses=args.losses, params=asdict(params),
        plant_friction=plant_losses and asdict(plant_losses),
        sim_dt_s=sim_dt, duration_s=duration, secondary_metrics_start_s=SECONDARY_METRICS_START_S,
        repository_commit=git("rev-parse", "HEAD"), repository_status=git("status", "--short"),
        gato_commit=git("rev-parse", "HEAD", cwd=ROOT/"external/gato"),
        gato_status=git("status", "--short", cwd=ROOT/"external/gato"),
        versions={k:importlib.metadata.version(k) for k in ("gato-mpc", "grid-rbd", "pin", "numpy")},
        timing=dict(mode=args.solve_time_mode, fixed_s=args.solve_time_s,
                    measured_source="MPCController.step wall_s",
                    control_update_period_s=update_period_s,
                    previous_plan_during_solve=True, initial_control="zero torque"),
        reference_alignment="At solve time t controller reference knots are t+k*dt, k=0..N-1; state errors use reference at the same state timestamp, including t=0 and t=24s.",
        predictor_integration="upstream constant acceleration (named trapezoidal); native plant uses RK4 substeps",
        orientation="unconstrained", collisions="not checked", end_effector_frame=params.ee_frame,
        initial_warm_start="state hold, zero actuator torque; identical across cohorts",
        torque_convention="Saved applied torque is actuator torque; passive resisting torque is subtracted internally in ABA at every RK4 stage.")
    times, states, xyzs, refs, controls, passive, solve_us, wall_s, work, row_viol = [], [], [], [], [], [], [], [], [], []
    reference_horizons, planned_states, planned_torques = [], [], []
    charged_s, update_s, solve_start_s = [], [], []
    exit_code = 1
    try:
        mpc = MpcInProcess(params)
        # Rigid inertias match; passive plant losses are explicit ABA input.
        predictor_path, gpu_path, signs, _, _ = runtime_urdf(cache_root(), params.friction)
        model = pin.buildModelFromUrdf(str(predictor_path))
        data, fid = model.createData(), model.getFrameId(params.ee_frame)
        plant_path, _, _, _, _ = runtime_urdf(cache_root(), plant_losses)
        shutil.copyfile(plant_path, out/"plant.urdf")
        shutil.copyfile(gpu_path, out/"prediction.urdf")
        q0 = Q_CHECK.copy(); x = np.r_[q0, np.zeros(7)]
        start_xyz = ee_position(model, data, q0, fid)
        task = replace(TASK, target_xyz_m=tuple(start_xyz))
        result.update(q_start_rad=q0.tolist(), start_xyz_m=start_xyz.tolist(), gpu_joint_coordinate_signs=signs.tolist(),
            task=asdict(task), generator_adapter=mpc.details["generator_adapter"],
            model_checks=mpc.details["model_step_max_abs_errors"])
        result["model_receipt"] = model_receipt(mpc.controller.solver, mpc.details["plant_name"], gpu_path)
        (out/"model_receipt.json").write_text(json.dumps(result["model_receipt"], indent=2)+"\n")
        if args.build_only:
            result["status"] = "model_checked"
            return 0
        result.update(raw_limits=dict(lower_position_rad=model.lowerPositionLimit.tolist(), upper_position_rad=model.upperPositionLimit.tolist(),
                            velocity_rad_s=model.velocityLimit.tolist(), effort_nm=model.effortLimit.tolist()))
        def record(t):
            times.append(t); states.append(x.copy())
            xyzs.append(ee_position(model, data, x[:7], fid))
            refs.append(reference(np.array([t]), start_xyz, task)[0])
            if args.stop_on_raw_limit and (np.any(x[:7] < model.lowerPositionLimit-1e-5)
                    or np.any(x[:7] > model.upperPositionLimit+1e-5)
                    or np.any(np.abs(x[7:]) > model.velocityLimit+1e-5)):
                raise RuntimeError("Sampled native state exceeds a raw joint position/speed limit")
        record(0.)
        t, k = 0., 0
        previous_plan = None
        while t < duration-1e-9:
            horizon = np.zeros((n, 6), dtype=np.float32)
            horizon[:, :3] = reference(t+np.arange(n)*dt, start_xyz, task)
            solve_start_s.append(t)
            # The plan starts at the time of the state it was computed from; result()
            # rejects plans with nonfinite knots or torques beyond the raw effort limits.
            mpc.request(State(t, x[:7], x[7:]), horizon.ravel())
            plan = mpc.result(block=True)
            stats = plan.stats
            intervals = control_intervals(update_period_s, stats["wall_s"], mode=args.solve_time_mode,
                                          fixed_s=args.solve_time_s, remaining_s=duration-t)
            segments = control_segments(previous_plan, plan, intervals)
            reference_horizons.append(horizon.copy())
            planned_states.append(plan.states); planned_torques.append(plan.torques)
            solve_us.append(stats["solve_time_us"]); wall_s.append(stats["wall_s"]); row_viol.append(stats["row_viol"])
            work.append({key: stats[key] for key in ("sqp", "pcg", "admm", "admm_linsys")})
            charged_s.append(intervals.charged_s); update_s.append(intervals.elapsed_s)
            interval_start = t
            for applied_u, span in segments:
                interval_end = t+span
                while t < interval_end-1e-12:
                    h = min(sim_dt, interval_end-t)
                    passive.append(passive_torque(x[7:], plant_losses)); controls.append(applied_u.copy())
                    q, v = plant_rk4(model, data, x[:7], x[7:], applied_u, h, plant_losses); x = np.r_[q, v]
                    if not np.isfinite(x).all():
                        raise FloatingPointError("Nonfinite native plant state")
                    t += h; record(t)
            # Snap to the update boundary instead of accumulating RK4 rounding.
            t = min(interval_start + intervals.elapsed_s, duration)
            times[-1] = t
            previous_plan = plan
            if k % 200 == 0:
                print(f"{args.losses}: t={t:.2f}s error={np.linalg.norm(np.asarray(xyzs[-1])-refs[-1])*1000:.2f}mm", flush=True)
            k += 1
        state_array, torque_array = np.asarray(states), np.asarray(controls)
        error = np.linalg.norm(np.asarray(xyzs)-refs, axis=1)
        excess = dict(position_rad=float(max(0., np.max(model.lowerPositionLimit-state_array[:, :7]), np.max(state_array[:, :7]-model.upperPositionLimit))),
            speed_rad_s=float(max(0., np.max(np.abs(state_array[:, 7:])-model.velocityLimit))),
            actuator_torque_nm=float(max(0., np.max(np.abs(torque_array)-model.effortLimit))))
        result.update(status="completed", metrics_full_cycle=metrics(error),
            metrics_after_initial_1s=(metrics(error[np.asarray(times)>=SECONDARY_METRICS_START_S])
                if np.any(np.asarray(times)>=SECONDARY_METRICS_START_S) else None),
            max_limit_excess=excess, limits_pass=bool(max(excess.values())<=1e-5),
            solve_calls=len(solve_us), solve_time_median_ms=float(np.median(solve_us)/1000),
            controller_wall_median_ms=float(np.median(wall_s)*1000),
            max_abs_actuator_torque_nm=float(np.max(np.abs(torque_array))),
            work_totals={key:sum(w[key] for w in work) for key in work[0]})
        exit_code = 0 if result["limits_pass"] else 1
    except Exception:
        result.update(status="error", error=traceback.format_exc()); traceback.print_exc()
    finally:
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (out/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        (out/"solve_work.json").write_text(json.dumps(work)+"\n")
        np.savez_compressed(out/"trajectory.npz", time_s=times, state=states, ee_xyz_m=xyzs,
            reference_xyz_m=refs, applied_actuator_torque_nm=controls, passive_resisting_torque_nm=passive,
            solve_time_us=solve_us, controller_wall_s=wall_s, planned_row_violation=row_viol,
            reference_horizon=reference_horizons, planned_states=planned_states, planned_torques=planned_torques,
            solve_start_s=solve_start_s, charged_solve_time_s=charged_s, control_interval_s=update_s)
        if times:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xyz = np.asarray(xyzs); ref = np.asarray(refs)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(ref[:, 0], ref[:, 1], label="reference"); axes[0].plot(xyz[:, 0], xyz[:, 1], label=args.losses)
            axes[0].axis("equal"); axes[0].set(xlabel="x (m)", ylabel="y (m)"); axes[0].legend()
            axes[1].plot(times, 1000*np.linalg.norm(xyz-ref, axis=1)); axes[1].set(xlabel="Time (s)", ylabel="Tracking error (mm)")
            fig.tight_layout(); fig.savefig(out/"tracking.png", dpi=150); plt.close(fig)
        print(json.dumps(result, indent=2), flush=True)
        print(f"Saved {out}", flush=True)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
