#!/usr/bin/env python3
"""Horizontal figure-eight, with explicit plant/predictor losses."""
from __future__ import annotations
import argparse
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
from tiago_mpc.configuration import load_configuration
from tiago_mpc.controller import build_controller
from tiago_mpc.joint_dynamics import validate, passive_torque, plant_rk4, model_receipt
from tiago_mpc.model_runtime import URDF, EE_FRAME, Q_CHECK, ee_position, runtime_urdf
from tiago_mpc.plant_cache import cache_root
from tiago_mpc.protocol import figure8_reference as reference
from tiago_mpc.playback import MODES, Plan, knot_torques
from tiago_mpc.simulation import control_intervals, control_segments
DEFAULT_PROTOCOL = ROOT/"configs/tracking.json"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def metrics(error):
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), median_m=float(np.median(error)),
                p95_m=float(np.percentile(error, 95)), max_m=float(np.max(error)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", choices=("neither", "plant-only", "smooth-predictor"),
                        default="smooth-predictor", help="Default: hard-sign plant and smooth GATO predictor")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--stop-on-raw-limit", action="store_true",
                        help="Retain and stop an episode on a sampled raw physical-limit violation")
    parser.add_argument("--solve-time-mode", choices=("fixed", "measured"), default="fixed",
                        help="charge fixed latency or measured controller wall time to the plant")
    parser.add_argument("--solve-time-s", type=float, default=0., help="fixed solve latency in seconds (default zero)")
    parser.add_argument("--control-period-s", type=float, help="nominal seconds between solves (default: the knot interval dt_s)")
    parser.add_argument("--command-mode", choices=MODES, default="playback",
                        help="play each plan back by knot (default), or hold its first torque as before")
    args = parser.parse_args()
    try:
        control_intervals(1. if args.control_period_s is None else args.control_period_s, 0.,
                          mode=args.solve_time_mode, fixed_s=args.solve_time_s)
    except ValueError as exc:
        parser.error(str(exc))
    _, settings, effective = load_configuration(args.config)
    losses = validate(settings["joint_dynamics"])
    plant_losses = losses if args.losses != "neither" else None
    prediction_losses = effective["prediction_joint_dynamics"] if args.losses == "smooth-predictor" else None
    effective["prediction_joint_dynamics"] = prediction_losses
    n, dt, sim_dt, duration = (settings["N"], settings["dt_s"], settings["sim_dt_s"], settings["duration_s"])
    period = dt if args.control_period_s is None else args.control_period_s
    if (int(n) != n or n < 2 or not np.isfinite([dt, sim_dt, duration]).all()
            or min(dt, sim_dt, duration) <= 0 or sim_dt > dt):
        raise ValueError("Invalid tracking dimensions/times")
    if abs(round(duration/period)*period-duration) > 1e-10:
        raise ValueError("Duration must contain complete controller intervals")
    args.output.mkdir(parents=True, exist_ok=False); out = args.output
    for source in (Path(__file__), *(ROOT/"tiago_mpc"/name for name in
            ("configuration.py", "controller.py", "model_runtime.py", "joint_dynamics.py", "simulation.py",
             "protocol.py", "playback.py"))):
        shutil.copyfile(source, out/source.name)
    shutil.copyfile(args.config, out/"protocol.json")
    result = dict(status="setup", losses=args.losses, configuration=settings,
        repository_commit=git("rev-parse", "HEAD"), repository_status=git("status", "--short"),
        gato_commit=git("rev-parse", "HEAD", cwd=ROOT/"external/gato"),
        gato_status=git("status", "--short", cwd=ROOT/"external/gato"),
        versions={k:importlib.metadata.version(k) for k in ("gato-mpc", "grid-rbd", "pin", "numpy")},
        plant_joint_dynamics=plant_losses, prediction_joint_dynamics=prediction_losses,
        timing=dict(mode=args.solve_time_mode, fixed_s=args.solve_time_s,
                    measured_source="MPCController.step wall_s",
                    control_period_s=period, command_mode=args.command_mode,
                    previous_plan_during_solve=True, initial_control="zero torque",
                    past_plan_end="playback holds the final knot, flagged per plant step"),
        reference_alignment="At solve time t controller reference knots are t+k*dt, k=0..N-1; state errors use reference at the same state timestamp, including t=0 and t=24s.",
        predictor_integration="upstream constant acceleration (named trapezoidal); native plant uses RK4 substeps",
        orientation="unconstrained", collisions="not checked", end_effector_frame=EE_FRAME,
        initial_warm_start="state hold, zero actuator torque; identical across cohorts",
        torque_convention="Saved applied torque is actuator torque; passive resisting torque is subtracted internally in ABA at every RK4 stage.")
    times, states, xyzs, refs, controls, passive, solve_us, wall_s, work, row_viol = [], [], [], [], [], [], [], [], [], []
    reference_horizons, planned_xu = [], []
    charged_s, update_s, solve_start_s = [], [], []
    applied_knots, applied_plan_starts, past_plan_end = [], [], []
    exit_code = 1
    try:
        setup = build_controller(effective)
        controller = setup.controller
        solver = controller.solver
        model, data, fid, signs = setup.model, setup.data, setup.frame_id, setup.signs
        # Rigid inertias match; passive plant losses are explicit ABA input.
        plant_path, _, _, _, _ = runtime_urdf(cache_root(), plant_losses)
        shutil.copyfile(plant_path, out/"plant.urdf")
        shutil.copyfile(setup.gpu_urdf_path, out/"prediction.urdf")
        ss = np.tile(signs, 2)
        q0 = Q_CHECK.copy(); x = np.r_[q0, np.zeros(7)]
        start_xyz = ee_position(model, data, q0, fid)
        result.update(q_start_rad=q0.tolist(), start_xyz_m=start_xyz.tolist(), gpu_joint_coordinate_signs=signs.tolist(),
            effective_configuration=effective, generator_adapter=setup.details["generator_adapter"],
            effective_solver_params=effective["solver_params"],
            model_checks=setup.details["model_step_max_abs_errors"])
        result["model_receipt"] = model_receipt(solver, setup.details["plant_name"], setup.gpu_urdf_path)
        (out/"model_receipt.json").write_text(json.dumps(result["model_receipt"], indent=2)+"\n")
        if args.build_only:
            result["status"] = "model_checked"
            return 0
        result.update(admm_effective=effective["controller"]["limits_kw"],
            admm_linsys_policy=controller.admm_linsys, reset_rho_each_step=controller.reset_rho_each_step,
            raw_limits=dict(lower_position_rad=model.lowerPositionLimit.tolist(), upper_position_rad=model.upperPositionLimit.tolist(),
                            velocity_rad_s=model.velocityLimit.tolist(), effort_nm=model.effortLimit.tolist()))
        controller.reset(x*ss)
        def record(t):
            times.append(t); states.append(x.copy())
            xyzs.append(ee_position(model, data, x[:7], fid))
            refs.append(reference(np.array([t]), start_xyz, settings["figure8"])[0])
            if args.stop_on_raw_limit and (np.any(x[:7] < model.lowerPositionLimit-1e-5)
                    or np.any(x[:7] > model.upperPositionLimit+1e-5)
                    or np.any(np.abs(x[7:]) > model.velocityLimit+1e-5)):
                raise RuntimeError("Sampled native state exceeds a raw joint position/speed limit")
        record(0.)
        t, k, elapsed = 0., 0, None
        previous_plan = None
        while t < duration-1e-9:
            horizon = np.zeros((n, 6), dtype=np.float32)
            horizon[:, :3] = reference(t+np.arange(n)*dt, start_xyz, settings["figure8"])
            solve_start_s.append(t)
            step = controller.step(x*ss, horizon.ravel(), dt_elapsed=elapsed)
            u = np.asarray(step.u, dtype=float)*signs
            if not np.isfinite(u).all() or not np.isfinite(step.solve.xu).all():
                np.savez_compressed(out/"failure_solve.npz", state=x, reference=horizon, xu=step.solve.xu, torque_nm=u)
                raise FloatingPointError("Nonfinite predicted trajectory or actuator torque")
            # The plan starts at the time of the state it was computed from.
            plan = Plan(t, dt, knot_torques(step.xu_best, signs), args.command_mode)
            intervals = control_intervals(period, step.wall_s, mode=args.solve_time_mode,
                                          fixed_s=args.solve_time_s, remaining_s=duration-t)
            segments = control_segments(previous_plan, plan, intervals)
            if args.stop_on_raw_limit and any(np.any(np.abs(s[2]) > model.effortLimit+1e-5) for s in segments):
                np.savez_compressed(out/"failure_solve.npz", state=x, reference=horizon, xu=step.solve.xu, torque_nm=u)
                raise RuntimeError("Actuator torque due in this interval exceeds a raw effort limit before application")
            reference_horizons.append(horizon.copy()); planned_xu.append(np.asarray(step.solve.xu).copy())
            solve_us.append(float(step.solve.solve_time_us)); wall_s.append(float(step.wall_s)); row_viol.append(np.asarray(step.row_viol).copy())
            stats = step.solve.stats
            work.append({key:int(np.sum(getattr(stats, key))) for key in ("sqp_iters", "pcg_iters", "admm_iters", "admm_linsys_iters")})
            charged_s.append(intervals.charged_s); update_s.append(intervals.elapsed_s)
            interval_start = t
            for source, knot, applied_u, span, beyond in segments:
                interval_end = t+span
                while t < interval_end-1e-12:
                    h = min(sim_dt, interval_end-t)
                    passive.append(passive_torque(x[7:], plant_losses)); controls.append(applied_u.copy())
                    applied_knots.append(knot); past_plan_end.append(beyond)
                    applied_plan_starts.append(np.nan if source is None else source.start_s)
                    q, v = plant_rk4(model, data, x[:7], x[7:], applied_u, h, plant_losses); x = np.r_[q, v]
                    if not np.isfinite(x).all():
                        raise FloatingPointError("Nonfinite native plant state")
                    t += h; record(t)
            # Snap to the update boundary instead of accumulating RK4 rounding.
            t = min(interval_start + intervals.elapsed_s, duration)
            times[-1] = t
            # dt_elapsed is the time between the two plans' states.
            previous_plan, elapsed = plan, intervals.elapsed_s
            if k % 200 == 0:
                print(f"{args.losses}: t={t:.2f}s error={np.linalg.norm(np.asarray(xyzs[-1])-refs[-1])*1000:.2f}mm", flush=True)
            k += 1
        state_array, torque_array = np.asarray(states), np.asarray(controls)
        error = np.linalg.norm(np.asarray(xyzs)-refs, axis=1)
        excess = dict(position_rad=float(max(0., np.max(model.lowerPositionLimit-state_array[:, :7]), np.max(state_array[:, :7]-model.upperPositionLimit))),
            speed_rad_s=float(max(0., np.max(np.abs(state_array[:, 7:])-model.velocityLimit))),
            actuator_torque_nm=float(max(0., np.max(np.abs(torque_array)-model.effortLimit))))
        result.update(status="completed", metrics_full_cycle=metrics(error),
            metrics_after_initial_1s=(metrics(error[np.asarray(times)>=settings["secondary_metrics_start_s"]])
                if np.any(np.asarray(times)>=settings["secondary_metrics_start_s"]) else None),
            max_limit_excess=excess, limits_pass=bool(max(excess.values())<=1e-5),
            solve_calls=len(solve_us), solve_time_median_ms=float(np.median(solve_us)/1000),
            controller_wall_median_ms=float(np.median(wall_s)*1000),
            max_abs_actuator_torque_nm=float(np.max(np.abs(torque_array))),
            plant_steps_past_plan_end=int(np.sum(past_plan_end)),
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
            reference_horizon=reference_horizons, planned_xu=planned_xu,
            solve_start_s=solve_start_s, charged_solve_time_s=charged_s, control_interval_s=update_s,
            applied_knot=np.asarray(applied_knots, dtype=int),
            applied_plan_start_s=np.asarray(applied_plan_starts, dtype=float),
            applied_past_plan_end=np.asarray(past_plan_end, dtype=bool))
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
