#!/usr/bin/env python3
"""Fixed-clock horizontal figure-eight, with explicit plant/predictor losses."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pinocchio as pin
import reach_sim as common
from reaching_config import LEGACY_PARAMS
from joint_dynamics import validate, passive_torque, plant_rk4, register_robot, model_receipt

ROOT = common.ROOT
DEFAULT_PROTOCOL = ROOT/"configs/tracking.json"


def reference(times, start_xyz, shape):
    phase = 2*np.pi*np.asarray(times)/shape["period_s"]
    xyz = np.broadcast_to(start_xyz, (len(phase), 3)).copy()
    xyz[:, 0] += shape["edge_clearance_m"] + .5*shape["x_span_m"]*(1-np.cos(phase))
    xyz[:, 1] += shape["y_amplitude_m"]*np.sin(2*phase)
    return xyz


def model_checks(solver, model, data, signs, prediction_losses):
    """Value/A-block checks against the same constant-acceleration predictor step."""
    q = common.Q_START_RAD.copy()
    v = np.array([.3, -.35, .4, -.45, .5, -.55, .6])
    u = pin.rnea(model, data, q, v, np.zeros(7)) + passive_torque(v, prediction_losses)
    x = np.r_[q, v]; ss = np.tile(signs, 2)
    def native_predictor(state, h):
        qn, vn = state[:7], state[7:]
        acceleration = pin.aba(model, data, qn, vn, u-passive_torque(vn, prediction_losses)).copy()
        return np.r_[qn+vn*h+.5*acceleration*h*h, vn+acceleration*h]
    checks = {}
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
    args = parser.parse_args()
    settings = json.loads(args.config.read_text()); losses = validate(settings["joint_dynamics"])
    plant_losses = losses if args.losses != "neither" else None
    prediction_losses = None
    if args.losses == "smooth-predictor":
        prediction_losses = validate(dict(losses, model="coulomb_tanh_viscous",
            smoothing_rad_s=settings["predictor_smoothing_rad_s"]))
    n, dt, sim_dt, duration = (settings["N"], settings["dt_s"], settings["sim_dt_s"], settings["duration_s"])
    if (int(n) != n or n < 2 or not np.isfinite([dt, sim_dt, duration]).all()
            or min(dt, sim_dt, duration) <= 0 or sim_dt > dt):
        raise ValueError("Invalid tracking dimensions/times")
    if abs(round(duration/dt)*dt-duration) > 1e-10:
        raise ValueError("Duration must contain complete controller intervals")
    args.output.mkdir(parents=True, exist_ok=False); out = args.output
    for source in (Path(__file__), ROOT/"examples/joint_dynamics.py", ROOT/"examples/reach_sim.py"):
        shutil.copyfile(source, out/source.name)
    shutil.copyfile(args.config, out/"protocol.json")
    result = dict(status="setup", losses=args.losses, configuration=settings,
        protocol_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        repository_commit=common.git("rev-parse", "HEAD"), repository_status=common.git("status", "--short"),
        gato_commit=common.git("rev-parse", "HEAD", cwd=ROOT/"external/gato"),
        gato_status=common.git("status", "--short", cwd=ROOT/"external/gato"),
        versions={k:importlib.metadata.version(k) for k in ("gato-mpc", "grid-rbd", "pin", "numpy")},
        plant_joint_dynamics=plant_losses, prediction_joint_dynamics=prediction_losses,
        timing="fixed simulation clock; compute latency does not advance plant",
        reference_alignment="At time t controller reference knots are t+k*dt, k=0..N-1; state errors use reference at the same state timestamp, including t=0 and t=24s.",
        predictor_integration="upstream constant acceleration (named trapezoidal); native plant uses RK4 substeps",
        orientation="unconstrained", collisions="not checked", initial_warm_start="state hold, zero actuator torque; identical across cohorts",
        torque_convention="Saved applied torque is actuator torque; passive resisting torque is subtracted internally in ABA at every RK4 stage.",
        original_urdf_sha256=hashlib.sha256(common.URDF.read_bytes()).hexdigest())
    times, states, xyzs, refs, controls, passive, solve_us, wall_s, work, row_viol = [], [], [], [], [], [], [], [], [], []
    reference_horizons, planned_xu = [], []
    exit_code = 1
    try:
        path, gpu_path, signs = common.runtime_urdf(prediction_losses)
        # Both URDFs have the same rigid inertias; plant losses are explicit ABA input.
        plant_path, _, _ = common.runtime_urdf(plant_losses)
        shutil.copyfile(plant_path, out/"plant.urdf")
        shutil.copyfile(gpu_path, out/"prediction.urdf")
        model = pin.buildModelFromUrdf(str(plant_path)); data = model.createData()
        if list(model.names)[1:] != [f"arm_right_{i}_joint" for i in range(1, 8)]:
            raise ValueError("Unexpected joint order")
        fid = model.getFrameId(common.EE_FRAME); ss = np.tile(signs, 2)
        q0 = common.Q_START_RAD.copy(); x = np.r_[q0, np.zeros(7)]
        start_xyz = common.ee_position(model, data, q0, fid)
        result.update(q_start_rad=q0.tolist(), start_xyz_m=start_xyz.tolist(), gpu_joint_coordinate_signs=signs.tolist())
        predictor_name = "tiago_mpc_right_smooth" if prediction_losses else common.PLANT
        result["generator_adapter"] = register_robot(predictor_name, gpu_path, n_points=n,
            ee_frame=common.EE_FRAME, config=prediction_losses)
        params = LEGACY_PARAMS.replace(**settings["solver_params"])
        result["effective_solver_params"] = params.asdict()
        result.update(status="building", gpu_urdf_sha256=hashlib.sha256(gpu_path.read_bytes()).hexdigest())
        (out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
        solver = common.gato.BSQP(str(gpu_path), batch_size=1, N=n, dt=dt,
                                  plant_type=predictor_name, params=params)
        result["model_receipt"] = model_receipt(solver, predictor_name, gpu_path)
        (out/"model_receipt.json").write_text(json.dumps(result["model_receipt"], indent=2)+"\n")
        result["model_checks"] = model_checks(solver, model, data, signs, prediction_losses)
        if args.build_only:
            result["status"] = "model_checked"
            return 0
        controller = common.gato.MPCController(solver, limits="admm", linsys="pcg", **settings["controller"])
        bounds = settings["limit_bounds"]
        for group, key, raw in ((1, "velocity_fraction", model.velocityLimit), (2, "effort_fraction", model.effortLimit)):
            if not 0 < bounds[key] <= 1:
                raise ValueError("Limit fractions must be in (0,1]")
            solver.set_row_group_bounds(group, -bounds[key]*raw, bounds[key]*raw)
        result.update(admm_effective={**controller.LIMIT_DEFAULTS["admm"], **settings["controller"].get("limits_kw", {})},
            admm_linsys_policy=controller.admm_linsys, reset_rho_each_step=controller.reset_rho_each_step,
            raw_limits=dict(lower_position_rad=model.lowerPositionLimit.tolist(), upper_position_rad=model.upperPositionLimit.tolist(),
                            velocity_rad_s=model.velocityLimit.tolist(), effort_nm=model.effortLimit.tolist()))
        controller.reset(x*ss)
        def record(t):
            times.append(t); states.append(x.copy())
            xyzs.append(common.ee_position(model, data, x[:7], fid))
            refs.append(reference(np.array([t]), start_xyz, settings["figure8"])[0])
            if args.stop_on_raw_limit and (np.any(x[:7] < model.lowerPositionLimit-1e-5)
                    or np.any(x[:7] > model.upperPositionLimit+1e-5)
                    or np.any(np.abs(x[7:]) > model.velocityLimit+1e-5)):
                raise RuntimeError("Sampled native state exceeds a raw joint position/speed limit")
        record(0.)
        for k in range(round(duration/dt)):
            t = k*dt
            horizon = np.zeros((n, 6), dtype=np.float32)
            horizon[:, :3] = reference(t+np.arange(n)*dt, start_xyz, settings["figure8"])
            step = controller.step(x*ss, horizon.ravel())
            u = np.asarray(step.u, dtype=float)*signs
            if not np.isfinite(u).all() or not np.isfinite(step.solve.xu).all():
                np.savez_compressed(out/"failure_solve.npz", state=x, reference=horizon, xu=step.solve.xu, torque_nm=u)
                raise FloatingPointError("Nonfinite predicted trajectory or actuator torque")
            if args.stop_on_raw_limit and np.any(np.abs(u) > model.effortLimit+1e-5):
                np.savez_compressed(out/"failure_solve.npz", state=x, reference=horizon, xu=step.solve.xu, torque_nm=u)
                raise RuntimeError("Returned actuator torque exceeds a raw effort limit before application")
            reference_horizons.append(horizon.copy()); planned_xu.append(np.asarray(step.solve.xu).copy())
            solve_us.append(float(step.solve.solve_time_us)); wall_s.append(float(step.wall_s)); row_viol.append(np.asarray(step.row_viol).copy())
            stats = step.solve.stats
            work.append({key:int(np.sum(getattr(stats, key))) for key in ("sqp_iters", "pcg_iters", "admm_iters", "admm_linsys_iters")})
            local = 0.
            while local < dt-1e-12:
                h = min(sim_dt, dt-local)
                passive.append(passive_torque(x[7:], plant_losses)); controls.append(u.copy())
                q, v = plant_rk4(model, data, x[:7], x[7:], u, h, plant_losses); x = np.r_[q, v]
                if not np.isfinite(x).all():
                    raise FloatingPointError("Nonfinite native plant state")
                local += h; record(t+local)
            if k % 200 == 0:
                print(f"{args.losses}: t={t+dt:.2f}s error={np.linalg.norm(np.asarray(xyzs[-1])-refs[-1])*1000:.2f}mm", flush=True)
        state_array, torque_array = np.asarray(states), np.asarray(controls)
        error = np.linalg.norm(np.asarray(xyzs)-refs, axis=1)
        excess = dict(position_rad=float(max(0., np.max(model.lowerPositionLimit-state_array[:, :7]), np.max(state_array[:, :7]-model.upperPositionLimit))),
            speed_rad_s=float(max(0., np.max(np.abs(state_array[:, 7:])-model.velocityLimit))),
            actuator_torque_nm=float(max(0., np.max(np.abs(torque_array)-model.effortLimit))))
        result.update(status="completed", metrics_full_cycle=metrics(error),
            metrics_after_initial_1s=metrics(error[np.asarray(times)>=settings["secondary_metrics_start_s"]]),
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
            reference_horizon=reference_horizons, planned_xu=planned_xu)
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
