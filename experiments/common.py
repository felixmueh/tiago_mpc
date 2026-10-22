"""What the experiments share: run_experiment records one run, classifies, saves and
exports it; arguments and launch are the command line of their mains."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import signal
import traceback

import numpy as np

from experiments.replay import export_replay
from tiago_mpc.loops import Fixed, Measured, run_robot, run_simulation
from tiago_mpc.mpc import MpcProcess
from tiago_mpc.record import Record
from tiago_mpc.robot import RosPlant
from tiago_mpc.tiago import limits


def limit_excess(trajectory):
    """Largest excess of the recorded states and sent commands over the raw limits."""
    raw = limits()
    q, qd = trajectory["state"][:, :7], trajectory["state"][:, 7:]
    command = trajectory["command"][np.isfinite(trajectory["command"]).all(axis=1)]
    return dict(position_rad=float(max(0., np.max(raw.lower - q, initial=0.),
                                       np.max(q - raw.upper, initial=0.))),
                speed_rad_s=float(max(0., np.max(np.abs(qd) - raw.velocity, initial=0.))),
                torque_nm=float(max(0., np.max(np.abs(command) - raw.effort, initial=0.))))


def solve_summary(trajectory):
    """A run without solves has no median and zero work."""
    times = trajectory.get("plan_solve_time_us")
    return dict(solve_calls=len(trajectory["plan_t0_s"]),
                solve_time_median_ms=None if times is None else float(np.median(times) / 1000),
                work_totals={k: int(np.sum(trajectory.get(f"plan_{k}", 0)))
                             for k in ("sqp", "pcg", "admm", "admm_linsys", "zero_pcg_rows")})


def run_experiment(plant, mpc, task, law, run, *, duration, command_period=None, update_period=None,
                   out=None, done_status="done", evaluate=None):
    """Run task with mpc on plant for at most duration seconds, save the Record in out
    (default: _expr_data/<task>/<UTC time>/) and export its replay; returns the Record.
    The periods default to the knot interval mpc.dt.

    The status is done_status when the task is done, "timeout" when the duration ends
    first, "interrupted" (Ctrl-C, SIGTERM) or "error" (both raise again). A run that
    ends gets the raw-limit excess, the largest torque and the solve work, then
    evaluate(record) adds the experiment's metrics. The caller closes plant and mpc.
    """
    record = Record(out, task=task, params=mpc.params, plant=plant)
    try:
        done = run(plant, mpc, task, law, callbacks=[record], command_period=command_period or mpc.dt,
                   update_period=update_period or mpc.dt, duration=duration)
        trajectory = record.trajectory
        excess = limit_excess(trajectory)
        record.summary.update(
            status=done_status if done else "timeout", max_limit_excess=excess,
            limits_pass=max(excess.values()) <= 1e-5,
            max_abs_torque_nm=float(np.nanmax(np.abs(trajectory["command"]), initial=0.)),
            **solve_summary(trajectory))
        if evaluate is not None:
            evaluate(record)
    except BaseException as exc:  # also Ctrl-C and SIGTERM, which end robot runs
        record.summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                              error=traceback.format_exc())
        raise
    finally:
        record.save()
        export_replay(record)
    return record


def latency(text):
    """'measured', or fixed seconds."""
    return Measured() if text == "measured" else Fixed(float(text))


def arguments(description, argv):
    """The options of the experiments' mains."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--plant", choices=("sim", "robot"), default="sim")
    parser.add_argument("--execute", action="store_true",
                        help="robot: effort control after typed confirmation (default: checks only)")
    parser.add_argument("--sea-absolute", action="store_true", help="robot: SEA absolute-position feedback")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X_M", "Y_M", "Z_M"),
                        help="reach target or figure-eight start, metres in torso_lift_link (required with --execute)")
    parser.add_argument("--latency", type=latency, default=Fixed(), help="sim: seconds per solve, or 'measured'")
    parser.add_argument("--command-period", type=float, help="seconds (default: the knot interval)")
    parser.add_argument("--update-period", type=float, help="seconds between control updates (default: the knot interval)")
    parser.add_argument("--duration", type=float, help="seconds: cap the task's own duration")
    parser.add_argument("--output", type=Path, help="fresh directory (default: under _expr_data/<task>/)")
    args = parser.parse_args(argv)
    if args.target is not None and not np.isfinite(args.target).all():
        parser.error("--target must be finite")
    if args.execute and (args.plant != "robot" or args.target is None):
        parser.error("--execute needs --plant robot and an explicit --target")
    return args


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def launch(args, experiment, sim_plant, params):
    """experiment(plant, mpc, run=..., out=..., command_period=..., update_period=...,
    duration=...) from args, on args.plant with MpcProcess.compile(params); launch closes
    both and prints the summary. The robot only runs the startup checks (plant.start() and
    plant.stop()) unless --execute is confirmed; returns the Record or None."""
    if args.plant == "sim":
        plant, run = sim_plant(), partial(run_simulation, latency=args.latency)
        params = replace(params, max_sqp_time_s=None)  # no wall-time limit on the simulation clock
    else:
        if args.execute:
            print("Needs a stationary base and grippers and the arm at rest at comfortable_high_clearance;\n"
                  "there is no automatic reset. The effort receiver's command timeout is unverified.")
            if input("Type EXECUTE to enable supervised effort control: ").strip() != "EXECUTE":
                print("Cancelled before creating the ROS client.")
                raise SystemExit(2)
        signal.signal(signal.SIGTERM, _interrupt)  # stop and hand back as on Ctrl-C
        plant, run = RosPlant(execute=args.execute, sea_absolute=args.sea_absolute), run_robot
        if not args.execute:
            try:
                plant.start()
                plant.stop()
            finally:
                plant.close()
            print(json.dumps(dict(status="checked", plant=plant.settings), indent=2, default=str))
            return None
    try:
        mpc = MpcProcess.compile(params)  # compiles GATO's module once per variant (minutes)
        try:
            record = experiment(plant, mpc, run=run, out=args.output, command_period=args.command_period,
                                update_period=args.update_period, duration=args.duration)
        finally:
            mpc.close()
    finally:
        plant.close()
    print(json.dumps({k: v for k, v in record.summary.items() if k not in ("params", "versions")},
                     indent=2, default=str))
    print(f"Saved {record.out}")
    return record
