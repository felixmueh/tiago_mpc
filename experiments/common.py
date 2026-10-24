"""What the experiments share: run_experiment records one run, classifies, saves and
exports it; arguments and launch are the command line of their mains.

A library, not a command: run experiments.reach, experiments.figure8 and the others with
python -m experiments.<name> from the checkout's root."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import json
from pathlib import Path
import signal
from types import FrameType
import traceback
from typing import Any, Literal

import numpy as np

from experiments.replay import export_replay
import tiago_mpc as tm


def limit_excess(trajectory: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Largest excess of the recorded states and sent commands over the raw limits."""
    raw = tm.limits()
    q, qd = trajectory["state"][:, :7], trajectory["state"][:, 7:]
    command = trajectory["command"][np.isfinite(trajectory["command"]).all(axis=1)]
    return dict(position_rad=float(max(0., np.max(raw.lower - q, initial=0.),
                                       np.max(q - raw.upper, initial=0.))),
                speed_rad_s=float(max(0., np.max(np.abs(qd) - raw.velocity, initial=0.))),
                torque_nm=float(max(0., np.max(np.abs(command) - raw.effort, initial=0.))))


def solve_summary(trajectory: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """A run without solves has no median and zero work."""
    times = trajectory.get("plan_solve_time_us")
    return dict(solve_calls=len(trajectory["plan_t0_s"]),
                solve_time_median_ms=None if times is None else float(np.median(times) / 1000),
                work_totals={k: int(np.sum(trajectory.get(f"plan_{k}", 0)))
                             for k in ("sqp", "pcg", "admm", "admm_linsys", "zero_pcg_rows")})


def run_experiment(
    plant: tm.PlantInterface,
    mpc: tm.MpcInterface,
    task: tm.TaskInterface,
    law: tm.Law,
    run: Callable[..., tm.Outcome],
    *,
    duration: float,
    command_period: float | None = None,
    update_period: float | None = None,
    solve_latency: float | Literal["measured"] | None = None,
    out: str | Path | None = None,
    done_status: str = "done",
    evaluate: Callable[[tm.Record], object] | None = None,
) -> tm.Record:
    """Run task with mpc on plant for at most duration seconds with the loop run
    (tm.run_virtual_time, tm.run_real_time or an experiment's own), save the Record in out
    (default: _expr_data/<task>/<UTC time>/) and export its replay; returns the Record.
    The command period defaults to 2 ms (500 Hz), the update period to the knot
    interval mpc.dt (0: back to back). solve_latency goes to run unless None (the
    loop then needs none, as run_real_time).

    The summary records the loop's timing (timing: the loop, its periods,
    solve_latency, and its Outcome's late ticks and overruns). The status is
    done_status when the task is done, "timeout" when the duration ends first,
    "interrupted" (Ctrl-C, SIGTERM) or "error" (both raise again). A run that ends gets
    the raw-limit excess, the largest torque and the solve work, then evaluate(record)
    adds the experiment's metrics. The caller closes plant and mpc.
    """
    command_period = command_period or .002
    update_period = mpc.dt if update_period is None else update_period
    latency = {} if solve_latency is None else dict(solve_latency=solve_latency)
    record = tm.Record(out, task=task, params=mpc.params, plant=plant)
    record.summary["timing"] = dict(loop=getattr(run, "func", run).__name__, command_period_s=command_period,
                                    update_period_s=update_period, **latency)
    try:
        outcome = run(plant, mpc, task, law, callbacks=[record], command_period=command_period,
                      update_period=update_period, duration=duration, **latency)
        record.summary["timing"].update(late_ticks=outcome.late_ticks, overruns=outcome.overruns)
        trajectory = record.trajectory
        excess = limit_excess(trajectory)
        record.summary.update(
            status=done_status if outcome.done else "timeout", max_limit_excess=excess,
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


def parse_solve_latency(text: str) -> float | Literal["measured"]:
    """'measured', or declared seconds."""
    return "measured" if text == "measured" else float(text)


def arguments(description: str, argv: Sequence[str] | None) -> argparse.Namespace:
    """The options of the experiments' mains."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--plant", choices=("sim", "rehearsal", "robot"), default="sim",
                        help="sim: run_virtual_time with Mpc; rehearsal: SimPlant on run_real_time "
                             "with MpcProcess; robot: RosPlant on run_real_time (default: sim)")
    parser.add_argument("--execute", action="store_true",
                        help="robot: effort control after typed confirmation (default: checks only)")
    parser.add_argument("--sea-absolute", action="store_true", help="robot: SEA absolute-position feedback")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X_M", "Y_M", "Z_M"),
                        help="reach target or figure-eight start, metres in torso_lift_link (required with --execute)")
    parser.add_argument("--solve-latency", type=parse_solve_latency, metavar="SECONDS|measured",
                        help="sim: seconds from a request to its plan, or 'measured' (default: 0)")
    parser.add_argument("--command-period", type=float, help="seconds (default: 0.002, 500 Hz)")
    parser.add_argument("--update-period", type=float,
                        help="seconds between control updates (default: the knot interval; 0: back to back)")
    parser.add_argument("--duration", type=float, help="seconds: cap the task's own duration")
    parser.add_argument("--output", type=Path, help="fresh directory (default: under _expr_data/<task>/)")
    args = parser.parse_args(argv)
    if args.target is not None and not np.isfinite(args.target).all():
        parser.error("--target must be finite")
    if args.execute and (args.plant != "robot" or args.target is None):
        parser.error("--execute needs --plant robot and an explicit --target")
    if args.plant != "sim" and args.solve_latency is not None:
        parser.error("--solve-latency is for --plant sim; on wall time a solve takes as long as it takes")
    if args.plant == "sim" and args.solve_latency is None:
        args.solve_latency = 0.
    if isinstance(args.solve_latency, float) and not (np.isfinite(args.solve_latency) and args.solve_latency >= 0):
        parser.error("--solve-latency must be finite and nonnegative, or 'measured'")
    return args


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def launch(
    args: argparse.Namespace,
    experiment: Callable[..., tm.Record],
    sim_plant: Callable[[], tm.SimPlant],
    params: tm.MpcParams,
) -> tm.Record | None:
    """experiment(plant, mpc, run=..., out=..., command_period=..., update_period=...,
    duration=..., solve_latency=...) from args, on args.plant, and closes both; prints the
    summary. sim: run_virtual_time with an in-process Mpc.compile(params), without the solver's
    wall-time budget for a declared --solve-latency (max_sqp_time_s=None), with it for
    'measured'. rehearsal and robot: run_real_time with
    MpcProcess.compile(params) and the budget. The robot only runs the startup checks
    (plant.start(WallClock()) and plant.stop()) unless --execute is confirmed; returns the
    Record or None."""
    latency: float | Literal["measured"] | None = args.solve_latency  # None on wall time
    plant: tm.SimPlant | tm.RosPlant
    run: Callable[..., tm.Outcome]
    solver: type[tm.Mpc] | type[tm.MpcProcess]
    if args.plant == "sim":
        plant, run, solver = sim_plant(), tm.run_virtual_time, tm.Mpc  # in this process
        if latency != "measured":
            params = replace(params, max_sqp_time_s=None)  # a declared latency: the solve's wall time is irrelevant
    elif args.plant == "rehearsal":
        plant, run, solver = sim_plant(), tm.run_real_time, tm.MpcProcess  # concurrent, as on the robot
    else:
        if args.execute:
            print("Needs a stationary base and grippers and the arm at rest at comfortable_high_clearance;\n"
                  "there is no automatic reset. The effort receiver's command timeout is unverified.")
            if input("Type EXECUTE to enable supervised effort control: ").strip() != "EXECUTE":
                print("Cancelled before creating the ROS client.")
                raise SystemExit(2)
        signal.signal(signal.SIGTERM, _interrupt)  # stop and hand back as on Ctrl-C
        plant = tm.RosPlant(execute=args.execute, sea_absolute=args.sea_absolute)
        run, solver = tm.run_real_time, tm.MpcProcess
        if not args.execute:
            try:
                plant.start(tm.WallClock())
                plant.stop()
            finally:
                plant.close()
            print(json.dumps(dict(status="checked", plant=plant.settings), indent=2, default=str))
            return None
    try:
        mpc = solver.compile(params)  # compiles GATO's module once per variant (minutes)
        try:
            record = experiment(plant, mpc, run=run, out=args.output, command_period=args.command_period,
                                update_period=args.update_period, duration=args.duration, solve_latency=latency)
        finally:
            mpc.close()
    finally:
        plant.close()
    print(json.dumps({k: v for k, v in record.summary.items() if k not in ("params", "versions")},
                     indent=2, default=str))
    print(f"Saved {record.out}")
    return record
