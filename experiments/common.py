"""What the experiments share: run_experiment builds the MPC from parameters, records and
runs one run, then saves and exports it; common_results are the results every run has;
arguments, setup and handle_stop_signals are the command line of their mains.

A library, not a command: run experiments.reach, experiments.figure8 and the others with
python -m experiments.<name> from the checkout's root."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys
from types import FrameType
import traceback
from typing import Any, Literal, TypeVar

import numpy as np

from experiments.replay import export_replay
import tiago_mpc as tm

TimingT = TypeVar("TimingT", bound=tm.Timing)  # setup returns the main's own kind of timing


def limit_excess(ticks: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Largest excess of the recorded states and sent commands over the raw limits."""
    raw = tm.limits()
    q, qd = ticks["q"], ticks["qd"]
    command = ticks["command"][np.isfinite(ticks["command"]).all(axis=1)]
    return dict(position_rad=float(max(0., np.max(raw.lower - q, initial=0.),
                                       np.max(q - raw.upper, initial=0.))),
                speed_rad_s=float(max(0., np.max(np.abs(qd) - raw.velocity, initial=0.))),
                torque_nm=float(max(0., np.max(np.abs(command) - raw.effort, initial=0.))))


def common_results(record: tm.RunRecord) -> dict[str, Any]:
    """The results every run has, from its ticks, plans and events: the raw-limit excess,
    the largest torque, the solves and their work (a run without solves has no median and
    zero work), the counts of ticks and plans, and of the loops' event kinds."""
    ticks, plans = record.ticks, record.plans
    excess, times = limit_excess(ticks), plans.get("solve_time_us")
    kinds = Counter(event.kind for event in record.events)
    return dict(limits_pass=max(excess.values()) <= 1e-5, max_limit_excess=excess,
                max_abs_torque_nm=float(np.nanmax(np.abs(ticks["command"]), initial=0.)),
                solve_calls=len(plans["t0"]),
                solve_time_median_ms=None if times is None else float(np.median(times) / 1000),
                work_totals={k: int(np.sum(plans.get(k, 0))) for k in ("sqp", "pcg", "admm", "admm_linsys", "zero_pcg_rows")},
                ticks=len(ticks["t"]), plans=len(plans["t0"]),
                late_ticks=kinds["late_tick"], overruns=kinds["overrun"], late_plans=kinds["late_plan"])


def run_experiment(
    plant: tm.SimConfig | tm.RobotConfig,
    mpc: tm.MpcParams | tm.MpcInterface,
    task: tm.TaskInterface,
    law: tm.ControlLaw,
    timing: tm.Timing,
    *,
    duration: float | None = None,
    out: str | Path | None = None,
    **pieces: Any,
) -> tm.RunRecord:
    """Run task with law on the plant built from its config, on timing, for the task's own
    duration (task.duration_s) or at most duration seconds, and return the RunRecord (in
    out, default _expr_data/<task>/<UTC time>/).

    mpc is parameters or a built MPC. Given MpcParams, run_experiment compiles and builds
    the MPC, which may take minutes unless GATO's module is cached (compile_binary): after
    creating the record and before building the plant, MpcProcess on RealTime (it solves
    concurrently, as on the robot), else the in-process Mpc; it closes it, also after an
    error. The wall-time budget (max_sqp_time_s) stays only where the solve's wall time is
    the latency, RealTime and VirtualTime with "measured"; every other timing gets None.
    A built MPC (a fake, or one MpcProcess per case from one compiled binary) is the
    caller's, which closes it.

    The record holds the pieces as the run's configuration: task, params (what the MPC
    loaded), plant, law, timing, duration (the one the run got) and the extra keyword
    pieces. Its status is the loop's Outcome ("done", "timeout", "fault", "interrupted")
    with its reason and end time, and the common results; a real error is "error" with the
    traceback (the hard abort, a second Ctrl-C, "interrupted") and is raised again after
    saving. The plant's events after the loop (the handback) and its run_state are
    recorded, the plant is closed, the run saved and its replay exported, also after a
    fault or an error. The experiment's evaluate adds its results and saves again.
    """
    if isinstance(mpc, tm.MpcParams):
        measured = isinstance(timing, tm.VirtualTime) and timing.solve_latency_s == "measured"
        params = mpc if isinstance(timing, tm.RealTime) or measured else replace(mpc, max_sqp_time_s=None)
    else:
        params = mpc.params
    duration = task.duration_s if duration is None else min(task.duration_s, duration)
    record = tm.RunRecord(out, task=task, params=params, plant=plant, law=law, timing=timing,
                          duration=duration, **pieces)
    console, built = tm.Console(), None
    solver = None if isinstance(mpc, tm.MpcParams) else mpc
    try:
        if solver is None:  # inside the try: a failed compile is a saved error
            solver = (tm.MpcProcess.compile(params) if isinstance(timing, tm.RealTime)  # concurrent, as on the robot
                      else tm.Mpc.compile(params))
        built = plant.build()  # inside the try: a failed build is a saved error
        outcome = timing.run(built, solver, task, law, [record, console], duration=duration)
        record.metadata.update(status=outcome.status, reason=outcome.reason, end_t=outcome.t,
                               results=common_results(record))
    except BaseException as exc:  # a real error, or the hard abort
        record.metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                               error=traceback.format_exc())
        raise
    finally:
        if built is not None:
            late = built.take_events()  # the handback, after the last tick
            record.add_events(late)
            console.add_events(late)
            record.metadata["plant"] = built.run_state
            built.close()
        if isinstance(mpc, tm.MpcParams) and solver is not None:  # built here; a passed one is the caller's
            solver.close()
        record.save()  # a faulted or failed run is never lost
        export_replay(record)  # so a faulted run keeps its replay
    return record


def parse_solve_latency(text: str) -> float | Literal["measured"]:
    """'measured', or declared seconds."""
    return "measured" if text == "measured" else float(text)


def arguments(
    description: str,
    argv: Sequence[str] | None,
    plants: Sequence[str] = ("sim", "rehearsal", "robot"),
) -> argparse.Namespace:
    """The options of the experiments' mains; --plant accepts only plants, the ones the
    experiment supports; a timing option not given is None."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--plant", choices=plants, default="sim",
                        help="sim: run_virtual_time with Mpc; rehearsal: SimPlant on run_real_time "
                             "with MpcProcess; robot: RosPlant on run_real_time (default: sim)")
    parser.add_argument("--execute", action="store_true",
                        help="robot: effort control after typed confirmation (default: checks only)")
    parser.add_argument("--sea-absolute", action="store_true", help="robot: SEA absolute-position feedback")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X_M", "Y_M", "Z_M"),
                        help="reach target or figure-eight start, metres in torso_lift_link (required with --execute)")
    parser.add_argument("--solve-latency", type=parse_solve_latency, metavar="SECONDS|measured",
                        help="sim: seconds from a request to its plan, or 'measured' (default: the experiment's timing)")
    parser.add_argument("--command-period", type=float, help="seconds (default: the timing's, 0.002, 500 Hz)")
    parser.add_argument("--update-period", type=float,
                        help="seconds between control updates (default: the timing's, 0.01; 0: back to back)")
    parser.add_argument("--duration", type=float, help="seconds: cap the task's own duration")
    parser.add_argument("--output", type=Path, help="fresh directory (default: under _expr_data/<task>/)")
    args = parser.parse_args(argv)
    if args.target is not None and not np.isfinite(args.target).all():
        parser.error("--target must be finite")
    if args.execute and (args.plant != "robot" or args.target is None):
        parser.error("--execute needs --plant robot and an explicit --target")
    if args.plant != "sim" and args.solve_latency is not None:
        parser.error("--solve-latency is for --plant sim; on wall time a solve takes as long as it takes")
    if isinstance(args.solve_latency, float) and not (np.isfinite(args.solve_latency) and args.solve_latency >= 0):
        parser.error("--solve-latency must be finite and nonnegative, or 'measured'")
    return args


def handle_stop_signals() -> None:
    """Clear tm.STOP, then let SIGINT (Ctrl-C) and SIGTERM set it: the run ends at its next
    tick as interrupted, and every later run of this process at its first. A second signal
    restores Python's default handlers and raises KeyboardInterrupt, the hard abort for
    the blocking calls that never reach a tick (compiling, the robot's first state). Every
    main that runs the simulation or the arm calls it first."""
    tm.STOP.clear()

    def stop(signum: int, frame: FrameType | None) -> None:
        if tm.STOP.is_set():
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise KeyboardInterrupt(f"signal {signum}")
        tm.STOP.set()
        print(f"signal {signum}: stopping at the next command tick (again: abort)", file=sys.stderr)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def setup(
    args: argparse.Namespace,
    plant: tm.SimConfig,
    timing: TimingT,
) -> tuple[tm.SimConfig | tm.RobotConfig, TimingT | tm.RealTime] | int:
    """The plant config and timing of a main's run from args, or its exit status.

    plant and timing are the main's defaults: its SimConfig and its virtual-time timing
    (VirtualTime, or its own with command_period, update_period and solve_latency_s, as
    Compensated). sim: that plant and that timing with the given --command-period,
    --update-period and --solve-latency replaced. rehearsal: that plant on RealTime with
    the given periods. robot: the robot's gate, before anything compiles. Without
    --execute, the check-only run: RobotConfig(execute=False)'s startup checks, then exit
    status 0. With it, the typed EXECUTE confirmation before any ROS client exists
    (anything else: exit status 2), and only then RobotConfig(execute=True, sea_absolute)
    on RealTime with the given periods.
    """
    periods = {key: value for key, value in (("command_period", args.command_period),
                                              ("update_period", args.update_period)) if value is not None}
    if args.plant == "sim":
        latency = {} if args.solve_latency is None else dict(solve_latency_s=args.solve_latency)
        return plant, replace(timing, **periods, **latency)
    if args.plant == "rehearsal":
        return plant, replace(tm.RealTime(), **periods)  # MpcProcess, concurrent, as on the robot
    if not args.execute:  # check-only: the startup checks, then exit
        robot = tm.RobotConfig(sea_absolute=args.sea_absolute).build()
        try:
            robot.start(tm.WallClock())
            robot.stop()
        finally:
            robot.close()
        print(json.dumps(dict(status="checked", plant=robot.run_state), indent=2, default=str))
        return 0
    print("Needs a stationary base and grippers and the arm at rest at comfortable_high_clearance;\n"
          "there is no automatic reset. The effort receiver's command timeout is unverified.")
    if input("Type EXECUTE to enable supervised effort control: ").strip() != "EXECUTE":
        print("Cancelled before creating the ROS client.")
        return 2
    return tm.RobotConfig(execute=True, sea_absolute=args.sea_absolute), replace(tm.RealTime(), **periods)
