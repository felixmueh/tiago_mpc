#!/usr/bin/env python3
"""One position reach from the comfortable pose, in simulation or on the robot (checks only by default)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import signal
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.replay import export_replay
from tiago_mpc.laws import sample_knot
from tiago_mpc.loops import Fixed, Measured, run_robot, run_simulation
from tiago_mpc.mpc import TIAGO, MpcProcess
from tiago_mpc.record import Record
from tiago_mpc.robot import RosPlant
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import GRASP_FRAME, Q_REACH, fk, limits


@dataclass(frozen=True)
class Reach:
    """Reach target_m with ee_frame; done within the position and joint-speed L1 tolerances.

    The default target is the frame's position at a nearby posture (not a posture goal).
    """
    target_m: tuple[float, float, float] = tuple(map(float, fk(Q_REACH)))
    ee_frame: str = GRASP_FRAME
    position_tol_m: float = .05
    speed_l1_tol_rad_s: float = 1.
    timeout_s: float = 8.

    def reference(self, t, n, dt):
        out = np.zeros((n, 6), dtype=np.float32)  # orientation unconstrained
        out[:, :3] = self.target_m
        return out

    def distance(self, q):
        return float(np.linalg.norm(fk(q, self.ee_frame) - self.target_m))

    def done(self, state, t):
        return bool(self.distance(state.q) <= self.position_tol_m
                    and np.abs(state.qd).sum() <= self.speed_l1_tol_rad_s)


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
    return dict(solve_calls=len(trajectory["plan_t0_s"]),
                solve_time_median_ms=float(np.median(trajectory["plan_solve_time_us"]) / 1000),
                work_totals={k: int(np.sum(trajectory[f"plan_{k}"]))
                             for k in ("sqp", "pcg", "admm", "admm_linsys", "zero_pcg_rows")})


def reach(plant, run=run_simulation, task=Reach(), params=TIAGO, law=sample_knot, out=None,
          command_period=None, update_period=None, duration=None):
    """command_period and update_period default to the knot interval params.dt_s;
    duration (seconds) caps the task's timeout."""
    if task.ee_frame != params.ee_frame:
        raise ValueError("the task and the MPC must track the same frame")
    record = Record(out, task=task, params=params, plant=plant)
    mpc = None
    try:
        mpc = MpcProcess(params)
        arrived = run(plant, mpc, task, law, callbacks=[record], command_period=command_period or params.dt_s,
                      update_period=update_period or params.dt_s,
                      duration=task.timeout_s if duration is None else min(task.timeout_s, duration))
        trajectory = record.trajectory
        distance = np.array([task.distance(q) for q in trajectory["state"][:, :7]])
        excess = limit_excess(trajectory)
        record.arrays["distance_m"] = distance
        record.summary.update(
            status="arrived" if arrived else "timeout", reached=arrived, duration_cap_s=duration,
            simulated_time_s=float(trajectory["time_s"][-1]), initial_distance_m=distance[0],
            final_distance_m=distance[-1], max_distance_m=distance.max(),
            final_speed_l1_rad_s=float(np.abs(trajectory["state"][-1, 7:]).sum()),
            max_limit_excess=excess, limits_pass=max(excess.values()) <= 1e-5,
            max_abs_torque_nm=float(np.nanmax(np.abs(trajectory["command"]))),
            **solve_summary(trajectory))
    except BaseException as exc:  # also Ctrl-C and SIGTERM, which end robot runs
        record.summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                              error=traceback.format_exc())
        raise
    finally:
        if mpc is not None:
            mpc.close()
        plant.close()
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


def launch(args, experiment, sim_plant):
    """experiment(plant, run=...) on args.plant. The robot only runs the startup checks
    (plant.start() and plant.stop()) unless --execute is confirmed; returns the Record or None."""
    if args.plant == "sim":
        return experiment(sim_plant(), run=partial(run_simulation, latency=args.latency))
    if args.execute:
        print("Needs a stationary base and grippers and the arm at rest at comfortable_high_clearance;\n"
              "there is no automatic reset. The effort receiver's command timeout is unverified.")
        if input("Type EXECUTE to enable supervised effort control: ").strip() != "EXECUTE":
            print("Cancelled before creating the ROS client.")
            raise SystemExit(2)
    signal.signal(signal.SIGTERM, _interrupt)  # stop and hand back as on Ctrl-C
    plant = RosPlant(execute=args.execute, sea_absolute=args.sea_absolute)
    if args.execute:
        return experiment(plant, run=run_robot)
    try:
        plant.start()
        plant.stop()
    finally:
        plant.close()
    print(json.dumps(dict(status="checked", plant=plant.settings), indent=2, default=str))
    return None


def main(argv=None):
    args = arguments(__doc__, argv)
    task = Reach() if args.target is None else Reach(tuple(args.target))
    record = launch(args, partial(reach, task=task, out=args.output, command_period=args.command_period,
                                  update_period=args.update_period, duration=args.duration),
                    lambda: SimPlant(substep=.003))
    if record is None:
        return 0
    print(json.dumps({k: v for k, v in record.summary.items() if k not in ("params", "versions")},
                     indent=2, default=str))
    print(f"Saved {record.out}")
    return 0 if record.summary["reached"] and record.summary["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
