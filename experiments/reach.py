#!/usr/bin/env python3
"""One position reach from the comfortable pose (simulation; the robot comes later)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cache, partial
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pinocchio as pin
from tiago_mpc.laws import sample_knot
from tiago_mpc.loops import Fixed, Measured, run_simulation
from tiago_mpc.model_runtime import EE_FRAME, Q_CHECK, Q_DELTA_CHECK, URDF, ee_position
from tiago_mpc.mpc import TIAGO, MpcWorker
from tiago_mpc.record import Record
from tiago_mpc.sim import SimPlant


@cache
def _arm():
    model = pin.buildModelFromUrdf(str(URDF))
    return model, model.createData()


def fk(q, frame=EE_FRAME):
    """Position of frame in the reduced arm's root frame (torso_lift_link), metres."""
    model, data = _arm()
    return ee_position(model, data, np.asarray(q, dtype=float), model.getFrameId(frame))


@dataclass(frozen=True)
class Reach:
    """Reach target_m with ee_frame; done within the position and joint-speed L1 tolerances.

    The default target is the frame's position at a nearby posture (not a posture goal).
    """
    target_m: tuple[float, float, float] = tuple(map(float, fk(Q_CHECK + Q_DELTA_CHECK)))
    ee_frame: str = EE_FRAME
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
    """Largest excess of the recorded states and sent commands over the raw URDF limits."""
    model, _ = _arm()
    q, qd = trajectory["state"][:, :7], trajectory["state"][:, 7:]
    command = trajectory["command"][np.isfinite(trajectory["command"]).all(axis=1)]
    return dict(position_rad=float(max(0., np.max(model.lowerPositionLimit - q, initial=0.),
                                       np.max(q - model.upperPositionLimit, initial=0.))),
                speed_rad_s=float(max(0., np.max(np.abs(qd) - model.velocityLimit, initial=0.))),
                torque_nm=float(max(0., np.max(np.abs(command) - model.effortLimit, initial=0.))))


def solve_summary(trajectory):
    return dict(solve_calls=len(trajectory["plan_t0_s"]),
                solve_time_median_ms=float(np.median(trajectory["plan_solve_time_us"]) / 1000),
                work_totals={k: int(np.sum(trajectory[f"plan_{k}"]))
                             for k in ("sqp", "pcg", "admm", "admm_linsys", "zero_pcg_rows")})


def reach(plant, run=run_simulation, task=Reach(), params=TIAGO, law=sample_knot, out=None,
          command_period=None, update_period=None):
    """command_period and update_period default to the knot interval params.dt_s."""
    if task.ee_frame != params.ee_frame:
        raise ValueError("the task and the MPC must track the same frame")
    record = Record(out, task=task, params=params, plant=plant)
    mpc = None
    try:
        mpc = MpcWorker(params)
        arrived = run(plant, mpc, task, law, callbacks=[record], command_period=command_period or params.dt_s,
                      update_period=update_period or params.dt_s, duration=task.timeout_s)
        trajectory = record.trajectory()
        distance = np.array([task.distance(q) for q in trajectory["state"][:, :7]])
        excess = limit_excess(trajectory)
        record.arrays["distance_m"] = distance
        record.summary.update(
            status="arrived" if arrived else "timeout", reached=arrived,
            simulated_time_s=float(trajectory["time_s"][-1]), initial_distance_m=distance[0],
            final_distance_m=distance[-1], max_distance_m=distance.max(),
            final_speed_l1_rad_s=float(np.abs(trajectory["state"][-1, 7:]).sum()),
            max_limit_excess=excess, limits_pass=max(excess.values()) <= 1e-5,
            max_abs_torque_nm=float(np.nanmax(np.abs(trajectory["command"]))),
            **solve_summary(trajectory))
    except Exception:
        record.summary.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mpc is not None:
            mpc.close()
        plant.close()
        record.save()
    return record


def latency(text):
    """'measured', or fixed seconds."""
    return Measured() if text == "measured" else Fixed(float(text))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", choices=("sim",), default="sim")
    parser.add_argument("--latency", type=latency, default=Fixed(), help="seconds per solve, or 'measured'")
    parser.add_argument("--command-period", type=float, help="seconds (default: the knot interval)")
    parser.add_argument("--update-period", type=float, help="seconds between control updates (default: the knot interval)")
    parser.add_argument("--output", type=Path, help="fresh directory (default: under _expr_data/reach/)")
    args = parser.parse_args(argv)
    record = reach(SimPlant(substep=.003), run=partial(run_simulation, latency=args.latency), out=args.output,
                   command_period=args.command_period, update_period=args.update_period)
    print(json.dumps({k: v for k, v in record.summary.items() if k not in ("params", "versions")},
                     indent=2, default=str))
    print(f"Saved {record.out}")
    return 0 if record.summary["reached"] and record.summary["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
