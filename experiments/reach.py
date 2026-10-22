#!/usr/bin/env python3
"""One position reach from the comfortable pose, in simulation or on the robot (checks only by default)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import arguments, launch, run_experiment
from tiago_mpc.laws import sample_knot
from tiago_mpc.loops import run_simulation
from tiago_mpc.mpc import TIAGO
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import GRASP_FRAME, Q_REACH, fk


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


def reach(plant, mpc, run=run_simulation, task=Reach(), law=sample_knot, out=None,
          command_period=None, update_period=None, duration=None):
    """Reach task with mpc on plant (common.run_experiment); duration (seconds) caps
    the task's timeout."""
    if task.ee_frame != mpc.params.ee_frame:
        raise ValueError("the task and the MPC must track the same frame")

    def evaluate(record):
        trajectory = record.trajectory
        distance = np.array([task.distance(q) for q in trajectory["state"][:, :7]])
        record.arrays["distance_m"] = distance
        record.summary.update(
            reached=record.summary["status"] == "arrived", duration_cap_s=duration,
            simulated_time_s=float(trajectory["time_s"][-1]), initial_distance_m=distance[0],
            final_distance_m=distance[-1], max_distance_m=distance.max(),
            final_speed_l1_rad_s=float(np.abs(trajectory["state"][-1, 7:]).sum()))

    return run_experiment(plant, mpc, task, law, run, command_period=command_period,
                          update_period=update_period,
                          duration=task.timeout_s if duration is None else min(task.timeout_s, duration),
                          out=out, done_status="arrived", evaluate=evaluate)


def main(argv=None):
    args = arguments(__doc__, argv)
    task = Reach() if args.target is None else Reach(tuple(args.target))
    record = launch(args, partial(reach, task=task), lambda: SimPlant(substep=.003), TIAGO)
    return 0 if record is None or record.summary["reached"] and record.summary["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
