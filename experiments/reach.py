"""One position reach from the comfortable pose, in simulation or on the robot (checks only by default).

Run from the checkout's root: python -m experiments.reach [--help]
(on the robot: python -m experiments.reach --plant robot)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import sys

import numpy as np

from experiments.common import arguments, launch, run_experiment
import tiago_mpc as tm


@dataclass(frozen=True)
class Reach:
    """Reach the grasp-frame position target_m; done within the position and joint-speed
    L1 tolerances.

    The default target is the position at a nearby posture (not a posture goal).
    """
    target_m: tuple[float, float, float] = tuple(map(float, tm.ee_position(tm.Q_REACH)))
    position_tol_m: float = .05
    speed_l1_tol_rad_s: float = 1.
    timeout_s: float = 8.

    def at(self, t, n, dt):
        out = np.zeros((n, 6), dtype=np.float32)  # orientation unconstrained
        out[:, :3] = self.target_m
        return out, None

    def distance(self, q):
        return float(np.linalg.norm(tm.ee_position(q) - self.target_m))

    def done(self, state, t):
        return bool(self.distance(state.q) <= self.position_tol_m
                    and np.abs(state.qd).sum() <= self.speed_l1_tol_rad_s)


def reach(plant, mpc, run=tm.run_simulation, task=Reach(), law=tm.sample_knot, out=None,
          command_period=None, update_period=None, duration=None):
    """Reach task with mpc on plant (common.run_experiment); duration (seconds) caps
    the task's timeout."""
    if mpc.params.ee_frame != tm.GRASP_FRAME:
        raise ValueError(f"the reach task measures the grasp frame {tm.GRASP_FRAME}; "
                         f"the MPC tracks {mpc.params.ee_frame}")

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
    record = launch(args, partial(reach, task=task), lambda: tm.SimPlant(substep=.003), tm.TIAGO)
    return 0 if record is None or record.summary["reached"] and record.summary["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
