"""One position reach from the comfortable pose, in simulation or on the robot (checks only by default).

Run from the checkout's root: python -m experiments.reach [--help]
(on the robot: python -m experiments.reach --plant robot)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import sys

import numpy as np

from experiments.common import arguments, handle_stop_signals, run_experiment, setup
import tiago_mpc as tm


@dataclass(frozen=True)
class Reach:
    """Reach the grasp-frame position target_m; done within the position tolerance
    with the grasp point slower than the speed tolerance.

    The default target is the position at a nearby posture (not a posture goal).
    Orientation affects the MPC objective and recorded metrics; arrival uses position and speed.
    """
    target_m: tuple[float, float, float] = tuple(map(float, tm.ee_position(tm.Q_REACH)))
    position_tol_m: float = .05
    speed_tol_m_s: float = .02
    timeout_s: float = 8.
    orientation_rad: tuple[float, float, float] | None = None  # Grasp-frame rotation vector in ROOT_FRAME; None: free.

    def __post_init__(self):
        object.__setattr__(self, "target_m", tuple(map(float, self.target_m)))
        if self.orientation_rad is not None:
            object.__setattr__(self, "orientation_rad", tuple(map(float, self.orientation_rad)))

    @property
    def duration_s(self) -> float:
        return self.timeout_s

    def at(self, t, n, dt):
        out = np.full((n, 6), np.nan, dtype=np.float32)  # NaN orientation: none
        out[:, :3] = self.target_m
        if self.orientation_rad is not None:
            out[:, 3:] = self.orientation_rad
        return out, None

    def distance(self, q):
        return float(np.linalg.norm(tm.ee_position(q) - self.target_m))

    def done(self, state, t):
        return bool(self.distance(state.q) <= self.position_tol_m
                    and np.linalg.norm(tm.ee_velocity(state.q, state.qd)) <= self.speed_tol_m_s)


def evaluate(record: tm.RunRecord) -> None:
    """Add reaching metrics to the record and save it.

    A completed task is labelled arrived. A run with ticks records position error and final
    hand speed, plus angular error when the task has an orientation target.
    """
    task, ticks, results = record.config["task"], record.ticks, record.metadata["results"]
    if record.metadata["status"] == "done":
        record.metadata["status"] = "arrived"
    if len(ticks["t"]):  # none: stopped before the first tick
        distance = np.array([task.distance(q) for q in ticks["q"]])
        ticks["distance_m"] = distance
        results["initial_distance_m"] = distance[0]
        results["final_distance_m"] = distance[-1]
        results["max_distance_m"] = distance.max()
        results["final_speed_m_s"] = float(np.linalg.norm(tm.ee_velocity(ticks["q"][-1], ticks["qd"][-1])))
        if task.orientation_rad is not None:
            error = np.array([tm.orientation_error(q, task.orientation_rad) for q in ticks["q"]])
            ticks["orientation_error_rad"] = error
            results["initial_orientation_error_rad"] = error[0]
            results["final_orientation_error_rad"] = error[-1]
            results["max_orientation_error_rad"] = error.max()
    record.save()


def main(argv=None):
    handle_stop_signals()
    args = arguments(__doc__, argv)
    task = Reach() if args.target is None else Reach(tuple(args.target))
    pieces = setup(args, tm.SimConfig(substep=.003), tm.VirtualTime.like(tm.RealTime()))
    if isinstance(pieces, int):  # the robot's checks, or a cancelled execution
        return pieces
    plant, timing = pieces
    record = run_experiment(plant, tm.TIAGO, task, tm.PlanFeedback(), timing,
                            duration=args.duration, out=args.output)
    evaluate(record)
    print(json.dumps(record.metadata, indent=2, default=str))
    print(f"Saved {record.out}")
    return 0 if record.metadata["status"] == "arrived" and record.metadata["results"]["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
