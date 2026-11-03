"""Horizontal figure-eight tracking (robot: checks only by default).

In simulation with StickSlip Coulomb friction in the plant and Tanh in the predictor.

Run from the checkout's root: python -m experiments.figure8 [--help]."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import sys

import numpy as np

from experiments.common import arguments, handle_stop_signals, run_experiment, setup
import tiago_mpc as tm


@dataclass(frozen=True)
class Figure8:
    """One period of a horizontal figure eight in the reduced arm's root frame, metres.

    It starts edge_clearance_m ahead (+x) of start_m, by default the comfortable
    pose's grasp-frame position.
    """
    start_m: tuple[float, float, float] = tuple(map(float, tm.ee_position(tm.Q_START)))
    x_span_m: float = .28
    y_amplitude_m: float = .09
    edge_clearance_m: float = .02
    period_s: float = 24.
    orientation_rad: tuple[float, float, float] | None = None  # Grasp-frame rotation vector in ROOT_FRAME; None: free.

    def __post_init__(self):
        if self.orientation_rad is not None:
            object.__setattr__(self, "orientation_rad", tuple(map(float, self.orientation_rad)))

    @property
    def duration_s(self) -> float:
        return self.period_s

    def path(self, times):
        """(len(times), 3) positions at the given task times."""
        phase = 2*np.pi*np.asarray(times)/self.period_s
        xyz = np.broadcast_to(self.start_m, (len(phase), 3)).copy()
        xyz[:, 0] += self.edge_clearance_m + .5*self.x_span_m*(1-np.cos(phase))
        xyz[:, 1] += self.y_amplitude_m*np.sin(2*phase)
        return xyz

    def at(self, t, n, dt):
        out = np.full((n, 6), np.nan, dtype=np.float32)  # NaN orientation: none
        out[:, :3] = self.path(t + np.arange(n)*dt)
        if self.orientation_rad is not None:
            out[:, 3:] = self.orientation_rad
        return out, None

    def done(self, state, t):
        return t >= self.period_s - 1e-9  # absorbs the rounding of summed command periods


FIGURE8 = replace(tm.TIAGO, dt_s=.01, solver=replace(tm.TIAGO.solver, mu=1.),
                  cost=tm.default_cost(tm.TIAGO.N, velocity=.01, torque=1e-6),
                  admm=replace(tm.TIAGO.admm, rho_u=None),  # GATO's, for the control cost 1e-6
                  friction=tm.urdf_friction(), friction_law=tm.Tanh(.1))


def metrics(error, unit="m"):
    return {f"rmse_{unit}": float(np.sqrt(np.mean(error**2))), f"median_{unit}": float(np.median(error)),
            f"p95_{unit}": float(np.percentile(error, 95)), f"max_{unit}": float(np.max(error))}


def evaluate(record: tm.RunRecord) -> None:
    """Add figure-eight tracking metrics to the record and save it.

    A run with ticks records position error over the full cycle and after the first second,
    plus angular error when the task has an orientation target.
    """
    task, ticks, results = record.config["task"], record.ticks, record.metadata["results"]
    if len(ticks["t"]):  # none: stopped before the first tick
        t = ticks["state_t"]  # when each position was measured
        ticks["ee_xyz_m"] = np.array([tm.ee_position(q) for q in ticks["q"]])
        ticks["reference_xyz_m"] = task.path(t)
        error = np.linalg.norm(ticks["ee_xyz_m"] - ticks["reference_xyz_m"], axis=1)
        results["metrics_full_cycle"] = metrics(error)
        results["metrics_after_initial_1s"] = metrics(error[t >= 1.]) if np.any(t >= 1.) else None
        if task.orientation_rad is not None:
            angle = np.array([tm.orientation_error(q, task.orientation_rad) for q in ticks["q"]])
            ticks["orientation_error_rad"] = angle
            results["orientation_metrics_full_cycle"] = metrics(angle, "rad")
            results["orientation_metrics_after_initial_1s"] = metrics(angle[t >= 1.], "rad") if np.any(t >= 1.) else None
    record.save()


def main(argv=None):
    handle_stop_signals()
    args = arguments(__doc__, argv)
    task = Figure8() if args.target is None else Figure8(start_m=tuple(args.target))
    plant = tm.SimConfig(friction=tm.urdf_friction(), integrator="euler")
    pieces = setup(args, plant, tm.VirtualTime.like(tm.RealTime()))
    if isinstance(pieces, int):  # the robot's checks, or a cancelled execution
        return pieces
    plant, timing = pieces
    record = run_experiment(plant, FIGURE8, task, tm.PlanFeedback(), timing,
                            duration=args.duration, out=args.output)
    evaluate(record)
    print(json.dumps(record.metadata, indent=2, default=str))
    print(f"Saved {record.out}")
    ended = record.metadata["status"] in ("done", "timeout")  # timeout: a --duration cap; not a fault or a stop
    return 0 if ended and record.metadata["results"]["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
