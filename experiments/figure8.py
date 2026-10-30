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
        out = np.zeros((n, 6), dtype=np.float32)  # orientation unconstrained
        out[:, :3] = self.path(t + np.arange(n)*dt)
        return out, None

    def done(self, state, t):
        return t >= self.period_s - 1e-9  # absorbs the rounding of summed command periods


# ======================================================================
# TODO(cost menu): GATO 0.1.5's cost menu (gato.cost_menu) will replace these cost overrides
# (docs/decisions.md, 2026-10-24): qd_cost and u_cost become the menu's state and input
# weights; mu, rho_u and the predictor friction are not costs and stay.
# ======================================================================
FIGURE8 = replace(tm.TIAGO, dt_s=.01, solver=tm.TIAGO.solver.replace(mu=1., qd_cost=.01, u_cost=1e-6),
                  admm=replace(tm.TIAGO.admm, rho_u=1e-5),  # GATO's value, for the control cost 1e-6
                  friction=tm.urdf_friction(), friction_law=tm.Tanh(.03))


def metrics(error):
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), median_m=float(np.median(error)),
                p95_m=float(np.percentile(error, 95)), max_m=float(np.max(error)))


def evaluate(record: tm.RunRecord) -> None:
    """The figure eight's results on a run of a Figure8, live or saved
    (tm.load_run(path, classes=[Figure8])): the ee_xyz_m and reference_xyz_m columns and the
    tracking error over the full cycle and after the first second, unless the run has no
    ticks. Saves the record."""
    task, ticks, results = record.config["task"], record.ticks, record.metadata["results"]
    if len(ticks["t"]):  # none: stopped before the first tick
        t = ticks["state_t"]  # when each position was measured
        ticks["ee_xyz_m"] = np.array([tm.ee_position(q) for q in ticks["q"]])
        ticks["reference_xyz_m"] = task.path(t)
        error = np.linalg.norm(ticks["ee_xyz_m"] - ticks["reference_xyz_m"], axis=1)
        results["metrics_full_cycle"] = metrics(error)
        results["metrics_after_initial_1s"] = metrics(error[t >= 1.]) if np.any(t >= 1.) else None
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
