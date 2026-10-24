"""Horizontal figure-eight tracking (robot: checks only by default).

In simulation with StickSlip Coulomb friction in the plant and Tanh in the predictor.

Run from the checkout's root: python -m experiments.figure8 [--help]."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import sys

import numpy as np

from experiments.common import arguments, handle_stop_signals, launch, run_experiment
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
                  friction=tm.urdf_friction(tm.Tanh(.03)))


def metrics(error):
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), median_m=float(np.median(error)),
                p95_m=float(np.percentile(error, 95)), max_m=float(np.max(error)))


def figure8(
    plant: tm.SimConfig | tm.RobotConfig,
    mpc: tm.MpcInterface,
    *,
    task: Figure8 = Figure8(),
    law: tm.Law = tm.SampleKnot(),
    timing: tm.Timing = tm.VirtualTime.ideal(),
    duration: float | None = None,
    out: str | Path | None = None,
) -> tm.RunRecord:
    """Track task with mpc on plant (common.run_experiment); duration (seconds) caps the
    task's period."""
    record = run_experiment(plant, mpc, task, law, timing, out=out,
                            duration=task.period_s if duration is None else min(task.period_s, duration))
    ticks, results = record.ticks, record.metadata["results"]
    t = ticks["state_t"]  # when each position was measured
    ticks["ee_xyz_m"] = np.array([tm.ee_position(q) for q in ticks["q"]])
    ticks["reference_xyz_m"] = task.path(t)
    error = np.linalg.norm(ticks["ee_xyz_m"] - ticks["reference_xyz_m"], axis=1)
    results["metrics_full_cycle"] = metrics(error)
    results["metrics_after_initial_1s"] = metrics(error[t >= 1.]) if np.any(t >= 1.) else None
    record.save()
    return record


def main(argv=None):
    handle_stop_signals()
    args = arguments(__doc__, argv)
    task = Figure8() if args.target is None else Figure8(start_m=tuple(args.target))
    record = launch(args, partial(figure8, task=task), tm.SimConfig(friction=tm.urdf_friction(), integrator="euler"),
                    FIGURE8, tm.VirtualTime.ideal())
    if isinstance(record, int):  # the robot's checks, or a cancelled execution
        return record
    ended = record.metadata["status"] in ("done", "timeout")  # timeout: a --duration cap; not a fault or a stop
    return 0 if ended and record.metadata["results"]["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
