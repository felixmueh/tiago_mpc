#!/usr/bin/env python3
"""Horizontal figure-eight tracking (robot: checks only by default).

In simulation with friction in the plant (hard sign) and the predictor (tanh)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import json
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.reach import arguments, launch, limit_excess, solve_summary
from experiments.replay import export_replay
from tiago_mpc.laws import sample_knot
from tiago_mpc.loops import run_simulation
from tiago_mpc.mpc import TIAGO, MpcWorker
from tiago_mpc.record import Record
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import Q_START, fk, urdf_friction


@dataclass(frozen=True)
class Figure8:
    """One period of a horizontal figure eight in the reduced arm's root frame, metres.

    It starts edge_clearance_m ahead (+x) of start_m, by default the comfortable
    pose's position of the default end-effector frame.
    """
    start_m: tuple[float, float, float] = tuple(map(float, fk(Q_START)))
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

    def reference(self, t, n, dt):
        out = np.zeros((n, 6), dtype=np.float32)  # orientation unconstrained
        out[:, :3] = self.path(t + np.arange(n)*dt)
        return out

    def done(self, state, t):
        return t >= self.period_s


FIGURE8 = replace(TIAGO, dt_s=.01, solver=TIAGO.solver.replace(mu=1., qd_cost=.01, u_cost=1e-6),
                  admm=replace(TIAGO.admm, rho_q=.1, rho_qd=.03), friction=urdf_friction(.03))


def metrics(error):
    return dict(rmse_m=float(np.sqrt(np.mean(error**2))), median_m=float(np.median(error)),
                p95_m=float(np.percentile(error, 95)), max_m=float(np.max(error)))


def figure8(plant, run=run_simulation, task=Figure8(), params=FIGURE8, law=sample_knot, out=None,
            command_period=None, update_period=None, duration=None):
    """command_period and update_period default to the knot interval params.dt_s;
    duration (seconds) caps the task's period."""
    record = Record(out, task=task, params=params, plant=plant)
    mpc = None
    try:
        mpc = MpcWorker(params)
        run(plant, mpc, task, law, callbacks=[record], command_period=command_period or params.dt_s,
            update_period=update_period or params.dt_s,
            duration=task.period_s if duration is None else min(task.period_s, duration))
        trajectory = record.trajectory
        t = trajectory["state_time_s"]  # when each position was measured
        xyz = np.array([fk(q, params.ee_frame) for q in trajectory["state"][:, :7]])
        error = np.linalg.norm(xyz - task.path(t), axis=1)
        excess = limit_excess(trajectory)
        record.arrays.update(ee_xyz_m=xyz, reference_xyz_m=task.path(t))
        record.summary.update(
            status="completed", duration_cap_s=duration, metrics_full_cycle=metrics(error),
            metrics_after_initial_1s=metrics(error[t >= 1.]) if np.any(t >= 1.) else None,
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


def main(argv=None):
    args = arguments(__doc__, argv)
    task = Figure8() if args.target is None else Figure8(start_m=tuple(args.target))
    record = launch(args, partial(figure8, task=task, out=args.output, command_period=args.command_period,
                                  update_period=args.update_period, duration=args.duration),
                    lambda: SimPlant(substep=1e-4, friction=urdf_friction()))
    if record is None:
        return 0
    print(json.dumps({k: v for k, v in record.summary.items() if k not in ("params", "versions")},
                     indent=2, default=str))
    print(f"Saved {record.out}")
    return 0 if record.summary["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
