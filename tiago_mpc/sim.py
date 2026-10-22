"""SimPlant: the right arm as a Pinocchio RK4 plant in native coordinates.

It holds each command until the next tick and integrates in RK4 substeps of at
most substep seconds. friction=None is loss-free (GATO's RK4); a Friction adds
its passive torque at every RK4 stage (hard sign without a width).
"""
from __future__ import annotations

from dataclasses import asdict
import time

from gato.common import rk4
import numpy as np
import pinocchio as pin

from .joint_dynamics import plant_rk4
from .model_runtime import Q_CHECK, URDF
from .plan import State


class SimPlant:
    """Starts at rest in q (default: the comfortable pose) at t = 0.

    realtime=True makes advance wait for the wall clock (for run_robot).
    model and data are the Pinocchio handles.
    """

    def __init__(self, q=Q_CHECK, qd=None, *, substep, friction=None, realtime=False):
        self.model = pin.buildModelFromUrdf(str(URDF))
        self.data = self.model.createData()
        self.friction, self.substep, self.realtime = friction, substep, realtime
        self.settings = dict(friction=friction and asdict(friction), substep_s=substep,
                             clock="wall" if realtime else "simulation")
        self.t, self.q = 0., np.array(q, dtype=float)
        self.qd = np.zeros(7) if qd is None else np.array(qd, dtype=float)
        self.command, self._wall_start = np.zeros(7), None

    def start(self):
        self._wall_start = time.monotonic() - self.t

    def state(self):
        return State(self.t, self.q.copy(), self.qd.copy())

    def apply(self, command):
        self.command = np.array(command, dtype=float)

    def advance(self, t):
        """Integrate to t with the held command; realtime: then wait until t."""
        while self.t < t - 1e-12:
            h = min(self.substep, t - self.t)
            if self.friction is None:
                q, qd = rk4(self.model, self.data, self.q, self.qd, self.command, h)
            else:
                q, qd = plant_rk4(self.model, self.data, self.q, self.qd, self.command, h, self.friction)
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise FloatingPointError("Nonfinite simulated state")
            self.q, self.qd, self.t = q, qd, self.t + h
        self.t = t  # the tick's time, instead of accumulating substep rounding
        if self.realtime:
            time.sleep(max(0., self._wall_start + t - time.monotonic()))

    def stop(self):
        self.command = np.zeros(7)

    def close(self):
        pass
