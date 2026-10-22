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

from .clock import Clock
from .dynamics import Friction, plant_rk4
from .plan import State
from .tiago import ARM_URDF, Q_START


class SimPlant:
    """Starts at rest in q (default: the comfortable pose) when the run starts.

    clock is Clock.simulation() by default; Clock.wall() runs in real time (for
    run_robot). State.t is the integrated time, in the clock's domain. model and
    data are the Pinocchio handles.
    """

    def __init__(self, q: np.ndarray = Q_START, qd: np.ndarray | None = None, *, substep: float,
                 friction: Friction | None = None, clock: Clock | None = None):
        self.model = pin.buildModelFromUrdf(str(ARM_URDF))
        self.data = self.model.createData()
        self.friction, self.substep = friction, substep
        self.clock = Clock.simulation() if clock is None else clock
        self.settings = dict(friction=friction and asdict(friction), substep_s=substep, clock=self.clock.kind)
        self.q = np.array(q, dtype=float)
        self.qd = np.zeros(7) if qd is None else np.array(qd, dtype=float)
        self.command: np.ndarray = np.zeros(7)

    def start(self) -> None:
        self.t = self.clock.now()

    def now(self) -> float:
        return self.clock.now()

    def state(self) -> State:
        return State(self.t, self.q.copy(), self.qd.copy())

    def apply(self, command: np.ndarray) -> None:
        self.command = np.array(command, dtype=float)

    def advance(self, t: float) -> None:
        """Integrate to t with the held command, then advance the simulation clock to t
        or wait until the wall clock reaches t (late: return at once)."""
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
        if self.clock.kind == "simulation":
            self.clock.advance(t - self.clock.now())
        else:
            time.sleep(max(0., t - self.clock.now()))

    def stop(self) -> None:
        self.command = np.zeros(7)

    def close(self) -> None:
        pass
