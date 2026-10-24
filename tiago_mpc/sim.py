"""SimPlant: the right arm as a Pinocchio plant in native coordinates.

It holds each command until the next tick. integrator 'rk4' is GATO's loss-free RK4;
'euler' is semi-implicit Euler with implicit damping and tanh, sign or constraint
Coulomb friction (loss-free without a Friction). The default is RK4 when loss-free,
Euler with friction. The default maximum substep is 1 ms (1 kHz), independent of
command/control rates.
"""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Literal

from gato.common import rk4
import numpy as np
import pinocchio as pin

from .clock import Clock
from .dynamics import CoulombLaw, Friction, check_coulomb_law, plant_euler
from .plan import State
from .tiago import Q_START, arm_urdf


class SimPlant:
    """Starts at rest in q (default: the comfortable pose) when the run starts and stays
    there until the first command.

    It integrates with implicit Euler (semi-implicit, with implicit damping) when it has
    friction, RK4 otherwise.

    clock is Clock.simulation() by default; Clock.wall() runs in real time (for
    run_robot). state() first integrates up to the clock's time, so State.t is when
    the state is read. model and data are the Pinocchio handles.
    """

    def __init__(self, q: np.ndarray = Q_START, qd: np.ndarray | None = None, *, substep: float = .001,
                 friction: Friction | None = None, coulomb_law: CoulombLaw = "constraint",
                 integrator: Literal["rk4", "euler"] | None = None, clock: Clock | None = None):
        if not np.isfinite(substep) or substep <= 0:
            raise ValueError("substep must be finite and positive")
        integrator = integrator or ("rk4" if friction is None else "euler")
        if integrator not in ("rk4", "euler"):
            raise ValueError(f"Unknown integrator: {integrator}")
        if integrator == "rk4" and friction is not None:
            raise ValueError("RK4 is loss-free; simulate friction with integrator='euler'")
        check_coulomb_law(friction, coulomb_law)
        self.model = pin.buildModelFromUrdf(str(arm_urdf()))
        self.data = self.model.createData()
        self.friction, self.substep, self.integrator = friction, substep, integrator
        self.coulomb_law = coulomb_law
        self.clock = Clock.simulation() if clock is None else clock
        self.settings = dict(friction=friction and asdict(friction), substep_s=substep, clock=self.clock.kind,
                             integrator=integrator, coulomb_law=None if friction is None else coulomb_law)
        self.q = np.array(q, dtype=float)
        self.qd = np.zeros(7) if qd is None else np.array(qd, dtype=float)
        self.command: np.ndarray | None = None

    def start(self) -> None:
        self.t = self.clock.now()

    def now(self) -> float:
        return self.clock.now()

    def state(self) -> State:
        self._integrate(self.clock.now())
        return State(self.t, self.q.copy(), self.qd.copy())

    def apply(self, command: np.ndarray) -> None:
        """Hold command from now on, after integrating the held one up to now."""
        self._integrate(self.clock.now())
        self.command = np.array(command, dtype=float)

    def advance(self, t: float) -> None:
        """Integrate to t with the held command, then advance the simulation clock to t
        or wait until the wall clock reaches t (late: return at once)."""
        self._integrate(t)
        if self.clock.kind == "simulation":
            self.clock.advance(t - self.clock.now())
        else:
            time.sleep(max(0., t - self.clock.now()))

    def _integrate(self, t):
        """Integrate the held command from self.t to t in substeps; nothing unless t is later
        (state and apply on the simulation clock, or a deadline passed on the wall clock)."""
        if t - self.t <= 1e-12:
            return
        # No command yet: the state stays while time moves on, as the robot's onboard position
        # controller holds the arm until the first command activates effort control.
        while self.command is not None and self.t < t - 1e-12:
            h = min(self.substep, t - self.t)
            if self.integrator == "rk4":
                q, qd = rk4(self.model, self.data, self.q, self.qd, self.command, h)
            else:
                q, qd = plant_euler(self.model, self.data, self.q, self.qd, self.command, h,
                                    self.friction, self.coulomb_law)
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise FloatingPointError("Nonfinite simulated state")
            self.q, self.qd, self.t = q, qd, self.t + h
        self.t = t  # instead of accumulating substep rounding

    def stop(self) -> None:
        self.command = np.zeros(7)

    def close(self) -> None:
        pass
