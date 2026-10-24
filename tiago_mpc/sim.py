"""SimPlant: the right arm as a Pinocchio plant in native coordinates.

It holds each command until the next tick. The integrators and the Coulomb laws
(Friction.law) each takes:
- 'rk4': GATO's RK4 when loss-free; with a Tanh or Sign friction, RK4 with the passive
  torque at every stage (plant_rk4). StickSlip is Euler-only.
- 'euler': semi-implicit Euler with implicit damping (plant_euler), for every law: Tanh
  and Sign explicit in the old velocity, StickSlip as the coupled stick/slip solve;
  loss-free without a Friction.
The default is RK4 when loss-free, Euler with friction. The default maximum substep is
1 ms (1 kHz), independent of command/control rates.
"""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Literal
import warnings

import numpy as np
import pinocchio as pin
from gato.common import rk4

from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.dynamics import Friction, StickSlip, plant_euler, plant_rk4
from tiago_mpc.plan import State
from tiago_mpc.tiago import Q_START, arm_urdf


class SimPlant:
    """Starts at rest in q (default: the comfortable pose) when the run starts and stays
    there until the first command.

    The loop's clock comes at start: on virtual time state() and apply() find the plant
    at the tick's time, on wall time they first integrate up to the clock's time, so
    State.t is when the state is read. sync_real_time=True paces a virtual-time run for
    watching: advance also waits until the wall clock, counted from the first tick
    (after the warm-up), has caught up with the virtual time. It warns once when behind and never changes the results;
    on wall time it raises. model and data are the Pinocchio handles.
    """

    def __init__(
        self,
        q: np.ndarray = Q_START,
        qd: np.ndarray | None = None,
        *,
        substep: float = .001,
        friction: Friction | None = None,
        integrator: Literal["rk4", "euler"] | None = None,
        sync_real_time: bool = False,
    ):
        if not np.isfinite(substep) or substep <= 0:
            raise ValueError("substep must be finite and positive")
        integrator = integrator or ("rk4" if friction is None else "euler")
        if integrator not in ("rk4", "euler"):
            raise ValueError(f"Unknown integrator: {integrator}")
        if integrator == "rk4" and friction is not None and isinstance(friction.law, StickSlip):
            raise ValueError("StickSlip Coulomb friction is Euler-only; use integrator='euler'")
        self.model = pin.buildModelFromUrdf(str(arm_urdf()))
        self.data = self.model.createData()
        self.friction, self.substep, self.integrator = friction, substep, integrator
        self.sync_real_time = sync_real_time
        self.settings: dict[str, Any] = dict(friction=friction and asdict(friction), substep_s=substep,
                                             integrator=integrator, sync_real_time=sync_real_time)
        self.q = np.array(q, dtype=float)
        self.qd = np.zeros(7) if qd is None else np.array(qd, dtype=float)
        self.command: np.ndarray | None = None

    def start(self, clock: VirtualClock | WallClock) -> None:
        if self.sync_real_time and isinstance(clock, WallClock):
            raise ValueError("sync_real_time paces virtual time; on wall time the plant is in real time already")
        self.clock, self.t = clock, clock.now()
        self.settings["time"] = "wall" if isinstance(clock, WallClock) else "virtual"
        self._paced_from: tuple[float, float] | None = None  # pacing: wall and virtual time at the first tick
        self._behind = False

    def state(self) -> State:
        self._integrate(self.clock.now())
        return State(self.t, self.q.copy(), self.qd.copy())

    def apply(self, command: np.ndarray) -> None:
        """Hold command from now on, after integrating the held one up to now."""
        self._integrate(self.clock.now())
        self.command = np.array(command, dtype=float)

    def advance(self, t: float) -> None:
        """Integrate to t with the held command, then let the clock reach t: virtual time
        moves to t, wall time is waited for (late: at once). Paced, then also wait for the
        wall clock."""
        if self.sync_real_time and self._paced_from is None:  # from the first tick, after the warm-up
            self._paced_from = (time.monotonic(), self.t)
        self._integrate(t)
        self.clock.reach(t)
        if self._paced_from is not None:
            wall, start = self._paced_from
            wait = wall + (t - start) - time.monotonic()
            if wait < 0 and not self._behind:
                self._behind = True
                warnings.warn(f"sync_real_time: the run fell {-1e3 * wait:.1f} ms behind the wall clock "
                              "(warned once per run; the results are unaffected)")
            time.sleep(max(0., wait))

    def _integrate(self, t):
        """Integrate the held command from self.t to t in substeps; nothing unless t is later
        (state and apply on virtual time, or a deadline passed on wall time)."""
        if t - self.t <= 1e-12:
            return
        # No command yet: the state stays while time moves on, as the robot's onboard position
        # controller holds the arm until the first command activates effort control.
        while self.command is not None and self.t < t - 1e-12:
            h = min(self.substep, t - self.t)
            if self.integrator == "euler":
                q, qd = plant_euler(self.model, self.data, self.q, self.qd, self.command, h, self.friction)
            elif self.friction is not None:
                q, qd = plant_rk4(self.model, self.data, self.q, self.qd, self.command, h, self.friction)
            else:
                q, qd = rk4(self.model, self.data, self.q, self.qd, self.command, h)
            if not np.isfinite(q).all() or not np.isfinite(qd).all():
                raise FloatingPointError("Nonfinite simulated state")
            self.q, self.qd, self.t = q, qd, self.t + h
        self.t = t  # instead of accumulating substep rounding

    def stop(self) -> None:
        self.command = np.zeros(7)

    def close(self) -> None:
        pass
