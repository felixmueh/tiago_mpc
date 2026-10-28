"""SimConfig and SimPlant: the right arm as a Pinocchio plant in native coordinates.

It holds each command until the next tick. The integrators and the Coulomb laws
(SimConfig.friction_law) each takes:
- 'rk4': GATO's RK4 when loss-free; with a Tanh or Sign friction, RK4 with the passive
  torque at every stage (plant_rk4). StickSlip is Euler-only.
- 'euler': semi-implicit Euler with implicit damping (plant_euler), for every law: Tanh
  and Sign explicit in the old velocity, StickSlip as the coupled stick/slip solve;
  loss-free without a Friction.
The default is RK4; a plant with friction names its integrator. The default maximum
substep is 1 ms (1 kHz), independent of command/control rates.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

import numpy as np
import pinocchio as pin
from gato.common import rk4

from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.dynamics import CoulombLaw, Friction, StickSlip, plant_euler, plant_rk4
from tiago_mpc.interfaces import Event, State
from tiago_mpc.tiago import Q_START, arm_urdf


@dataclass(frozen=True)
class SimConfig:
    """The simulated arm, a value; build() is its SimPlant. q, qd: the start state (tuples;
    lists and arrays are converted), by default the comfortable pose at rest; substep:
    the longest integration step; friction: the plant's losses (None: loss-free);
    friction_law: their Coulomb evaluation or coupled stick/slip integration;
    armature: constant reflected joint inertia, passed to Pinocchio once;
    integrator: 'rk4' or 'euler' (the module docstring); sync_real_time: pace a
    virtual-time run for watching (SimPlant). SimPlant checks the values."""
    q: tuple[float, ...] = tuple(map(float, Q_START))
    qd: tuple[float, ...] = (0.,) * 7
    substep: float = .001
    friction: Friction | None = None
    friction_law: CoulombLaw = StickSlip()
    integrator: Literal["rk4", "euler"] = "rk4"
    sync_real_time: bool = False
    armature: tuple[float, ...] = (0.,) * 7  # reflected joint inertia, kg m²

    def __post_init__(self):
        object.__setattr__(self, "q", tuple(map(float, self.q)))
        object.__setattr__(self, "qd", tuple(map(float, self.qd)))
        values = tuple(float(value) for value in self.armature)
        if len(values) != 7 or not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError('armature must contain seven finite nonnegative values')
        object.__setattr__(self, 'armature', values)

    def build(self) -> SimPlant:
        return SimPlant(self)


class SimPlant:
    """Starts at rest in config.q (default: the comfortable pose) when the run starts and
    stays there until the first command.

    The loop's clock comes at start: on virtual time state() and apply() find the plant
    at the tick's time, on wall time they first integrate up to the clock's time, so
    State.t is when the state is read. sync_real_time=True paces a virtual-time run for
    watching: advance also waits until the wall clock, counted from the first tick
    (after the warm-up), has caught up with the virtual time. Each time it falls behind
    is a pacing_behind event; it never changes the results. On wall time it raises.
    model and data are the Pinocchio handles.
    """

    def __init__(self, config: SimConfig):
        if not np.isfinite(config.substep) or config.substep <= 0:
            raise ValueError("substep must be finite and positive")
        if config.integrator not in ("rk4", "euler"):
            raise ValueError(f"Unknown integrator: {config.integrator}")
        if (config.integrator == "rk4" and config.friction is not None
                and isinstance(config.friction_law, StickSlip)):
            raise ValueError("StickSlip Coulomb friction is Euler-only; use integrator='euler'")
        self.config = config
        self.model = pin.buildModelFromUrdf(str(arm_urdf()))
        self.model.armature = np.asarray(config.armature)  # Pinocchio includes it in ABA, CRBA and RNEA
        self.data = self.model.createData()
        self.q, self.qd = np.array(config.q, dtype=float), np.array(config.qd, dtype=float)
        self.command: np.ndarray | None = None
        self.run_state: dict[str, Any] = {}
        self._events: list[Event] = []

    def start(self, clock: VirtualClock | WallClock) -> None:
        if self.config.sync_real_time and isinstance(clock, WallClock):
            raise ValueError("sync_real_time paces virtual time; on wall time the plant is in real time already")
        self.clock, self.t = clock, clock.now()
        self._paced_from: tuple[float, float] | None = None  # pacing: wall and virtual time at the first tick
        self._behind = False

    def state(self) -> State:
        self._integrate(self.clock.now())
        return State(self.t, self.q.copy(), self.qd.copy())

    def apply(self, command: np.ndarray) -> None:
        """Hold command from now on, after integrating the held one up to now."""
        self._integrate(self.clock.now())
        self.command = np.array(command, dtype=float)

    def take_events(self) -> list[Event]:
        """The events since the last call (pacing_behind), forgotten here."""
        events, self._events = self._events, []
        return events

    def advance(self, t: float) -> None:
        """Integrate to t with the held command, then let the clock reach t: virtual time
        moves to t, wall time is waited for (late: at once). Paced, then also wait for the
        wall clock."""
        if self.config.sync_real_time and self._paced_from is None:  # from the first tick, after the warm-up
            self._paced_from = (time.monotonic(), self.t)
        self._integrate(t)
        self.clock.reach(t)
        if self._paced_from is not None:
            wall, start = self._paced_from
            wait = wall + (t - start) - time.monotonic()
            if wait < 0 and not self._behind:  # fell behind; caught up again when it can wait
                self._events.append(Event(t, "pacing_behind", f"sync_real_time: the run fell {-1e3 * wait:.1f} ms "
                                                              "behind the wall clock (the results are unaffected)",
                                          dict(behind_s=-wait)))
            self._behind = wait < 0
            time.sleep(max(0., wait))

    def _integrate(self, t):
        """Integrate the held command from self.t to t in substeps; nothing unless t is later
        (state and apply on virtual time, or a deadline passed on wall time)."""
        if t - self.t <= 1e-12:
            return
        friction = self.config.friction
        law = self.config.friction_law
        # No command yet: the state stays while time moves on, as the robot's onboard position
        # controller holds the arm until the first command activates effort control.
        while self.command is not None and self.t < t - 1e-12:
            h = min(self.config.substep, t - self.t)
            if self.config.integrator == "euler":
                q, qd = plant_euler(self.model, self.data, self.q, self.qd, self.command, h, friction, law)
            elif friction is not None:
                q, qd = plant_rk4(self.model, self.data, self.q, self.qd, self.command, h, friction, law)
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
