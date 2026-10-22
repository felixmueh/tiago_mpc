"""The plant's state, the MPC's plan and the loop's tick, and the interfaces of the
pieces: the protocols a plant, an MPC and a task satisfy (structurally, checked by
mypy, never at run time) and the types of a control law and a callback.

A plan holds GATO's knot states x_0..x_{N-1} and knot torques u_0..u_{N-2}
(native coordinates), its knot interval dt and its start time t0: the time of
the state it was computed from. Knot k covers t0 + k dt <= t < t0 + (k + 1) dt,
a zero-order hold as GATO's transcription assumes; past the last knot there is
no torque (PlanEnded).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

from .clock import Clock

if TYPE_CHECKING:
    from .mpc import MpcParams

# Knot boundaries computed as k * dt must select knot k despite rounding
# (a fraction of one knot interval).
_KNOT_TOLERANCE = 1e-9


class PlanEnded(RuntimeError):
    """The requested time lies past the plan's last knot."""


@dataclass(frozen=True)
class State:
    """t: the time the state was measured, in seconds in the plant clock's domain (robot:
    monotonic receipt time); arm q and qd in native coordinates.

    The robot path adds the full body's named joint values for the safety check."""
    t: float
    q: np.ndarray
    qd: np.ndarray
    joint_positions: dict[str, float] | None = None
    joint_velocities: dict[str, float] | None = None


@dataclass(frozen=True)
class Plan:
    t0: float
    dt: float
    states: np.ndarray   # (N, 14): q, qd per knot
    torques: np.ndarray  # (N - 1, 7)
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        states, torques = np.array(self.states, dtype=float), np.array(self.torques, dtype=float)
        if (torques.ndim != 2 or len(torques) < 1 or states.shape != (len(torques) + 1, 2 * torques.shape[1])
                or not np.isfinite(states).all() or not np.isfinite(torques).all()):
            raise ValueError("A plan needs N finite knot states and N - 1 finite knot torques")
        if not math.isfinite(self.t0) or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("A plan needs a finite start time and a positive knot interval")
        states.flags.writeable = torques.flags.writeable = False
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "torques", torques)

    def index(self, offset_s: float) -> int:
        """Zero-order-hold knot of a time offset from t0; may exceed the plan."""
        if not offset_s >= 0 or math.isinf(offset_s):
            raise ValueError("time precedes the plan's state or is not finite")
        return math.floor(offset_s / self.dt + _KNOT_TOLERANCE)


@dataclass(frozen=True)
class Tick:
    """What callbacks see once per command tick: its execution time t and the latest
    state (state.t may be older and repeat); command is None before the first plan
    and on the last tick; new marks the tick at which the plan took effect."""
    t: float
    state: State
    command: np.ndarray | None
    plan: Plan | None
    new: bool


class PlantInterface(Protocol):
    """SimPlant (sim.py) and RosPlant (robot.py). run starts it, then per command tick
    reads its state and execution time (now), applies a command, which it holds, and
    advances to the next tick; stop comes first on exit (zero torque, handback).
    settings: what Record saves about it."""
    clock: Clock
    settings: dict[str, Any]

    def start(self) -> None: ...
    def now(self) -> float: ...
    def state(self) -> State: ...
    def apply(self, command: np.ndarray) -> None: ...
    def advance(self, t: float) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class MpcInterface(Protocol):
    """Mpc and MpcProcess (mpc.py): plans of N knots dt apart. warmup prepares the
    solver from a state before the first request and returns when done (nothing when
    params.warmup is off); request raises while the previous plan is unread; result
    returns a plan once, None while it is pending, and raises if the solve failed."""
    params: MpcParams
    N: int
    dt: float

    def warmup(self, state: State, reference: np.ndarray) -> None: ...
    def request(self, state: State, reference: np.ndarray, updates: dict[str, Any] | None = None) -> None: ...
    def result(self, block: bool = False) -> Plan | None: ...
    def close(self) -> None: ...


class TaskInterface(Protocol):
    """A frozen dataclass; t counts from the run's first state. at: the (n, 6)
    end-effector reference at the n knots dt apart from t and the named solver updates
    (mpc.UPDATES) for the request at t, None when it changes nothing; done: whether
    it is met."""
    __dataclass_fields__: ClassVar[dict[str, Any]]  # a dataclass: Record saves its fields

    def at(self, t: float, n: int, dt: float) -> tuple[np.ndarray, dict[str, Any] | None]: ...
    def done(self, state: State, t: float) -> bool: ...


Law = Callable[[Plan, float, State], np.ndarray]  # law(plan, t, state) -> command (laws.py)
Callback = Callable[[Tick], None]  # sees every command tick, e.g. a Record
