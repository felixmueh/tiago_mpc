"""The data that crosses the pieces' interfaces (the plant's state, the MPC's plan, the
loop's tick and its events) and the interfaces themselves: the protocols a plant, an MPC,
a task and a timing satisfy (structurally, checked by mypy, never at run time) and the
types of a control law and a tick callback.

A plan holds GATO's knot states x_0..x_{N-1} and knot torques u_0..u_{N-2}
(native coordinates), its knot interval dt and its start time t0: the origin of
its initial condition, predicted when compensation is enabled. measured_t retains
the original observation time. Knot k covers t0 + k dt <= t < t0 + (k + 1) dt,
a zero-order hold as GATO's transcription assumes; past the last knot there is
no torque, and a plan's lease ends before it (loops.lease_fault).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

from tiago_mpc.clock import VirtualClock, WallClock

if TYPE_CHECKING:
    from tiago_mpc.loops import Outcome
    from tiago_mpc.mpc import MpcParams

# Knot boundaries computed as k * dt must select knot k despite rounding
# (a fraction of one knot interval).
_KNOT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class State:
    """t: the time the state was measured, in seconds on the run's time (robot:
    monotonic receipt time); arm q and qd in native coordinates.

    The robot path adds the full body's named joint values for the safety check."""
    t: float
    q: np.ndarray
    qd: np.ndarray
    joint_positions: dict[str, float] | None = None
    joint_velocities: dict[str, float] | None = None


@dataclass(frozen=True)
class Prediction:
    """Predicted native q/qd at t, carrying the original measurement time unchanged."""
    t: float
    q: np.ndarray
    qd: np.ndarray
    measured_t: float = field(kw_only=True)

    def __post_init__(self) -> None:
        if not math.isfinite(self.t):
            raise ValueError("prediction time must be finite")
        if not math.isfinite(self.measured_t) or self.measured_t > self.t:
            raise ValueError("prediction must not precede its finite measured time")
        for name in ("q", "qd"):
            value = np.array(getattr(self, name), dtype=float)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError("prediction needs seven finite native positions and velocities")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Plan:
    t0: float
    dt: float
    states: np.ndarray   # (N, 14): q, qd per knot
    torques: np.ndarray  # (N - 1, 7)
    stats: dict = field(default_factory=dict)
    measured_t: float = field(kw_only=True)  # source observation; prediction never renews its lease

    def __post_init__(self):
        states, torques = np.array(self.states, dtype=float), np.array(self.torques, dtype=float)
        if (torques.ndim != 2 or len(torques) < 1 or states.shape != (len(torques) + 1, 2 * torques.shape[1])
                or not np.isfinite(states).all() or not np.isfinite(torques).all()):
            raise ValueError("A plan needs N finite knot states and N - 1 finite knot torques")
        if not math.isfinite(self.t0) or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("A plan needs a finite start time and a positive knot interval")
        if not math.isfinite(self.measured_t) or self.measured_t > self.t0:
            raise ValueError("measured_t must be finite and no later than the plan origin")
        states.flags.writeable = torques.flags.writeable = False
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "torques", torques)

    def index(self, offset_s: float) -> int:
        """Zero-order-hold knot of a time offset from t0; may exceed the plan. An offset a rounding
        error below 0 is knot 0: summed command ticks reach a predicted t0 up to that early."""
        if not offset_s >= -_KNOT_TOLERANCE * self.dt or math.isinf(offset_s):
            raise ValueError("time precedes the plan's state or is not finite")
        return max(0, math.floor(offset_s / self.dt + _KNOT_TOLERANCE))


@dataclass(frozen=True)
class Event:
    """Something the loop or a plant detected, at its own time t on the loop's clock: its
    kind, the message the console prints and small JSON data (record.py stores them)."""
    t: float
    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tick:
    """What tick callbacks see once per command tick: its time t and the latest
    state (state.t may be older and repeat); command is None before the first plan,
    on the last tick and on a fault; new marks the tick at which the plan took effect,
    update the tick at which a control update started (its request); events: what the
    loop and the plant detected since the previous tick (usually none); faulted: the
    plan that arrived at this tick and faulted (loops.plan_fault), which never takes
    effect, else None. plan_request_t associates a new/faulted plan with its request
    tick, independently of solver statistics; None when no plan is introduced."""
    t: float
    state: State
    command: np.ndarray | None
    plan: Plan | None
    new: bool
    update: bool
    events: tuple[Event, ...]
    faulted: Plan | None
    plan_request_t: float | None = None  # update tick that requested the new/faulted plan


class PlantInterface(Protocol):
    """SimPlant (sim.py) and RosPlant (robot.py), built from their configs. A loop starts
    it on the loop's time (start(clock): VirtualClock or WallClock, which RosPlant needs),
    then per command tick reads its state, applies a command, which it holds, takes its
    events (take_events: those emitted since the last call, on the loop's clock) and
    advances to the next tick (the plant integrates to it, the clock reaches it); stop
    comes first on exit (zero torque, handback), and its events are taken after the loop.
    run_state: the plant's final values, which the record saves ({} for SimPlant)."""
    run_state: dict[str, Any]

    def start(self, clock: VirtualClock | WallClock) -> None: ...
    def state(self) -> State: ...
    def apply(self, command: np.ndarray) -> None: ...
    def take_events(self) -> list[Event]: ...
    def advance(self, t: float) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class MpcInterface(Protocol):
    """Mpc and MpcProcess (mpc.py): plans of N knots dt apart. warmup prepares the
    solver from a state before the first request and returns when done (nothing when
    params.warmup is off); request raises while the previous plan is unread; result
    returns a plan once, None while it is pending, and raises if the solve failed.
    A request takes the measured State or a Prediction of the future initial condition;
    either preserves the measurement time that bounds the resulting plan's lease."""
    params: MpcParams
    N: int
    dt: float

    def warmup(self, state: State, reference: np.ndarray) -> None: ...
    def request(
        self,
        state: State | Prediction,
        reference: np.ndarray,
        updates: dict[str, Any] | None = None,
    ) -> None: ...
    def result(self, block: bool = False) -> Plan | None: ...
    def close(self) -> None: ...


class TaskInterface(Protocol):
    """A frozen dataclass; t counts from the run's first state. at: the (n, 6)
    end-effector reference at the n knots dt apart from t and the named solver updates
    (mpc.UPDATES) for the request at t, None when it changes nothing; done: whether
    it is met; duration_s: the task's own duration in seconds, the run's unless the
    caller caps it (experiments.common.run_experiment). The loops never read it: they
    run for the duration they are given."""
    __dataclass_fields__: ClassVar[dict[str, Any]]  # a dataclass: the record saves its fields

    def at(self, t: float, n: int, dt: float) -> tuple[np.ndarray, dict[str, Any] | None]: ...
    def done(self, state: State, t: float) -> bool: ...
    @property
    def duration_s(self) -> float: ...


ControlLaw = Callable[[Plan, float, State], np.ndarray]  # law(plan, t, state) -> command (laws.py)
Callback = Callable[[Tick], None]  # sees every command tick, e.g. a RunRecord


class Timing(Protocol):
    """VirtualTime and RealTime (loops.py), or an experiment's own: a frozen dataclass of
    a loop's settings whose run calls that loop with them and returns its Outcome."""
    __dataclass_fields__: ClassVar[dict[str, Any]]  # a dataclass: recorded, and replaced by the mains' setup

    def run(
        self,
        plant: PlantInterface,
        mpc: MpcInterface,
        task: TaskInterface,
        law: ControlLaw,
        tick_callbacks: Sequence[Callback],
        *,
        duration: float,
    ) -> Outcome: ...
