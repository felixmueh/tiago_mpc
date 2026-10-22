"""Clock: a plant's source of execution time, in one of three domains.

Clock.wall() is the monotonic wall time; Clock.simulation(start) moves only by
advance(dt); Clock.external() takes the timestamps of an outside clock, such as
Gazebo's, by update(t). Neither moves backward, and an external clock may repeat
a time (a paused simulation).
"""
from __future__ import annotations

import math
import time
from typing import Literal


class Clock:
    """now() is the current execution time; wall_time() the monotonic wall time (for
    measuring computation), whatever the domain."""

    def __init__(self, kind: Literal["wall", "simulation", "external"]):
        self.kind = kind
        self._t: float | None = None

    @classmethod
    def wall(cls) -> Clock:
        return cls("wall")

    @classmethod
    def simulation(cls, start: float = 0.) -> Clock:
        clock = cls("simulation")
        clock._set(start)
        return clock

    @classmethod
    def external(cls) -> Clock:
        return cls("external")

    def now(self) -> float:
        if self.kind == "wall":
            return time.monotonic()
        if self._t is None:
            raise RuntimeError("the external clock has no time yet")
        return self._t

    @staticmethod
    def wall_time() -> float:
        return time.monotonic()

    def advance(self, dt: float) -> None:
        if self.kind != "simulation":
            raise RuntimeError(f"a {self.kind} clock cannot be advanced")
        self._set(self.now() + dt)

    def update(self, t: float) -> None:
        if self.kind != "external":
            raise RuntimeError(f"a {self.kind} clock cannot be updated")
        self._set(t)

    def _set(self, t):
        if not math.isfinite(t) or (self._t is not None and t < self._t):
            raise ValueError("clock time must be finite and must not move backward")
        self._t = t
