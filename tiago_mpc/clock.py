"""The two kinds of time a loop runs on, given to the plant at start (loops.py).

Stepped is time the loop keeps: nothing takes time unless the loop moves it on.
Wall is the computer's monotonic clock: everything takes its real time. Neither
moves backward.
"""
from __future__ import annotations

import math
import time


class Stepped:
    """now() is the last time reached; reach(t) moves it to t, finite and never backward."""

    def __init__(self, t: float = 0.):
        self.t = -math.inf
        self.reach(t)

    def now(self) -> float:
        return self.t

    def reach(self, t: float) -> None:
        if not math.isfinite(t) or t < self.t:
            raise ValueError("stepped time must be finite and must not move backward")
        self.t = t


class Wall:
    """now() is time.monotonic(); reach(t) sleeps until t, and returns at once when late."""

    def now(self) -> float:
        return time.monotonic()

    def reach(self, t: float) -> None:
        time.sleep(max(0., t - time.monotonic()))
