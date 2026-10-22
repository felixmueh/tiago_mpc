"""The plant's state and the MPC's plan, shared by the simulation loops and the ROS runner.

A plan holds GATO's knot states x_0..x_{N-1} and knot torques u_0..u_{N-2}
(native coordinates), its knot interval dt and its start time t0: the time of
the state it was computed from. Knot k covers t0 + k dt <= t < t0 + (k + 1) dt,
a zero-order hold as GATO's transcription assumes; past the last knot there is
no torque (PlanEnded).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# Knot boundaries computed as k * dt must select knot k despite rounding
# (a fraction of one knot interval).
_KNOT_TOLERANCE = 1e-9


class PlanEnded(RuntimeError):
    """The requested time lies past the plan's last knot."""


@dataclass(frozen=True)
class State:
    """t in seconds (robot: monotonic receipt time); arm q and qd in native coordinates.

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

    def controls_over(self, begin_s: float, span_s: float):
        """(torque, duration) pieces over offsets [begin_s, begin_s + span_s), split at knots.

        Raises PlanEnded if the span reaches past the plan's last knot.
        """
        if span_s <= 0:
            return []
        end = begin_s + span_s
        k, position, out = self.index(begin_s), begin_s, []
        while k < len(self.torques):
            boundary = (k + 1) * self.dt
            if boundary >= end - _KNOT_TOLERANCE * self.dt:
                return out + [(self.torques[k], end - position)]
            out.append((self.torques[k], boundary - position))
            position, k = boundary, k + 1
        raise PlanEnded("span reaches past the plan's last knot")
