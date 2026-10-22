"""The plan the execution controller samples, shared by the simulation loops and the ROS runner.

A plan holds GATO's knot torques u_0..u_{K-1} (K = N - 1, native coordinates),
its knot interval dt and its start time t0: the time of the state it was
computed from. ``control_at`` gives u_k for t0 + k dt <= t < t0 + (k + 1) dt,
a zero-order hold per knot as GATO's transcription assumes. Past the last
knot there is no control: ``control_at`` and ``controls_over`` raise PlanEnded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Knot boundaries computed as k * dt must select knot k despite rounding
# (a fraction of one knot interval).
_KNOT_TOLERANCE = 1e-9


class PlanEnded(RuntimeError):
    """The requested time lies past the plan's last knot."""


def knot_torques(xu, signs):
    """Native torques u_0..u_{N-2} from GATO's flat [x_0 u_0 x_1 ... x_{N-1}] plan."""
    signs = np.asarray(signs, dtype=float)
    nu = signs.size
    stride = 3 * nu  # fixed-base arm: nx = 2 nu
    xu = np.asarray(xu, dtype=float).reshape(-1)
    if (xu.size + nu) % stride or xu.size < stride + 2 * nu:
        raise ValueError("Malformed GATO trajectory")
    knots = (xu.size + nu) // stride - 1
    return xu[:knots * stride].reshape(knots, stride)[:, 2 * nu:] * signs


@dataclass(frozen=True)
class Plan:
    start_s: float
    dt_s: float
    torques: np.ndarray

    def __post_init__(self):
        torques = np.array(self.torques, dtype=float)
        if torques.ndim != 2 or len(torques) < 1 or not np.isfinite(torques).all():
            raise ValueError("A plan needs at least one finite knot torque")
        if not math.isfinite(self.start_s) or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("A plan needs a finite start time and a positive knot interval")
        torques.flags.writeable = False
        object.__setattr__(self, "torques", torques)

    @property
    def knots(self) -> int:
        return len(self.torques)

    @property
    def end_s(self) -> float:
        return self.start_s + self.knots * self.dt_s

    def index(self, offset_s: float) -> int:
        """Zero-order-hold knot of a time offset from the start; may exceed the plan."""
        if not offset_s >= 0 or math.isinf(offset_s):
            raise ValueError("time precedes the plan's state or is not finite")
        return math.floor(offset_s / self.dt_s + _KNOT_TOLERANCE)

    def control_at(self, t: float) -> tuple[int, np.ndarray]:
        """Knot and torque due at absolute time t (ROS: monotonic seconds)."""
        k = self.index(t - self.start_s)
        if k >= self.knots:
            raise PlanEnded("time is past the plan's last knot")
        return k, self.torques[k]

    def controls_over(self, begin_s: float, span_s: float):
        """(torque, duration) pieces over offsets [begin_s, begin_s + span_s), split at knots.

        Raises PlanEnded if the span reaches past the plan's last knot.
        """
        if span_s <= 0:
            return []
        end = begin_s + span_s
        k, position, out = self.index(begin_s), begin_s, []
        while k < self.knots:
            boundary = (k + 1) * self.dt_s
            if boundary >= end - _KNOT_TOLERANCE * self.dt_s:
                return out + [(self.torques[k], end - position)]
            out.append((self.torques[k], boundary - position))
            position, k = boundary, k + 1
        raise PlanEnded("span reaches past the plan's last knot")
