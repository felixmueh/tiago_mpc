"""Control laws: frozen dataclasses called law(plan, t, state) -> command, once per command
tick (plan.Law), so a run records them as its configuration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiago_mpc.plan import Plan, State


@dataclass(frozen=True)
class SampleKnot:
    """The torque of the knot due at t. The loops never ask past the plan's lease, which
    ends before its last knot does; past it, a broken invariant, this raises."""

    def __call__(self, plan: Plan, t: float, state: State) -> np.ndarray:
        k = plan.index(t - plan.t0)
        if k >= len(plan.torques):
            raise RuntimeError("time is past the plan's last knot, beyond its lease")
        return plan.torques[k]


@dataclass(frozen=True)
class HoldFirst:
    """The plan's first torque, whatever t is."""

    def __call__(self, plan: Plan, t: float, state: State) -> np.ndarray:
        return plan.torques[0]
