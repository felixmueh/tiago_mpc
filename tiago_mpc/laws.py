"""Control laws: law(plan, t, state) -> command, once per command tick (plan.Law)."""
from __future__ import annotations

import numpy as np

from .plan import Plan, PlanEnded, State


def sample_knot(plan: Plan, t: float, state: State) -> np.ndarray:
    """The torque of the knot due at t; past the plan's last knot this raises PlanEnded."""
    k = plan.index(t - plan.t0)
    if k >= len(plan.torques):
        raise PlanEnded("time is past the plan's last knot")
    return plan.torques[k]


def hold_first(plan: Plan, t: float, state: State) -> np.ndarray:
    """The plan's first torque, whatever t is."""
    return plan.torques[0]
