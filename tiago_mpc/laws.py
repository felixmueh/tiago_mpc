"""Control laws: law(plan, t, state) -> command, once per command tick."""
from .plan import PlanEnded


def sample_knot(plan, t, state):
    """The torque of the knot due at t; past the plan's last knot this raises PlanEnded."""
    k = plan.index(t - plan.t0)
    if k >= len(plan.torques):
        raise PlanEnded("time is past the plan's last knot")
    return plan.torques[k]


def hold_first(plan, t, state):
    """The plan's first torque, whatever t is."""
    return plan.torques[0]
