"""Solve latency charged to the simulated plant, independently of wall pacing.

The execution controller samples each plan by knot from its state's time
(tiago_mpc/plan.py), as the ROS runner does; the control update period may
differ from the knot interval. A charged solve longer than the whole plan would
need a torque past its last knot; that raises PlanEnded and ends the run.
TODO(torque command, Felix 2026-10-22): the execution controller still sends
the plan's zero-order-hold torque; smoothing or local feedback, a PAL real-time
controller and separate simulation/robot control modes remain deferred (see the
note above publish_checked in tiago_mpc/runner.py).
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ControlIntervals:
    charged_s: float
    previous_s: float
    new_s: float

    @property
    def elapsed_s(self) -> float:
        return self.previous_s + self.new_s


def control_intervals(update_period_s: float, solve_wall_s: float, *,
                      mode: str = "fixed", fixed_s: float = 0.0,
                      remaining_s: float = math.inf) -> ControlIntervals:
    """Continue the previous plan while solving, then apply the new plan.

    ``fixed`` charges ``fixed_s`` (zero gives instantaneous actuation).
    ``measured`` charges the wall duration of the complete controller step.
    Control updates remain one nominal period apart unless the solve overruns it.
    Clip both intervals at the simulation's remaining duration. The caller
    retains the new plan for the following solve, including after an overrun.
    """
    if not math.isfinite(update_period_s) or update_period_s <= 0:
        raise ValueError("update_period_s must be finite and positive")
    for name, value in (("solve_wall_s", solve_wall_s), ("fixed_s", fixed_s)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if math.isnan(remaining_s) or remaining_s < 0:
        raise ValueError("remaining_s must be nonnegative")
    if mode not in ("fixed", "measured"):
        raise ValueError("mode must be 'fixed' or 'measured'")

    charged = fixed_s if mode == "fixed" else solve_wall_s
    previous = min(charged, remaining_s)
    elapsed = min(max(update_period_s, charged), remaining_s)
    return ControlIntervals(charged, previous, elapsed - previous)


def control_segments(previous, plan, intervals):
    """Plant inputs over one control update, as (torque, duration) pieces.

    While the solve is charged, the previous plan continues from its own state
    time; before the first plan the plant receives zero torque. The new plan
    then runs from offset ``intervals.previous_s``.
    """
    if previous is None:
        head = ([(np.zeros(plan.torques.shape[1]), intervals.previous_s)]
                if intervals.previous_s > 0 else [])
    else:
        head = previous.controls_over(plan.start_s - previous.start_s, intervals.previous_s)
    return head + plan.controls_over(intervals.previous_s, intervals.new_s)
