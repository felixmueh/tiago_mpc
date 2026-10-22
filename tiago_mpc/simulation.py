"""Solve latency charged to the simulated plant, independently of wall pacing.

The plant plays back each plan by knot from its state's time (tiago_mpc/playback.py),
as the ROS runner does; the control period may differ from the knot interval.
Past a plan's last knot, which only charged solve time can reach, the plant
holds the final knot and records that it did. TODO(torque command): see the note
above publish_checked in tiago_mpc/runner.py for what remains deferred.
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


def control_intervals(period_s: float, solve_wall_s: float, *,
                      mode: str = "fixed", fixed_s: float = 0.0,
                      remaining_s: float = math.inf) -> ControlIntervals:
    """Continue the previous plan while solving, then apply the new plan.

    ``fixed`` charges ``fixed_s`` (zero gives instantaneous actuation).
    ``measured`` charges the wall duration of the complete controller step.
    Updates remain one nominal period apart unless the solve overruns it.
    Clip both intervals at the simulation's remaining duration. The caller
    retains the new plan for the following solve, including after an overrun.
    """
    if not math.isfinite(period_s) or period_s <= 0:
        raise ValueError("period_s must be finite and positive")
    for name, value in (("solve_wall_s", solve_wall_s), ("fixed_s", fixed_s)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if math.isnan(remaining_s) or remaining_s < 0:
        raise ValueError("remaining_s must be nonnegative")
    if mode not in ("fixed", "measured"):
        raise ValueError("mode must be 'fixed' or 'measured'")

    charged = fixed_s if mode == "fixed" else solve_wall_s
    previous = min(charged, remaining_s)
    elapsed = min(max(period_s, charged), remaining_s)
    return ControlIntervals(charged, previous, elapsed - previous)


def control_segments(previous, plan, intervals):
    """Plant inputs over one update, as (plan, knot, torque, duration, past_end).

    While the solve is charged, the previous plan continues from its own state
    time; before the first plan the plant receives zero torque (plan None,
    knot -1). The new plan then plays from offset ``intervals.previous_s``.
    """
    if previous is None:
        head = ([(None, -1, np.zeros(plan.torques.shape[1]), intervals.previous_s, False)]
                if intervals.previous_s > 0 else [])
    else:
        head = [(previous, *segment) for segment in
                previous.segments(plan.start_s - previous.start_s, intervals.previous_s)]
    return head + [(plan, *segment) for segment in
                   plan.segments(intervals.previous_s, intervals.new_s)]
