"""Solve latency charged to the simulated plant, independently of wall pacing."""
from dataclasses import dataclass
import math


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
    """Hold the previous control while solving, then apply the new control.

    ``fixed`` charges ``fixed_s`` (zero gives instantaneous actuation).
    ``measured`` charges the wall duration of the complete controller step.
    Updates remain one nominal period apart unless the solve overruns it.
    Clip both intervals at the simulation's remaining duration. The caller
    retains the new control for the following solve, including after an overrun.
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
