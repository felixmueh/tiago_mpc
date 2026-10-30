"""Optional latency prediction: an independent physical model and sent-command history."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pinocchio as pin

from tiago_mpc.dynamics import CoulombLaw, Friction, StickSlip, plant_euler
from tiago_mpc.interfaces import ControlLaw, Plan, Prediction, State
from tiago_mpc.tiago import ARMATURE_KG_M2, arm_urdf

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LatencyCompensation:
    """Prediction settings, on by default in RealTime() (and VirtualTime.like(RealTime())).

    latency_s is a fixed forecast, or 'previous': the last completed request-to-result
    delay, bootstrapped by initial_latency_s. It never uses the current solve's duration.
    The physical propagation model is independent of the MPC and plant settings; by default
    loss-free with the fitted armature, as they are.
    Command publication is modeled as immediate; actuator/network delays are not known.
    Future feedback-dependent commands assume fresh predicted feedback at each tick;
    the current command always uses the actual measured feedback.
    """
    latency_s: float | Literal["previous"] = "previous"
    initial_latency_s: float = .005
    friction: Friction | None = None
    friction_law: CoulombLaw = StickSlip()
    armature: tuple[float, ...] = ARMATURE_KG_M2
    substep: float = .001

    def __post_init__(self) -> None:
        if not (self.latency_s == "previous" or isinstance(self.latency_s, (int, float))
                and math.isfinite(self.latency_s) and self.latency_s >= 0):
            raise ValueError("latency_s must be finite nonnegative seconds or 'previous'")
        if not math.isfinite(self.initial_latency_s) or self.initial_latency_s < 0:
            raise ValueError("initial_latency_s must be finite and nonnegative")
        if not math.isfinite(self.substep) or self.substep <= 0:
            raise ValueError("prediction substep must be finite and positive")
        armature = tuple(float(value) for value in self.armature)
        if len(armature) != 7 or not all(math.isfinite(value) and value >= 0 for value in armature):
            raise ValueError("armature must contain seven finite nonnegative values")
        object.__setattr__(self, "armature", armature)


def _tick_at_or_after(t: float, first: float, period: float) -> float:
    return first + max(0, math.ceil((t - first) / period - _TOLERANCE)) * period


class _Predictor:
    """One run's physical model and bounded successful-publication history."""

    def __init__(
        self,
        config: LatencyCompensation,
        lease_s: float,
        command_fault: Callable[[np.ndarray], str | None],
    ):
        self.config, self.lease_s = config, lease_s
        self.command_fault = command_fault
        self.model = pin.buildModelFromUrdf(str(arm_urdf()))
        self.model.armature = np.asarray(config.armature)
        self.data = self.model.createData()
        self.history: deque[tuple[float, np.ndarray | None]] = deque([(-math.inf, None)])
        self.previous_latency_s = config.initial_latency_s

    @property
    def latency_s(self) -> float:
        delay = self.config.latency_s
        return self.previous_latency_s if delay == "previous" else float(delay)

    def sent(self, t: float, command: np.ndarray) -> None:
        value = np.array(command, dtype=float)
        value.flags.writeable = False
        self.history.append((t, value))
        # Retain the command already held at the oldest usable observation time.
        while len(self.history) > 1 and self.history[1][0] <= t - self.lease_s:
            self.history.popleft()

    def arrived(self, delay_s: float) -> None:
        self.previous_latency_s = delay_s

    def predict(
        self,
        state: State,
        now: float,
        at: float,
        plan: Plan | None,
        law: ControlLaw,
        command: np.ndarray | None,
        next_tick: float,
        command_period: float,
    ) -> Prediction | None:
        """Replay publications to now, then forecast from the actually held command.

        None: the observation/history or committed plan cannot support the forecast.
        Unknown startup effort is onboard position hold, matching SimPlant.command=None.
        No observation or plan timestamp is changed.
        """
        if (state.t > now + _TOLERANCE or state.t < self.history[0][0]
                or at < now - _TOLERANCE or at >= state.t + self.lease_s):
            return None
        if plan is not None:
            if at > min(plan.measured_t + self.lease_s, plan.t0 + len(plan.torques) * plan.dt) + _TOLERANCE:
                return None
        q, qd, t = np.array(state.q, copy=True), np.array(state.qd, copy=True), state.t

        def advance(until: float, torque: np.ndarray | None) -> None:
            nonlocal q, qd, t
            while t < until - 1e-12:
                h = min(self.config.substep, until - t)
                if torque is not None:
                    q, qd = plant_euler(self.model, self.data, q, qd, torque, h,
                                       self.config.friction, self.config.friction_law)
                t += h
            t = until

        held = None
        for changed_at, torque in self.history:
            if changed_at <= state.t:
                held = torque
            elif changed_at <= now:
                advance(changed_at, held)
                held = torque
            else:
                break
        advance(now, held)
        held = command
        tick = _tick_at_or_after(now + 1e-9, next_tick, command_period)
        while t < at - 1e-12:
            advance(min(at, tick), held)
            if t >= at - 1e-12:
                break
            if plan is not None:
                if t >= min(plan.measured_t + self.lease_s, plan.t0 + len(plan.torques) * plan.dt):
                    return None
                held = np.asarray(law(plan, t, State(t, q, qd)), dtype=float)
                if self.command_fault(held) is not None:
                    return None
            tick += command_period
        return Prediction(at, q, qd, measured_t=state.t)
