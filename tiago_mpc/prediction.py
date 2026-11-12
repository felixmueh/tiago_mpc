"""Optional latency prediction under the current command and future control updates."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np
import pinocchio as pin

from tiago_mpc.dynamics import CoulombLaw, Friction, StickSlip, plant_euler
from tiago_mpc.interfaces import ControlLaw, EffortHandling, Fault, Plan, Prediction, SafetyInterface, State
from tiago_mpc.laws import ExecutionSafety, RejectOnEffort, decide_command
from tiago_mpc.tiago import ARMATURE_KG_M2, arm_urdf

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LatencyCompensation:
    """The independent physical model and delay forecast for latency compensation.

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


@dataclass(frozen=True)
class _Predictor:
    """Predict from explicit inputs with one reusable physical model and workspace."""

    # TODO: decide if this should be kept as simple as it is or combined with the simulator.
    config: LatencyCompensation
    safety: SafetyInterface = ExecutionSafety()
    effort_handling: EffortHandling | None = RejectOnEffort()
    model: pin.Model = field(init=False, repr=False, compare=False)
    data: pin.Data = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        model = pin.buildModelFromUrdf(str(arm_urdf()))
        model.armature = np.asarray(self.config.armature)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "data", model.createData())

    def __call__(
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
        """Propagate the held command, then decide future commands on predicted states.

        The current held command approximates the whole stale-measurement interval;
        no earlier publications are retained. Future ticks use the current plan.
        None means the observation or committed plan cannot support the forecast.
        Unknown startup effort is onboard position hold, matching SimPlant.command=None.
        During startup, execution safety is checked when the first plan arrives.
        No observation or plan timestamp is changed.
        """
        if state.t > now + _TOLERANCE or at < now - _TOLERANCE:
            return None
        if plan is not None:
            expires = plan.t0 + len(plan.torques)*plan.dt
            if at > expires + _TOLERANCE:
                return None
        q, qd, t = np.array(state.q, copy=True), np.array(state.qd, copy=True), state.t
        held = command
        tick = _tick_at_or_after(now + 1e-9, next_tick, command_period)
        while t < at - 1e-12:
            until = min(at, tick)
            while t < until - 1e-12:
                h = min(self.config.substep, until - t)
                if held is not None:
                    q, qd = plant_euler(self.model, self.data, q, qd, held, h,
                                       self.config.friction, self.config.friction_law)
                t += h
            t = until
            if t >= at - 1e-12:
                break
            if plan is not None:
                decision = decide_command(plan, t, State(t, q, qd), law,
                                          safety=self.safety, effort_handling=self.effort_handling)
                if isinstance(decision, Fault):
                    return None
                held = decision.applied
            tick += command_period
        return Prediction(at, q, qd, measured_t=state.t)
