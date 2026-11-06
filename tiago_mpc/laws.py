"""Pure control laws return command arrays for measured or forecast states.

The frozen dataclasses are recorded as run configuration. SampleKnot and
HoldFirst select plan torques. PlanFeedback adds PD around the interpolated plan;
FrictionFeedforward adds explicit joint losses at its planned velocity. All use native
joint coordinates and can be composed independently of the experiment or plant."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiago_mpc.dynamics import Friction, Sign, Tanh, passive_torque
from tiago_mpc.interfaces import ControlLaw, EffortHandling, Plan, State
from tiago_mpc.tiago import PD_KD, PD_KP, limits


def _apply_effort(command: np.ndarray, handling: EffortHandling | None) -> tuple[np.ndarray | None, str | None]:
    """Validate a command, then apply the simulation's effort policy."""
    if command.shape != (7,) or not np.isfinite(command).all():
        return None, "invalid_command"
    if handling is None:
        return command, None
    applied = handling(command)
    if applied is None:
        return None, "effort_limit"
    applied = np.asarray(applied, dtype=float)
    if applied.shape != (7,) or not np.isfinite(applied).all():
        return None, "invalid_command"
    return applied, None


def _effort_vector(value: tuple[float, ...]) -> np.ndarray:
    effort = np.asarray(value, dtype=float)
    if effort.shape != (7,) or not np.isfinite(effort).all() or np.any(effort <= 0):
        raise ValueError("effort requires seven finite positive torque limits")
    return effort


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
class EffortClip:
    """Saturate a complete torque command at the configured effort limits."""
    effort: tuple[float, ...] = tuple(map(float, limits().effort))

    def __post_init__(self) -> None:
        object.__setattr__(self, "effort", tuple(map(float, _effort_vector(self.effort))))

    def __call__(self, command: np.ndarray) -> np.ndarray:
        effort = np.asarray(self.effort)
        return np.clip(command, -effort, effort)


@dataclass(frozen=True)
class RejectOnEffort:
    """Refuse a torque command beyond the configured effort limits."""
    effort: tuple[float, ...] = tuple(map(float, limits().effort))

    def __post_init__(self) -> None:
        object.__setattr__(self, "effort", tuple(map(float, _effort_vector(self.effort))))

    def __call__(self, command: np.ndarray) -> np.ndarray | None:
        return None if np.any(np.abs(command) > np.asarray(self.effort)) else command


@dataclass(frozen=True)
class HoldFirst:
    """The plan's first torque, whatever t is."""

    def __call__(self, plan: Plan, t: float, state: State) -> np.ndarray:
        return plan.torques[0]


@dataclass(frozen=True)
class PlanFeedback:
    """Plan torque plus kp*(q_plan-q) + kd*(qd_plan-qd) at each command tick.

    The plan follows GATO's constant-acceleration transcription between knots:
    a=(qd_next-qd)/dt, q_plan=q+qd*s+a*s²/2 and qd_plan=qd+a*s. Position
    meets the next knot only as far as the solve closed its defects. Feedback
    keeps its measurement time; this law does not extrapolate held observations.
    Gains are seven per-joint values or scalars, in native coordinates.
    """
    kp: tuple[float, ...] | float = PD_KP
    kd: tuple[float, ...] | float = PD_KD

    def __call__(self, plan: Plan, t: float, state: State) -> np.ndarray:
        k = plan.index(t - plan.t0)
        if k >= len(plan.torques):
            raise RuntimeError("time is past the plan's last knot, beyond its lease")
        s = min(max(t - plan.t0 - k * plan.dt, 0.), plan.dt)
        q, qd = plan.states[k, :7], plan.states[k, 7:]
        a = (plan.states[k + 1, 7:] - qd) / plan.dt
        q_plan, qd_plan = q + qd * s + .5 * a * s * s, qd + a * s
        return (plan.torques[k] + np.asarray(self.kp) * (q_plan - state.q)
                + np.asarray(self.kd) * (qd_plan - state.qd))


@dataclass(frozen=True)
class FrictionFeedforward:
    """Add explicit friction at interpolated planned velocity to base_law's command.

    Tanh and Sign are explicit torque laws; StickSlip is a plant integration
    constraint and cannot be added here. At zero planned velocity the added
    torque is zero, with no independent breakaway kick. Keep predictor friction
    off when using this law to compensate the same joint losses externally.
    """
    friction: Friction
    base_law: ControlLaw = SampleKnot()
    evaluation: Tanh | Sign = Tanh(.03)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, (Tanh, Sign)):
            raise ValueError("Feedforward needs an explicit Tanh or Sign torque law")

    def __call__(self, plan: Plan, t: float, state: State) -> np.ndarray:
        command = self.base_law(plan, t, state)
        k = plan.index(t - plan.t0)
        if k >= len(plan.torques):
            raise RuntimeError("time is past the plan's last knot, beyond its lease")
        s = min(max(t - plan.t0 - k * plan.dt, 0.), plan.dt)
        velocity = plan.states[k, 7:]
        planned_velocity = velocity + s * (plan.states[k + 1, 7:] - velocity) / plan.dt
        feedforward = passive_torque(planned_velocity, self.friction, self.evaluation)
        return command + feedforward
