"""Pure control laws decide commands for measured or forecast states.

The frozen dataclasses are recorded as run configuration. SampleKnot and
HoldFirst select plan torques. PlanFeedback adds PD around the interpolated plan;
FrictionFeedforward adds explicit joint losses at its planned velocity. All use native
joint coordinates and can be composed independently of the experiment or plant."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tiago_mpc.dynamics import Friction, Sign, Tanh, passive_torque
from tiago_mpc.interfaces import Command, ControlLaw, EffortHandling, Fault, Plan, SafetyInterface, State
from tiago_mpc.tiago import PD_KD, PD_KP, limits


LEASE_S = .1  # A plan may be commanded at most this long after its measured source state.


def _validate_lease(plan: Plan, t: float, max_plan_age_s: float | None) -> Fault | None:
    expires = plan.t0 + len(plan.torques) * plan.dt
    if max_plan_age_s is not None:
        expires = min(plan.measured_t + max_plan_age_s, expires)
    return Fault("lease_expired") if t >= expires else None


def validate_plan(plan: Plan) -> Fault | None:
    """Refuse a plan with any knot torque beyond the robot's raw effort limits.

    Every knot is checked on arrival because it may be commanded before the lease ends.
    """
    return Fault("effort_limit") if np.any(np.abs(plan.torques) > limits().effort) else None


@dataclass(frozen=True)
class ExecutionSafety:
    """Check a plan's finite horizon and age from its original measured observation.

    None disables the age limit. decide_command also enforces the finite horizon
    independently of the supplied safety policy.
    """
    max_plan_age_s: float | None = LEASE_S  # Seconds since the measured source state.

    def __post_init__(self) -> None:
        if (self.max_plan_age_s is not None and
                (not math.isfinite(self.max_plan_age_s) or self.max_plan_age_s <= 0)):
            raise ValueError("max_plan_age_s must be finite positive seconds or None")

    def validate(self, plan: Plan, t: float, state: State) -> Fault | None:
        return _validate_lease(plan, t, self.max_plan_age_s)


def _validate_command(command: Command, effort: np.ndarray | None = None) -> Fault | None:
    for value in (command.requested, command.applied):
        if value.shape != (7,) or not np.isfinite(value).all():
            return Fault("invalid_command", command.requested)
        if effort is not None and np.any(np.abs(value) > effort):
            return Fault("effort_limit", command.requested)
    return None


def _effort_vector(value: tuple[float, ...]) -> np.ndarray:
    effort = np.asarray(value, dtype=float)
    if effort.shape != (7,) or not np.isfinite(effort).all() or np.any(effort <= 0):
        raise ValueError("effort requires seven finite positive torque limits")
    return effort


@dataclass(frozen=True)
class SampleKnot:
    """The torque of the knot due at t. The loops never ask past the plan's lease, which
    ends before its last knot does; past it, a broken invariant, this raises."""

    def __call__(self, plan: Plan, t: float, state: State) -> Command:
        k = plan.index(t - plan.t0)
        if k >= len(plan.torques):
            raise RuntimeError("time is past the plan's last knot, beyond its lease")
        return Command(plan.torques[k], plan.torques[k])


@dataclass(frozen=True)
class EffortClip:
    """Saturate a complete torque command at the configured effort limits."""
    effort: tuple[float, ...] = tuple(map(float, limits().effort))

    def __post_init__(self) -> None:
        object.__setattr__(self, "effort", tuple(map(float, _effort_vector(self.effort))))

    def __call__(self, command: Command) -> Command:
        effort = np.asarray(self.effort)
        return Command(command.requested, np.clip(command.applied, -effort, effort))


@dataclass(frozen=True)
class RejectOnEffort:
    """Refuse a torque command beyond the configured effort limits."""
    effort: tuple[float, ...] = tuple(map(float, limits().effort))

    def __post_init__(self) -> None:
        object.__setattr__(self, "effort", tuple(map(float, _effort_vector(self.effort))))

    def __call__(self, command: Command) -> Command | Fault:
        if np.any(np.abs(command.applied) > np.asarray(self.effort)):
            return Fault("effort_limit", command.requested)
        return command


@dataclass(frozen=True)
class RealTimeSafety:
    """Check the mandatory 100 ms lease and finite horizon before user safety."""
    safety: SafetyInterface = ExecutionSafety()

    def validate(self, plan: Plan, t: float, state: State) -> Fault | None:
        fault = ExecutionSafety().validate(plan, t, state)
        return fault if fault is not None else self.safety.validate(plan, t, state)


@dataclass(frozen=True)
class RealTimeEffort:
    """Check raw requested and applied effort before and after the user's policy."""
    effort_handling: EffortHandling | None = RejectOnEffort()

    def __post_init__(self) -> None:
        if isinstance(self.effort_handling, EffortClip):
            raise ValueError("Real-time execution cannot clip effort")

    def __call__(self, command: Command) -> Command | Fault:
        fault = _validate_command(command, limits().effort)
        if fault is not None:
            return fault
        checked = command if self.effort_handling is None else self.effort_handling(command)
        if isinstance(checked, Fault):
            return checked
        fault = _validate_command(checked, limits().effort)
        return fault if fault is not None else checked


@dataclass(frozen=True)
class HoldFirst:
    """The plan's first torque, whatever t is."""

    def __call__(self, plan: Plan, t: float, state: State) -> Command:
        return Command(plan.torques[0], plan.torques[0])


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

    def __call__(self, plan: Plan, t: float, state: State) -> Command:
        k = plan.index(t - plan.t0)
        if k >= len(plan.torques):
            raise RuntimeError("time is past the plan's last knot, beyond its lease")
        s = min(max(t - plan.t0 - k * plan.dt, 0.), plan.dt)
        q, qd = plan.states[k, :7], plan.states[k, 7:]
        a = (plan.states[k + 1, 7:] - qd) / plan.dt
        q_plan, qd_plan = q + qd * s + .5 * a * s * s, qd + a * s
        torque = (plan.torques[k] + np.asarray(self.kp) * (q_plan - state.q)
                  + np.asarray(self.kd) * (qd_plan - state.qd))
        return Command(torque, torque)


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

    def __call__(self, plan: Plan, t: float, state: State) -> Command | Fault:
        command = self.base_law(plan, t, state)
        if isinstance(command, Fault):
            return command
        k = plan.index(t - plan.t0)
        if k >= len(plan.torques):
            raise RuntimeError("time is past the plan's last knot, beyond its lease")
        s = min(max(t - plan.t0 - k * plan.dt, 0.), plan.dt)
        velocity = plan.states[k, 7:]
        planned_velocity = velocity + s * (plan.states[k + 1, 7:] - velocity) / plan.dt
        feedforward = passive_torque(planned_velocity, self.friction, self.evaluation)
        return Command(command.requested + feedforward, command.applied + feedforward)


def decide_command(
    plan: Plan,
    t: float,
    state: State,
    law: ControlLaw,
    *,
    safety: SafetyInterface = ExecutionSafety(),
    effort_handling: EffortHandling | None = EffortClip(),
) -> Command | Fault:
    """Check the plan's lease, evaluate its law, then handle the requested effort.

    A refused lease never evaluates the law. Laws and effort policies must be pure.
    """
    fault = _validate_lease(plan, t, None)
    if fault is not None:
        return fault
    fault = safety.validate(plan, t, state)
    if fault is not None:
        return fault
    command = law(plan, t, state)
    if isinstance(command, Fault):
        return command
    fault = _validate_command(command)
    if fault is not None:
        return fault
    if effort_handling is not None:
        command = effort_handling(command)
        if isinstance(command, Fault):
            return command
        fault = _validate_command(command)
        if fault is not None:
            return fault
    return command
