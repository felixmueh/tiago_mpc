"""Joint friction, its Coulomb laws, and the plant's Euler and RK4 steps."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from typing_extensions import assert_never


@dataclass(frozen=True)
class Tanh:
    """Smooth Coulomb c*tanh(v/width_rad_s); the only law GATO predicts."""
    width_rad_s: float

    def __post_init__(self):
        if not (math.isfinite(self.width_rad_s) and self.width_rad_s > 0):
            raise ValueError("width_rad_s must be finite and positive")


@dataclass(frozen=True)
class Sign:
    """Coulomb c*sign(v), explicit in the old velocity; it may chatter."""


@dataclass(frozen=True)
class StickSlip:
    """Coulomb as the Euler step's coupled stick/slip solve (_stick_slip); no torque law."""


CoulombLaw = Tanh | Sign | StickSlip  # record.to_dict tells them apart by their type names


@dataclass(frozen=True)
class Friction:
    """Coulomb and viscous coefficients per native arm joint; each consumer selects its law."""
    coulomb_nm: tuple[float, ...]
    viscous_nm_s_per_rad: tuple[float, ...]

    def __post_init__(self):
        for key in ("coulomb_nm", "viscous_nm_s_per_rad"):
            values = tuple(float(v) for v in getattr(self, key))
            if len(values) != 7 or not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"{key} must contain seven finite nonnegative coefficients")
            object.__setattr__(self, key, values)


def passive_torque(velocity: np.ndarray, friction: Friction | None, law: Tanh | Sign) -> np.ndarray:
    """Positive resisting bias: subtract from actuator torque before ABA. Tanh and Sign only.

    Friction only. No gravity.
    """
    v = np.asarray(velocity, dtype=float)
    if friction is None:
        return np.zeros_like(v)
    match law:
        case Tanh(width_rad_s=width):
            coulomb = np.tanh(v / width)
        case Sign():
            coulomb = np.sign(v)
        case _:
            raise ValueError("passive_torque needs an explicit Tanh or Sign torque law")
    return np.asarray(friction.viscous_nm_s_per_rad)*v + np.asarray(friction.coulomb_nm)*coulomb


def plant_euler(
    model: Any,
    data: Any,
    q: np.ndarray,
    velocity: np.ndarray,
    actuator_torque: np.ndarray,
    dt: float,
    friction: Friction | None,
    friction_law: CoulombLaw,
) -> tuple[np.ndarray, np.ndarray]:
    """Native-coordinate Euler with implicit viscous damping; inertia and bias at (q, v).

    M(q) includes model.armature via Pinocchio CRBA; do not add it twice.
    A = M(q) + dt*diag(b), r = M(q)*v + dt*(torque - bias(q, v)).
    Tanh and Sign solve A*v_next = r - dt*c*tanh(v/width) or r - dt*c*sign(v): Coulomb is
    explicit in the old velocity, and sign can chatter.
    StickSlip minimises .5*v_next.T*A*v_next - r.T*v_next + dt*sum(c*abs(v_next)):
    at zero velocity friction lies in [-c, c]; slipping friction is c*sign(v_next).
    friction=None is loss-free. All advance position with v_next. This is not fully implicit
    rigid-body Euler. URDF friction/damping tags do not enter CRBA/RNEA; losses are applied here once.
    """
    import pinocchio as pin

    mass = pin.crba(model, data, q)
    mass = np.triu(mass) + np.triu(mass, 1).T
    bias = pin.rnea(model, data, q, velocity, np.zeros(model.nv))
    momentum = mass @ velocity + dt*(actuator_torque - bias)
    if friction is None:
        new = np.linalg.solve(mass, momentum)
    else:
        system = mass + dt*np.diag(friction.viscous_nm_s_per_rad)
        threshold = dt*np.asarray(friction.coulomb_nm)
        match friction_law:
            case Tanh(width_rad_s=width):
                new = np.linalg.solve(system, momentum - threshold*np.tanh(velocity/width))
            case Sign():
                new = np.linalg.solve(system, momentum - threshold*np.sign(velocity))
            case StickSlip():
                new = _stick_slip(system, momentum, threshold)
            case _:
                assert_never(friction_law)
    return pin.integrate(model, q, dt*new), new


def plant_rk4(
    model: Any,
    data: Any,
    q: np.ndarray,
    velocity: np.ndarray,
    actuator_torque: np.ndarray,
    dt: float,
    friction: Friction,
    friction_law: CoulombLaw,
) -> tuple[np.ndarray, np.ndarray]:
    """Native-coordinate RK4 with the passive torque recomputed at every stage (Tanh or Sign)."""
    import pinocchio as pin

    if isinstance(friction_law, StickSlip):
        raise ValueError("StickSlip Coulomb friction is Euler-only; use plant_euler")

    def acceleration(q_stage, v_stage):
        return pin.aba(model, data, q_stage, v_stage,
                       actuator_torque-passive_torque(v_stage, friction, friction_law)).copy()
    k1q = velocity
    k1v = acceleration(q, velocity)
    k2q = velocity + dt*k1v/2
    k2v = acceleration(pin.integrate(model, q, dt*k1q/2), k2q)
    k3q = velocity + dt*k2v/2
    k3v = acceleration(pin.integrate(model, q, dt*k2q/2), k3q)
    k4q = velocity + dt*k3v
    k4v = acceleration(pin.integrate(model, q, dt*k3q), k4q)
    return (pin.integrate(model, q, dt*(k1q+2*k2q+2*k3q+k4q)/6),
            velocity + dt*(k1v+2*k2v+2*k3v+k4v)/6)


def _stick_slip(system: np.ndarray, momentum: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Coupled convex solve: active sets, with coordinate descent if sets cycle.

    threshold is the Coulomb impulse dt*c. A slipping joint that reverses sticks;
    a stuck joint whose holding impulse exceeds its bound breaks away.
    """
    new = np.linalg.solve(system, momentum)
    direction = np.sign(new)
    slipping = (new != 0) | (threshold == 0)
    tolerance = 1e-12 * (1 + np.max(np.abs(momentum)))  # impulse residual
    for _ in range(4*len(new)):
        new = np.zeros_like(momentum)
        index = np.flatnonzero(slipping)
        if index.size:
            new[index] = np.linalg.solve(system[np.ix_(index, index)],
                                         momentum[index] - threshold[index]*direction[index])
        impulse = momentum - system @ new
        reversed_ = slipping & (threshold > 0) & (new*direction <= 0)
        breaking = ~slipping & (np.abs(impulse) > threshold + tolerance)
        if not (reversed_.any() or breaking.any()):
            return new
        slipping = (slipping & ~reversed_) | breaking
        direction = np.where(breaking, np.sign(impulse), direction)
    for _ in range(1000):
        for i in range(len(new)):
            push = momentum[i] - system[i] @ new + system[i, i]*new[i]
            new[i] = np.sign(push)*max(abs(push) - threshold[i], 0.)/system[i, i]
        impulse = momentum - system @ new
        error = np.where(new != 0, np.abs(impulse - threshold*np.sign(new)),
                         np.maximum(np.abs(impulse) - threshold, 0.))
        if np.max(error) <= tolerance:
            return new
    raise FloatingPointError("Stick/slip friction solve did not converge")
