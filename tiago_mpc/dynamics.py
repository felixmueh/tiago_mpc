"""Joint friction laws and the plant's semi-implicit Euler step."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

CoulombSolver = Literal["sign", "constraint"]


@dataclass(frozen=True)
class Friction:
    """Joint losses: Coulomb plus viscous, per native arm joint.

    With width_rad_s the Coulomb term is tanh(v/width_rad_s), the law GATO predicts.
    Without a width the plant uses these coefficients with sign or constraint Coulomb.
    """
    coulomb_nm: tuple[float, ...]
    viscous_nm_s_per_rad: tuple[float, ...]
    width_rad_s: float | None = None

    def __post_init__(self):
        for key in ("coulomb_nm", "viscous_nm_s_per_rad"):
            values = tuple(float(v) for v in getattr(self, key))
            if len(values) != 7 or not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"{key} must contain seven finite nonnegative coefficients")
            object.__setattr__(self, key, values)
        if self.width_rad_s is not None and not (math.isfinite(self.width_rad_s) and self.width_rad_s > 0):
            raise ValueError("width_rad_s must be finite and positive")


def passive_torque(velocity: np.ndarray, friction: Friction | None) -> np.ndarray:
    """Positive resisting bias: subtract from actuator torque before ABA."""
    v = np.asarray(velocity, dtype=float)
    if friction is None:
        return np.zeros_like(v)
    coulomb = np.sign(v) if friction.width_rad_s is None else np.tanh(v / friction.width_rad_s)
    return np.asarray(friction.viscous_nm_s_per_rad)*v + np.asarray(friction.coulomb_nm)*coulomb


def plant_euler(model: Any, data: Any, q: np.ndarray, velocity: np.ndarray, actuator_torque: np.ndarray,
                dt: float, friction: Friction, solver: CoulombSolver = "constraint") -> tuple[np.ndarray, np.ndarray]:
    """Native-coordinate Euler with implicit viscous damping; inertia and bias at (q, v).

    A = M(q) + dt*diag(b), r = M(q)*v + dt*(torque - bias(q, v)).
    'sign' solves A*v_next = r - dt*c*sign(v): Coulomb is explicit and can chatter.
    'constraint' minimises .5*v_next.T*A*v_next - r.T*v_next + dt*sum(c*abs(v_next)):
    at zero velocity friction lies in [-c, c]; slipping friction is c*sign(v_next).
    Both advance position with v_next. This is not fully implicit rigid-body Euler.
    URDF friction/damping tags do not enter CRBA/RNEA; losses are applied here once.
    """
    import pinocchio as pin

    if friction.width_rad_s is not None:
        raise ValueError("plant_euler uses hard Coulomb friction; smoothing is predictor-only")
    mass = pin.crba(model, data, q)
    mass = np.triu(mass) + np.triu(mass, 1).T
    bias = pin.rnea(model, data, q, velocity, np.zeros(model.nv))
    system = mass + dt*np.diag(friction.viscous_nm_s_per_rad)
    momentum = mass @ velocity + dt*(actuator_torque - bias)
    threshold = dt*np.asarray(friction.coulomb_nm)
    if solver == "sign":
        new = np.linalg.solve(system, momentum - threshold*np.sign(velocity))
    elif solver == "constraint":
        new = _stick_slip(system, momentum, threshold)
    else:
        raise ValueError(f"Unknown Coulomb solver: {solver}")
    return pin.integrate(model, q, dt*new), new


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
