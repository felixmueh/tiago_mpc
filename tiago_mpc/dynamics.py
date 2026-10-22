"""Joint friction laws (hard-sign plant, smooth GATO predictor) and the plant's RK4 step."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Friction:
    """Joint losses: Coulomb plus viscous, per native arm joint.

    The Coulomb term is tanh(v/width_rad_s), the smooth law GATO predicts, or
    sign(v) without a width (hard sign, simulated plant only).
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


def plant_rk4(model, data, q: np.ndarray, velocity: np.ndarray, actuator_torque: np.ndarray,
              dt: float, friction: Friction) -> tuple[np.ndarray, np.ndarray]:
    """Native-coordinate RK4 with passive torque recomputed at every stage."""
    import pinocchio as pin

    def acceleration(q_stage, v_stage):
        return pin.aba(model, data, q_stage, v_stage,
                       actuator_torque-passive_torque(v_stage, friction)).copy()
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
