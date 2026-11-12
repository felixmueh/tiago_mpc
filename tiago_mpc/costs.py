"""Convenience constructors for native GATO cost menus.

Use default_cost for the usual tracking terms and with_reference to update their
grasp-frame targets. Callers can also construct a gato.cost_menu.CostMenu directly
for custom objectives.
"""
from __future__ import annotations

from dataclasses import replace
import math

from gato.cost_menu import CostMenu, Term
import numpy as np


def default_cost(
    n: int = 32,
    *,
    position: float = 4.,
    terminal_position: float = 8.,
    velocity: float = .05,
    torque: float = 1e-4,
    orientation: float = 0.,
    terminal_orientation: float = 0.,
) -> CostMenu:
    """Build the TIAGo tracking menu for n knots, with zero position and identity rotation targets.

    Zero-weight terms are omitted; an all-zero objective keeps one zero input term
    because GATO requires a nonempty menu. A different MPC horizon needs a new menu.
    """
    if not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    weights = (position, terminal_position, velocity, torque, orientation, terminal_orientation)
    if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
        raise ValueError("tracking weights must be finite and nonnegative")
    menu = CostMenu()
    if position or terminal_position:
        menu.frame_pos = Term(weight=((float(position),) * 3,) * (n - 1)
                              + ((float(terminal_position),) * 3,), target=(0.,) * 3)
    if velocity:
        menu.state = Term(weight=(0.,) * 7 + (float(velocity),) * 7)
    if torque:
        menu.input = Term(weight=float(torque))
    if orientation or terminal_orientation:
        menu.orientation = Term(weight=((float(orientation),) * 3,) * (n - 1)
                                + ((float(terminal_orientation),) * 3,),
                                target=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    if not menu.terms():
        menu.input = Term(weight=0.)
    return menu


def with_reference(menu: CostMenu, reference: np.ndarray) -> CostMenu:
    """Copy a native menu with grasp-frame targets from an (N, 6) reference.

    Positions and rotation vectors are in ROOT_FRAME. Three NaN rotation entries
    disable orientation at that knot. Always retarget the original menu so its
    orientation weights survive later references with valid rotations. Other terms
    and activation settings are shared unchanged; caller data is never modified.
    """
    if not isinstance(menu, CostMenu):
        raise TypeError("menu must be a native gato.cost_menu.CostMenu")
    reference = np.asarray(reference, dtype=float)
    if (reference.ndim != 2 or reference.shape[1] != 6 or len(reference) < 2
            or not np.isfinite(reference[:, :3]).all()):
        raise ValueError("the reference must be (N, 6), N >= 2, with finite positions")
    result = replace(menu)
    if menu.frame_pos is not None:
        result.frame_pos = replace(menu.frame_pos, target=reference[:, :3].copy())
    if menu.orientation is None:
        return result
    result.orientation = replace(menu.orientation)

    rotation = reference[:, 3:]
    targeted = np.isfinite(rotation).all(axis=1)
    if not (targeted | np.isnan(rotation).all(axis=1)).all():
        raise ValueError("an orientation target is three finite values, or three NaN for none")
    n = len(reference)
    weight = np.asarray(result.orientation.weight, dtype=float)
    if weight.ndim <= 2:
        weight = np.broadcast_to(weight, (n, 3)).copy()
    elif weight.ndim == 3 and weight.shape[1:] == (n, 3):
        weight = weight.copy()
    else:
        raise ValueError("orientation weights must be scalar, (3,), (N, 3), or (B, N, 3)")
    weight[..., ~targeted, :] = 0.
    result.orientation.weight = weight
    active = np.any(weight != 0., axis=-1)
    if weight.ndim == 3:
        active = np.any(active, axis=0)
    if not active.any():
        result.orientation.target = np.eye(3)
        return result

    import pinocchio as pin

    rotations = rotation[active]
    if np.all(rotations == rotations[0]):
        result.orientation.target = pin.exp3(rotations[0])
    else:
        result.orientation.target = np.array([pin.exp3(r) if enabled else np.eye(3)
                                              for r, enabled in zip(rotation, active)])
    return result
