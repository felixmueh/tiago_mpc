"""Stored model values and optional, explicitly requested gain estimation."""
import ast

import numpy as np
import pinocchio as pin
import pytest

import tiago_mpc as tm
from tiago_mpc import tiago, util


def test_gain_estimate_uses_the_requested_pose_and_armature_once():
    q = tm.Q_START + np.linspace(-.08, .06, 7)
    armature = np.linspace(.1, .7, 7)
    model = pin.buildModelFromUrdf(str(tm.arm_urdf()))
    inertia = np.diag(pin.crba(model, model.createData(), q)) + armature
    kp, kd = util.inertia_gains(17., 1.3, q, armature)
    np.testing.assert_allclose(kp, inertia * 17.**2)
    np.testing.assert_allclose(kd, inertia * 2 * 1.3 * 17.)
    assert all(type(value) is float for vector in (kp, kd) for value in vector)
    assert ast.literal_eval(repr((kp, kd))) == (kp, kd)  # directly pasteable Python literals


def test_gain_estimate_defaults_to_start_pose_and_link_only_inertia_without_changing_stored_gains():
    stored = tiago.PD_KP, tiago.PD_KD
    model = pin.buildModelFromUrdf(str(tm.arm_urdf()))
    inertia = np.diag(pin.crba(model, model.createData(), tm.Q_START))
    kp, kd = util.inertia_gains(20., 1.4)
    np.testing.assert_allclose(kp, inertia * 20.**2)
    np.testing.assert_allclose(kd, inertia * 2 * 1.4 * 20.)
    assert (tiago.PD_KP, tiago.PD_KD) == stored
    assert tm.inertia_gains is util.inertia_gains
    assert tm.ARMATURE_KG_M2 is tiago.ARMATURE_KG_M2
    assert tm.PD_KP is tiago.PD_KP and tm.PD_KD is tiago.PD_KD


@pytest.mark.parametrize("kwargs, message", [
    ({"bandwidth_rad_s": 0.}, "bandwidth"),
    ({"bandwidth_rad_s": float("nan")}, "bandwidth"),
    ({"damping_ratio": 0.}, "damping"),
    ({"damping_ratio": float("inf")}, "damping"),
    ({"armature": (0.,) * 6}, "armature"),
    ({"armature": (-1.,) * 7}, "armature"),
    ({"armature": (float("nan"),) * 7}, "armature"),
])
def test_gain_estimate_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        util.inertia_gains(**{"bandwidth_rad_s": 20., **kwargs})
