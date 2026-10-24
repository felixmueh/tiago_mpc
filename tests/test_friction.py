"""Physical friction regressions, independent of MPC and CUDA."""
from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc.clock import VirtualClock
from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh, plant_euler
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import Q_START, urdf_friction


@pytest.mark.parametrize("scale", [.5, 1., 1.5])
def test_subthreshold_load_holds_exactly_with_stick_slip_but_moves_with_sign(scale):
    f = urdf_friction()
    f = replace(f, coulomb_nm=tuple(scale*np.array(f.coulomb_nm)),
                viscous_nm_s_per_rad=tuple(scale*np.array(f.viscous_nm_s_per_rad)))
    for law in (Sign(), StickSlip()):
        plant = SimPlant(friction=replace(f, law=law))
        assert plant.substep == .001
        gravity = pin.computeGeneralizedGravity(plant.model, plant.data, plant.q).copy()
        plant.start(VirtualClock())
        plant.apply(gravity + .5*np.array(f.coulomb_nm))
        plant.advance(.2)
        if law == StickSlip():
            np.testing.assert_array_equal(plant.q, Q_START)
            np.testing.assert_array_equal(plant.qd, np.zeros(7))
        else:
            assert np.max(np.abs(plant.q - Q_START)) > 1e-5


@pytest.mark.parametrize("scale", [.5, 1., 1.5])
def test_coupled_friction_torque_satisfies_stick_slip_conditions(scale):
    plant = SimPlant()
    f = urdf_friction()
    f = replace(f, coulomb_nm=tuple(scale*np.array(f.coulomb_nm)),
                viscous_nm_s_per_rad=tuple(scale*np.array(f.viscous_nm_s_per_rad)))
    rng = np.random.default_rng(17)
    for _ in range(40):
        q, v, h = Q_START + rng.uniform(-.1, .1, 7), rng.uniform(-.2, .2, 7), .001
        u = rng.uniform(-3., 3., 7) + pin.rnea(plant.model, plant.data, q, v, np.zeros(7))
        q_new, v_new = plant_euler(plant.model, plant.data, q, v, u, h, f)
        # RNEA with the step acceleration recovers the applied torque minus passive losses.
        resisting = u - pin.rnea(plant.model, plant.data, q, v, (v_new - v)/h)
        coulomb = resisting - np.array(f.viscous_nm_s_per_rad)*v_new
        moving = v_new != 0
        np.testing.assert_allclose(coulomb[moving], np.array(f.coulomb_nm)[moving]*np.sign(v_new[moving]),
                                   atol=2e-8, rtol=0)
        assert np.all(np.abs(coulomb[~moving]) <= np.array(f.coulomb_nm)[~moving] + 2e-8)
        np.testing.assert_allclose(q_new, pin.integrate(plant.model, q, h*v_new), atol=1e-14)


@pytest.mark.parametrize("friction", [None, Friction((0.,)*7, (0.,)*7, Tanh(.03)), Friction((0.,)*7, (0.,)*7, Sign()),
                                      Friction((0.,)*7, (0.,)*7, StickSlip())])
def test_zero_losses_agree_with_aba_despite_nonzero_urdf_tags(friction):
    plant = SimPlant()
    v, u, h = np.linspace(-.1, .1, 7), np.linspace(-1., 1., 7), .001
    a = pin.aba(plant.model, plant.data, Q_START, v, u).copy()
    _, v_new = plant_euler(plant.model, plant.data, Q_START, v, u, h, friction)
    np.testing.assert_allclose(v_new, v + h*a, atol=1e-12)


def test_tanh_coulomb_approaches_sign_as_the_width_shrinks():
    plant, f = SimPlant(), urdf_friction()
    v, u, h = np.array([-.05, -.01, -.002, 0., .002, .01, .05]), np.linspace(-1., 1., 7), .001

    def step(law):
        return plant_euler(plant.model, plant.data, Q_START, v, u, h, replace(f, law=law))[1]
    sign = step(Sign())
    np.testing.assert_allclose(step(Tanh(1e-9)), sign, atol=1e-12)
    smooth = step(Tanh(.03))
    assert np.max(np.abs(smooth - sign)) > 1e-5  # weaker Coulomb below the width


def test_breakaway_and_stopping_without_numerical_reversal():
    # Seven uncoupled unit-inertia revolute joints, zero gravity: analytic Coulomb solution.
    model = pin.Model()
    model.gravity.setZero()
    for i in range(7):
        j = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), f"joint{i}")
        model.appendBodyToJoint(j, pin.Inertia(1., np.zeros(3), np.eye(3)), pin.SE3.Identity())
    data, q, v = model.createData(), np.zeros(7), np.zeros(7)
    f = Friction((1.,)*7, (0.,)*7)
    q, v = plant_euler(model, data, q, v, np.full(7, 2.), .001, f)
    np.testing.assert_allclose(v, .001, atol=1e-14)  # (2 Nm - 1 Nm)/1 kg m²
    for _ in range(3):
        q, v = plant_euler(model, data, q, v, np.zeros(7), .001, f)
        np.testing.assert_allclose(v, 0., atol=1e-14)
    _, v = plant_euler(model, data, q, v, np.full(7, -2.), .001, f)
    np.testing.assert_allclose(v, -.001, atol=1e-14)
