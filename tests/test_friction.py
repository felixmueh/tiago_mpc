"""Physical friction regressions, independent of MPC and CUDA."""
from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc.clock import VirtualClock
from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh, passive_torque, plant_euler, plant_rk4
from tiago_mpc.sim import SimConfig
from tiago_mpc.tiago import Q_START, urdf_friction


@pytest.mark.parametrize("scale", [.5, 1., 1.5])
def test_subthreshold_load_holds_exactly_with_stick_slip_but_moves_with_sign(scale):
    f = urdf_friction()
    f = replace(f, coulomb_nm=tuple(scale*np.array(f.coulomb_nm)),
                viscous_nm_s_per_rad=tuple(scale*np.array(f.viscous_nm_s_per_rad)))
    for law in (Sign(), StickSlip()):
        plant = SimConfig(friction=f, friction_law=law, integrator="euler").build()
        assert plant.config.substep == .001
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
    plant = SimConfig().build()
    f = urdf_friction()
    f = replace(f, coulomb_nm=tuple(scale*np.array(f.coulomb_nm)),
                viscous_nm_s_per_rad=tuple(scale*np.array(f.viscous_nm_s_per_rad)))
    rng = np.random.default_rng(17)
    for _ in range(40):
        q, v, h = Q_START + rng.uniform(-.1, .1, 7), rng.uniform(-.2, .2, 7), .001
        u = rng.uniform(-3., 3., 7) + pin.rnea(plant.model, plant.data, q, v, np.zeros(7))
        q_new, v_new = plant_euler(plant.model, plant.data, q, v, u, h, f, StickSlip())
        # RNEA with the step acceleration recovers the applied torque minus passive losses.
        resisting = u - pin.rnea(plant.model, plant.data, q, v, (v_new - v)/h)
        coulomb = resisting - np.array(f.viscous_nm_s_per_rad)*v_new
        moving = v_new != 0
        np.testing.assert_allclose(coulomb[moving], np.array(f.coulomb_nm)[moving]*np.sign(v_new[moving]),
                                   atol=2e-8, rtol=0)
        assert np.all(np.abs(coulomb[~moving]) <= np.array(f.coulomb_nm)[~moving] + 2e-8)
        np.testing.assert_allclose(q_new, pin.integrate(plant.model, q, h*v_new), atol=1e-14)


@pytest.mark.parametrize("friction", [None, Friction((0.,)*7, (0.,)*7)])
@pytest.mark.parametrize("law", [Tanh(.03), Sign(), StickSlip()])
def test_zero_losses_agree_with_aba_despite_nonzero_urdf_tags(friction, law):
    plant = SimConfig().build()
    v, u, h = np.linspace(-.1, .1, 7), np.linspace(-1., 1., 7), .001
    a = pin.aba(plant.model, plant.data, Q_START, v, u).copy()
    _, v_new = plant_euler(plant.model, plant.data, Q_START, v, u, h, friction, law)
    np.testing.assert_allclose(v_new, v + h*a, atol=1e-12)


def test_tanh_coulomb_approaches_sign_as_the_width_shrinks():
    plant, f = SimConfig().build(), urdf_friction()
    v, u, h = np.array([-.05, -.01, -.002, 0., .002, .01, .05]), np.linspace(-1., 1., 7), .001

    def step(law):
        return plant_euler(plant.model, plant.data, Q_START, v, u, h, f, law)[1]
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
    q, v = plant_euler(model, data, q, v, np.full(7, 2.), .001, f, StickSlip())
    np.testing.assert_allclose(v, .001, atol=1e-14)  # (2 Nm - 1 Nm)/1 kg m²
    for _ in range(3):
        q, v = plant_euler(model, data, q, v, np.zeros(7), .001, f, StickSlip())
        np.testing.assert_allclose(v, 0., atol=1e-14)
    _, v = plant_euler(model, data, q, v, np.full(7, -2.), .001, f, StickSlip())
    np.testing.assert_allclose(v, -.001, atol=1e-14)


def test_armature_enters_mass_bias_and_coupled_friction_once():
    values = np.array([.746, .882, .211, .648, .326, .343, .27])
    bare = SimConfig().build()
    plant = SimConfig(armature=tuple(values)).build()
    q, v = Q_START + np.linspace(-.05, .05, 7), np.linspace(-.1, .1, 7)
    mass = pin.crba(bare.model, bare.data, q).copy()
    augmented = pin.crba(plant.model, plant.data, q).copy()
    np.testing.assert_allclose(augmented, mass + np.diag(values), atol=1e-12)
    bias = pin.rnea(bare.model, bare.data, q, v, np.zeros(7)).copy()
    np.testing.assert_allclose(pin.rnea(plant.model, plant.data, q, v, np.zeros(7)), bias, atol=1e-12)
    u = bias + np.linspace(-2., 2., 7)
    h = .001
    _, new = plant_euler(plant.model, plant.data, q, v, u, h, None, StickSlip())
    np.testing.assert_allclose(new, v + h*np.linalg.solve(augmented, u-bias), atol=1e-12)
    friction = urdf_friction()
    _, new = plant_euler(plant.model, plant.data, q, v, u, h, friction, StickSlip())
    resistance = u - pin.rnea(plant.model, plant.data, q, v, (new-v)/h)
    coulomb = resistance - np.array(friction.viscous_nm_s_per_rad)*new
    moving = new != 0
    np.testing.assert_allclose(coulomb[moving], np.array(friction.coulomb_nm)[moving]*np.sign(new[moving]),
                               atol=1e-7, rtol=0)
    assert np.all(np.abs(coulomb[~moving]) <= np.array(friction.coulomb_nm)[~moving]+1e-7)


def test_shared_coefficients_have_separate_explicit_torque_evaluations():
    friction = Friction((2.,)*7, (.5,)*7)
    v = np.array([-.05, -.01, -.002, 0., .002, .01, .05])
    np.testing.assert_array_equal(passive_torque(v, friction, Sign()), 2*np.sign(v) + .5*v)
    np.testing.assert_array_equal(passive_torque(v, friction, Tanh(.03)), 2*np.tanh(v/.03) + .5*v)
    np.testing.assert_array_equal(passive_torque(v, None, Tanh(.03)), np.zeros(7))
    with pytest.raises(ValueError, match="Tanh or Sign"):
        passive_torque(v, friction, StickSlip())


def test_rk4_rejects_stick_slip_before_evaluating_any_stage():
    with pytest.raises(ValueError, match="Euler-only"):
        plant_rk4(None, None, Q_START, np.zeros(7), np.zeros(7), .001,
                  Friction((1.,)*7, (0.,)*7), StickSlip())
