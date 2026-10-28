"""Package control laws and their recorded configuration."""
import json

import numpy as np
import pytest

import tiago_mpc as tm
from tiago_mpc.laws import FrictionFeedforward, HoldFirst, PlanFeedback, SampleKnot
from tiago_mpc.interfaces import Plan

sample_knot, hold_first = SampleKnot(), HoldFirst()
PLAN = Plan(10., .002, np.zeros((6, 4)), 1. + np.arange(5)[:, None] * np.ones(2))


def test_sample_knot_holds_each_knot_from_the_state_time():
    assert sample_knot(PLAN, 10., None)[0] == 1.
    assert sample_knot(PLAN, 10.0019, None)[0] == 1.
    # 10 + 3 * .002 rounds below the boundary; it still selects knot 3.
    assert sample_knot(PLAN, 10. + 3 * .002, None)[0] == 4.
    np.testing.assert_array_equal(sample_knot(PLAN, 10.0099, None), [5., 5.])
    with pytest.raises(RuntimeError, match="last knot"):  # beyond the lease: a broken invariant
        sample_knot(PLAN, 10.01, None)
    with pytest.raises(ValueError):
        sample_knot(PLAN, 9.999, None)


def test_hold_first_ignores_the_time():
    for t in (10., 10.005, 11.):
        np.testing.assert_array_equal(hold_first(PLAN, t, None), [1., 1.])


@pytest.mark.parametrize("law", [
    PlanFeedback(),
    PlanFeedback(kp=3., kd=(1., 2., 3., 4., 5., 6., 7.)),
    FrictionFeedforward(tm.Friction((1.,)*7, (.5,)*7)),
    FrictionFeedforward(tm.Friction((1.,)*7, (.5,)*7), PlanFeedback(kp=3., kd=2.), tm.Sign()),
])
def test_public_package_laws_rebuild_from_recorded_type_names_without_extra_classes(law):
    assert tm.PlanFeedback is PlanFeedback and tm.FrictionFeedforward is FrictionFeedforward
    assert type(law).__module__ == "tiago_mpc.laws"
    payload = json.loads(json.dumps(tm.to_dict(law)))
    rebuilt = tm.from_dict(payload)
    assert rebuilt == law and tm.to_dict(rebuilt) == payload
    states = np.arange(42., dtype=float).reshape(3, 14)/100
    plan = tm.Plan(1., .01, states, np.ones((2, 7)))
    measured = tm.State(1., np.zeros(7), np.full(7, -.1))
    np.testing.assert_array_equal(rebuilt(plan, 1.014, measured), law(plan, 1.014, measured))


def test_feedforward_checks_horizon_even_when_its_base_law_holds_the_first_torque():
    plan = tm.Plan(1., .01, np.zeros((3, 14)), np.ones((2, 7)))
    law = tm.FrictionFeedforward(tm.Friction((1.,)*7, (0.,)*7), tm.HoldFirst())
    state = tm.State(1.03, np.zeros(7), np.zeros(7))
    with pytest.raises(RuntimeError, match="last knot"):
        law(plan, 1.03, state)


def test_plan_feedback_defaults_are_the_model_constants():
    assert PlanFeedback().kp == tm.PD_KP and PlanFeedback().kd == tm.PD_KD


@pytest.mark.parametrize("evaluation", [tm.Tanh(.03), tm.Sign()])
def test_feedforward_evaluates_interpolated_plan_velocity_independently_of_measured_velocity(evaluation):
    states = np.zeros((3, 14))
    states[:, 7:] = np.array([0., .02, .04])[:, None]
    plan = tm.Plan(1., .01, states, np.full((2, 7), 3.))
    friction = tm.Friction((1.,)*7, (.5,)*7)
    law = FrictionFeedforward(friction, tm.HoldFirst(), evaluation)
    t = 1.005
    velocity = np.full(7, (t - 1.)/.01*.02)
    expected = 3. + tm.passive_torque(velocity, friction, evaluation)
    for qd in (-10., 0., 10.):
        measured = tm.State(t, np.zeros(7), np.full(7, qd))
        np.testing.assert_allclose(law(plan, t, measured), expected, atol=1e-12)


def test_feedforward_rejects_a_plant_constraint_as_its_evaluation():
    with pytest.raises(ValueError, match="Tanh or Sign"):
        FrictionFeedforward(tm.Friction((1.,)*7, (0.,)*7), evaluation=tm.StickSlip())
