"""Package control laws and their recorded configuration."""
import json

import numpy as np
import pytest

import tiago_mpc as tm
from tiago_mpc.laws import (FrictionFeedforward, HoldFirst, PlanFeedback, RealTimeEffort, RealTimeSafety, SampleKnot,
                            decide_command)
from tiago_mpc.interfaces import Command, Fault, Plan, State
from tiago_mpc.dynamics import Friction, StickSlip, Tanh
from tiago_mpc.util import inertia_gains

sample_knot, hold_first = SampleKnot(), HoldFirst()
PLAN = Plan(10., .002, np.zeros((6, 4)), 1. + np.arange(5)[:, None] * np.ones(2), measured_t=10.)


def test_sample_knot_holds_each_knot_from_the_state_time():
    assert sample_knot(PLAN, 10., None).applied[0] == 1.
    assert sample_knot(PLAN, 10.0019, None).applied[0] == 1.
    assert sample_knot(PLAN, 10. - 1e-15, None).applied[0] == 1.  # summed ticks may reach t0 a rounding error early
    # 10 + 3 * .002 rounds below the boundary; it still selects knot 3.
    assert sample_knot(PLAN, 10. + 3 * .002, None).applied[0] == 4.
    np.testing.assert_array_equal(sample_knot(PLAN, 10.0099, None).applied, [5., 5.])
    with pytest.raises(RuntimeError, match="last knot"):  # beyond the lease: a broken invariant
        sample_knot(PLAN, 10.01, None)
    with pytest.raises(ValueError):
        sample_knot(PLAN, 9.999, None)


def test_hold_first_ignores_the_time():
    for t in (10., 10.005, 11.):
        np.testing.assert_array_equal(hold_first(PLAN, t, None).applied, [1., 1.])


@pytest.mark.parametrize("source", ["law", "handler"])
@pytest.mark.parametrize("field", ["requested", "applied"])
@pytest.mark.parametrize("bad, reason", [
    (np.r_[np.ones(6), 26.1], "effort_limit"),
    (np.r_[np.ones(6), np.nan], "invalid_command"),
    (np.ones(6), "invalid_command"),
])
def test_real_time_decision_checks_raw_commands_before_and_after_custom_handling(
    source,
    field,
    bad,
    reason,
):
    plan = Plan(0., .01, np.zeros((3, 14)), np.ones((2, 7)), measured_t=0.)
    state = tm.State(0., np.zeros(7), np.zeros(7))
    fields = dict(requested=np.ones(7), applied=np.ones(7))
    fields[field] = bad
    unsafe = Command(**fields)
    calls = []

    class Safety:
        def validate(self, plan, t, state):
            return None

    def handler(command):
        calls.append(command)
        return unsafe if source == "handler" else Command(command.requested, np.zeros(7))

    safety = RealTimeSafety(Safety())
    command = unsafe if source == "law" else Command(np.ones(7), np.ones(7))
    decision = decide_command(plan, 0., state, lambda *args: command, safety=safety,
                              effort_handling=RealTimeEffort(handler))
    assert isinstance(decision, Fault) and decision.reason == reason
    np.testing.assert_array_equal(decision.requested, unsafe.requested)
    assert len(calls) == (source == "handler")


def test_command_decision_refuses_expired_horizon_or_measurement_lease_before_evaluating_the_law():
    def unreachable(plan, t, state):
        pytest.fail("an expired plan must not evaluate the command law")

    state = tm.State(.1, np.zeros(7), np.zeros(7))
    short = Plan(0., .01, np.zeros((3, 14)), np.ones((2, 7)), measured_t=0.)
    long = Plan(.08, .03, np.zeros((32, 14)), np.ones((31, 7)), measured_t=0.)
    for plan, t, age in ((short, .02, None), (long, .1, .1)):
        decision = decide_command(plan, t, state, unreachable, safety=tm.ExecutionSafety(age),
                                  effort_handling=tm.EffortClip())
        assert isinstance(decision, Fault)
        assert decision.reason == "lease_expired" and decision.requested is None


def test_command_decision_handles_the_complete_feedback_command_and_preserves_its_request():
    states = np.zeros((3, 14))
    states[:, :7] = 1.
    plan = Plan(0., .01, states, np.ones((2, 7)), measured_t=0.)
    state = tm.State(0., np.zeros(7), np.zeros(7))
    law = PlanFeedback(kp=3., kd=0.)
    clipped = decide_command(plan, 0., state, law, safety=tm.ExecutionSafety(),
                             effort_handling=tm.EffortClip(effort=(2.,)*7))
    assert isinstance(clipped, Command) and clipped.clipped
    np.testing.assert_array_equal(clipped.requested, np.full(7, 4.))
    np.testing.assert_array_equal(clipped.applied, np.full(7, 2.))
    refused = decide_command(plan, 0., state, law, safety=tm.ExecutionSafety(),
                             effort_handling=tm.RejectOnEffort(effort=(2.,)*7))
    assert isinstance(refused, Fault) and refused.reason == "effort_limit"
    np.testing.assert_array_equal(refused.requested, clipped.requested)
    accepted = decide_command(plan, 0., state, law, safety=tm.ExecutionSafety(), effort_handling=None)
    assert isinstance(accepted, Command) and not accepted.clipped
    np.testing.assert_array_equal(accepted.applied, clipped.requested)
    np.testing.assert_array_equal(plan.torques, np.ones((2, 7)))


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
    plan = tm.Plan(1., .01, states, np.ones((2, 7)), measured_t=1.)
    measured = tm.State(1., np.zeros(7), np.full(7, -.1))
    np.testing.assert_array_equal(rebuilt(plan, 1.014, measured).applied, law(plan, 1.014, measured).applied)


def test_feedforward_checks_horizon_even_when_its_base_law_holds_the_first_torque():
    plan = tm.Plan(1., .01, np.zeros((3, 14)), np.ones((2, 7)), measured_t=1.)
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
    plan = tm.Plan(1., .01, states, np.full((2, 7), 3.), measured_t=1.)
    friction = tm.Friction((1.,)*7, (.5,)*7)
    law = FrictionFeedforward(friction, tm.HoldFirst(), evaluation)
    t = 1.005
    velocity = np.full(7, (t - 1.)/.01*.02)
    expected = 3. + tm.passive_torque(velocity, friction, evaluation)
    for qd in (-10., 0., 10.):
        measured = tm.State(t, np.zeros(7), np.full(7, qd))
        np.testing.assert_allclose(law(plan, t, measured).applied, expected, atol=1e-12)


def test_feedforward_rejects_a_plant_constraint_as_its_evaluation():
    with pytest.raises(ValueError, match="Tanh or Sign"):
        FrictionFeedforward(tm.Friction((1.,)*7, (0.,)*7), evaluation=tm.StickSlip())


def test_custom_safety_receives_the_actual_decision_inputs_before_law_and_effort():
    state = tm.State(.004, np.zeros(7), np.zeros(7))
    plan = Plan(0., .01, np.zeros((3, 14)), np.ones((2, 7)), measured_t=0.)
    refusal = Fault("fixture_clearance")
    seen = []

    class Safety:
        def validate(self, value, t, measured):
            seen.append((value, t, measured))
            return refusal

    def unreachable(*args):
        pytest.fail("safety refusal must precede law and effort handling")

    assert decide_command(plan, .004, state, unreachable, safety=Safety(),
                          effort_handling=unreachable) is refusal
    assert seen == [(plan, .004, state)]


def test_law_fault_passes_through_without_effort_handling():
    plan = Plan(0., .01, np.zeros((3, 14)), np.ones((2, 7)), measured_t=0.)
    state = tm.State(0., np.zeros(7), np.zeros(7))
    refusal = Fault("law_refused", np.full(7, 2.))

    def unreachable(command):
        pytest.fail("a refused law must not reach effort handling")

    assert decide_command(plan, 0., state, lambda *args: refusal,
                          effort_handling=unreachable) is refusal
    wrapped = FrictionFeedforward(tm.Friction((1.,)*7, (.5,)*7), lambda *args: refusal)
    assert wrapped(plan, 0., state) is refusal


def test_custom_effort_policy_receives_and_preserves_the_law_request():
    plan = Plan(0., .01, np.zeros((3, 14)), np.ones((2, 7)), measured_t=0.)
    state = tm.State(0., np.zeros(7), np.zeros(7))
    command = Command(np.full(7, 4.), np.full(7, 3.))
    seen = []

    def effort(value):
        seen.append(value)
        return Command(value.requested, value.applied * .5)

    decision = decide_command(plan, 0., state, lambda *args: command, effort_handling=effort)
    assert seen == [command]
    np.testing.assert_array_equal(decision.requested, np.full(7, 4.))
    np.testing.assert_array_equal(decision.applied, np.full(7, 1.5))


def test_custom_safety_cannot_extend_the_finite_plan_horizon():
    class Safety:
        def validate(self, plan, t, state):
            pytest.fail("the unconditional horizon guard must run first")

    plan = Plan(0., .01, np.zeros((3, 14)), np.ones((2, 7)), measured_t=0.)
    state = tm.State(.02, np.zeros(7), np.zeros(7))
    decision = decide_command(plan, .02, state, lambda *args: pytest.fail("expired law"),
                              safety=Safety(), effort_handling=None)
    assert isinstance(decision, Fault) and decision.reason == "lease_expired"


def test_pd_initialization_accounts_for_reflected_armature_once():
    armature = np.array([.7, .8, .2, .6, .3, .35, .27])
    old_kp, old_kd = inertia_gains(20., 1.4)
    kp, kd = inertia_gains(20., 1.4, armature=armature)
    np.testing.assert_allclose(np.subtract(kp, old_kp), armature*20.**2)
    np.testing.assert_allclose(np.subtract(kd, old_kd), armature*2*1.4*20.)
    with pytest.raises(ValueError, match='armature'):
        inertia_gains(20., armature=(-1.,)*7)


def test_friction_feedforward_uses_planned_velocity_and_native_signs():
    v0 = np.array([.1, -.1, .2, -.2, .3, -.3, 0.])
    v1 = 2*v0
    states = np.array([np.r_[np.zeros(7), v] for v in (v0, v1)])
    torque = np.arange(7.)[None]
    plan = Plan(1., .01, states, torque, measured_t=1.)
    measured = State(1.005, np.zeros(7), -v0)
    friction = Friction((1.,)*7, (.5,)*7)
    expected = torque[0] + np.tanh(1.5*v0/.03) + .5*1.5*v0
    np.testing.assert_allclose(FrictionFeedforward(friction, SampleKnot(), Tanh(.03))(plan, 1.005, measured).applied,
                               expected)
    pd = PlanFeedback(kp=0., kd=2.)
    np.testing.assert_allclose(FrictionFeedforward(friction, pd, Tanh(.03))(plan, 1.005, measured).applied,
                               expected + 2*(1.5*v0 - measured.qd))
    with pytest.raises(RuntimeError, match='last knot'):
        FrictionFeedforward(friction, SampleKnot())(plan, 1.02, measured)


def test_smooth_feedforward_has_no_breakaway_kick_at_rest():
    plan = Plan(0., .01, np.zeros((2, 14)), np.ones((1, 7)), measured_t=0.)
    law = FrictionFeedforward(Friction((3.,)*7, (2.,)*7))
    np.testing.assert_array_equal(law(plan, .004, State(.004, np.zeros(7), np.zeros(7))).applied, np.ones(7))
    with pytest.raises(ValueError, match='explicit'):
        FrictionFeedforward(Friction((1.,)*7, (0.,)*7), evaluation=StickSlip())


def step(x, u, h):
    """A double integrator, qdd = u, with GATO's constant-acceleration step (exact here)."""
    return np.r_[x[:7] + x[7:]*h + .5*u*h*h, x[7:] + u*h]


def test_plan_feedback_follows_the_plans_constant_acceleration_between_knots():
    x0, u = np.r_[np.zeros(7), np.ones(7)], np.full(7, 2.)
    x1 = step(x0, u, .01)
    plan = Plan(1., .01, np.array([x0, x1, step(x1, -u, .01)]), np.array([u, -u]), measured_t=1.)
    law = PlanFeedback(kp=10., kd=np.arange(7.))
    on = step(x0, u, .004)  # on the plan 4 ms into knot 0: the knot torque alone
    np.testing.assert_allclose(law(plan, 1.004, State(1.004, on[:7], on[7:])).applied, u)
    off = State(1.014, x1[:7] + .1, np.zeros(7))  # knot 1, 4 ms in
    expected = step(x1, -u, .004)
    np.testing.assert_allclose(law(plan, 1.014, off).applied, -u + 10.*(expected[:7] - off.q) + np.arange(7.)*expected[7:])
    kp, kd = inertia_gains(30.)
    assert len(kp) == 7 and all(p > 0 for p in kp) and kd[0] == pytest.approx(2 * kp[0] / 30.)
