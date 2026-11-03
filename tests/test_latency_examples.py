"""The two latency examples on fakes (CPU only): PlanFeedback, and run_compensated's timing
and prediction (experiments/plan_feedback.py, experiments/latency_compensation.py)."""
from functools import partial
import math

import numpy as np
import pytest

from experiments.latency_compensation import (Compensated, first_tick_of_each_plan, implicit_step,
                                              requested_states, run_compensated)
from experiments.latency_study import row
from test_loops import Mpc, Plant, Task, commanded, go
from tiago_mpc.dynamics import Tanh
from tiago_mpc.laws import PlanFeedback, SampleKnot
from tiago_mpc.util import inertia_gains
from tiago_mpc.interfaces import Plan, State
from tiago_mpc.record import RunRecord, from_dict, load_run, to_dict
from tiago_mpc.tiago import Q_START, arm_urdf, urdf_friction


@pytest.mark.parametrize("compensated", [False, True])
def test_study_retains_an_error_before_the_first_recorded_tick(tmp_path, compensated):
    record = RunRecord(tmp_path / "r", task=Task())
    record.metadata.update(status="error", error="Traceback (most recent call last):\nRuntimeError: first plan rejected")
    record.save()
    result = row(record.out, "reach-friction", "compensated" if compensated else "stock", 20., compensated)
    assert result["status"] == "error" and result["end_s"] is None
    assert result["fault"] == "RuntimeError: first plan rejected"


def step(x, u, h):
    """A double integrator, qdd = u, with GATO's constant-acceleration step (exact here)."""
    return np.r_[x[:7] + x[7:]*h + .5*u*h*h, x[7:] + u*h]


def test_plan_feedback_follows_the_plans_constant_acceleration_between_knots():
    x0, u = np.r_[np.zeros(7), np.ones(7)], np.full(7, 2.)
    x1 = step(x0, u, .01)
    plan = Plan(1., .01, np.array([x0, x1, step(x1, -u, .01)]), np.array([u, -u]), measured_t=1.)
    law = PlanFeedback(kp=10., kd=np.arange(7.))
    on = step(x0, u, .004)  # on the plan 4 ms into knot 0: the knot torque alone
    np.testing.assert_allclose(law(plan, 1.004, State(1.004, on[:7], on[7:])), u)
    off = State(1.014, x1[:7] + .1, np.zeros(7))  # knot 1, 4 ms in
    expected = step(x1, -u, .004)
    np.testing.assert_allclose(law(plan, 1.014, off), -u + 10.*(expected[:7] - off.q) + np.arange(7.)*expected[7:])
    kp, kd = inertia_gains(30.)
    assert len(kp) == 7 and all(p > 0 for p in kp) and kd[0] == pytest.approx(2 * kp[0] / 30.)


@pytest.mark.parametrize("latency", [0., .02, .025])
def test_uncompensated_it_times_requests_and_plans_as_run_virtual_time(latency):
    """The same request times and commands as run_virtual_time(solve_latency_s=latency)."""
    stock, mine = Mpc(), Mpc()
    _, a, _ = go(stock, solve_latency_s=latency, duration=.12)
    _, b, _ = go(mine, loop=partial(run_compensated, solve_latency_s=latency, compensate=False), duration=.12)
    assert stock.requests == mine.requests and commanded(a) == commanded(b)
    assert [(k.t, k.new, k.update) for k in a] == [(k.t, k.new, k.update) for k in b]
    assert [[e.kind for e in k.events] for k in a] == [[e.kind for e in k.events] for k in b]  # late_plan


class Moving(Plant):
    """A double integrator with the commanded acceleration; held until the first command."""
    def __init__(self):
        super().__init__()
        self.x, self.u = np.zeros(14), None

    def state(self):
        return State(self.clock.now(), self.x[:7].copy(), self.x[7:].copy())

    def apply(self, command):
        super().apply(command)
        self.u = command

    def advance(self, t):
        if self.u is not None:
            self.x = step(self.x, self.u, t - self.clock.now())
        super().advance(t)


class Predicting(Mpc):
    """test_loops' Mpc with the solver's sim_forward (the plant's model)."""

    class controller:
        class solver:
            @staticmethod
            def sim_forward(x, u, h):
                return step(x[0], u[0], h)[None].astype(np.float32)


@pytest.mark.parametrize("substep", [None, .004])  # one step per tick, or three substeps
def test_compensated_plans_start_at_the_predicted_state_and_run_their_knot_0(tmp_path, substep):
    plant, mpc, ticks = Moving(), Predicting(), []
    record = RunRecord(tmp_path / "r", task=Task())
    run_compensated(plant, mpc, Task(), SampleKnot(), tick_callbacks=[record, ticks.append],
                    command_period=.01, update_period=.01, duration=.1, solve_latency_s=.02,
                    prediction_substep_s=substep, predictor="gpu")  # the fake sim_forward is the plant's model
    # Requested at 0, 20, ..., 80 ms, each from the state predicted 20 ms on, dated then.
    assert [round(t, 6) for t, _ in mpc.requests] == [.02, .04, .06, .08, .1]
    assert commanded(ticks)[:4] == [(.02, 1.), (.03, 2.), (.04, 1.), (.05, 2.)]  # knot 0 runs
    assert all(e.kind == 'prediction_request' for k in ticks for e in k.events)  # no plan is late
    np.testing.assert_allclose(record.plans["update_t"], [0., .02, .04, .06, .08])
    np.testing.assert_allclose(record.plans["t0"], [.02, .04, .06, .08, .1])
    effect = first_tick_of_each_plan(record)
    np.testing.assert_allclose(record.ticks["t"][effect], [.02, .04, .06, .08, .1])
    actual = np.c_[record.ticks["q"][effect], record.ticks["qd"][effect]]
    np.testing.assert_allclose(requested_states(record), actual, atol=1e-6)
    assert 'requested_state' not in record.plans
    assert all(set(k.plan.stats) == {'wall_s'} for k in ticks if k.new)
    record.save()
    loaded = load_run(record.out)
    np.testing.assert_array_equal(requested_states(loaded), requested_states(record))
    assert actual[-1, 7] > 0  # the arm moved: the prediction was not trivial


def test_compensation_runs_on_virtual_time_and_returns_an_outcome():
    plant, _, outcome = go(Mpc(), loop=partial(run_compensated, solve_latency_s=.01, compensate=False), duration=.03)
    assert type(plant.clock).__name__ == "VirtualClock" and outcome.status == "timeout"
    timing = Compensated(solve_latency_s=.01, command_period=.01, compensate=False)
    plant, ticks = Plant(), []
    outcome = timing.run(plant, Mpc(), Task(), SampleKnot(), [ticks.append], duration=.03)  # as a timing
    assert outcome.status == "timeout" and commanded(ticks) == [(.01, 2.), (.02, 2.)]
    assert from_dict(to_dict(timing), [Compensated]) == timing


def test_faulted_plan_does_not_enter_active_prediction_error_or_replace_active_tick_plan(tmp_path):
    record, ticks = RunRecord(tmp_path/'faulted', task=Task()), []
    outcome = run_compensated(Moving(), Predicting(over=4), Task(), SampleKnot(), [record, ticks.append],
                              command_period=.01, update_period=.01, duration=.1, solve_latency_s=.02,
                              predictor='gpu')
    assert outcome.reason == 'effort_limit'
    assert ticks[-1].plan is ticks[-2].plan and not ticks[-1].new
    record.metadata.update(status=outcome.status, reason=outcome.reason, end_t=outcome.t)
    record.save()
    assert len(first_tick_of_each_plan(record)) == 2 and len(record.plans['t0']) == 3
    result = row(record.out, 'reach', 'compensated', 20., True)
    assert result['fault'] == 'effort_limit' and result['takeover_q_mrad'] == 0.
    assert result['takeover_qd_rad_s'] == 0.


@pytest.mark.parametrize("latency", ["measured", -.01, math.nan, math.inf])
def test_compensation_needs_a_declared_latency_before_it_starts_the_plant(latency):
    plant = Plant()
    with pytest.raises(ValueError, match="solve_latency_s"):
        go(Mpc(), plant=plant, loop=partial(run_compensated, solve_latency_s=latency))
    assert not plant.calls


def test_the_implicit_step_is_aba_without_friction_and_stable_with_it():
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(arm_urdf()))
    data, u, h = model.createData(), np.full(7, .5), .002
    x = np.r_[Q_START, np.full(7, .1)]
    a = pin.aba(model, data, Q_START, x[7:], u)
    np.testing.assert_allclose(implicit_step(None)(x, u, h), np.r_[Q_START + h*(x[7:] + h*a), x[7:] + h*a], atol=1e-12)
    # URDF friction: joint 7's time constant is tens of microseconds. At rest under the holding
    # torque its speed chatters at about +-h c / (m + h b) from step to step, but does not grow.
    hold, step = pin.rnea(model, data, Q_START, np.zeros(7), np.zeros(7)), implicit_step(urdf_friction(), Tanh(.03))
    x, speeds = np.r_[Q_START, np.zeros(6), .1], []
    for _ in range(30):
        x = step(x, hold, h)
        speeds.append(x[13])
    c, b = urdf_friction().coulomb_nm[6], urdf_friction().viscous_nm_s_per_rad[6]
    chatter = h * c / (pin.crba(model, data, Q_START)[6, 6] + h * b)
    # It does not creep: after an even number of steps, joint 7 is back within 5 % of one step's travel.
    assert max(np.abs(speeds)) < 1.1 * chatter and abs(x[6] - Q_START[6]) < .05 * h * chatter
