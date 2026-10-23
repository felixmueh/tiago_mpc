"""The two latency examples on fakes (CPU only): PlanFeedback, and run_compensated's timing
and prediction (experiments/plan_feedback.py, experiments/latency_compensation.py)."""
from functools import partial

import numpy as np
import pytest

from experiments.latency_compensation import implicit_step, run_compensated
from experiments.plan_feedback import PlanFeedback, inertia_gains
from test_loops import Mpc, Plant, Task, commanded, go
from tiago_mpc.laws import sample_knot
from tiago_mpc.loops import Fixed, run_simulation
from tiago_mpc.mpc import TIAGO
from tiago_mpc.plan import Plan, State
from tiago_mpc.record import Record
from tiago_mpc.tiago import Q_START, arm_urdf, urdf_friction

pytestmark = pytest.mark.filterwarnings("ignore:a plan took effect")  # run_simulation's late plans


def step(x, u, h):
    """A double integrator, qdd = u, with GATO's constant-acceleration step (exact here)."""
    return np.r_[x[:7] + x[7:]*h + .5*u*h*h, x[7:] + u*h]


def test_plan_feedback_follows_the_plans_constant_acceleration_between_knots():
    x0, u = np.r_[np.zeros(7), np.ones(7)], np.full(7, 2.)
    x1 = step(x0, u, .01)
    plan = Plan(1., .01, np.array([x0, x1, step(x1, -u, .01)]), np.array([u, -u]))
    law = PlanFeedback(kp=10., kd=np.arange(7.))
    on = step(x0, u, .004)  # on the plan 4 ms into knot 0: the knot torque alone
    np.testing.assert_allclose(law(plan, 1.004, State(1.004, on[:7], on[7:])), u)
    off = State(1.014, x1[:7] + .1, np.zeros(7))  # knot 1, 4 ms in
    expected = step(x1, -u, .004)
    np.testing.assert_allclose(law(plan, 1.014, off), -u + 10.*(expected[:7] - off.q) + np.arange(7.)*expected[7:])
    kp, kd = inertia_gains(30.)
    assert len(kp) == 7 and all(p > 0 for p in kp) and kd[0] == pytest.approx(2 * kp[0] / 30.)


@pytest.mark.parametrize("latency", [0., .02, .025])
def test_uncompensated_it_times_requests_and_plans_as_run_simulation(latency):
    """The same request times and commands as run_simulation(latency=Fixed(latency))."""
    stock, mine = Mpc(), Mpc()
    _, a, _ = go(stock, latency=Fixed(latency), duration=.12)
    _, b, _ = go(mine, loop=partial(run_compensated, latency_s=latency, compensate=False), duration=.12)
    assert stock.requests == mine.requests and commanded(a) == commanded(b)
    assert [(k.t, k.new) for k in a] == [(k.t, k.new) for k in b]


class Moving(Plant):
    """A double integrator with the commanded acceleration; held until the first command."""
    def __init__(self):
        super().__init__()
        self.x, self.u = np.zeros(14), None

    def state(self):
        return State(self.t, self.x[:7].copy(), self.x[7:].copy())

    def apply(self, command):
        super().apply(command)
        self.u = command

    def advance(self, t):
        if self.u is not None:
            self.x = step(self.x, self.u, t - self.t)
        self.t = t


class Predicting(Mpc):
    """test_loops' Mpc with the solver's sim_forward (the plant's model) and a sign map."""
    signs = np.array([1., 1., 1., 1., 1., -1., 1.])

    class controller:
        class solver:
            @staticmethod
            def sim_forward(x, u, h):
                return step(x[0], u[0], h)[None].astype(np.float32)


@pytest.mark.parametrize("substep", [None, .004])  # one step per tick, or three substeps
def test_compensated_plans_start_at_the_predicted_state_and_run_their_knot_0(tmp_path, substep):
    plant, mpc, ticks = Moving(), Predicting(), []
    record = Record(tmp_path / "r", task=Task(), params=TIAGO, plant=plant)
    run_compensated(plant, mpc, Task(), sample_knot, callbacks=[record, ticks.append], command_period=.01,
                    update_period=.01, duration=.1, latency_s=.02, prediction_substep_s=substep,
                    predictor="gpu")  # the fake sim_forward is the plant's model
    assert record.summary["loop"]["prediction_substeps"] == (1 if substep is None else 3)
    # Requested at 0, 20, ..., 80 ms, each from the state predicted 20 ms on, dated then.
    assert [round(t, 6) for t, _ in mpc.requests] == [.02, .04, .06, .08, .1]
    assert commanded(ticks)[:4] == [(.02, 1.), (.03, 2.), (.04, 1.), (.05, 2.)]  # knot 0 runs
    assert record.summary["loop"]["compensate"] and record.summary["loop"]["effective_latency_s"] == .02
    np.testing.assert_allclose(record.arrays["effect_time_s"], [.02, .04, .06, .08, .1])
    np.testing.assert_allclose(record.arrays["requested_state"], record.arrays["effect_state"], atol=1e-6)
    assert record.arrays["effect_state"][-1, 7] > 0  # the arm moved: the prediction was not trivial


def test_compensation_refuses_a_wall_clock():
    with pytest.raises(ValueError, match="simulation clock"):
        go(Mpc(), loop=partial(run_compensated, latency_s=.01), plant=Plant(kind="wall"))


def test_the_implicit_step_is_aba_without_friction_and_stable_with_it():
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(arm_urdf()))
    data, u, h = model.createData(), np.full(7, .5), .002
    x = np.r_[Q_START, np.full(7, .1)]
    a = pin.aba(model, data, Q_START, x[7:], u)
    np.testing.assert_allclose(implicit_step(None)(x, u, h), np.r_[Q_START + h*(x[7:] + h*a), x[7:] + h*a], atol=1e-12)
    # URDF friction: joint 7's time constant is about 36 us. At rest under the holding torque its
    # speed chatters at about +-h c / (m + h b) = 0.15 rad/s from step to step, but does not grow.
    hold, step = pin.rnea(model, data, Q_START, np.zeros(7), np.zeros(7)), implicit_step(urdf_friction(.03))
    x, speeds = np.r_[Q_START, np.zeros(6), .1], []
    for _ in range(30):
        x = step(x, hold, h)
        speeds.append(x[13])
    assert max(np.abs(speeds)) < .16 and abs(x[6] - Q_START[6]) < 1e-5
