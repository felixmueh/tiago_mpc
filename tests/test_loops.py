"""The loop on a fake plant and a fake Mpc (CPU only)."""
from dataclasses import dataclass
import math

import numpy as np
import pytest

from tiago_mpc.laws import hold_first
from tiago_mpc.loops import Fixed, Measured, Schedule, guard, run, run_robot, run_simulation
from tiago_mpc.plan import Plan, State

EFFORT = np.full(7, 50.)


class Plant:
    def __init__(self):
        self.t, self.calls, self.applied = 0., [], []

    def start(self):
        self.calls.append("start")

    def state(self):
        return State(self.t, np.zeros(7), np.zeros(7))

    def apply(self, command):
        self.applied.append((self.t, command))

    def advance(self, t):
        self.t = t

    def stop(self):
        self.calls.append("stop")


class Mpc:
    """Plans whose knot k has torque k + 1. delay: result() calls before a reply;
    replies: how many requests get one (later ones hang)."""
    details = dict(effort_limits=EFFORT.tolist())

    def __init__(self, N=11, dt=.01, delay=0, replies=math.inf, wall=0.):
        self.N, self.dt, self.delay, self.replies, self.wall = N, dt, delay, replies, wall
        self.requests, self.waiting, self.dead = [], None, False

    def request(self, state, reference, updates=None):
        assert self.waiting is None, "a request is pending"
        assert reference.shape == (self.N, 6)
        self.requests.append((state.t, updates))
        self.waiting = [self.delay, state.t]

    def result(self, block=False):
        if self.dead:
            raise RuntimeError("MPC worker exited unexpectedly")
        if self.waiting is None or len(self.requests) > self.replies:
            return None
        if self.waiting[0] > 0 and not block:
            self.waiting[0] -= 1
            return None
        t, self.waiting = self.waiting[1], None
        torques = np.arange(1., self.N)[:, None] * np.ones(7)
        return Plan(t, self.dt, np.zeros((self.N, 14)), torques, dict(wall_s=self.wall))


@dataclass
class Task:
    done_at: float = math.inf

    def reference(self, t, n, dt):
        return np.zeros((n, 6))

    def done(self, state, t):
        return t >= self.done_at - 1e-9


def go(mpc, task=Task(), loop=run_simulation, duration=.1, update_period=.01, **kw):
    plant, ticks = Plant(), []
    result = loop(plant, mpc, task, callbacks=[ticks.append], command_period=.01,
                  update_period=update_period, duration=duration, **kw)
    return plant, ticks, result


def commanded(ticks):
    """(t, knot torque) of each tick with a command."""
    return [(round(k.state.t, 6), k.command[0]) for k in ticks if k.command is not None]


def test_zero_latency_commands_each_plan_at_its_own_tick():
    plant, ticks, done = go(Mpc(), duration=.03)
    assert not done and plant.calls == ["start", "stop"]
    assert commanded(ticks) == [(0., 1.), (.01, 1.), (.02, 1.)]
    assert [k.new for k in ticks] == [True, True, True, False]
    assert ticks[-1].command is None and ticks[-1].state.t == pytest.approx(.03)


@pytest.mark.parametrize("latency", [Fixed(.025), Measured()])
def test_a_plan_takes_effect_at_the_first_tick_after_its_charge_and_blocks_requests(latency):
    mpc = Mpc(wall=.025)
    plant, ticks, _ = go(mpc, latency=latency, duration=.08)
    # Requested at 0, effective at 30 ms (knot 3); no request while it is pending.
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .03, .06]
    assert commanded(ticks)[:4] == [(.03, 4.), (.04, 5.), (.05, 6.), (.06, 4.)]
    assert all(k.command is None for k in ticks[:3])


def test_measured_latency_charges_the_plans_own_wall_time():
    mpc = Mpc(wall=.004)
    _, ticks, _ = go(mpc, latency=Measured(), duration=.03)
    assert commanded(ticks) == [(.01, 2.), (.02, 2.)]  # 4 ms charge: from the next tick


def test_no_command_before_the_first_plan_on_the_wall_clock():
    mpc = Mpc(delay=2)
    _, ticks, _ = go(mpc, loop=run_robot, duration=.06)
    assert [k.command is None for k in ticks[:3]] == [True, True, False]
    assert commanded(ticks)[0] == (.02, 3.)
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .02, .04]


def test_done_is_checked_before_any_command():
    mpc = Mpc()
    plant, ticks, done = go(mpc, Task(done_at=0.))
    assert done and not mpc.requests and not plant.applied and len(ticks) == 1
    plant, ticks, done = go(Mpc(), Task(done_at=.02))
    assert done and [t for t, _ in plant.applied] == [0., .01] and ticks[-1].command is None


def test_the_lease_ends_100_ms_after_the_plans_state():
    mpc = Mpc(N=32, replies=1)  # 310 ms plan; later solves hang
    with pytest.raises(RuntimeError, match="lease expired"):
        go(mpc, loop=run_robot, duration=1.)
    assert mpc.requests == [(0., None), (.01, None)]


def test_past_the_plans_end_faults_and_stops():
    plant, ticks = Plant(), []
    with pytest.raises(RuntimeError, match="lease expired"):  # 30 ms plan: lease ends with it
        run_robot(plant, Mpc(N=4, replies=1), Task(), hold_first, callbacks=[ticks.append],
                  command_period=.01, update_period=.01, duration=1.)
    assert [round(t, 6) for t, _ in plant.applied] == [0., .01, .02]
    assert plant.calls == ["start", "stop"]


def test_a_dead_worker_is_noticed_within_one_tick_and_stops_the_plant():
    mpc, plant = Mpc(), Plant()

    def die(tick):
        mpc.dead = tick.state.t >= .02

    with pytest.raises(RuntimeError, match="exited"):
        run_robot(plant, mpc, Task(), callbacks=[die], command_period=.01, update_period=.05, duration=1.)
    assert plant.calls == ["start", "stop"] and plant.t == pytest.approx(.03)


def test_task_updates_reach_the_request_and_the_schedule_sets_the_control_rate():
    class Updating(Task):
        def updates(self, t):
            return {"cost_weights": {"q_cost": t}}

    mpc = Mpc()
    go(mpc, Updating(), duration=.06, update_period=.02)
    assert [(t, u["cost_weights"]["q_cost"]) for t, u in mpc.requests] == pytest.approx(
        [(0., 0.), (.02, .02), (.04, .04)])


def test_schedule_skips_missed_slots():
    due = Schedule(.01)
    assert [due(t) for t in (5., 5.004, 5.01, 5.035, 5.039, 5.04)] == [True, False, True, True, False, True]


def test_guard_rejects_nonfinite_and_excess_commands():
    plan = Plan(0., .01, np.zeros((32, 14)), np.ones((31, 7)))
    np.testing.assert_array_equal(guard(np.ones(7), plan, .099, EFFORT), np.ones(7))
    for bad in (np.r_[np.ones(6), np.nan], np.r_[np.ones(6), 51.], np.ones(6)):
        with pytest.raises(RuntimeError, match="effort"):
            guard(bad, plan, 0., EFFORT)
    with pytest.raises(RuntimeError, match="expired"):
        guard(np.ones(7), plan, .1, EFFORT)
