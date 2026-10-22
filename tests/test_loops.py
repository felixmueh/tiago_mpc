"""The loop on a fake plant and a fake Mpc (CPU only)."""
from dataclasses import dataclass
import math

import numpy as np
import pytest

from tiago_mpc.clock import Clock
from tiago_mpc.laws import hold_first
from tiago_mpc.loops import Fixed, Measured, Schedule, guard, run, run_robot, run_simulation
from tiago_mpc.plan import Plan, State


class Plant:
    """On a simulation clock, or labelled kind ("wall" for run_robot); measure(t) is the
    time of the latest state at time t."""
    def __init__(self, measure=lambda t: t, kind="simulation"):
        self.t, self.measure, self.calls, self.applied = 0., measure, [], []
        self.clock, self.settings = Clock(kind), {}

    def start(self):
        self.calls.append("start")

    def now(self):
        return self.t

    def state(self):
        return State(self.measure(self.t), np.zeros(7), np.zeros(7))

    def apply(self, command):
        self.applied.append((self.t, command))

    def advance(self, t):
        self.t = t

    def stop(self):
        self.calls.append("stop")

    def close(self):
        pass


class Mpc:
    """Plans whose knot k has torque k + 1. delay: result() calls before a reply;
    replies: how many requests get one (later ones hang)."""
    def __init__(self, N=11, dt=.01, delay=0, replies=math.inf, wall=0.):
        self.params, self.N, self.dt = None, N, dt
        self.delay, self.replies, self.wall = delay, replies, wall
        self.requests, self.waiting, self.dead = [], None, False

    def request(self, state, reference, updates=None):
        assert self.waiting is None, "a request is pending"
        assert reference.shape == (self.N, 6)
        self.requests.append((state.t, updates))
        self.waiting = [self.delay, state.t]

    def result(self, block=False):
        if self.dead:
            raise RuntimeError("MPC process exited unexpectedly")
        if self.waiting is None or len(self.requests) > self.replies:
            return None
        if self.waiting[0] > 0 and not block:
            self.waiting[0] -= 1
            return None
        t, self.waiting = self.waiting[1], None
        torques = np.arange(1., self.N)[:, None] * np.ones(7)
        return Plan(t, self.dt, np.zeros((self.N, 14)), torques, dict(wall_s=self.wall))

    def close(self):
        pass


@dataclass
class Task:
    done_at: float = math.inf

    def reference(self, t, n, dt):
        return np.zeros((n, 6))

    def done(self, state, t):
        return t >= self.done_at - 1e-9


def go(mpc, task=Task(), loop=run_simulation, duration=.1, update_period=.01, plant=None, **kw):
    plant, ticks = plant or Plant(kind="wall" if loop is run_robot else "simulation"), []
    result = loop(plant, mpc, task, callbacks=[ticks.append], command_period=.01,
                  update_period=update_period, duration=duration, **kw)
    return plant, ticks, result


def commanded(ticks):
    """(t, knot torque) of each tick with a command."""
    return [(round(k.t, 6), k.command[0]) for k in ticks if k.command is not None]


def test_zero_latency_commands_each_plan_at_its_own_tick():
    plant, ticks, done = go(Mpc(), duration=.03)
    assert not done and plant.calls == ["start", "stop"]
    assert commanded(ticks) == [(0., 1.), (.01, 1.), (.02, 1.)]
    assert [k.new for k in ticks] == [True, True, True, False]
    assert ticks[-1].command is None and ticks[-1].t == pytest.approx(.03)


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


def test_plans_are_sampled_at_execution_time_when_measurements_repeat():
    """Commands every 10 ms, a new measurement every 20 ms, one plan (10 ms knots)."""
    plant = Plant(measure=lambda t: math.floor(t / .02 + 1e-9) * .02)
    _, ticks, _ = go(Mpc(replies=1), plant=plant, duration=.04, update_period=1.)
    assert [round(k.state.t, 6) for k in ticks[:4]] == [0., 0., .02, .02]
    assert commanded(ticks) == [(0., 1.), (.01, 2.), (.02, 3.), (.03, 4.)]  # knots 0, 1, 2, 3


def test_a_stale_state_cannot_extend_the_lease():
    """The state's time stands still; each new plan starts from it, so the lease ends
    100 ms after it, on execution time."""
    mpc, plant = Mpc(N=32), Plant(measure=lambda t: 0.)
    with pytest.raises(RuntimeError, match="lease expired"):
        go(mpc, plant=plant, duration=1.)
    assert all(t == 0. for t, _ in mpc.requests)  # every plan starts at the stale state
    assert len(plant.applied) >= 10 and max(t for t, _ in plant.applied) < .1


def test_ticks_keep_their_rate_and_a_late_tick_skips_the_missed_ones():
    class Late(Plant):  # returns 5 ms after the deadline at 20 ms, 15 ms after the one at 40 ms
        def advance(self, t):
            self.t = {.02: .025, .04: .055}.get(round(t, 6), t)

    _, ticks, _ = go(Mpc(), loop=run_robot, plant=Late(kind="wall"), duration=.07)
    assert [round(k.t, 6) for k in ticks] == [0., .01, .025, .03, .055, .065, .075]


def test_past_the_plans_end_faults_and_stops():
    plant, ticks = Plant(kind="wall"), []
    with pytest.raises(RuntimeError, match="lease expired"):  # 30 ms plan: lease ends with it
        run_robot(plant, Mpc(N=4, replies=1), Task(), hold_first, callbacks=[ticks.append],
                  command_period=.01, update_period=.01, duration=1.)
    assert [round(t, 6) for t, _ in plant.applied] == [0., .01, .02]
    assert plant.calls == ["start", "stop"]


def test_a_dead_mpc_process_is_noticed_within_one_tick_and_stops_the_plant():
    mpc, plant = Mpc(), Plant(kind="wall")

    def die(tick):
        mpc.dead = tick.t >= .02

    with pytest.raises(RuntimeError, match="exited"):
        run_robot(plant, mpc, Task(), callbacks=[die], command_period=.01, update_period=.05, duration=1.)
    assert plant.calls == ["start", "stop"] and plant.t == pytest.approx(.03)


def test_each_loop_refuses_a_plant_on_the_other_clock():
    for loop, kind in ((run_simulation, "wall"), (run_simulation, "external"), (run_robot, "simulation")):
        plant = Plant(kind=kind)
        with pytest.raises(ValueError, match="clock"):
            go(Mpc(), loop=loop, plant=plant)
        assert not plant.calls  # never started


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
    np.testing.assert_array_equal(guard(np.ones(7), plan, .099), np.ones(7))
    for bad in (np.r_[np.ones(6), np.nan], np.r_[np.ones(6), 26.1], np.ones(6)):  # joint 7: 26 Nm
        with pytest.raises(RuntimeError, match="effort"):
            guard(bad, plan, 0.)
    with pytest.raises(RuntimeError, match="expired"):
        guard(np.ones(7), plan, .1)


def test_a_500_hz_tick_fits_well_inside_its_2_ms(tmp_path):
    """The loop body with sample_knot, guard, a Reach task and Record; fake plant and Mpc
    (a new plan at every tick). Measured: median 0.06 ms, p99 0.08 ms, max 0.1 ms."""
    import time
    from experiments.reach import Reach
    from tiago_mpc.mpc import TIAGO
    from tiago_mpc.record import Record

    class Timed(Plant):
        settings = dict(clock="simulation")

        def advance(self, t):
            stamps.append(time.perf_counter())
            super().advance(t)

    stamps, plant, task = [], Timed(), Reach(timeout_s=10.)
    record = Record(tmp_path / "r", task=task, params=TIAGO, plant=plant)
    run_simulation(plant, Mpc(N=32, dt=.002), task, callbacks=[record], command_period=.002,
                   update_period=.002, duration=2.)
    body = np.diff(stamps) * 1e3
    print(f"tick body ms: median {np.median(body):.3f}, p99 {np.percentile(body, 99):.3f}, max {body.max():.3f}")
    assert len(body) == 999 and all(k.command is not None for k in record.ticks[:-1])
    # One late tick is allowed for a loaded machine; a full garbage collection gives several.
    assert np.median(body) < .5 and np.percentile(body, 99) < 1. and np.sum(body > 2.) <= 1
