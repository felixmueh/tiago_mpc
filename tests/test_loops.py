"""The two loops on a fake plant and a fake Mpc (CPU only); run_real_time on faked wall time."""
from dataclasses import dataclass, replace
import math

import numpy as np
import pytest

from tiago_mpc import clock as clock_module
from tiago_mpc import loops
from tiago_mpc.laws import FrictionFeedforward, HoldFirst, PlanFeedback, SampleKnot
from tiago_mpc.dynamics import Friction
from tiago_mpc.mpc import TIAGO
from tiago_mpc.loops import (STOP, RealTime, Schedule, VirtualTime, command_fault, lease_fault, plan_fault,
                             run_real_time, run_virtual_time)
from tiago_mpc.interfaces import Plan, Prediction, State
from tiago_mpc.record import Event


class FakeTime:
    """time.monotonic and time.sleep for Wall: sleeping moves the time on."""
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


@pytest.fixture
def wall(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(clock_module, "time", fake)
    return fake


class Plant:
    """On the loop's clock; measure(t) is the time of the latest state at time t; events:
    what take_events hands out next."""
    def __init__(self, measure=lambda t: t):
        self.measure, self.calls, self.applied, self.run_state, self.events = measure, [], [], {}, []

    def start(self, clock):
        self.clock = clock
        self.calls.append("start")

    def state(self):
        return State(self.measure(self.clock.now()), np.zeros(7), np.zeros(7))

    def apply(self, command):
        self.applied.append((self.clock.now(), command))

    def take_events(self):
        events, self.events = self.events, []
        return events

    def advance(self, t):
        self.clock.reach(t)

    def stop(self):
        self.calls.append("stop")

    def close(self):
        pass


class Mpc:
    """Plans whose knot k has torque k + 1 on joint 1 and 1 on the others (within the
    raw effort limits for N <= 43). delay: result() calls before a reply;
    replies: how many requests get one (later ones hang); over: the request whose
    plan has 0 on joint 2 and its last knot above joint 7's raw effort limit (26 Nm);
    wall: the solve's own wall time (stats["wall_s"])."""
    def __init__(self, N=11, dt=.01, delay=0, replies=math.inf, wall=0., over=None):
        self.params, self.N, self.dt, self.over = None, N, dt, over
        self.delay, self.replies, self.wall = delay, replies, wall
        self.warmups, self.requests, self.waiting, self.dead = [], [], None, False
        self.predictions, self.requested_states = [], []

    def warmup(self, state, reference):
        assert not self.requests and reference.shape == (self.N, 6)
        self.warmups.append(state.t)

    def request(self, state, reference, updates=None):
        assert self.waiting is None, "a request is pending"
        assert reference.shape == (self.N, 6)
        self.requests.append((state.t, updates))
        self.predictions.append(state if isinstance(state, Prediction) else None)
        self.requested_states.append(state)
        self.waiting = [self.delay, state.t, state.measured_t if isinstance(state, Prediction) else state.t]

    def result(self, block=False):
        if self.dead:
            raise RuntimeError("MPC process exited unexpectedly")
        if self.waiting is None or len(self.requests) > self.replies:
            return None
        if self.waiting[0] > 0 and not block:
            self.waiting[0] -= 1
            return None
        t, measured_t = self.waiting[1:]
        self.waiting = None
        torques = np.ones((self.N - 1, 7))
        torques[:, 0] = np.arange(1., self.N)
        if len(self.requests) == self.over:
            torques[:, 1], torques[-1, 6] = 0., 26.1
        stats = dict(wall_s=self.wall)
        return Plan(t, self.dt, np.zeros((self.N, 14)), torques, stats, measured_t=measured_t)

    def close(self):
        pass


@dataclass
class Task:
    done_at: float = math.inf

    @property
    def duration_s(self):
        return 1.

    def at(self, t, n, dt):
        return np.zeros((n, 6)), None

    def done(self, state, t):
        return t >= self.done_at - 1e-9


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
@pytest.mark.parametrize("feedback", [False, True])
def test_duplicate_mpc_and_external_friction_is_rejected_before_start(loop, feedback, wall):
    plant, mpc = Plant(), Mpc()
    # Different estimates still compensate the same losses twice.
    mpc.params = replace(TIAGO, friction=Friction((2.,)*7, (1.,)*7))
    base = PlanFeedback() if feedback else SampleKnot()
    law = FrictionFeedforward(Friction((1.,)*7, (.5,)*7), base)
    settings = dict(command_period=.002, update_period=.02, duration=.01)
    if loop is run_virtual_time:
        settings['solve_latency_s'] = .0055
    with pytest.raises(ValueError, match="double compensation"):
        loop(plant, mpc, Task(), law, **settings)
    assert not plant.calls and not plant.applied and not mpc.warmups and not mpc.requests


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
def test_nested_external_compensation_is_rejected_before_start(loop, wall):
    plant, mpc = Plant(), Mpc()
    mpc.params = TIAGO
    friction = Friction((1.,)*7, (.5,)*7)
    law = FrictionFeedforward(friction, FrictionFeedforward(friction))
    settings = dict(command_period=.002, update_period=.02, duration=.01)
    if loop is run_virtual_time:
        settings['solve_latency_s'] = .0055
    with pytest.raises(ValueError, match="another friction feedforward"):
        loop(plant, mpc, Task(), law, **settings)
    assert not plant.calls and not mpc.warmups and not mpc.requests


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
@pytest.mark.parametrize("mode", ["external", "external_pd", "internal_pd"])
def test_valid_compensation_choices_still_execute(loop, mode, wall):
    plant, mpc = Plant(), Mpc(wall=.004)
    friction = Friction((1.,)*7, (.5,)*7)
    mpc.params = replace(TIAGO, friction=friction if mode == "internal_pd" else None)
    base = PlanFeedback() if mode.endswith('pd') else SampleKnot()
    law = FrictionFeedforward(friction, base) if mode.startswith('external') else base
    settings = dict(command_period=.002, update_period=.02, duration=.01)
    if loop is run_virtual_time:
        settings['solve_latency_s'] = .004
    outcome = loop(plant, mpc, Task(), law, **settings)
    assert outcome.status == 'timeout' and plant.applied
    assert plant.calls == ['start', 'stop'] and mpc.requests


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
def test_physical_friction_and_armature_do_not_count_as_double_compensation(loop, wall):
    from tiago_mpc.sim import SimConfig
    from tiago_mpc.tiago import ARMATURE_KG_M2

    friction = Friction((1.,)*7, (.5,)*7)
    plant = SimConfig(friction=friction, integrator='euler', armature=ARMATURE_KG_M2).build()
    mpc = Mpc(wall=.004)
    mpc.params = TIAGO
    settings = dict(command_period=.002, update_period=.02, duration=.006)
    if loop is run_virtual_time:
        settings['solve_latency_s'] = .004
    outcome = loop(plant, mpc, Task(), FrictionFeedforward(friction), **settings)
    assert outcome.status == 'timeout' and mpc.requests


def go(mpc, task=Task(), loop=run_virtual_time, duration=.1, update_period=.01, plant=None, **kw):
    """10 ms command ticks; run_virtual_time needs solve_latency_s in kw."""
    plant, ticks = plant or Plant(), []
    outcome = loop(plant, mpc, task, tick_callbacks=[ticks.append], command_period=.01,
                   update_period=update_period, duration=duration, **kw)
    return plant, ticks, outcome


def commanded(ticks):
    """(t, knot torque) of each tick with a command."""
    return [(round(k.t, 6), k.command[0]) for k in ticks if k.command is not None]


def events(ticks, kind):
    """The events of kind on the ticks."""
    return [e for k in ticks for e in k.events if e.kind == kind]


# Virtual time (run_virtual_time)

def test_zero_latency_commands_each_plan_at_its_own_tick():
    mpc = Mpc()
    plant, ticks, outcome = go(mpc, duration=.03, solve_latency_s=0.)
    assert outcome == (outcome.__class__("timeout", None, pytest.approx(.03)))
    assert plant.calls == ["start", "stop"] and mpc.warmups == [0.]
    assert commanded(ticks) == [(0., 1.), (.01, 1.), (.02, 1.)]
    assert [k.new for k in ticks] == [True, True, True, False]
    assert [k.update for k in ticks] == [True, True, True, False]  # a control update at each tick but the last
    assert ticks[-1].command is None and ticks[-1].t == pytest.approx(.03)
    assert not any(k.events for k in ticks)


def test_each_plan_that_skips_its_knot_0_is_a_late_plan_event():
    _, ticks, _ = go(Mpc(), solve_latency_s=.01, duration=.08)  # every plan from its knot 1
    late = events(ticks, "late_plan")
    assert [e.data["plan"] for e in late] == list(range(7)) and late[0].data["delay_s"] == pytest.approx(.01)
    assert [round(e.t, 6) for e in late] == [round(k.t, 6) for k in ticks[:-1] if k.new]  # the last sends nothing
    assert "10.0 ms after its state" in late[0].message and "docs/robot.md" in late[0].message
    _, ticks, _ = go(Mpc(), duration=.08, solve_latency_s=0.)  # zero latency: every plan from its knot 0
    assert not events(ticks, "late_plan")


@pytest.mark.parametrize("solve_latency", [.025, "measured"])
def test_a_plan_takes_effect_at_the_first_tick_after_its_latency_and_blocks_requests(solve_latency):
    mpc = Mpc(wall=.025)
    plant, ticks, outcome = go(mpc, solve_latency_s=solve_latency, duration=.08)
    # Requested at 0, effective at 30 ms (knot 3); no request while it is pending.
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .03, .06]
    assert [round(k.t, 6) for k in ticks if k.update] == [0., .03, .06]
    assert commanded(ticks)[:4] == [(.03, 4.), (.04, 5.), (.05, 6.), (.06, 4.)]
    assert all(k.command is None for k in ticks[:3]) and not events(ticks, "overrun")


def test_measured_latency_uses_the_plans_own_wall_time():
    mpc = Mpc(wall=.004)
    _, ticks, _ = go(mpc, solve_latency_s="measured", duration=.03)
    assert commanded(ticks) == [(.01, 2.), (.02, 2.)]  # 4 ms: from the next tick


def test_each_solve_longer_than_a_declared_latency_is_an_overrun_event():
    _, ticks, _ = go(Mpc(wall=.025), solve_latency_s=.01, duration=.08)
    overruns = events(ticks, "overrun")
    assert len(overruns) == 8  # requests at 0, 10, ..., 70 ms
    assert overruns[0].data == dict(plan=0, wall_s=.025, solve_latency_s=.01)
    assert "longer than solve_latency_s=10 ms" in overruns[0].message
    assert commanded(ticks)[0] == (.01, 2.)  # the declared latency, not the solve's time
    _, ticks, _ = go(Mpc(wall=.025), solve_latency_s=0., duration=.08)  # the explicit ideal
    assert not events(ticks, "overrun")


def test_update_period_0_requests_at_the_tick_the_previous_plan_takes_effect():
    mpc = Mpc(wall=.015)
    _, ticks, _ = go(mpc, solve_latency_s="measured", duration=.08, update_period=0.)
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .02, .04, .06]  # back to back
    mpc = Mpc()
    go(mpc, solve_latency_s=0., duration=.03, update_period=0.)
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .01, .02]  # zero latency: every tick


def test_done_is_checked_before_any_command():
    mpc = Mpc()
    plant, ticks, outcome = go(mpc, Task(done_at=0.), solve_latency_s=0.)
    assert outcome.status == "done" and not mpc.requests and not plant.applied and len(ticks) == 1
    plant, ticks, outcome = go(Mpc(), Task(done_at=.02), solve_latency_s=0.)
    assert outcome.status == "done" and [t for t, _ in plant.applied] == [0., .01] and ticks[-1].command is None


def test_plans_are_sampled_at_the_ticks_time_when_measurements_repeat():
    """Commands every 10 ms, a new measurement every 20 ms, one plan (10 ms knots)."""
    plant = Plant(measure=lambda t: math.floor(t / .02 + 1e-9) * .02)
    _, ticks, _ = go(Mpc(replies=1), plant=plant, duration=.04, update_period=1., solve_latency_s=0.)
    assert [round(k.state.t, 6) for k in ticks[:4]] == [0., 0., .02, .02]
    assert commanded(ticks) == [(0., 1.), (.01, 2.), (.02, 3.), (.03, 4.)]  # knots 0, 1, 2, 3


def test_a_stale_state_cannot_extend_the_lease():
    """The state's time stands still; each new plan starts from it, so the lease ends
    100 ms after it, on the run's time."""
    mpc, plant = Mpc(N=32), Plant(measure=lambda t: 0.)
    _, ticks, outcome = go(mpc, plant=plant, duration=1., solve_latency_s=0.)
    assert outcome.status == "fault" and outcome.reason == "lease_expired"
    assert all(t == 0. for t, _ in mpc.requests)  # every plan starts at the stale state
    assert len(plant.applied) >= 10 and max(t for t, _ in plant.applied) < .1
    assert ticks[-1].t >= .1 and ticks[-1].command is None and plant.calls[-1] == "stop"


def test_task_updates_reach_the_request_and_the_schedule_sets_the_control_rate():
    class Updating(Task):
        def at(self, t, n, dt):
            return np.zeros((n, 6)), {"cost_weights": {"q_cost": t}}

    mpc = Mpc()
    go(mpc, Updating(), duration=.06, update_period=.02, solve_latency_s=0.)
    assert [(t, u["cost_weights"]["q_cost"]) for t, u in mpc.requests] == pytest.approx(
        [(0., 0.), (.02, .02), (.04, .04)])


@pytest.mark.parametrize("solve_latency", [None, -.01, math.nan, math.inf, "real"])
def test_run_virtual_time_needs_a_valid_solve_latency_before_it_starts_the_plant(solve_latency):
    plant = Plant()
    with pytest.raises(ValueError, match="solve_latency_s"):
        go(Mpc(), plant=plant, solve_latency_s=solve_latency)
    assert not plant.calls
    with pytest.raises(TypeError, match="solve_latency_s"):  # no default
        go(Mpc())


@pytest.mark.parametrize("loop, kw", [(run_virtual_time, dict(solve_latency_s=0.)), (run_real_time, {})])
def test_a_run_longer_than_max_runtime_raises_before_it_starts_the_plant(wall, monkeypatch, loop, kw):
    monkeypatch.setattr(loops, "MAX_RUNTIME_MIN", .001)  # 60 ms: a broken check costs a 0.1 s run
    plant = Plant()
    with pytest.raises(ValueError, match="Not validated for long runs"):
        go(Mpc(), loop=loop, duration=.1, plant=plant, **kw)
    assert not plant.calls


# Wall time (run_real_time)

def test_no_command_before_the_first_plan_and_each_plan_from_the_tick_after_it_arrives(wall):
    mpc = Mpc(delay=2)  # arrives at the third poll after its request
    _, ticks, outcome = go(mpc, loop=run_real_time, duration=.06)
    assert [k.command is None for k in ticks[:4]] == [True, True, True, False]
    assert commanded(ticks)[0] == (.03, 4.)
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .03]
    assert [round(k.t, 6) for k in ticks if k.update] == [0., .03]  # requested after the tick callbacks
    assert outcome.status == "timeout" and not events(ticks, "late_tick")


def test_the_lease_ends_100_ms_after_the_plans_state(wall):
    mpc = Mpc(N=32, replies=1)  # 310 ms plan; later solves hang
    _, ticks, outcome = go(mpc, loop=run_real_time, duration=1.)
    assert outcome.status == "fault" and outcome.reason == "lease_expired" and outcome.t == pytest.approx(.11)
    assert mpc.requests == [(0., None), (.01, None)] and ticks[-1].command is None


class Late(Plant):
    """advance(t) returns at returns[t] instead of t (default: 5 ms after the deadline at
    20 ms, 15 ms after the one at 40 ms)."""
    def __init__(self, wall, returns={.02: .025, .04: .055}):
        super().__init__()
        self.wall, self.returns = wall, returns

    def advance(self, t):
        super().advance(t)
        self.wall.now = self.returns.get(round(t, 6), self.wall.now)


def test_ticks_keep_their_rate_and_a_late_tick_skips_the_missed_ones(wall):
    _, ticks, _ = go(Mpc(), loop=run_real_time, plant=Late(wall), duration=.07)
    assert [round(k.t, 6) for k in ticks] == [0., .01, .025, .03, .055, .065, .075]
    late, = events(ticks, "late_tick")  # 5 ms is not late
    assert late.t == pytest.approx(.055) and late.data == dict(late_s=pytest.approx(.015), skipped=1)
    assert "15.0 ms after its deadline" in late.message


def test_the_tick_after_the_first_command_is_late_without_an_event(wall):
    class Activating(Plant):  # the robot's first command switches controllers: 25 ms
        def apply(self, command):
            super().apply(command)
            wall.now += .025 if len(self.applied) == 1 else 0.

    _, ticks, _ = go(Mpc(), loop=run_real_time, plant=Activating(), duration=.05)
    assert [round(k.t, 6) for k in ticks][:4] == [0., .01, .035, .045] and not events(ticks, "late_tick")

    class Both(Activating, Late):  # and 15 ms late at 65 ms: that one is an event
        pass

    wall.now = 0.
    _, ticks, _ = go(Mpc(), loop=run_real_time, plant=Both(wall, {.065: .08}), duration=.1)
    assert [round(e.t, 6) for e in events(ticks, "late_tick")] == [.08]


def test_past_the_plans_end_faults_and_stops(wall):
    plant, ticks = Plant(), []
    outcome = run_real_time(plant, Mpc(N=4, replies=1), Task(), HoldFirst(), tick_callbacks=[ticks.append],
                            command_period=.01, update_period=.01, duration=1.)
    assert outcome.status == "fault" and outcome.reason == "lease_expired"  # 30 ms plan: lease ends with it
    assert [round(t, 6) for t, _ in plant.applied] == [.01, .02]  # from the tick after it arrived
    assert plant.calls == ["start", "stop"] and round(ticks[-1].t, 6) == .03 and ticks[-1].command is None


def test_a_dead_mpc_process_is_noticed_within_one_tick_and_stops_the_plant(wall):
    mpc, plant = Mpc(), Plant()

    def die(tick):
        mpc.dead = tick.t >= .02

    with pytest.raises(RuntimeError, match="exited"):
        run_real_time(plant, mpc, Task(), tick_callbacks=[die], command_period=.01, update_period=.05,
                      duration=1.)
    assert plant.calls == ["start", "stop"] and wall.now == pytest.approx(.03)


class Blocking(Mpc):
    """In-process on wall time: request k takes solves[k] seconds (the last one repeats)."""
    def __init__(self, wall, solves, **kw):
        super().__init__(**kw)
        self.wall, self.solves = wall, solves

    def request(self, state, reference, updates=None):
        super().request(state, reference, updates)
        self.wall.now += self.solves[min(len(self.requests), len(self.solves)) - 1]


def test_a_blocking_solve_ends_the_run_or_the_lease_at_the_tick_after_it(wall):
    """A 120 ms solve, requested at 0, outlasts the 50 ms run and the plan's 100 ms lease."""
    _, ticks, outcome = go(Blocking(wall, [.12]), loop=run_real_time, duration=.05)
    assert outcome.status == "timeout" and len(events(ticks, "late_tick")) == 1
    assert [(round(k.t, 6), k.command) for k in ticks] == [(0., None), (.12, None)]
    plant = Plant()  # a 310 ms plan: its lease ends first
    _, ticks, outcome = go(Blocking(wall, [.12], N=32), loop=run_real_time, plant=plant, duration=1.)
    assert outcome.reason == "lease_expired" and len(events(ticks, "late_tick")) == 1
    assert not plant.applied and plant.calls == ["start", "stop"]


def test_after_a_short_blocking_solve_the_next_tick_is_behind_but_not_late(wall):
    """One 12 ms solve, requested at 0: the tick due at 10 ms runs at 12 ms with knot 1,
    2 ms behind but within its period, so the ticks keep their times."""
    plant = Plant()
    _, ticks, _ = go(Blocking(wall, [.012]), loop=run_real_time, plant=plant, duration=.03, update_period=1.)
    assert commanded(ticks) == [(.012, 2.), (.02, 3.)] and not events(ticks, "late_tick")


def test_a_blocking_solve_comes_after_the_command_so_no_command_goes_out_past_its_lease(wall):
    """The plan of the request at 0 takes effect at 10 ms; the request there blocks for
    120 ms, past that plan's lease (100 ms). Its command went out before, at 10 ms; the
    next tick, at 130 ms, faults on the next plan's lease (10 ms + 100 ms)."""
    plant = Plant()
    _, ticks, outcome = go(Blocking(wall, [0., .12], N=32), loop=run_real_time, plant=plant, duration=1.)
    assert outcome.reason == "lease_expired" and not events(ticks, "late_tick")  # late after the first command
    assert [round(t, 6) for t, _ in plant.applied] == [.01]


def test_update_period_0_on_wall_time_requests_at_the_tick_each_plan_is_used(wall):
    mpc = Mpc(delay=1)  # arrives at the second poll after its request
    go(mpc, loop=run_real_time, duration=.06, update_period=0.)
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .02, .04]


def test_the_warmup_runs_once_before_the_first_tick_and_counts_against_no_lease(wall):
    """A 150 ms warm-up on wall time; the first plan starts at the state after it."""
    class Warming(Mpc):
        def warmup(self, state, reference):
            super().warmup(state, reference)
            wall.now += .15

    plant, mpc = Plant(), Warming(N=32)
    _, ticks, _ = go(mpc, loop=run_real_time, plant=plant, duration=.05)
    assert mpc.warmups == [0.] and [round(t, 6) for t, _ in mpc.requests][:2] == [.15, .16]
    assert commanded(ticks)[0] == (.16, 2.) and plant.calls == ["start", "stop"]


# Both

def test_schedule_skips_missed_slots_and_period_0_is_due_at_every_call():
    due = Schedule(.01)
    assert [due(t) for t in (5., 5.004, 5.01, 5.035, 5.039, 5.04)] == [True, False, True, True, False, True]
    due = Schedule(0)
    assert all(due(t) for t in (5., 5., 5.004))


@pytest.mark.parametrize("loop, kw", [(run_virtual_time, dict(solve_latency_s=0.)),
                                      (run_virtual_time, dict(solve_latency_s=.02)),
                                      (run_real_time, {})])
def test_a_plan_above_the_raw_effort_limit_ends_the_run_before_any_of_its_knots_is_commanded(wall, loop, kw):
    """Only plan 2's last knot is over the limit; the run stops on its arrival, sending nothing."""
    mpc, plant, ticks = Mpc(delay=1, over=2), Plant(), []
    outcome = loop(plant, mpc, Task(), tick_callbacks=[ticks.append], command_period=.01, update_period=.01,
                   duration=.1, **kw)
    assert outcome.status == "fault" and outcome.reason == "effort_limit"
    assert len(mpc.requests) == 2 and mpc.waiting is None and plant.calls[-1] == "stop"
    assert all(command[1] == 1. for _, command in plant.applied)  # plan 2 is never commanded
    assert ticks[-1].command is None and all(k.plan is None or k.plan.torques[0, 1] == 1. for k in ticks)
    assert ticks[-1].faulted.torques[-1, 6] == 26.1 and all(k.faulted is None for k in ticks[:-1])  # for the record


def test_the_faults_name_the_reason_and_none_without_one():
    plan = Plan(0., .01, np.zeros((32, 14)), np.ones((31, 7)), measured_t=0.)
    assert plan_fault(plan) is None and command_fault(np.ones(7)) is None and lease_fault(plan, .099) is None
    assert plan_fault(Plan(0., .01, np.zeros((3, 14)), np.r_[np.ones((1, 7)), [[1.] * 6 + [26.1]]], measured_t=0.)) == "effort_limit"
    assert command_fault(np.r_[np.ones(6), 26.1]) == "effort_limit"  # joint 7: 26 Nm
    for bad in (np.r_[np.ones(6), np.nan], np.ones(6)):
        assert command_fault(bad) == "invalid_command"
    assert lease_fault(plan, .1) == "lease_expired"
    assert lease_fault(Plan(0., .01, np.zeros((4, 14)), np.ones((3, 7)), measured_t=0.), .03) == "lease_expired"  # its end


@pytest.mark.parametrize("loop, kw", [(run_virtual_time, dict(solve_latency_s=0.)), (run_real_time, {})])
def test_a_stop_request_ends_the_run_at_the_next_tick_without_a_command(wall, loop, kw):
    def stop(tick):
        if round(tick.t, 6) == .02:
            STOP.set()

    plant, ticks = Plant(), []
    outcome = loop(plant, Mpc(), Task(), tick_callbacks=[stop, ticks.append], command_period=.01,
                   update_period=.01, duration=1., **kw)
    assert outcome.status == "interrupted" and outcome.reason == "stop_requested" and outcome.t == pytest.approx(.03)
    assert [round(k.t, 6) for k in ticks] == [0., .01, .02, .03] and ticks[-1].command is None
    assert not ticks[-1].update and plant.calls == ["start", "stop"]


def test_the_plants_events_ride_on_the_next_tick(wall):
    """take_events once per tick, before the tick is built: an event from advance arrives with the next."""
    class Emitting(Plant):
        def advance(self, t):
            super().advance(t)
            self.events.append(Event(t, "pacing_behind", "behind", dict(behind_s=.001)))

    _, ticks, _ = go(Mpc(), plant=Emitting(), duration=.02, solve_latency_s=0.)
    assert [[e.t for e in k.events] for k in ticks] == [[], [.01], [pytest.approx(.02)]]


def test_the_timings_run_their_loops_with_their_settings(wall):
    assert VirtualTime.ideal() == VirtualTime(.002, .01, 0.)
    assert VirtualTime.constant_latency() == VirtualTime(.002, .01, .05)
    assert VirtualTime.constant_latency(.02, update_period=.02) == VirtualTime(.002, .02, .02)
    assert VirtualTime.as_fast_as_possible() == VirtualTime(.002, 0., "measured")
    assert VirtualTime.constant_rate() == VirtualTime(.002, .01, .01) and RealTime() == RealTime(.002, .01)
    for timing, clock in ((VirtualTime(.01, .01, .025), "VirtualClock"), (RealTime(.01, .01), "WallClock")):
        plant, ticks = Plant(), []
        outcome = timing.run(plant, Mpc(delay=2), Task(), HoldFirst(), [ticks.append], duration=.08)
        assert outcome.status == "timeout" and len(ticks) == 9 and type(plant.clock).__name__ == clock
        assert [round(k.t, 6) for k in ticks if k.new][0] == .03  # 25 ms declared; the third poll


def test_a_500_hz_tick_fits_well_inside_its_2_ms(tmp_path):
    """The loop body with SampleKnot, the fault checks, a Reach task and a RunRecord; fake plant
    and Mpc (a new plan at every tick). Measured: median 0.06 ms, p99 0.08 ms, max 0.1 ms."""
    import time
    from experiments.reach import Reach
    from tiago_mpc.mpc import TIAGO
    from tiago_mpc.record import RunRecord

    class Timed(Plant):
        def advance(self, t):
            stamps.append(time.perf_counter())
            super().advance(t)

    stamps, plant, task = [], Timed(), Reach(timeout_s=10.)
    record = RunRecord(tmp_path / "r", task=task, params=TIAGO)
    run_virtual_time(plant, Mpc(N=32, dt=.002), task, tick_callbacks=[record], command_period=.002,
                update_period=.002, duration=2., solve_latency_s=0.)
    body = np.diff(stamps) * 1e3
    print(f"tick body ms: median {np.median(body):.3f}, p99 {np.percentile(body, 99):.3f}, max {body.max():.3f}")
    assert len(body) == 999 and np.isfinite(record.ticks["command"][:-1]).all()
    # One late tick is allowed for a loaded machine; a full garbage collection gives several.
    assert np.median(body) < .5 and np.percentile(body, 99) < 1. and np.sum(body > 2.) <= 1
