"""The two loops on a fake plant and a fake Mpc (CPU only); run_real_time on faked wall time."""
from dataclasses import dataclass, replace
import math
import warnings

import numpy as np
import pytest

from tiago_mpc import clock as clock_module
from tiago_mpc import loops
from tiago_mpc.laws import FrictionFeedforward, HoldFirst, PlanFeedback, SampleKnot
from tiago_mpc.dynamics import Friction, Tanh
from tiago_mpc.mpc import TIAGO
from tiago_mpc.prediction import LatencyCompensation
from tiago_mpc.loops import STOP, RealTimeSettings, Schedule, VirtualTimeSettings, run_real_time, run_virtual_time
from tiago_mpc.interfaces import Command, Diagnostic, Plan, Prediction, State


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
    """On the loop's clock; measure(t) gives the latest state's measurement time."""
    def __init__(self, measure=lambda t: t):
        self.measure, self.calls, self.applied, self.run_state, self.diagnostics = measure, [], [], {}, []

    def start(self, clock):
        self.clock = clock
        self.calls.append("start")

    def state(self):
        return State(self.measure(self.clock.now()), np.zeros(7), np.zeros(7))

    def apply(self, command):
        self.applied.append((self.clock.now(), command))

    def take_diagnostics(self):
        diagnostics, self.diagnostics = self.diagnostics, []
        return diagnostics

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

    def warmup(self, state, reference=None):
        assert not self.requests and (reference is None or reference.shape == (self.N, 6))
        self.warmups.append(state.t)

    def request(self, state, reference=None, updates=None):
        assert self.waiting is None, "a request is pending"
        assert reference is None or reference.shape == (self.N, 6)
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
    settings = dict(command_period=.002, update_period=.02)
    if loop is run_virtual_time:
        settings = dict(settings=VirtualTimeSettings(**settings, solve_latency_s=.0055))
    else:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=RealTimeSettings(**settings), compensation=compensation)
    with pytest.raises(ValueError, match="double compensation"):
        loop(plant, mpc, Task(), law, duration=.01, **settings)
    assert not plant.calls and not plant.applied and not mpc.warmups and not mpc.requests


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
def test_nested_external_compensation_is_rejected_before_start(loop, wall):
    plant, mpc = Plant(), Mpc()
    mpc.params = TIAGO
    friction = Friction((1.,)*7, (.5,)*7)
    law = FrictionFeedforward(friction, FrictionFeedforward(friction))
    settings = dict(command_period=.002, update_period=.02)
    if loop is run_virtual_time:
        settings = dict(settings=VirtualTimeSettings(**settings, solve_latency_s=.0055))
    else:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=RealTimeSettings(**settings), compensation=compensation)
    with pytest.raises(ValueError, match="another friction feedforward"):
        loop(plant, mpc, Task(), law, duration=.01, **settings)
    assert not plant.calls and not mpc.warmups and not mpc.requests


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
@pytest.mark.parametrize("mode", ["external", "external_pd", "internal_pd"])
def test_valid_compensation_choices_still_execute(loop, mode, wall):
    plant, mpc = Plant(), Mpc(wall=.004)
    friction = Friction((1.,)*7, (.5,)*7)
    mpc.params = replace(TIAGO, friction=friction if mode == "internal_pd" else None)
    base = PlanFeedback() if mode.endswith('pd') else SampleKnot()
    law = FrictionFeedforward(friction, base) if mode.startswith('external') else base
    settings = dict(command_period=.002, update_period=.02)
    if loop is run_virtual_time:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=VirtualTimeSettings(**settings, solve_latency_s=.004), compensation=compensation)
    else:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=RealTimeSettings(**settings), compensation=compensation)
    outcome = loop(plant, mpc, Task(), law, duration=.01, **settings)
    assert outcome.status == 'timeout' and plant.applied
    assert plant.calls == ['start', 'stop'] and mpc.requests


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
@pytest.mark.parametrize("feedforward", [False, True])
def test_plan_feedback_without_latency_compensation_warns(loop, feedforward, wall):
    mpc = Mpc(wall=.004)
    mpc.params = TIAGO
    law = FrictionFeedforward(Friction((1.,)*7, (.5,)*7), PlanFeedback()) if feedforward else PlanFeedback()
    settings = dict(command_period=.002, update_period=.02)
    if loop is run_virtual_time:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=VirtualTimeSettings(**settings, solve_latency_s=.004), compensation=compensation)
    else:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=RealTimeSettings(**settings), compensation=compensation)
    with pytest.warns(RuntimeWarning, match="PlanFeedback without latency compensation"):
        outcome = loop(Plant(), mpc, Task(), law, duration=.01, **settings)
    assert outcome.status == 'timeout'  # a warning, not a rejection


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
@pytest.mark.parametrize("case", ["sample_knot", "feedforward", "compensated_pd", "compensated_both"])
def test_no_feedback_warning_without_plan_feedback_or_with_compensation(loop, case, wall):
    mpc = Mpc(wall=.004)
    mpc.params = TIAGO
    friction = Friction((1.,)*7, (.5,)*7)
    law = {"sample_knot": SampleKnot(), "feedforward": FrictionFeedforward(friction, SampleKnot()),
           "compensated_pd": PlanFeedback(), "compensated_both": FrictionFeedforward(friction, PlanFeedback())}[case]
    settings = dict(command_period=.002, update_period=.02)
    if case.startswith("compensated"):
        settings['compensation'] = LatencyCompensation(initial_latency_s=.004)
    if loop is run_virtual_time:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=VirtualTimeSettings(**settings, solve_latency_s=.004), compensation=compensation)
    else:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=RealTimeSettings(**settings), compensation=compensation)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        outcome = loop(Plant(), mpc, Task(), law, duration=.01, **settings)
    assert outcome.status == 'timeout'


@pytest.mark.parametrize("loop", [run_real_time, run_virtual_time])
def test_physical_friction_and_armature_do_not_count_as_double_compensation(loop, wall):
    from tiago_mpc.sim import SimConfig
    from tiago_mpc.tiago import ARMATURE_KG_M2

    friction = Friction((1.,)*7, (.5,)*7)
    plant = SimConfig(friction=friction, integrator='euler', armature=ARMATURE_KG_M2).build()
    mpc = Mpc(wall=.004)
    mpc.params = TIAGO
    settings = dict(command_period=.002, update_period=.02)
    if loop is run_virtual_time:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=VirtualTimeSettings(**settings, solve_latency_s=.004), compensation=compensation)
    else:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=RealTimeSettings(**settings), compensation=compensation)
    outcome = loop(plant, mpc, Task(), FrictionFeedforward(friction, SampleKnot(), Tanh(.03)), duration=.006,
                   **settings)
    assert outcome.status == 'timeout' and mpc.requests


def go(
    mpc,
    task=Task(),
    loop=run_virtual_time,
    duration=.1,
    update_period=.01,
    plant=None,
    diagnostics=None,
    **kw,
):
    """Run test fakes at 10 ms command ticks with the chosen loop's settings."""
    plant, ticks = plant or Plant(), []
    options = {key: kw.pop(key) for key in ("compensation", "safety", "effort_handling") if key in kw}
    settings = dict(command_period=.01, update_period=update_period, **kw)
    if loop is run_virtual_time:
        settings = dict(settings=VirtualTimeSettings(**settings))
        options.setdefault("compensation", None)  # these traces are uncompensated
    elif loop is run_real_time:
        compensation = settings.pop("compensation", None)
        settings = dict(settings=RealTimeSettings(**settings), compensation=compensation)
    outcome = loop(plant, mpc, task, tick_callbacks=[ticks.append],
                   diagnostic_callbacks=[] if diagnostics is None else [diagnostics.append],
                   duration=duration, **settings, **options, law=SampleKnot())
    return plant, ticks, outcome


def commanded(ticks):
    """(t, knot torque) of each tick with a command."""
    return [(round(k.t, 6), k.command[0]) for k in ticks if k.command is not None]


def diagnostics_of(diagnostics, kind):
    """The diagnostics of one kind."""
    return [diagnostic for diagnostic in diagnostics if diagnostic.kind == kind]


# Virtual time (run_virtual_time)

def test_zero_latency_commands_each_plan_at_its_own_tick():
    diagnostics = []
    mpc = Mpc()
    plant, ticks, outcome = go(mpc, diagnostics=diagnostics, duration=.03, solve_latency_s=0.)
    assert outcome == (outcome.__class__("timeout", None, pytest.approx(.03)))
    assert plant.calls == ["start", "stop"] and mpc.warmups == [0.]
    assert commanded(ticks) == [(0., 1.), (.01, 1.), (.02, 1.)]
    assert [k.plan_activated for k in ticks] == [True, True, True, False]
    assert [k.update for k in ticks] == [True, True, True, False]  # a control update at each tick but the last
    assert ticks[-1].command is None and ticks[-1].t == pytest.approx(.03)
    assert not diagnostics


def test_each_plan_that_skips_its_knot_0_is_a_late_plan_event():
    diagnostics = []
    _, ticks, _ = go(Mpc(), diagnostics=diagnostics, solve_latency_s=.01, duration=.08)  # every plan from its knot 1
    late = diagnostics_of(diagnostics, "late_plan")
    assert [e.data["plan"] for e in late] == list(range(7)) and late[0].data["delay_s"] == pytest.approx(.01)
    assert [round(e.t, 6) for e in late] == [round(k.t, 6) for k in ticks[:-1] if k.plan_activated]  # the last sends nothing
    assert "10.0 ms after its state" in late[0].message and "docs/robot.md" in late[0].message
    diagnostics.clear()
    _, ticks, _ = go(Mpc(), diagnostics=diagnostics, duration=.08, solve_latency_s=0.)  # zero latency: every plan from its knot 0
    assert not diagnostics_of(diagnostics, "late_plan")


@pytest.mark.parametrize("solve_latency", [.025, "measured"])
def test_a_plan_takes_effect_at_the_first_tick_after_its_latency_and_blocks_requests(solve_latency):
    diagnostics = []
    mpc = Mpc(wall=.025)
    plant, ticks, outcome = go(mpc, diagnostics=diagnostics, solve_latency_s=solve_latency, duration=.08)
    # Requested at 0, effective at 30 ms (knot 3); no request while it is pending.
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .03, .06]
    assert [round(k.t, 6) for k in ticks if k.update] == [0., .03, .06]
    assert commanded(ticks)[:4] == [(.03, 4.), (.04, 5.), (.05, 6.), (.06, 4.)]
    assert all(k.command is None for k in ticks[:3]) and not diagnostics_of(diagnostics, "overrun")


def test_measured_latency_uses_the_plans_own_wall_time():
    mpc = Mpc(wall=.004)
    _, ticks, _ = go(mpc, solve_latency_s="measured", duration=.03)
    assert commanded(ticks) == [(.01, 2.), (.02, 2.)]  # 4 ms: from the next tick


def test_measured_zero_waits_one_tick_while_explicit_ideal_zero_is_immediate():
    for latency, first in [('measured', .01), (0., 0.)]:
        mpc = Mpc(wall=0.)
        _, ticks, _ = go(mpc, solve_latency_s=latency, duration=.03)
        assert commanded(ticks)[0][0] == first
        assert [round(t, 6) for t, _ in mpc.requests] == [0., .01, .02]


def test_each_solve_longer_than_a_declared_latency_is_an_overrun_event():
    diagnostics = []
    _, ticks, _ = go(Mpc(wall=.025), diagnostics=diagnostics, solve_latency_s=.01, duration=.08)
    overruns = diagnostics_of(diagnostics, "overrun")
    assert len(overruns) == 8  # requests at 0, 10, ..., 70 ms
    assert overruns[0].data == dict(plan=0, wall_s=.025, solve_latency_s=.01)
    assert "longer than solve_latency_s=10 ms" in overruns[0].message
    assert commanded(ticks)[0] == (.01, 2.)  # the declared latency, not the solve's time
    diagnostics.clear()
    _, ticks, _ = go(Mpc(wall=.025), diagnostics=diagnostics, solve_latency_s=0., duration=.08)  # the explicit ideal
    assert not diagnostics_of(diagnostics, "overrun")


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
            return np.zeros((n, 6)), {"external_force": np.full(6, t)}

    mpc = Mpc()
    go(mpc, Updating(), duration=.06, update_period=.02, solve_latency_s=0.)
    assert [(t, u["external_force"][0]) for t, u in mpc.requests] == pytest.approx(
        [(0., 0.), (.02, .02), (.04, .04)])


@pytest.mark.parametrize("solve_latency", [None, -.01, math.nan, math.inf, "real"])
def test_virtual_settings_need_a_valid_solve_latency(solve_latency):
    with pytest.raises(ValueError, match="solve_latency_s"):
        VirtualTimeSettings(.01, .01, solve_latency)


def test_each_loop_defaults_to_its_settings_compensation_and_the_feedforward_law():
    import inspect
    from tiago_mpc.tiago import urdf_friction
    for loop, settings in ((run_virtual_time, VirtualTimeSettings.constant_rate()), (run_real_time, RealTimeSettings())):
        defaults = {name: p.default for name, p in inspect.signature(loop).parameters.items()}
        assert defaults["settings"] == settings and defaults["compensation"] == "default"
        assert defaults["law"] is None  # resolved to FrictionFeedforward() at the call
    assert FrictionFeedforward() == FrictionFeedforward(urdf_friction(), PlanFeedback(), Tanh(.01))


@pytest.mark.parametrize('loop, settings', [
    (run_virtual_time, VirtualTimeSettings.ideal()),
    (run_real_time, RealTimeSettings()),
])
def test_each_loop_requires_a_duration_before_starting_the_plant(loop, settings):
    plant = Plant()
    with pytest.raises(TypeError, match='duration'):
        loop(plant, Mpc(), Task(), settings=settings, law=SampleKnot())
    assert not plant.calls


@pytest.mark.parametrize('duration', [None, -1., math.inf, -math.inf, math.nan])
@pytest.mark.parametrize('loop, settings', [
    (run_virtual_time, VirtualTimeSettings.ideal()),
    (run_real_time, RealTimeSettings()),
])
def test_invalid_duration_is_rejected_before_startup(loop, settings, duration):
    plant, mpc = Plant(), Mpc()
    with pytest.raises(ValueError, match='duration'):
        loop(plant, mpc, Task(), settings=settings, duration=duration, law=SampleKnot())
    assert not plant.calls and not mpc.warmups and not mpc.requests


@pytest.mark.parametrize('loop, settings', [
    (run_virtual_time, VirtualTimeSettings.ideal()),
    (run_real_time, RealTimeSettings()),
])
def test_zero_duration_and_exact_ten_minute_limit_are_accepted(wall, loop, settings):
    assert loops.MAX_RUNTIME_MIN == 10
    for duration in (0., 600.):
        plant = Plant()
        outcome = loop(plant, Mpc(), Task(done_at=0.), settings=settings, duration=duration, law=SampleKnot())
        assert outcome.status == 'done' and outcome.t == 0.
        assert plant.calls == ['start', 'stop'] and not plant.applied
    plant = Plant()
    with pytest.raises(ValueError, match='Not validated for long runs'):
        loop(plant, Mpc(), Task(), settings=settings, duration=600.001, law=SampleKnot())
    assert not plant.calls


@pytest.mark.parametrize("settings_type", [VirtualTimeSettings, RealTimeSettings])
@pytest.mark.parametrize("piece", ["compensation", "safety", "effort_handling", "max_plan_age_s"])
def test_timing_settings_reject_execution_policies(settings_type, piece):
    with pytest.raises(TypeError, match=piece):
        settings_type(**{piece: None})


@pytest.mark.parametrize("setting, value", [
    ("command_period", .01), ("update_period", .01), ("solve_latency_s", 0.), ("max_plan_age_s", .1),
])
def test_run_virtual_time_requires_timing_in_settings(setting, value):
    plant = Plant()
    with pytest.raises(TypeError, match=setting):
        run_virtual_time(plant, Mpc(), Task(), settings=VirtualTimeSettings.ideal(),
                         duration=.01, **{setting: value}, law=SampleKnot())
    assert not plant.calls


@pytest.mark.parametrize("loop, kw", [(run_virtual_time, dict(solve_latency_s=0.)), (run_real_time, {})])
def test_a_run_longer_than_max_runtime_raises_before_it_starts_the_plant(wall, monkeypatch, loop, kw):
    monkeypatch.setattr(loops, "MAX_RUNTIME_MIN", .001)  # 60 ms: a broken check costs a 0.1 s run
    plant = Plant()
    with pytest.raises(ValueError, match="Not validated for long runs"):
        go(Mpc(), loop=loop, duration=.1, plant=plant, **kw)
    assert not plant.calls


# Wall time (run_real_time)

def test_no_command_before_the_first_plan_and_each_plan_from_the_tick_after_it_arrives(wall):
    diagnostics = []
    mpc = Mpc(delay=2)  # arrives at the third poll after its request
    _, ticks, outcome = go(mpc, diagnostics=diagnostics, loop=run_real_time, duration=.06)
    assert [k.command is None for k in ticks[:4]] == [True, True, True, False]
    assert commanded(ticks)[0] == (.03, 4.)
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .03]
    assert [round(k.t, 6) for k in ticks if k.update] == [0., .03]  # requested after the tick callbacks
    assert outcome.status == "timeout" and not diagnostics_of(diagnostics, "late_tick")


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
    diagnostics = []
    _, ticks, _ = go(Mpc(), diagnostics=diagnostics, loop=run_real_time, plant=Late(wall), duration=.07)
    assert [round(k.t, 6) for k in ticks] == [0., .01, .025, .03, .055, .065, .075]
    late, = diagnostics_of(diagnostics, "late_tick")  # 5 ms is not late
    assert late.t == pytest.approx(.055) and late.data == dict(late_s=pytest.approx(.015), skipped=1)
    assert "15.0 ms after its deadline" in late.message


def test_the_tick_after_the_first_command_is_late_without_an_event(wall):
    diagnostics = []
    class Activating(Plant):  # the robot's first command switches controllers: 25 ms
        def apply(self, command):
            super().apply(command)
            wall.now += .025 if len(self.applied) == 1 else 0.

    _, ticks, _ = go(Mpc(), diagnostics=diagnostics, loop=run_real_time, plant=Activating(), duration=.05)
    assert [round(k.t, 6) for k in ticks][:4] == [0., .01, .035, .045] and not diagnostics_of(diagnostics, "late_tick")

    class Both(Activating, Late):  # and 15 ms late at 65 ms: that one is an event
        pass

    wall.now = 0.
    diagnostics.clear()
    _, ticks, _ = go(Mpc(), diagnostics=diagnostics, loop=run_real_time, plant=Both(wall, {.065: .08}), duration=.1)
    assert [round(e.t, 6) for e in diagnostics_of(diagnostics, "late_tick")] == [.08]


def test_past_the_plans_end_faults_and_stops(wall):
    plant, ticks = Plant(), []
    outcome = run_real_time(plant, Mpc(N=4, replies=1), Task(), HoldFirst(), tick_callbacks=[ticks.append],
                            settings=RealTimeSettings(.01, .01), compensation=None, duration=1.)
    assert outcome.status == "fault" and outcome.reason == "lease_expired"  # 30 ms plan: lease ends with it
    assert [round(t, 6) for t, _ in plant.applied] == [.01, .02]  # from the tick after it arrived
    assert plant.calls == ["start", "stop"] and round(ticks[-1].t, 6) == .03 and ticks[-1].command is None


def test_a_dead_mpc_process_is_noticed_within_one_tick_and_stops_the_plant(wall):
    mpc, plant = Mpc(), Plant()

    def die(tick):
        mpc.dead = tick.t >= .02

    with pytest.raises(RuntimeError, match="exited"):
        run_real_time(plant, mpc, Task(), tick_callbacks=[die], settings=RealTimeSettings(.01, .05), compensation=None,
                      duration=1., law=SampleKnot())
    assert plant.calls == ["start", "stop"] and wall.now == pytest.approx(.03)


class Blocking(Mpc):
    """In-process on wall time: request k takes solves[k] seconds (the last one repeats)."""
    def __init__(self, wall, solves, **kw):
        super().__init__(**kw)
        self.wall, self.solves = wall, solves

    def request(self, state, reference=None, updates=None):
        super().request(state, reference, updates)
        self.wall.now += self.solves[min(len(self.requests), len(self.solves)) - 1]


def test_a_blocking_solve_ends_the_run_or_the_lease_at_the_tick_after_it(wall):
    """A 120 ms solve, requested at 0, outlasts the 50 ms run and the plan's 100 ms lease."""
    diagnostics = []
    _, ticks, outcome = go(Blocking(wall, [.12]), diagnostics=diagnostics, loop=run_real_time, duration=.05)
    assert outcome.status == "timeout" and len(diagnostics_of(diagnostics, "late_tick")) == 1
    assert [(round(k.t, 6), k.command) for k in ticks] == [(0., None), (.12, None)]
    diagnostics.clear()
    plant = Plant()  # a 310 ms plan: its lease ends first
    _, ticks, outcome = go(Blocking(wall, [.12], N=32), diagnostics=diagnostics, loop=run_real_time, plant=plant, duration=1.)
    assert outcome.reason == "lease_expired" and len(diagnostics_of(diagnostics, "late_tick")) == 1
    assert not plant.applied and plant.calls == ["start", "stop"]


def test_after_a_short_blocking_solve_the_next_tick_is_behind_but_not_late(wall):
    """One 12 ms solve, requested at 0: the tick due at 10 ms runs at 12 ms with knot 1,
    2 ms behind but within its period, so the ticks keep their times."""
    diagnostics = []
    plant = Plant()
    _, ticks, _ = go(Blocking(wall, [.012]), diagnostics=diagnostics, loop=run_real_time, plant=plant, duration=.03, update_period=1.)
    assert commanded(ticks) == [(.012, 2.), (.02, 3.)] and not diagnostics_of(diagnostics, "late_tick")


def test_a_blocking_solve_comes_after_the_command_so_no_command_goes_out_past_its_lease(wall):
    """The plan of the request at 0 takes effect at 10 ms; the request there blocks for
    120 ms, past that plan's lease (100 ms). Its command went out before, at 10 ms; the
    next tick, at 130 ms, faults on the next plan's lease (10 ms + 100 ms)."""
    diagnostics = []
    plant = Plant()
    _, ticks, outcome = go(Blocking(wall, [0., .12], N=32), diagnostics=diagnostics, loop=run_real_time, plant=plant, duration=1.)
    assert outcome.reason == "lease_expired" and not diagnostics_of(diagnostics, "late_tick")  # late after the first command
    assert [round(t, 6) for t, _ in plant.applied] == [.01]


def test_update_period_0_on_wall_time_requests_at_the_tick_each_plan_is_used(wall):
    mpc = Mpc(delay=1)  # arrives at the second poll after its request
    go(mpc, loop=run_real_time, duration=.06, update_period=0.)
    assert [round(t, 6) for t, _ in mpc.requests] == [0., .02, .04]


def test_the_warmup_runs_once_before_the_first_tick_and_counts_against_no_lease(wall):
    """A 150 ms warm-up on wall time; the first plan starts at the state after it."""
    class Warming(Mpc):
        def warmup(self, state, reference=None):
            super().warmup(state, reference)
            wall.now += .15

    plant, mpc = Plant(), Warming(N=32)
    _, ticks, _ = go(mpc, loop=run_real_time, plant=plant, duration=.05)
    assert mpc.warmups == [0.] and [round(t, 6) for t, _ in mpc.requests][:2] == [.15, .16]
    assert commanded(ticks)[0] == (.16, 2.) and plant.calls == ["start", "stop"]


# Both

def test_schedule_skips_missed_slots_without_shifting_its_phase():
    updates = Schedule(.01, start_at=5.)
    for past, next_at in ((5., 5.01), (5.035, 5.04), (5.0799999995, 5.09)):
        updates.advance(past=past)
        assert updates.next_at == pytest.approx(next_at, abs=1e-12)
    updates.advance()
    assert updates.next_at == pytest.approx(5.1, abs=1e-12)


@pytest.mark.parametrize('origin', [0., 1e6])
def test_schedule_alignment_uses_the_rolling_cursor_and_absolute_tolerance_without_consuming_it(origin):
    period = .002
    commands = Schedule(period, start_at=origin)
    for _ in range(137):  # The real command grid accumulates floating-point rounding.
        commands.advance()
    eligible = commands.next_at
    for target, expected in ((eligible-.001, eligible),
                             (eligible+.0001, eligible+period),
                             (eligible+period+.5e-9, eligible+period),
                             (eligible+period+2e-9, eligible+2*period)):
        assert commands.align(target) == expected
    assert commands.next_at == eligible


def test_zero_period_schedule_stays_ready_and_alignment_does_not_advance_it():
    updates = Schedule(0., start_at=5.)
    updates.advance()
    assert updates.next_at == 5.
    updates.advance(past=5.004)
    updates.advance(past=5.)
    assert updates.next_at == 5.004
    assert updates.align(5.002) == 5.004
    assert updates.align(5.02) == 5.02
    assert updates.next_at == 5.004


@pytest.mark.parametrize("compensation", [None, LatencyCompensation(latency_s=.01)])
def test_real_time_uses_and_replaces_a_plan_with_an_unused_excessive_future_knot(wall, compensation):
    mpc, plant, ticks = Mpc(delay=1, over=2), Plant(), []
    outcome = run_real_time(plant, mpc, Task(), tick_callbacks=[ticks.append],
                            settings=RealTimeSettings(.01, .01), compensation=compensation, duration=.08, law=SampleKnot())
    assert outcome.status == "timeout" and outcome.reason is None
    selected = next(k.plan for k in ticks if k.plan_activated and k.plan.torques[0, 1] == 0.)
    assert selected.torques[-1, 6] == 26.1
    assert [(round(k.t, 6), k.command[1]) for k in ticks if k.plan is selected] == [(.04, 0.), (.05, 0.)]
    assert any(k.command is not None and k.plan is not selected for k in ticks if k.t > .05)
    assert all(command[6] == 1. for _, command in plant.applied)
    assert plant.calls == ["start", "stop"]


@pytest.mark.parametrize("compensation", [None, LatencyCompensation(latency_s=.01)])
@pytest.mark.parametrize("transform", [False, True])
def test_real_time_checks_the_selected_command_after_its_law(wall, compensation, transform):
    def scaled(plan, t, state):
        command = SampleKnot()(plan, t, state)
        return Command(command.requested * .5, command.applied * .5)

    plant, ticks = Plant(), []
    origin = 0. if compensation is None else .01
    outcome = run_real_time(plant, Mpc(N=4, replies=1, over=1), Task(),
                            scaled if transform else SampleKnot(), [ticks.append],
                            settings=RealTimeSettings(.01, 1.), compensation=compensation, duration=origin + .03)
    assert any(k.plan is not None and k.plan.torques[-1, 6] == 26.1 for k in ticks)
    assert plant.calls == ["start", "stop"]
    selected = next(k for k in ticks if abs(k.t - origin - .02) < 1e-9)
    if transform:
        assert outcome.status == "timeout" and outcome.reason is None
        assert selected.command[6] == pytest.approx(13.05)
        assert plant.applied[-1][0] == selected.t
    else:
        assert outcome.status == "fault" and outcome.reason == "effort_limit"
        assert ticks[-1] is selected and selected.command is None
        assert selected.requested_command[6] == 26.1
        assert all(t < selected.t for t, _ in plant.applied)


@pytest.mark.parametrize("loop, kw", [
    (run_virtual_time, dict(settings=VirtualTimeSettings(.01, .01, 0.))),
    (run_real_time, dict(settings=RealTimeSettings(.01, .01))),
])
def test_a_stop_request_ends_the_run_without_a_later_command(wall, loop, kw):
    def stop(tick):
        if round(tick.t, 6) == .02:
            STOP.set()

    plant, ticks = Plant(), []
    outcome = loop(plant, Mpc(), Task(), tick_callbacks=[stop, ticks.append], duration=1., **kw, law=SampleKnot())
    assert outcome.status == "interrupted" and outcome.reason == "stop_requested"
    assert all(t <= .02 for t, _ in plant.applied)
    assert plant.calls == ["start", "stop"]


def test_plant_diagnostics_arrive_before_the_next_tick_once(wall):
    class Emitting(Plant):
        def advance(self, t):
            super().advance(t)
            self.diagnostics.append(Diagnostic(t, "pacing_behind", "behind", dict(behind_s=.001)))

    plant, ticks, diagnostics, order = Emitting(), [], [], []

    def diagnostic(value):
        diagnostics.append(value)
        order.append(('diagnostic', value.t))

    def tick(value):
        ticks.append(value)
        order.append(('tick', value.t))

    run_virtual_time(plant, Mpc(), Task(), tick_callbacks=[tick], diagnostic_callbacks=[diagnostic],
                     settings=VirtualTimeSettings(.01, .01, 0.), duration=.02, law=SampleKnot(), compensation=None)
    assert [value.t for value in diagnostics] == pytest.approx([.01, .02])
    assert order == [('tick', 0.), ('diagnostic', .01), ('tick', .01),
                     ('diagnostic', pytest.approx(.02)), ('tick', pytest.approx(.02))]
    assert all(isinstance(value, Diagnostic) for value in diagnostics)
    assert plant.take_diagnostics() == []


def test_timing_presets_and_explicit_loop_selection(wall):
    assert VirtualTimeSettings.ideal() == VirtualTimeSettings(.002, .01, 0.)
    assert VirtualTimeSettings.constant_latency() == VirtualTimeSettings(.002, .01, .05)
    assert VirtualTimeSettings.constant_latency(.02, update_period=.02) == VirtualTimeSettings(.002, .02, .02)
    assert VirtualTimeSettings.as_fast_as_possible() == VirtualTimeSettings(.002, 0., "measured")
    assert VirtualTimeSettings.constant_rate() == VirtualTimeSettings(.002, .01, .01)
    assert RealTimeSettings() == RealTimeSettings(.002, .01)
    assert VirtualTimeSettings.like(RealTimeSettings()) == VirtualTimeSettings(.002, .01, .01)
    for timing, clock in ((VirtualTimeSettings(.01, .01, .025), "VirtualClock"), (RealTimeSettings(.01, .01), "WallClock")):
        plant, ticks = Plant(), []
        loop = run_real_time if isinstance(timing, RealTimeSettings) else run_virtual_time
        outcome = loop(plant, Mpc(delay=2), Task(), HoldFirst(), [ticks.append], settings=timing,
                       compensation=None, duration=.08)
        assert outcome.status == "timeout" and len(ticks) == 9 and type(plant.clock).__name__ == clock
        assert [round(k.t, 6) for k in ticks if k.plan_activated][0] == .03  # 25 ms declared; the third poll


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
    run_virtual_time(plant, Mpc(N=32, dt=.002), task, tick_callbacks=[record],
                     solve_completed_callbacks=[record.solve_completed],
                     settings=VirtualTimeSettings(.002, .002, 0.), duration=2., law=SampleKnot(), compensation=None)
    body = np.diff(stamps) * 1e3
    print(f"tick body ms: median {np.median(body):.3f}, p99 {np.percentile(body, 99):.3f}, max {body.max():.3f}")
    assert len(body) == 999 and np.isfinite(record.ticks["command"][:-1]).all()
    # One late tick is allowed for a loaded machine; a full garbage collection gives several.
    assert np.median(body) < .5 and np.percentile(body, 99) < 1. and np.sum(body > 2.) <= 1


def test_compensated_plans_take_effect_at_a_tick_a_rounding_error_before_their_origin():
    """50 Hz control updates with a fixed 20 ms forecast on virtual time: after about 2 s of summed
    2 ms ticks, the tick at a predicted origin t + 20 ms lands about 2e-15 s before it, and its plan
    still takes effect there."""
    timing, ticks = VirtualTimeSettings(.002, .02, .02), []
    outcome = run_virtual_time(Plant(), Mpc(N=32, dt=.03), Task(), SampleKnot(), [ticks.append], settings=timing,
                               compensation=LatencyCompensation(latency_s=.02), duration=2.2)
    assert outcome.status == "timeout" and sum(k.plan_activated for k in ticks) == 110
