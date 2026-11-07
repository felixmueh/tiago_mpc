"""Shared simulation laws, observation and cancellation; no compiled solver."""
import gc
import threading
from dataclasses import replace

import numpy as np
import pytest

import tiago_mpc as tm
from tiago_mpc import loops
from tiago_mpc.prediction import _Predictor
from test_loops import Mpc, Plant, Task, wall


SETTINGS = tm.VirtualTimeSettings(.002, .002, 0.)


def run(mpc=None, plant=None, law=None, *, settings=SETTINGS, duration=.01, **options):
    args = () if law is None else (law,)
    return tm.run_virtual_time(plant or Plant(), mpc or Mpc(), Task(), *args,
                               settings=settings, duration=duration, **options)


@pytest.mark.parametrize('law', [tm.SampleKnot(), tm.PlanFeedback(kp=0., kd=0.)])
def test_simulation_keeps_over_limit_predictions_and_clips_explicit_laws(law):
    class Excess(Mpc):
        def result(self, block=False):
            plan = super().result(block)
            return None if plan is None else tm.Plan(plan.t0, plan.dt, plan.states,
                np.full_like(plan.torques, 1000.), plan.stats, measured_t=plan.measured_t)

    plant, messages = Plant(), []
    outcome = run(Excess(), plant, law, tick_callbacks=[messages.append],
                  solve_completed_callbacks=[messages.append])
    assert outcome.status == 'timeout'
    solves = [message for message in messages if isinstance(message, tm.SolveCompleted)]
    ticks = [message for message in messages if isinstance(message, tm.Tick) and message.command is not None]
    assert solves and np.all(solves[0].plan.torques == 1000.)
    assert ticks and all(tick.clipped for tick in ticks)
    for tick in ticks:
        np.testing.assert_array_equal(tick.requested_command, np.full(7, 1000.))
        np.testing.assert_array_equal(tick.command, tm.limits().effort)
    assert all(np.all(command <= tm.limits().effort) for _, command in plant.applied)


def test_reject_on_effort_is_a_normal_command_refusal_with_recorded_request():
    messages, plant = [], Plant()
    outcome = run(plant=plant, law=lambda plan, t, state: tm.Command(np.full(7, 1000.), np.full(7, 1000.)),
                  effort_handling=tm.RejectOnEffort(),
                  tick_callbacks=[messages.append], diagnostic_callbacks=[messages.append])
    assert outcome.status == 'fault' and outcome.reason == 'effort_limit'
    assert not plant.applied
    tick = next(message for message in messages if isinstance(message, tm.Tick))
    assert tick.command is None
    np.testing.assert_array_equal(tick.requested_command, np.full(7, 1000.))
    event = next(message for message in messages if isinstance(message, tm.Diagnostic))
    assert event.kind == 'command_rejected' and event.data['reason'] == 'effort_limit'


@pytest.mark.parametrize('handling', [tm.EffortClip(), tm.RejectOnEffort(), None])
@pytest.mark.parametrize('bad', [np.full(7, np.inf), np.full(7, np.nan), np.ones(6)])
def test_effort_handling_never_conceals_invalid_commands(handling, bad):
    plant, ticks = Plant(), []
    outcome = run(plant=plant, law=lambda plan, t, state: tm.Command(bad, bad),
                  effort_handling=handling, tick_callbacks=[ticks.append])
    assert outcome.status == 'fault' and outcome.reason == 'invalid_command'
    assert not plant.applied and ticks[-1].command is None


def test_clipping_limits_feedback_and_feedforward_together():
    class Reference(Mpc):
        def result(self, block=False):
            plan = super().result(block)
            return None if plan is None else tm.Plan(plan.t0, plan.dt, np.ones_like(plan.states),
                                                     plan.torques, plan.stats, measured_t=plan.measured_t)

    mpc, ticks = Reference(), []
    mpc.params = tm.TIAGO
    law = tm.FrictionFeedforward(tm.Friction((1.,)*7, (1.,)*7), tm.PlanFeedback(kp=10., kd=10.))
    with pytest.warns(RuntimeWarning, match='without latency compensation'):
        outcome = run(mpc=mpc, law=law, effort_handling=tm.EffortClip(effort=(2.,)*7),
                      tick_callbacks=[ticks.append], duration=.002)
    assert outcome.status == 'timeout'
    assert np.all(ticks[0].requested_command > 2.) and ticks[0].clipped
    np.testing.assert_array_equal(ticks[0].command, np.full(7, 2.))


@pytest.mark.parametrize('handling, expected', [
    (tm.EffortClip(), tm.limits().effort),
    (None, np.full(7, 1000.)),
])
def test_virtual_loop_applies_explicit_effort_policy_to_explicit_law(handling, expected):
    plant, ticks = Plant(), []
    outcome = tm.run_virtual_time(plant, Mpc(), Task(),
        lambda plan, t, state: tm.Command(np.full(7, 1000.), np.full(7, 1000.)),
        [ticks.append], settings=SETTINGS, effort_handling=handling, duration=.004)
    assert outcome.status == 'timeout' and plant.applied
    for tick in ticks:
        if tick.command is not None:
            np.testing.assert_array_equal(tick.command, expected)
            np.testing.assert_array_equal(tick.requested_command, np.full(7, 1000.))
            assert tick.clipped == (handling is not None)


@pytest.mark.parametrize('handling, applied', [
    (tm.EffortClip(effort=(2.,)*7), 2.),
    (tm.RejectOnEffort(effort=(2.,)*7), None),
    (None, 1000.),
])
def test_forecasting_uses_the_same_effort_policy_and_keeps_refusal_predictive(monkeypatch, handling, applied):
    from tiago_mpc import prediction
    monkeypatch.setattr(prediction, 'plant_euler',
                        lambda model, data, q, v, u, h, friction, law: (q+u*h, v+u*h))
    predictor = _Predictor(tm.LatencyCompensation(), safety=tm.ExecutionSafety(None), effort_handling=handling)
    initial = tm.State(0., np.zeros(7), np.zeros(7))
    plan = tm.Plan(0., .03, np.zeros((32, 14)), np.ones((31, 7)), measured_t=0.)
    base = lambda plan, t, state: tm.Command(np.full(7, 1000.), np.full(7, 1000.))
    predicted = predictor(initial, 0., .006, plan, base,
                          np.full(7, 2. if applied is None else applied), .002, .002)
    if applied is None:
        assert predicted is None
    else:
        np.testing.assert_allclose(predicted.q, .006*applied)
        np.testing.assert_allclose(predicted.qd, .006*applied)
    np.testing.assert_array_equal(initial.q, np.zeros(7))


def test_future_refusal_does_not_emit_an_actual_command_rejection():
    settings = replace(SETTINGS, update_period=.008)
    messages, plant = [], Plant()
    law = lambda plan, t, state: tm.Command(np.full(7, 1. if t < .009 else 1000.), np.full(7, 1. if t < .009 else 1000.))
    outcome = run(plant=plant, law=law, settings=settings, duration=.02,
                  compensation=tm.LatencyCompensation(latency_s=.004),
                  effort_handling=tm.RejectOnEffort(effort=(2.,)*7),
                  tick_callbacks=[messages.append], diagnostic_callbacks=[messages.append])
    assert outcome.status == 'fault' and outcome.reason == 'prediction_unavailable'
    assert outcome.t == pytest.approx(.008)
    assert plant.applied and all(np.all(command == 1.) for _, command in plant.applied)
    assert not any(isinstance(message, tm.Diagnostic) and message.kind == 'command_rejected'
                   for message in messages)
    tick = [message for message in messages if isinstance(message, tm.Tick)][-1]
    np.testing.assert_array_equal(tick.requested_command, np.ones(7))


def test_forecasting_does_not_copy_a_pure_callable_with_a_lock(monkeypatch):
    from tiago_mpc import prediction
    monkeypatch.setattr(prediction, 'plant_euler',
                        lambda model, data, q, v, u, h, friction, law: (q+u*h, v+u*h))

    class Law:
        def __init__(self):
            self.lock = threading.Lock()

        def __call__(self, plan, t, state):
            with self.lock:
                return tm.Command(plan.torques[0], plan.torques[0])

    predictor = _Predictor(tm.LatencyCompensation(), safety=tm.ExecutionSafety(None))
    initial = tm.State(0., np.zeros(7), np.zeros(7))
    plan = tm.Plan(0., .01, np.zeros((4, 14)), np.ones((3, 7)), measured_t=0.)
    predicted = predictor(initial, 0., .006, plan, Law(), np.ones(7), .002, .002)
    np.testing.assert_allclose(predicted.q, .006)
    np.testing.assert_allclose(predicted.qd, .006)


def test_solve_messages_keep_unused_plans_and_exact_reference_snapshots():
    class Reference(Task):
        def __init__(self): self.reference = np.zeros((11, 6))
        def at(self, t, n, dt):
            self.reference[:] = t
            return self.reference, dict(external_force=np.full(6, t))

    messages, ticks, task = [], [], Reference()
    tm.run_virtual_time(Plant(), Mpc(), task, tick_callbacks=[ticks.append],
        solve_completed_callbacks=[messages.append], settings=replace(SETTINGS, solve_latency_s=.006), duration=.008)
    solves = [message for message in messages if isinstance(message, tm.SolveCompleted)]
    active = {id(tick.plan) for tick in ticks if tick.command is not None}
    assert len(solves) == 2 and sum(id(message.plan) not in active for message in solves) == 1
    np.testing.assert_array_equal(solves[0].reference, np.zeros((11, 6)))
    assert not solves[0].reference.flags.writeable
    assert solves[1].request_time_s == pytest.approx(.006)
    np.testing.assert_array_equal(solves[1].updates['external_force'], np.full(6, .006))


@pytest.mark.parametrize('loop, settings', [
    (tm.run_virtual_time, SETTINGS),
    (tm.run_real_time, tm.RealTimeSettings(.002, .002)),
])
def test_each_notification_stream_receives_its_payload_once(loop, settings, wall):
    class Emitting(Plant):
        def advance(self, t):
            super().advance(t)
            self.diagnostics.append(tm.Diagnostic(t, 'physics', 'integrated'))

    plant, mpc, ticks, solves, diagnostics = Emitting(), Mpc(), [], [], []

    def tick(value):
        assert isinstance(value, tm.Tick)
        ticks.append(value)

    def solved(value):
        assert isinstance(value, tm.SolveCompleted)
        solves.append(value)

    def diagnostic(value):
        assert isinstance(value, tm.Diagnostic)
        diagnostics.append(value)

    outcome = loop(plant, mpc, Task(), settings=settings, compensation=None, duration=.006,
                   tick_callbacks=[tick], solve_completed_callbacks=[solved], diagnostic_callbacks=[diagnostic])
    assert outcome.status == 'timeout'
    assert [value.t for value in ticks] == pytest.approx([0., .002, .004, .006])
    assert [value.request_time_s for value in solves] == pytest.approx([0., .002, .004])
    assert [value.t for value in diagnostics if value.kind == 'physics'] == pytest.approx([.002, .004, .006])
    assert len({id(value) for value in [*ticks, *solves, *diagnostics]}) == len(ticks)+len(solves)+len(diagnostics)
    for value, initial in zip(solves, mpc.requested_states):
        assert value.initial is initial and value.plan.t0 == initial.t
        assert not value.initial.q.flags.writeable and not value.reference.flags.writeable
        np.testing.assert_array_equal(value.reference, np.zeros((11, 6)))
        assert value.updates is None
    assert plant.take_diagnostics() == []


@pytest.mark.parametrize('loop, settings', [
    (tm.run_virtual_time, SETTINGS),
    (tm.run_real_time, tm.RealTimeSettings(.002, .002)),
])
def test_requests_own_stable_snapshots_without_observers(loop, settings, wall):
    class MutablePlant(Plant):
        def __init__(self):
            super().__init__()
            self.q, self.qd = np.zeros(7), np.zeros(7)
            self.positions = {'joint': 0.}

        def state(self):
            return tm.State(self.clock.now(), self.q, self.qd, self.positions)

    class MutableTask(Task):
        def __init__(self):
            self.reference = np.zeros((11, 6))
            self.updates = {'external_force': np.zeros(6), 'nested': {'values': []}}

        def at(self, t, n, dt):
            self.reference.fill(t)
            self.updates['external_force'].fill(t)
            return self.reference, self.updates

    plant, task, submitted = MutablePlant(), MutableTask(), []

    class Capturing(Mpc):
        def request(self, state, reference=None, updates=None):
            assert not state.q.flags.writeable and not state.qd.flags.writeable
            assert not reference.flags.writeable
            submitted.append((state, reference, updates))
            plant.q += 1.
            plant.qd += 2.
            plant.positions['joint'] += 1.
            task.reference.fill(100.)
            task.updates['external_force'].fill(200.)
            task.updates['nested']['values'].append(len(submitted))
            super().request(state, reference, updates)

    outcome = loop(plant, Capturing(), task, settings=settings, compensation=None, duration=.006)
    assert outcome.status == 'timeout' and len(submitted) == 3
    for index, (state, reference, updates) in enumerate(submitted):
        np.testing.assert_array_equal(state.q, np.full(7, index))
        np.testing.assert_array_equal(state.qd, np.full(7, 2*index))
        assert state.joint_positions == {'joint': index}
        np.testing.assert_array_equal(reference, np.full((11, 6), state.t))
        np.testing.assert_array_equal(updates['external_force'], np.full(6, state.t))
        assert updates['nested']['values'] == list(range(1, index+1))


@pytest.mark.parametrize('real', [False, True])
def test_each_actual_request_forecasts_once_while_an_early_plan_waits(monkeypatch, wall, real):
    forecasts = []

    def predict(self, state, now, at, *args):
        forecasts.append((now, at))
        return tm.Prediction(at, state.q, state.qd, measured_t=state.t)

    class CheckingMpc(Mpc):
        def request(self, state, reference=None, updates=None):
            assert len(forecasts) == len(self.requests)+1
            assert forecasts[-1][1] == state.t
            super().request(state, reference, updates)

    monkeypatch.setattr(_Predictor, '__call__', predict)
    plant, mpc, ticks = Plant(), CheckingMpc(), []
    loop = tm.run_real_time if real else tm.run_virtual_time
    settings = tm.RealTimeSettings(.002, 0.) if real else tm.VirtualTimeSettings(.002, 0., .002)
    outcome = loop(plant, mpc, Task(), settings=settings, duration=.018,
                   compensation=tm.LatencyCompensation(latency_s=.006), tick_callbacks=[ticks.append])
    assert outcome.status == 'timeout'
    assert len(forecasts) == len(mpc.requests) == 3
    assert [now for now, _ in forecasts] == pytest.approx([0., .006, .012])
    assert [at for _, at in forecasts] == pytest.approx([.006, .012, .018])
    assert all(tick.command is None for tick in ticks if tick.t < .006)


def test_real_observers_follow_publication(wall):
    order = []

    class Emitting(Plant):
        def apply(self, command):
            super().apply(command)
            order.append(('apply', self.clock.now()))
            self.diagnostics.append(tm.Diagnostic(self.clock.now(), 'activated', 'activated'))

    def observer(value):
        order.append((type(value).__name__, wall.now))
        wall.now += .12

    plant = Emitting()
    outcome = tm.run_real_time(plant, Mpc(), Task(), settings=tm.RealTimeSettings(.002, 1.),
        compensation=tm.LatencyCompensation(latency_s=.002), duration=1.,
        solve_completed_callbacks=[observer], diagnostic_callbacks=[observer])
    assert outcome.reason == 'lease_expired'
    assert order[:3] == [('apply', .002), ('SolveCompleted', .002), ('Diagnostic', pytest.approx(.122))]
    assert [t for t, _ in plant.applied] == [.002]


@pytest.mark.parametrize('where', ['result', 'solve_message', 'tick_message', 'tick_callback'])
def test_cancellation_during_a_tick_finishes_its_command_and_physics(where):
    cancelled, plant, messages, boundaries = threading.Event(), Plant(), [], []

    class Cancelling(Mpc):
        def result(self, block=False):
            result = super().result(block)
            if where == 'result':
                cancelled.set()
            return result

    def callback(message):
        messages.append(message)
        if ((where == 'solve_message' and isinstance(message, tm.SolveCompleted)) or
                (where == 'tick_message' and isinstance(message, tm.Tick))):
            cancelled.set()

    def tick_callback(tick):
        if where == 'tick_callback':
            cancelled.set()

    outcome = run(Cancelling(), plant, cancelled=cancelled.is_set, before_tick_callbacks=[boundaries.append],
                  solve_completed_callbacks=[callback], tick_callbacks=[callback, tick_callback])
    assert outcome == tm.Outcome('interrupted', 'stop_requested', .002)
    assert [state.t for state in boundaries] == [0., .002]
    assert [t for t, _ in plant.applied] == [0.] and plant.clock.now() == .002
    assert [type(message) for message in messages] == [tm.SolveCompleted, tm.Tick]
    assert messages[-1].command is not None and plant.calls == ['start', 'stop']


def test_before_tick_callbacks_run_in_order_before_cancellation_prevents_work():
    calls, advances, cancelled = [], [], threading.Event()

    class TracedPlant(Plant):
        def advance(self, t):
            advances.append(t)
            super().advance(t)

    def first(state):
        calls.append(('first', state))
        cancelled.set()

    def second(state):
        calls.append(('second', state))

    plant, mpc = TracedPlant(), Mpc()
    outcome = run(mpc, plant, before_tick_callbacks=[first, second], cancelled=cancelled.is_set)
    assert outcome == tm.Outcome('interrupted', 'stop_requested', 0.)
    assert [name for name, _ in calls] == ['first', 'second']
    assert calls[0][1] is calls[1][1] and isinstance(calls[0][1], tm.State) and calls[0][1].t == 0.
    assert not mpc.requests and not plant.applied and not advances
    assert plant.calls == ['start', 'stop']


def test_paused_callback_wakes_for_cancellation_without_an_extra_command():
    parked, wake, cancelled = threading.Event(), threading.Event(), threading.Event()
    plant, mpc, outcome = Plant(), Mpc(), []

    def before(state):
        if abs(state.t-.006) < 1e-12:
            parked.set()
            assert wake.wait(5)

    thread = threading.Thread(target=lambda: outcome.append(run(mpc, plant, duration=1.,
        before_tick_callbacks=[before], cancelled=cancelled.is_set)))
    thread.start()
    try:
        assert parked.wait(5)
        before_count = len(mpc.requests), len(plant.applied)
        cancelled.set()
        wake.set()
        thread.join(5)
        assert not thread.is_alive()
        assert (len(mpc.requests), len(plant.applied)) == before_count
        assert mpc.warmups == [0.] and plant.clock.now() == pytest.approx(.006)
        assert outcome[0].status == 'interrupted'
    finally:
        cancelled.set()
        wake.set()
        thread.join(5)


@pytest.mark.parametrize('enabled', [False, True])
def test_local_runtime_limit_override_keeps_gc_disabled_and_restores_its_policy(monkeypatch, enabled):
    was_enabled = gc.isenabled()
    gc.enable() if enabled else gc.disable()
    monkeypatch.setattr(loops, 'MAX_RUNTIME_MIN', 20)
    policies = []
    try:
        outcome = run(Mpc(N=32, dt=100.), duration=700.,
            settings=tm.VirtualTimeSettings(100., 0., 0.), safety=tm.ExecutionSafety(None),
            before_tick_callbacks=[lambda state: policies.append(gc.isenabled())])
        assert outcome == tm.Outcome('timeout', None, 700.)
        assert policies == [False]*8 and gc.isenabled() == enabled
    finally:
        gc.enable() if was_enabled else gc.disable()


def test_virtual_gc_policy_is_restored_when_a_callback_raises():
    was_enabled = gc.isenabled()
    gc.enable()

    def fail(tick):
        assert not gc.isenabled()
        raise RuntimeError('observer failed')

    plant = Plant()
    try:
        with pytest.raises(RuntimeError, match='observer failed'):
            run(plant=plant, tick_callbacks=[fail])
        assert gc.isenabled() and plant.calls == ['start', 'stop']
    finally:
        gc.enable() if was_enabled else gc.disable()


@pytest.mark.parametrize('age, end, reason', [(.1, .1, 'lease_expired'),
                                             (.006, .006, 'lease_expired'), (None, .12, None)])
def test_plan_age_is_configurable_independently_of_long_horizon(age, end, reason):
    outcome = run(Mpc(N=32, dt=.03), settings=replace(SETTINGS, update_period=1.), safety=tm.ExecutionSafety(age),
                  duration=.12)
    assert outcome.t == pytest.approx(end) and outcome.reason == reason


@pytest.mark.parametrize('age', [0., -1., np.inf, np.nan])
def test_invalid_plan_age_is_rejected_before_start(age):
    plant = Plant()
    with pytest.raises(ValueError, match='max_plan_age_s'):
        run(plant=plant, safety=tm.ExecutionSafety(age))
    assert not plant.calls


def test_anchor_forecast_uses_the_actual_initial_held_command():
    plant = tm.SimConfig(integrator='euler').build()
    mpc, boundaries = Mpc(), []
    settings = replace(SETTINGS, solve_latency_s=.002)
    outcome = run(mpc, plant, duration=.004, settings=settings,
        compensation=tm.LatencyCompensation(latency_s=.002),
        start_time_s=10., initial_command=np.zeros(7), before_tick_callbacks=[boundaries.append])
    assert outcome.status == 'timeout' and mpc.warmups == [10.]
    predicted = mpc.requested_states[0]
    assert predicted.t == pytest.approx(10.002) and predicted.measured_t == 10.
    np.testing.assert_allclose(predicted.q, boundaries[1].q, atol=1e-12)
    np.testing.assert_allclose(predicted.qd, boundaries[1].qd, atol=1e-12)


def test_physics_diagnostics_keep_detection_time_and_arrive_before_the_next_boundary():
    plant, observed, order = tm.SimConfig(integrator='euler').build(), [], []

    def step(state):
        observed.append(state.t)
        return [tm.Diagnostic(state.t, 'limit_breach', 'detected', dict(q=state.q.tolist()))] if state.t < .0015 else []

    plant.step_callback = step

    def diagnostic(value):
        order.append(('diagnostic', value.t, plant.t))

    run(plant=plant, initial_command=np.zeros(7), duration=.004,
        diagnostic_callbacks=[diagnostic],
        before_tick_callbacks=[lambda state: order.append(('boundary', state.t, plant.t))])
    assert observed == pytest.approx([.001, .002, .003, .004])
    assert order.index(('diagnostic', .001, .002)) < order.index(('boundary', .002, .002))


def test_real_time_rejects_an_excessive_law_command_without_clipping(wall):
    plant, ticks = Plant(), []
    outcome = tm.run_real_time(plant, Mpc(), Task(), lambda plan, t, state: tm.Command(np.full(7, 1000.), np.full(7, 1000.)),
                               settings=tm.RealTimeSettings(.002, .002), compensation=None, duration=.01,
                               tick_callbacks=[ticks.append])
    assert outcome.status == 'fault' and outcome.reason == 'effort_limit'
    assert not plant.applied and plant.calls == ['start', 'stop']
    assert ticks[-1].command is None and not ticks[-1].clipped
    np.testing.assert_array_equal(ticks[-1].requested_command, np.full(7, 1000.))


@pytest.mark.parametrize('done_at, expected_status', [(float('inf'), 'timeout'), (.002, 'done')])
def test_normal_completion_collects_an_async_result_and_leaves_the_mpc_reusable(tmp_path, done_at, expected_status):
    class Deferred(Mpc):
        def warmup(self, state, reference=None):
            self.warmups.append(state.t)

        def result(self, block=False):
            assert isinstance(messages[-1], tm.Tick) and messages[-1].command is None
            return super().result(True) if block else None

    mpc = Deferred()
    for index in range(2):
        plant, messages = Plant(), []
        record = tm.RunRecord(tmp_path / str(index))
        outcome = tm.run_virtual_time(plant, mpc, Task(done_at), settings=replace(SETTINGS, solve_latency_s=.1),
            duration=.004, tick_callbacks=[record, messages.append],
            solve_completed_callbacks=[record.solve_completed, messages.append])
        assert outcome.status == expected_status and mpc.waiting is None
        assert not plant.applied and plant.calls == ['start', 'stop']
        solved = [message for message in messages if isinstance(message, tm.SolveCompleted)]
        assert len(solved) == 1 and solved[0].request_time_s == 0.
        assert messages[-1] is solved[0] and isinstance(messages[-2], tm.Tick)
        terminal = messages[-2]
        assert terminal.command is None and terminal.plan is None
        assert plant.clock.now() == terminal.t == pytest.approx(.002 if done_at == .002 else .004)
        assert [message.t for message in messages if isinstance(message, tm.Tick)] == pytest.approx(
            [0., .002] if done_at == .002 else [0., .002, .004])
        record.save()
        assert record.plans['update_t'] == [0.] and record.plans['t0'] == [0.]
        np.testing.assert_array_equal(record.plans['torques'][0], solved[0].plan.torques)
        assert np.all(record.ticks['plan'] == -1)
        loaded = tm.load_run(record.out)
        np.testing.assert_array_equal(loaded.ticks['plan'], record.ticks['plan'])
        np.testing.assert_array_equal(loaded.plans['torques'], record.plans['torques'])


def test_effort_fault_collects_the_outstanding_solve_without_applying_its_plan():
    class Deferred(Mpc):
        def warmup(self, state, reference=None):
            self.warmups.append(state.t)

        def result(self, block=False):
            return super().result(True) if block else None

    mpc = Deferred()
    for _ in range(2):
        plant, messages = Plant(), []
        outcome = run(mpc, plant, lambda plan, t, state: tm.Command(np.full(7, 1000.), np.full(7, 1000.)),
                      settings=replace(SETTINGS, solve_latency_s=.002), effort_handling=tm.RejectOnEffort(),
                      tick_callbacks=[messages.append],
                      solve_completed_callbacks=[messages.append])
        assert outcome.status == 'fault' and outcome.reason == 'effort_limit'
        assert outcome.t == pytest.approx(.002) and mpc.waiting is None
        assert not plant.applied and plant.clock.now() == pytest.approx(.002)
        solved = [message for message in messages if isinstance(message, tm.SolveCompleted)]
        assert [message.request_time_s for message in solved] == pytest.approx([0., .002])
        assert isinstance(messages[-2], tm.Tick) and messages[-2].command is None
        assert messages[-1] is solved[-1] and messages[-2].t == plant.clock.now()


@pytest.mark.parametrize('latency, wall_s, first_command, forecast', [
    (0., .001, 0., None), (.002, .001, .002, None), (.0055, .001, .006, None),
    ('measured', .003, .004, None), ('measured', 0., .002, None),
    (.002, .001, .008, .008), (0., .001, .008, .008), ('measured', .003, .008, .008),
])
def test_async_solver_waits_at_the_first_possible_activation_tick(latency, wall_s, first_command, forecast):
    plant, blocking, notified = Plant(), [], []

    class Deferred(Mpc):
        def request(self, state, reference=None, updates=None):
            self.request_t = plant.clock.now()
            super().request(state, reference, updates)

        def result(self, block=False):
            assert block and self.waiting is not None
            earliest = max(self.request_t+(.002 if latency == 'measured' else latency), self.waiting[1])
            blocking.append(plant.clock.now())
            # Normal completion collects a remaining result without advancing
            # virtual time to its later activation boundary.
            assert plant.clock.now() >= min(earliest, .012)-1e-9
            return super().result(block=True)

    def observe(message):
        if isinstance(message, tm.SolveCompleted):
            notified.append((message.request_time_s, plant.clock.now()))

    mpc = Deferred(wall=wall_s)
    outcome = run(mpc, plant, settings=replace(SETTINGS, solve_latency_s=latency), duration=.012,
                  compensation=None if forecast is None else tm.LatencyCompensation(latency_s=forecast),
                  solve_completed_callbacks=[observe])
    assert outcome.status == 'timeout'
    assert plant.applied[0][0] == pytest.approx(first_command)
    assert blocking[0] == pytest.approx(max(forecast or 0., .002 if latency == 'measured' else first_command))
    assert len(notified) == len(mpc.requests) == len(set(request_t for request_t, _ in notified))
    assert [t for _, t in notified] == blocking


def test_zero_delay_requests_once_per_boundary_and_uses_each_result_immediately():
    plant, boundaries, ticks, solves = Plant(), [], [], []

    class Distinct(Mpc):
        def request(self, state, reference=None, updates=None):
            assert len(self.requests) < len(boundaries)
            super().request(state, reference, updates)

        def result(self, block=False):
            plan = super().result(block)
            return tm.Plan(plan.t0, plan.dt, plan.states, np.full_like(plan.torques, len(self.requests)),
                           plan.stats, measured_t=plan.measured_t)

    mpc = Distinct()
    outcome = run(mpc, plant, settings=replace(SETTINGS, update_period=0.), duration=.006,
                  before_tick_callbacks=[lambda state: boundaries.append(state.t)],
                  tick_callbacks=[ticks.append], solve_completed_callbacks=[solves.append])
    assert outcome.status == 'timeout'
    assert boundaries == pytest.approx([0., .002, .004, .006])
    assert [request_t for request_t, _ in mpc.requests] == pytest.approx([0., .002, .004])
    assert [tick.t for tick in ticks] == pytest.approx(boundaries)
    assert len(solves) == len(mpc.requests) == len(plant.applied) == 3
    for index, (tick, solved) in enumerate(zip(ticks, solves), start=1):
        assert tick.plan_activated and tick.update and tick.plan is solved.plan
        assert tick.t == solved.plan.t0 == solved.request_time_s
        np.testing.assert_array_equal(tick.command, np.full(7, index))
    assert ticks[-1].command is None and not ticks[-1].update


def test_an_immediate_ideal_result_replaces_the_plan_activated_at_the_same_boundary():
    class Distinct(Mpc):
        def result(self, block=False):
            plan = super().result(block)
            return tm.Plan(plan.t0, plan.dt, plan.states, np.full_like(plan.torques, len(self.requests)),
                           plan.stats, measured_t=plan.measured_t)

    plant, messages, mpc = Plant(), [], Distinct()
    outcome = run(mpc, plant, duration=.006, compensation=tm.LatencyCompensation(initial_latency_s=.004),
                  tick_callbacks=[messages.append], solve_completed_callbacks=[messages.append])
    assert outcome.status == 'timeout'
    solves = [message for message in messages if isinstance(message, tm.SolveCompleted)]
    assert [message.request_time_s for message in solves] == [0., .004]
    assert plant.applied[0][0] == .004
    np.testing.assert_array_equal(plant.applied[0][1], np.full(7, 2.))
    tick = next(message for message in messages if isinstance(message, tm.Tick) and message.command is not None)
    assert tick.plan_activated and tick.plan is solves[1].plan
    assert solves[1].request_time_s == .004


class MovingPlant(Plant):
    """A constant-acceleration plant with one event per integrated interval."""
    def __init__(self):
        super().__init__()
        self.q, self.qd = np.linspace(.1, .7, 7), np.linspace(-.02, .02, 7)
        self.command, self.advanced = None, []

    def state(self):
        return tm.State(self.clock.now(), self.q.copy(), self.qd.copy())

    def apply(self, command):
        super().apply(command.copy())
        self.command = command.copy()

    def advance(self, t):
        h = t-self.clock.now()
        if self.command is not None:
            self.q = self.q+self.qd*h+.5*self.command*h*h
            self.qd = self.qd+self.command*h
        self.diagnostics.append(tm.Diagnostic(t, 'physics', 'integrated', dict(q=self.q.tolist())))
        self.advanced.append(t)
        super().advance(t)


class DeferredAsync(Mpc):
    """Identical synchronous or worker solves, with a reply retained until collection."""
    def __init__(self, plant, first_solve, latencies, *, asynchronous=True):
        super().__init__(N=11, dt=.004)
        self.plant, self.first_solve, self.latencies = plant, first_solve, latencies
        self.asynchronous = asynchronous
        self.workers, self.reads, self.inputs, self.warm_states = [], [], [], []
        self.ready, self.reply, self.warm_state = threading.Event(), None, 0.

    def request(self, state, reference=None, updates=None):
        super().request(state, reference, updates)
        self.inputs.append((state_trace(state), reference.tolist(), updates))
        index = len(self.requests)-1
        self.ready = threading.Event()

        def solve():
            if index == 0:
                assert self.first_solve.wait(5)
            self.warm_state += .05+.01*float(np.sum(state.q))
            self.warm_states.append(self.warm_state)
            states = np.tile(np.r_[state.q, state.qd], (self.N, 1))
            torques = .1*np.arange(1, self.N)[:, None]+self.warm_state+.1*state.q[None, :]
            self.reply = tm.Plan(state.t, self.dt, states, torques,
                dict(wall_s=self.latencies[index % len(self.latencies)], warm_start=self.warm_state),
                measured_t=state.measured_t if isinstance(state, tm.Prediction) else state.t)
            self.ready.set()

        if self.asynchronous:
            worker = threading.Thread(target=solve)
            self.workers.append(worker)
            worker.start()
        else:
            solve()

    def result(self, block=False):
        self.reads.append((self.plant.clock.now(), block))
        if self.waiting is None or not block:
            return None
        assert self.ready.wait(5)
        self.waiting = None
        reply, self.reply = self.reply, None
        return reply


def state_trace(state):
    return (type(state).__name__, state.t, getattr(state, 'measured_t', state.t),
            state.q.tolist(), state.qd.tolist())


def test_async_worker_finishes_inside_an_independent_physics_interval():
    integrating = threading.Event()

    class Integrating(MovingPlant):
        def advance(self, t):
            if not integrating.is_set():
                assert not mpc.ready.is_set() and not mpc.reads
                integrating.set()
                assert mpc.ready.wait(5)
                assert mpc.reply is not None and not mpc.reads
            super().advance(t)

    plant = Integrating()
    mpc = DeferredAsync(plant, integrating, (.001,))
    try:
        outcome = run(mpc, plant, settings=replace(SETTINGS, update_period=.02, solve_latency_s=.006),
                      duration=.008, initial_command=np.full(7, .2))
        assert outcome.status == 'timeout'
        assert mpc.reads == [(pytest.approx(.006), True)]
        assert plant.advanced == pytest.approx([.002, .004, .006, .008])
    finally:
        integrating.set()
        for worker in mpc.workers:
            worker.join(5)


def numerical_trace(
    monkeypatch,
    latency,
    native_latencies,
    compensated,
    *,
    paused=False,
    asynchronous=True,
    observed=True,
):
    from tiago_mpc import prediction
    monkeypatch.setattr(prediction, 'plant_euler', lambda model, data, q, v, u, h, friction, law:
                        (q+v*h+.5*u*h*h, v+u*h))
    first_solve, parked, wake, pause_requested = (threading.Event() for _ in range(4))
    if not paused:
        first_solve.set()
    plant, messages, ticks, boundaries, outcomes, errors = MovingPlant(), [], [], [], [], []
    mpc = DeferredAsync(plant, first_solve, native_latencies, asynchronous=asynchronous)

    def message(value):
        messages.append(value)
        if paused and isinstance(value, tm.Tick) and value.t == 0.:
            pause_requested.set()

    def before(state):
        boundaries.append(state.t)
        if pause_requested.is_set():
            pause_requested.clear()
            parked.set()
            assert wake.wait(5)

    def loop():
        try:
            outcomes.append(run(mpc, plant, duration=.024,
                settings=replace(SETTINGS, update_period=.004, solve_latency_s=latency),
                compensation=tm.LatencyCompensation(initial_latency_s=.004) if compensated else None,
                initial_command=np.full(7, .2), before_tick_callbacks=[before],
                tick_callbacks=[ticks.append]+([message] if observed else []),
                solve_completed_callbacks=[message] if observed else [],
                diagnostic_callbacks=[message] if observed else []))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=loop)
    thread.start()
    try:
        if paused:
            assert parked.wait(5)
            assert boundaries == [0., .002] and mpc.waiting is not None
            assert [value.t for value in messages if isinstance(value, tm.Diagnostic)
                    and value.kind == 'physics'] == [.002]
            snapshot = state_trace(plant.state()), len(plant.applied), len(mpc.inputs), list(mpc.reads)
            assert not mpc.ready.is_set()
            first_solve.set()
            assert mpc.ready.wait(5)  # The worker finishes while the loop stays parked.
            assert mpc.reply is not None and mpc.waiting is not None
            assert snapshot == (state_trace(plant.state()), len(plant.applied), len(mpc.inputs), mpc.reads)
            assert not any(isinstance(value, tm.SolveCompleted) for value in messages)
            wake.set()
        thread.join(5)
        assert not thread.is_alive()
        assert not errors, errors
        assert outcomes[0].status == 'timeout' and mpc.waiting is None
    finally:
        first_solve.set()
        wake.set()
        thread.join(5)
        for worker in mpc.workers:
            worker.join(5)

    solves = [value for value in messages if isinstance(value, tm.SolveCompleted)]
    diagnostics = [value for value in messages if isinstance(value, tm.Diagnostic)]
    assert boundaries == pytest.approx(np.arange(13)*.002)
    assert len(set(boundaries)) == len(boundaries) and mpc.warmups == [0.]
    assert len(mpc.inputs) == len(mpc.warm_states) > 2
    assert all(a < b for a, b in zip(mpc.warm_states, mpc.warm_states[1:]))
    assert len(plant.advanced) == 12
    if observed:
        assert [value for value in messages if isinstance(value, tm.Tick)] == ticks
        assert len(solves) == len(mpc.inputs)
        assert len({id(value) for value in messages}) == len(messages)
        assert len([event for event in diagnostics if event.kind == 'physics']) == len(plant.advanced)
        overruns = [event for event in diagnostics if event.kind == 'overrun']
        assert len(overruns) == (len(solves) if latency == .002 else 0)
        assert len({event.data['plan'] for event in overruns}) == len(overruns)
    else:
        assert not messages
    assert any(np.any(tick.state.qd != ticks[0].state.qd) for tick in ticks[1:])
    if compensated:
        assert all(isinstance(value, tm.Prediction) for value in mpc.requested_states)
        assert any(value.t > value.measured_t for value in mpc.requested_states)
    else:
        assert all(isinstance(value, tm.State) for value in mpc.requested_states)
    # Foreground wall duration and notification interleaving are observer-dependent;
    # native solve durations, event timestamps and the numerical trace are not.
    return dict(outcome=outcomes[0], requests=mpc.inputs, warm_states=mpc.warm_states,
        commands=[(t, command.tolist()) for t, command in plant.applied],
        ticks=[(tick.t, state_trace(tick.state), None if tick.command is None else tick.command.tolist(),
                tick.plan_activated, tick.update, None if tick.plan is None else tick.plan.t0,
                None if tick.requested_command is None else tick.requested_command.tolist(), tick.clipped)
               for tick in ticks],
        solves=[(value.request_time_s, state_trace(value.initial), value.reference.tolist(), value.updates,
                 value.plan.t0, value.plan.measured_t, value.plan.states.tolist(), value.plan.torques.tolist(),
                 value.plan.stats) for value in solves], diagnostics=diagnostics, boundaries=boundaries)


@pytest.mark.parametrize('latency, native_latencies', [(.002, (.003,)), (.0055, (.001,)),
                                                      ('measured', (.003, .005, .001, .004))])
@pytest.mark.parametrize('compensated', [False, True])
def test_pause_preserves_the_full_async_numerical_trace(monkeypatch, latency, native_latencies, compensated):
    assert numerical_trace(monkeypatch, latency, native_latencies, compensated, paused=True) == (
        numerical_trace(monkeypatch, latency, native_latencies, compensated))


@pytest.mark.parametrize('latency, native_latencies', [(0., (0.,)), (.0055, (.001,)),
                                                      ('measured', (0., .005, .001, .004))])
@pytest.mark.parametrize('compensated', [False, True])
def test_synchronous_and_asynchronous_solves_keep_the_same_numerical_trace(
    monkeypatch,
    latency,
    native_latencies,
    compensated,
):
    assert numerical_trace(monkeypatch, latency, native_latencies, compensated, asynchronous=False) == (
        numerical_trace(monkeypatch, latency, native_latencies, compensated))


@pytest.mark.parametrize('latency, native_latencies', [(0., (0.,)), (.0055, (.001,)),
                                                      ('measured', (.003, .005, .001, .004))])
def test_message_observation_preserves_the_numerical_trace(monkeypatch, latency, native_latencies):
    observed = numerical_trace(monkeypatch, latency, native_latencies, True)
    unobserved = numerical_trace(monkeypatch, latency, native_latencies, True, observed=False)
    for trace in (observed, unobserved):
        del trace['solves'], trace['diagnostics']
    assert observed == unobserved


def test_cancellation_wakes_a_parked_gate_and_leaves_its_reply_for_caller_teardown():
    first_solve, parked, wake, cancelled = (threading.Event() for _ in range(4))
    plant, outcomes, messages = MovingPlant(), [], []
    mpc = DeferredAsync(plant, first_solve, (.003,))

    def before(state):
        if state.t == pytest.approx(.002):
            parked.set()
            assert wake.wait(5)

    thread = threading.Thread(target=lambda: outcomes.append(run(mpc, plant, duration=1.,
        settings=replace(SETTINGS, solve_latency_s=.006), before_tick_callbacks=[before],
        tick_callbacks=[messages.append], solve_completed_callbacks=[messages.append],
        diagnostic_callbacks=[messages.append], cancelled=cancelled.is_set)))
    thread.start()
    try:
        assert parked.wait(5)
        snapshot = state_trace(plant.state()), list(mpc.reads), list(plant.advanced)
        first_solve.set()
        assert mpc.ready.wait(5)
        cancelled.set()
        wake.set()
        thread.join(5)
        assert not thread.is_alive()
        assert outcomes == [tm.Outcome('interrupted', 'stop_requested', .002)]
        assert snapshot == (state_trace(plant.state()), mpc.reads, plant.advanced)
        assert not plant.applied and plant.calls == ['start', 'stop']
        assert mpc.warmups == [0.] and len(mpc.requests) == 1 and mpc.waiting is not None
        assert not any(isinstance(value, tm.SolveCompleted) for value in messages)
        assert mpc.result(block=True).t0 == 0.  # The caller may collect during teardown.
        assert mpc.waiting is None and mpc.result(block=True) is None
    finally:
        first_solve.set()
        cancelled.set()
        wake.set()
        thread.join(5)
        for worker in mpc.workers:
            worker.join(5)


def test_custom_safety_refusal_is_an_execution_fault_before_the_law():
    plant, ticks = Plant(), []

    class Safety:
        def validate(self, plan, t, state):
            return tm.Fault('custom_clearance')

    def unreachable(*args):
        pytest.fail('a refused plan must not evaluate the law')

    outcome = run(plant=plant, law=unreachable, safety=Safety(), tick_callbacks=[ticks.append])
    assert outcome.reason == 'custom_clearance' and not plant.applied
    assert ticks[-1].command is None


@pytest.mark.parametrize('refusal', ['safety', 'law'])
def test_forecast_custom_refusal_emits_no_actual_rejection(refusal):
    messages, plant = [], Plant()

    class Safety:
        def validate(self, plan, t, state):
            return tm.Fault('custom_forecast') if refusal == 'safety' and t >= .009 else None

    def law(plan, t, state):
        if refusal == 'law' and t >= .009:
            return tm.Fault('custom_forecast')
        return tm.Command(np.ones(7), np.ones(7))

    outcome = run(plant=plant, law=law, safety=Safety(), settings=replace(SETTINGS, update_period=.008),
                  compensation=tm.LatencyCompensation(latency_s=.004), duration=.02,
                  diagnostic_callbacks=[messages.append])
    assert outcome.reason == 'prediction_unavailable' and outcome.t == pytest.approx(.008)
    assert plant.applied and all(np.all(command == 1.) for _, command in plant.applied)
    assert not any(isinstance(value, tm.Diagnostic) and value.kind == 'command_rejected' for value in messages)


@pytest.mark.parametrize('handling', [None, lambda value: tm.Command(value.requested, np.zeros(7))])
def test_real_raw_effort_cannot_be_weakened_by_custom_policy(wall, handling):
    plant = Plant()
    law = lambda *args: tm.Command(np.full(7, 1000.), np.ones(7))
    outcome = tm.run_real_time(plant, Mpc(), Task(), law, settings=tm.RealTimeSettings(.002, .002),
                               compensation=None, effort_handling=handling, duration=.01)
    assert outcome.reason == 'effort_limit' and not plant.applied


def test_real_time_retains_100ms_lease_with_custom_permissive_safety(wall):
    class Safety:
        def validate(self, plan, t, state):
            return None

    plant = Plant()
    outcome = tm.run_real_time(plant, Mpc(N=32, replies=1), Task(),
                               settings=tm.RealTimeSettings(.002, 1.), compensation=None,
                               safety=Safety(), effort_handling=None, duration=.2)
    assert outcome.reason == 'lease_expired' and outcome.t == pytest.approx(.1)
    assert all(t < .1 for t, _ in plant.applied)


@pytest.mark.parametrize('knots, expiry', [(4, .012), (32, .1)])
def test_real_waiting_plan_expires_before_user_safety_or_law_runs(tmp_path, wall, knots, expiry):
    class SkippingPlant(Plant):
        def advance(self, t):
            super().advance(t)
            if t >= .004:
                wall.now = expiry

    class Safety:
        def validate(self, plan, t, state):
            pytest.fail('a waiting or expired plan must not reach user safety')

    def unreachable(*args):
        pytest.fail('a waiting or expired plan must not evaluate its law')

    plant, ticks, solves = SkippingPlant(), [], []
    record = tm.RunRecord(tmp_path / 'expired')
    outcome = tm.run_real_time(plant, Mpc(N=knots, dt=.002), Task(), unreachable,
                               settings=tm.RealTimeSettings(.002, 1.),
                               compensation=tm.LatencyCompensation(latency_s=.006),
                               safety=Safety(), duration=.2, tick_callbacks=[record, ticks.append],
                               solve_completed_callbacks=[record.solve_completed, solves.append])
    assert outcome.reason == 'lease_expired' and outcome.t == pytest.approx(expiry)
    assert not plant.applied and all(tick.plan is None for tick in ticks)
    assert len(solves) == 1 and solves[0].plan.t0 == pytest.approx(.006)
    assert solves[0].request_time_s == 0. and isinstance(solves[0].initial, tm.Prediction)
    assert solves[0].initial.t == pytest.approx(.006) and solves[0].initial.measured_t == 0.
    record.save()
    assert record.plans['update_t'] == [0.] and record.plans['t0'] == pytest.approx([.006])
    assert np.all(record.ticks['plan'] == -1)
    loaded = tm.load_run(record.out)
    np.testing.assert_array_equal(loaded.ticks['plan'], record.ticks['plan'])
    np.testing.assert_array_equal(loaded.plans['torques'][0], solves[0].plan.torques)


def test_real_time_rejects_clipping_before_plant_start():
    plant = Plant()
    with pytest.raises(ValueError, match='cannot clip'):
        tm.run_real_time(plant, Mpc(), Task(), settings=tm.RealTimeSettings(),
                         effort_handling=tm.EffortClip(), duration=.01)
    assert not plant.calls
