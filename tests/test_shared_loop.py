"""Shared simulation laws, observation, cancellation and streaming; no compiled solver."""
import gc
import threading

import numpy as np
import pytest

import tiago_mpc as tm
from tiago_mpc.prediction import _Predictor
from test_loops import Mpc, Plant, Task, wall


def run(mpc=None, plant=None, law=None, **settings):
    options = dict(command_period=.002, update_period=.002, duration=.01, solve_latency_s=0.)
    options.update(settings)
    args = () if law is None else (law,)
    return tm.run_virtual_time(plant or Plant(), mpc or Mpc(), Task(), *args, **options)


@pytest.mark.parametrize('law', [tm.SampleKnot(), tm.PlanFeedback(kp=0., kd=0.)])
def test_simulation_keeps_over_limit_predictions_and_clips_explicit_laws(law):
    class Excess(Mpc):
        def result(self, block=False):
            plan = super().result(block)
            return None if plan is None else tm.Plan(plan.t0, plan.dt, plan.states,
                np.full_like(plan.torques, 1000.), plan.stats, measured_t=plan.measured_t)

    plant, messages = Plant(), []
    outcome = run(Excess(), plant, law, message_callbacks=[messages.append])
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
    outcome = run(plant=plant, law=lambda plan, t, state: np.full(7, 1000.),
                  effort_handling=tm.RejectOnEffort(), message_callbacks=[messages.append])
    assert outcome.status == 'fault' and outcome.reason == 'effort_limit'
    assert not plant.applied
    tick = next(message for message in messages if isinstance(message, tm.Tick))
    assert tick.command is None
    np.testing.assert_array_equal(tick.requested_command, np.full(7, 1000.))
    event = next(message for message in messages if isinstance(message, tm.Event))
    assert event.kind == 'command_rejected' and event.data['reason'] == 'effort_limit'


@pytest.mark.parametrize('handling', [tm.EffortClip(), tm.RejectOnEffort(), None])
@pytest.mark.parametrize('bad', [np.full(7, np.inf), np.full(7, np.nan), np.ones(6)])
def test_effort_handling_never_conceals_invalid_commands(handling, bad):
    plant, ticks = Plant(), []
    outcome = run(plant=plant, law=lambda plan, t, state: bad,
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


@pytest.mark.parametrize('timing, expected', [
    (tm.VirtualTime(.002, .002, 0.), tm.limits().effort),
    (tm.VirtualTime(.002, .002, 0., effort_handling=None), np.full(7, 1000.)),
])
def test_virtual_timing_applies_its_effort_policy_to_an_explicit_law(timing, expected):
    plant, ticks = Plant(), []
    outcome = timing.run(plant, Mpc(), Task(), lambda plan, t, state: np.full(7, 1000.),
                         [ticks.append], duration=.004)
    assert outcome.status == 'timeout' and plant.applied
    for tick in ticks:
        if tick.command is not None:
            np.testing.assert_array_equal(tick.command, expected)
            np.testing.assert_array_equal(tick.requested_command, np.full(7, 1000.))
            assert tick.clipped == (timing.effort_handling is not None)


@pytest.mark.parametrize('handling, applied', [
    (tm.EffortClip(effort=(2.,)*7), 2.),
    (tm.RejectOnEffort(effort=(2.,)*7), None),
    (None, 1000.),
])
def test_forecasting_uses_the_same_effort_policy_and_keeps_refusal_predictive(monkeypatch, handling, applied):
    from tiago_mpc import prediction
    monkeypatch.setattr(prediction, 'plant_euler',
                        lambda model, data, q, v, u, h, friction, law: (q+u*h, v+u*h))
    predictor = _Predictor(tm.LatencyCompensation(), None, handling)
    initial = tm.State(0., np.zeros(7), np.zeros(7))
    plan = tm.Plan(0., .03, np.zeros((32, 14)), np.ones((31, 7)), measured_t=0.)
    base = lambda plan, t, state: np.full(7, 1000.)
    predicted = predictor.predict(initial, 0., .006, plan, base,
                                  np.full(7, 2. if applied is None else applied), .002, .002)
    if applied is None:
        assert predicted is None
    else:
        np.testing.assert_allclose(predicted.q, .006*applied)
        np.testing.assert_allclose(predicted.qd, .006*applied)
    np.testing.assert_array_equal(initial.q, np.zeros(7))


def test_forecasting_does_not_copy_a_pure_callable_with_a_lock(monkeypatch):
    from tiago_mpc import prediction
    monkeypatch.setattr(prediction, 'plant_euler',
                        lambda model, data, q, v, u, h, friction, law: (q+u*h, v+u*h))

    class Law:
        def __init__(self):
            self.lock = threading.Lock()

        def __call__(self, plan, t, state):
            with self.lock:
                return plan.torques[0]

    predictor = _Predictor(tm.LatencyCompensation(), None)
    initial = tm.State(0., np.zeros(7), np.zeros(7))
    plan = tm.Plan(0., .01, np.zeros((4, 14)), np.ones((3, 7)), measured_t=0.)
    predicted = predictor.predict(initial, 0., .006, plan, Law(), np.ones(7), .002, .002)
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
        message_callbacks=[messages.append], command_period=.002, update_period=.002,
        duration=.008, solve_latency_s=.006)
    solves = [message for message in messages if isinstance(message, tm.SolveCompleted)]
    active = {id(tick.plan) for tick in ticks if tick.command is not None}
    assert len(solves) == 2 and sum(id(message.plan) not in active for message in solves) == 1
    np.testing.assert_array_equal(solves[0].reference, np.zeros((11, 6)))
    assert not solves[0].reference.flags.writeable
    assert solves[1].request_time_s == pytest.approx(.006)
    np.testing.assert_array_equal(solves[1].updates['external_force'], np.full(6, .006))


def test_cancel_after_native_return_prevents_command_and_obsolete_message():
    cancelled = threading.Event()

    class Cancelling(Mpc):
        def result(self, block=False):
            result = super().result(block)
            cancelled.set()
            return result

    plant, messages = Plant(), []
    outcome = run(Cancelling(), plant, cancelled=cancelled.is_set, message_callbacks=[messages.append])
    assert outcome.status == 'interrupted' and outcome.t == 0.
    assert not plant.applied and not messages and plant.clock.now() == 0.


def test_cancel_from_solve_callback_is_checked_before_activation():
    cancelled, messages, plant = threading.Event(), [], Plant()

    def callback(message):
        messages.append(message)
        if isinstance(message, tm.SolveCompleted): cancelled.set()

    outcome = run(plant=plant, cancelled=cancelled.is_set, message_callbacks=[callback])
    assert outcome.status == 'interrupted' and outcome.t == 0.
    assert not plant.applied and len(messages) == 1


def test_paused_callback_wakes_for_cancellation_without_an_extra_command():
    parked, wake, cancelled = threading.Event(), threading.Event(), threading.Event()
    plant, mpc, outcome = Plant(), Mpc(), []

    def before(state):
        if abs(state.t-.006) < 1e-12:
            parked.set()
            assert wake.wait(5)

    thread = threading.Thread(target=lambda: outcome.append(run(mpc, plant, duration=None,
        before_tick=before, cancelled=cancelled.is_set)))
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


def test_streaming_has_no_ten_minute_endpoint_and_restores_gc_policy():
    stopped = threading.Event()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        outcome = run(Mpc(N=32, dt=100.), duration=None, command_period=100., update_period=0.,
            max_plan_age_s=None, cancelled=stopped.is_set,
            before_tick=lambda state: stopped.set() if state.t >= 700. else None)
        assert outcome.status == 'interrupted' and outcome.t == 700.
        assert not gc.isenabled()
    finally:
        if was_enabled: gc.enable()


@pytest.mark.parametrize('age, end, reason', [(.1, .1, 'lease_expired'),
                                             (.006, .006, 'lease_expired'), (None, .12, None)])
def test_plan_age_is_configurable_independently_of_long_horizon(age, end, reason):
    outcome = run(Mpc(N=32, dt=.03), update_period=1., duration=.12, max_plan_age_s=age)
    assert outcome.t == pytest.approx(end) and outcome.reason == reason


@pytest.mark.parametrize('age', [0., -1., np.inf, np.nan])
def test_invalid_plan_age_is_rejected_before_start(age):
    plant = Plant()
    with pytest.raises(ValueError, match='max_plan_age_s'):
        run(plant=plant, max_plan_age_s=age)
    assert not plant.calls


def test_anchor_forecast_uses_the_actual_initial_held_command():
    plant = tm.SimConfig(integrator='euler').build()
    mpc, boundaries = Mpc(), []
    outcome = run(mpc, plant, duration=.004, solve_latency_s=.002,
        start_time_s=10., initial_command=np.zeros(7), before_tick=boundaries.append,
        compensation=tm.LatencyCompensation(latency_s=.002))
    assert outcome.status == 'timeout' and mpc.warmups == [10.]
    predicted = mpc.requested_states[0]
    assert predicted.t == pytest.approx(10.002) and predicted.measured_t == 10.
    np.testing.assert_allclose(predicted.q, boundaries[1].q, atol=1e-12)
    np.testing.assert_allclose(predicted.qd, boundaries[1].qd, atol=1e-12)


def test_physics_events_keep_detection_time_and_arrive_before_the_next_command():
    plant, observed, order = tm.SimConfig(integrator='euler').build(), [], []

    def step(state):
        observed.append(state.t)
        return [tm.Event(state.t, 'limit_breach', 'detected', dict(q=state.q.tolist()))] if state.t < .0015 else []

    plant.step_callback = step

    def message(value):
        if isinstance(value, tm.Event): order.append(('event', value.t, plant.t))

    run(plant=plant, initial_command=np.zeros(7), duration=.004,
        message_callbacks=[message], before_tick=lambda state: order.append(('boundary', state.t, plant.t)))
    assert observed == pytest.approx([.001, .002, .003, .004])
    assert order.index(('event', .001, .002)) < order.index(('boundary', .002, .002))


def test_real_time_rejects_an_excessive_law_command_without_clipping(wall):
    plant = Plant()
    outcome = tm.run_real_time(plant, Mpc(), Task(), lambda plan, t, state: np.full(7, 1000.),
                               command_period=.002, update_period=.002, duration=.01)
    assert outcome.status == 'fault' and outcome.reason == 'effort_limit'
    assert not plant.applied and plant.calls == ['start', 'stop']


@pytest.mark.parametrize('done_at, expected_status', [(float('inf'), 'timeout'), (.002, 'done')])
def test_normal_completion_collects_an_async_result_and_leaves_the_mpc_reusable(done_at, expected_status):
    class Deferred(Mpc):
        def warmup(self, state, reference=None):
            self.warmups.append(state.t)

        def result(self, block=False):
            return super().result(True) if block else None

    mpc = Deferred()
    for _ in range(2):
        plant, messages = Plant(), []
        outcome = tm.run_virtual_time(plant, mpc, Task(done_at), command_period=.002,
            update_period=.002, duration=.004, solve_latency_s=.1, message_callbacks=[messages.append])
        assert outcome.status == expected_status and mpc.waiting is None
        assert not plant.applied and plant.calls == ['start', 'stop']
        solved = [message for message in messages if isinstance(message, tm.SolveCompleted)]
        assert len(solved) == 1 and solved[0].request_time_s == 0.


def test_effort_fault_collects_the_outstanding_solve_without_applying_its_plan():
    class Deferred(Mpc):
        def warmup(self, state, reference=None):
            self.warmups.append(state.t)

        def result(self, block=False):
            return super().result(True) if block else None

    mpc = Deferred()
    for _ in range(2):
        plant, messages = Plant(), []
        outcome = run(mpc, plant, lambda plan, t, state: np.full(7, 1000.), solve_latency_s=.002,
                      effort_handling=tm.RejectOnEffort(), message_callbacks=[messages.append])
        assert outcome.status == 'fault' and outcome.reason == 'effort_limit'
        assert outcome.t == pytest.approx(.002) and mpc.waiting is None
        assert not plant.applied and plant.clock.now() == pytest.approx(.002)
        solved = [message for message in messages if isinstance(message, tm.SolveCompleted)]
        assert [message.request_time_s for message in solved] == pytest.approx([0., .002])


@pytest.mark.parametrize('latency,wall,first_command', [(0.,.001,0.),(.002,.001,.002),
                                                       (.0055,.001,.006),('measured',.003,.004)])
def test_async_solver_waits_at_the_first_possible_activation_tick(latency,wall,first_command):
    plant = Plant()
    blocking = []

    class Deferred(Mpc):
        def result(self,block=False):
            if self.waiting is None:return None
            request_t = self.waiting[1]
            earliest = request_t + (.002 if latency=='measured' else latency)
            if not block:return None
            blocking.append(plant.clock.now())
            # Normal completion collects a remaining result without advancing
            # virtual time to its later activation boundary.
            assert plant.clock.now() >= min(earliest, .012)-1e-9
            return super().result(block=True)

    mpc = Deferred(wall=wall)
    outcome = run(mpc,plant,solve_latency_s=latency,duration=.012)
    assert outcome.status == 'timeout'
    assert plant.applied[0][0] == pytest.approx(first_command)
    assert blocking[0] == pytest.approx(.002 if latency=='measured' else first_command)


def test_observer_can_request_quiescence_before_parking_with_a_future_plan():
    messages,plant = [],Plant()

    class Deferred(Mpc):
        def result(self,block=False):
            return super().result(True) if block else None

    outcome = run(Deferred(),plant,solve_latency_s=1.,duration=.004,
                  quiesce_at=lambda state:state.t>=.004,message_callbacks=[messages.append])
    assert outcome.status == 'timeout'
    solves=[message for message in messages if isinstance(message,tm.SolveCompleted)]
    assert len(solves)==1 and solves[0].request_time_s==0.
    assert not plant.applied


def test_tick_requested_pause_collects_new_request_before_same_boundary_gate():
    plant, messages, paused, stopped = Plant(), [], threading.Event(), threading.Event()

    class Deferred(Mpc):
        def result(self, block=False):
            return super().result(True) if block else None

    mpc = Deferred()

    def message(value):
        messages.append(value)
        if isinstance(value, tm.Tick) and value.command is not None:
            paused.set()

    def before(state):
        if paused.is_set():
            assert mpc.waiting is None
            assert len([m for m in messages if isinstance(m, tm.SolveCompleted)]) == 2
            assert state.t == pytest.approx(.002)
            stopped.set()

    outcome = run(mpc, plant, solve_latency_s=.002, duration=.01, message_callbacks=[message],
                  before_tick=before, quiesce_at=lambda state:paused.is_set(), cancelled=stopped.is_set)
    assert outcome.status == 'interrupted' and outcome.t == pytest.approx(.002)
    assert len(plant.applied) == 1 and len(mpc.requests) == 2


@pytest.mark.parametrize('cancel_at_pause', [False, True])
def test_quiescence_overruns_reach_message_observers_once_and_the_next_tick(cancel_at_pause):
    messages, ticks, plant, stopped = [], [], Plant(), threading.Event()
    paused = False

    class Deferred(Mpc):
        def result(self, block=False):
            return super().result(True) if block else None

    mpc = Deferred(wall=.003)

    def message(value):
        nonlocal paused
        messages.append(value)
        if isinstance(value, tm.Tick) and value.t == pytest.approx(.002):
            paused = True

    def before(state):
        nonlocal paused
        if paused:
            assert mpc.waiting is None
            paused = False
            if cancel_at_pause:
                stopped.set()

    outcome = run(mpc, plant, solve_latency_s=.002, duration=.006, tick_callbacks=[ticks.append],
                  message_callbacks=[message], before_tick=before, quiesce_at=lambda state: paused,
                  cancelled=stopped.is_set)
    observed = [value for value in messages if isinstance(value, tm.Event) and value.kind == 'overrun']
    recorded = [event for tick in ticks for event in tick.events if event.kind == 'overrun']
    if cancel_at_pause:
        assert outcome.status == 'interrupted' and outcome.t == pytest.approx(.002)
        assert plant.clock.now() == pytest.approx(.002)
        # No further Tick follows cancellation; the just-collected event stays
        # available to the message observer that requested quiescence.
        assert [event.t for event in observed] == pytest.approx([0., .002])
        assert [event.t for event in recorded] == pytest.approx([0.])
    else:
        assert outcome.status == 'timeout'
        assert [event.t for event in observed] == pytest.approx([0., .002, .004])
        assert recorded == observed
        assert [event.t for event in ticks[2].events if event.kind == 'overrun'] == pytest.approx([.002])
