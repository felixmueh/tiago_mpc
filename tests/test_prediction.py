"""Latency forecasts, physical propagation and both loop clocks; CPU-only known answers."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pinocchio as pin
import pytest

import tiago_mpc as tm
from tiago_mpc import prediction as module
from tiago_mpc.prediction import _Predictor
from test_loops import Mpc, Plant, Task, commanded, wall


def predictor(**settings):
    return _Predictor(tm.LatencyCompensation(**settings))


def plan(t0=0., measured_t=0., dt=.002, n=40):
    return tm.Plan(t0, dt, np.zeros((n, 14)), np.ones((n - 1, 7)), measured_t=measured_t)


def test_stale_measurement_uses_held_command_until_the_first_future_tick(monkeypatch):
    calls, ticks = [], []

    def step(model, data, q, qd, u, h, friction, law):
        calls.append((float(u[0]), h))
        return q + u*h, qd.copy()

    monkeypatch.setattr(module, 'plant_euler', step)
    p = predictor()
    state = tm.State(.0008, np.zeros(7), np.zeros(7))

    def law(plan, t, state):
        ticks.append(t)
        torque = np.full(7, 3. if t < .005 else 4.)
        return tm.Command(torque, torque)

    result = p(state, .0033, .0071, plan(), law, np.full(7, 2.), .004, .002)
    expected = 2*(.004-.0008) + 3*(.006-.004) + 4*(.0071-.006)
    np.testing.assert_allclose(result.q, expected, atol=1e-14)
    assert ticks == pytest.approx([.004, .006])
    assert sum(h for _, h in calls) == pytest.approx(.0071-.0008)
    assert result.t == .0071 and state.t == .0008
    np.testing.assert_array_equal(state.q, np.zeros(7))


def test_propagation_uses_its_explicit_coupled_friction_and_armature_once():
    f = tm.Friction((1.,)*7, (2.,)*7)
    p = predictor(friction=f, armature=tm.ARMATURE_KG_M2)
    np.testing.assert_array_equal(p.model.armature, tm.ARMATURE_KG_M2)
    q, v = tm.Q_START.copy(), np.linspace(-.1, .1, 7)
    u = pin.rnea(p.model, p.data, q, v, np.zeros(7)).copy() + np.linspace(-2., 2., 7)
    state = tm.State(0., q.copy(), v.copy())
    expected_q, expected_v = q, v
    for h in (.001, .001, .0003):
        expected_q, expected_v = module.plant_euler(p.model, p.data, expected_q, expected_v, u, h,
                                                   f, tm.StickSlip())
    result = p(state, 0., .0023, None, tm.SampleKnot(), u, .004, .002)
    np.testing.assert_allclose(result.q, expected_q, atol=1e-13)
    np.testing.assert_allclose(result.qd, expected_v, atol=1e-13)


def test_composed_ff_pd_holds_total_command_then_forecasts_future_feedback(monkeypatch):
    def step(model, data, q, qd, u, h, friction, law):
        return q + u*h, qd + u*h

    monkeypatch.setattr(module, 'plant_euler', step)
    f = tm.Friction((.2,)*7, (.1,)*7)
    composed = tm.FrictionFeedforward(f, tm.PlanFeedback(kp=.1, kd=.1))
    states = np.zeros((40, 14))
    states[:, 7:] = .1
    active = tm.Plan(0., .002, states, np.ones((39, 7)), measured_t=0.)
    raw = tm.State(.0007, np.full(7, .1), np.full(7, -.2))
    seen = []

    def law(plan, t, state):
        command = composed(plan, t, state)
        seen.append((t, state, command.applied))
        return command

    command = law(active, .002, raw).applied
    p = predictor(friction=f, armature=tm.ARMATURE_KG_M2)
    result = p(raw, .0033, .0061, active, law, command, .004, .002)
    assert result is not None
    assert [t for t, _, _ in seen] == pytest.approx([.002, .004, .006])
    assert seen[0][1] is raw
    q_at_first_future_tick = raw.q + command*(.004-raw.t)
    qd_at_first_future_tick = raw.qd + command*(.004-raw.t)
    np.testing.assert_allclose(seen[1][1].q, q_at_first_future_tick, atol=1e-14)
    np.testing.assert_allclose(seen[1][1].qd, qd_at_first_future_tick, atol=1e-14)
    assert np.max(np.abs(seen[1][2] - command)) > 1e-5
    np.testing.assert_allclose(result.q, seen[2][1].q+seen[2][2]*.0001, atol=1e-14)
    np.testing.assert_allclose(result.qd, seen[2][1].qd+seen[2][2]*.0001, atol=1e-14)
    np.testing.assert_array_equal(raw.q, np.full(7, .1))
    np.testing.assert_array_equal(raw.qd, np.full(7, -.2))


def test_initial_no_command_means_onboard_hold_instead_of_zero_effort():
    p = predictor()
    measured = tm.State(0., tm.Q_START.copy(), np.zeros(7))
    result = p(measured, .003, .009, None, tm.SampleKnot(), None, .004, .002)
    np.testing.assert_array_equal(result.q, measured.q)
    np.testing.assert_array_equal(result.qd, measured.qd)


def test_repeated_and_interleaved_calls_are_independent_and_preserve_their_sources():
    p = predictor()
    q, qd, command = tm.Q_START.copy(), np.linspace(-.1, .1, 7), np.linspace(-2., 2., 7)
    original_q, original_qd, original_command = q.copy(), qd.copy(), command.copy()
    state = tm.State(.0007, q, qd)
    active = plan()
    original_states, original_torques = active.states.copy(), active.torques.copy()

    def forecast():
        return p(state, .0033, .0061, active, tm.PlanFeedback(kp=.1, kd=.1), command, .004, .002)

    first, repeated = forecast(), forecast()
    assert first is not None and repeated is not None
    assert p(tm.State(1., tm.Q_START.copy(), np.zeros(7)), 1.003, 1.009, None,
             tm.SampleKnot(), np.full(7, 3.), 1.004, .002) is not None
    interleaved = forecast()
    assert interleaved is not None
    for value in (repeated, interleaved):
        assert value is not first and value.t == first.t and value.measured_t == state.t
        np.testing.assert_array_equal(value.q, first.q)
        np.testing.assert_array_equal(value.qd, first.qd)
    for value, original in ((q, original_q), (qd, original_qd), (command, original_command),
                            (active.states, original_states), (active.torques, original_torques)):
        np.testing.assert_array_equal(value, original)
    assert not np.shares_memory(first.q, q) and not np.shares_memory(first.qd, qd)


def test_predictor_configuration_is_frozen():
    p = predictor()
    for field, value in (('config', tm.LatencyCompensation(latency_s=.002)),
                         ('safety', tm.ExecutionSafety(None)), ('effort_handling', None)):
        with pytest.raises(FrozenInstanceError):
            setattr(p, field, value)


def test_forecast_cannot_cross_off_grid_old_plan_expiry_or_unsafe_future_command():
    p = predictor()
    state = tm.State(0., np.zeros(7), np.zeros(7))
    short = plan(dt=.0015, n=4)  # lease/horizon ends at 4.5 ms, between command ticks
    assert p(state, 0., .006, short, tm.SampleKnot(), np.ones(7), .002, .002) is None
    assert p(state, 0., .0045, short, tm.SampleKnot(), np.ones(7), .002, .002) is not None
    unsafe = lambda plan, t, state: tm.Command(np.full(7, 1000.), np.full(7, 1000.))
    assert p(state, 0., .006, plan(), unsafe, np.ones(7), .002, .002) is None


@pytest.mark.parametrize('field,value', [('t', np.nan), ('q', np.zeros(6)), ('qd', np.full(7, np.inf))])
def test_prediction_rejects_invalid_initial_conditions(field, value):
    fields = dict(t=1., q=np.zeros(7), qd=np.zeros(7), measured_t=0.)
    fields[field] = value
    with pytest.raises(ValueError):
        tm.Prediction(**fields)


def test_predicted_origin_does_not_renew_observation_lease_or_plan_horizon():
    safety = tm.ExecutionSafety()
    state = tm.State(.099, np.zeros(7), np.zeros(7))
    future = plan(t0=.08, measured_t=0.)
    assert safety.validate(future, .099, state) is None
    fault = safety.validate(future, .1, state)
    assert isinstance(fault, tm.Fault) and fault.reason == 'lease_expired' and fault.requested is None
    short = plan(t0=.01, measured_t=0., dt=.002, n=4)
    fault = safety.validate(short, .016, state)
    assert isinstance(fault, tm.Fault) and fault.reason == 'lease_expired' and fault.requested is None
    with pytest.raises(ValueError, match='measured_t'):
        plan(t0=.01, measured_t=.02)


def run(
    timing,
    mpc,
    duration=.024,
    plant=None,
    callbacks=(),
    *,
    compensation=None,
    solve_completed_callbacks=(),
    diagnostic_callbacks=(),
):
    ticks = []
    loop = tm.run_real_time if isinstance(timing, tm.RealTimeSettings) else tm.run_virtual_time
    result = loop(plant or Plant(), mpc, Task(), tm.SampleKnot(), [ticks.append, *callbacks],
                  settings=timing, compensation=compensation, duration=duration,
                  solve_completed_callbacks=solve_completed_callbacks,
                  diagnostic_callbacks=diagnostic_callbacks)
    return ticks, result


@pytest.mark.parametrize('real', [False, True])
@pytest.mark.parametrize('command_period, update_period, first_origin', [
    (.002, .006, .006),
    (.004, .011, .012),
    (.003, 0., .003),
])
def test_settings_based_compensation_seeds_from_the_control_period_and_activates_on_the_command_grid(
    real,
    command_period,
    update_period,
    first_origin,
    wall,
):
    mpc, ticks = Mpc(N=32, wall=.001), []
    settings = (tm.RealTimeSettings(command_period, update_period) if real else
                tm.VirtualTimeSettings(command_period, update_period, 'measured'))
    options = {} if real else dict(compensation=tm.default_compensation(settings))
    loop = tm.run_real_time if real else tm.run_virtual_time
    outcome = loop(Plant(), mpc, Task(), tick_callbacks=[ticks.append], settings=settings,
                   duration=first_origin+command_period, **options)
    assert outcome.status == 'timeout'
    assert mpc.predictions[0].t == pytest.approx(first_origin)
    assert mpc.predictions[0].measured_t == 0.
    first = next(tick for tick in ticks if tick.plan_activated)
    assert first.t == pytest.approx(first_origin) and first.plan.t0 == pytest.approx(first_origin)
    assert all(tick.command is None for tick in ticks if tick.t < first_origin-1e-9)


@pytest.mark.parametrize('latency, first_origin', [(0., 0.), (.0055, .006), (.02, .02)])
def test_declared_virtual_compensation_matches_the_declared_delay_even_when_native_solves_differ(
    latency,
    first_origin,
):
    settings = tm.VirtualTimeSettings(.002, .01, latency)
    mpc, solves = Mpc(N=32, wall=.001), []
    ticks, outcome = run(settings, mpc, duration=.032, compensation=tm.default_compensation(settings),
                         solve_completed_callbacks=[solves.append])
    assert outcome.status == 'timeout'
    first = next(tick for tick in ticks if tick.plan_activated)
    assert first.t == pytest.approx(first_origin) and first.plan.t0 == pytest.approx(first_origin)
    assert all(solved.plan.t0-solved.request_time_s == pytest.approx(first_origin) for solved in solves)
    assert all(tick.t == pytest.approx(tick.plan.t0) for tick in ticks if tick.plan_activated)


def test_real_default_forecasts_the_previous_observed_request_to_result_delay_including_ipc_and_polling(wall):
    class Variable(Mpc):
        def __init__(self):
            super().__init__(N=32, wall=.001)
            self.request_times, self.result_times = [], []

        def request(self, state, reference=None, updates=None):
            self.request_times.append(wall.now)
            super().request(state, reference, updates)
            wall.now += .0007  # request serialization and transport
            self.ready_at = wall.now + [.0039, .0103, .0011][min(len(self.requests)-1, 2)]

        def result(self, block=False):
            if self.waiting is None:
                return None
            wall.now += .0003  # receiving and polling work on the same clock
            if wall.now < self.ready_at:
                return None
            result = super().result(block)
            self.result_times.append(wall.now)
            return result

    mpc, ticks = Variable(), []
    outcome = tm.run_real_time(Plant(), mpc, Task(), settings=tm.RealTimeSettings(.002, .008),
                               compensation='default', duration=.034, tick_callbacks=[ticks.append])
    assert outcome.status == 'timeout'
    assert mpc.request_times == pytest.approx([0., .008, .0203])
    assert mpc.result_times == pytest.approx([.0063, .0203, .0223])
    assert [state.t for state in mpc.predictions] == pytest.approx([.008, .016, .034])
    assert [tick.t for tick in ticks if tick.plan_activated] == pytest.approx([.008, .02, .034])
    # The second forecast uses 6.3 ms already observed; the third uses 12.3 ms.
    # Neither substitutes native wall_s=1 ms or the currently unknown solve delay.
    assert [state.measured_t for state in mpc.predictions] == pytest.approx([0., .008, .02])


@pytest.mark.parametrize('compensation, origins, activations', [
    (None, [0., .006, .012, .018], [.006, .012, .018]),
    (tm.LatencyCompensation(latency_s=.008), [.008, .016, .024], [.008, .016]),
])
def test_real_explicit_fixed_forecast_and_none_override_the_settings_based_default(
    compensation,
    origins,
    activations,
    wall,
):
    mpc = Mpc(N=32, dt=.002, delay=2)
    ticks, outcome = run(tm.RealTimeSettings(.002, .006), mpc, duration=.02, compensation=compensation)
    assert outcome.status == 'timeout'
    assert [state.t for state in mpc.requested_states] == pytest.approx(origins)
    assert [tick.t for tick in ticks if tick.plan_activated] == pytest.approx(activations)
    assert all((state is None) == (compensation is None) for state in mpc.predictions)


@pytest.mark.parametrize('real', [False, True])
def test_early_results_wait_until_predicted_origin_and_block_next_request(real, wall):
    config = tm.LatencyCompensation(latency_s=.006)
    timing = (tm.RealTimeSettings(.002, 0.) if real else tm.VirtualTimeSettings(.002, 0., .002))
    mpc = Mpc(dt=.002)
    ticks, outcome = run(timing, mpc, duration=.016, compensation=config)
    assert outcome.status == 'timeout'
    assert [tick.t for tick in ticks if tick.plan_activated] == pytest.approx([.006, .012])
    assert [tick.t for tick in ticks if tick.update] == pytest.approx([0., .006, .012])
    assert all(tick.command is None for tick in ticks if tick.t < .006)
    assert all(tick.plan.t0 <= tick.t + 1e-12 for tick in ticks if tick.plan is not None)


@pytest.mark.parametrize('latency', [.1, .12])
def test_real_startup_forecasts_past_the_lease_then_refuses_expired_pending_plan(wall, latency):
    plant, mpc, solves = Plant(), Mpc(N=32), []
    ticks, outcome = run(tm.RealTimeSettings(.002, .02), mpc, plant=plant, duration=.2,
                         compensation=tm.LatencyCompensation(latency_s=latency),
                         solve_completed_callbacks=[solves.append])
    assert len(mpc.requests) == 1 and mpc.predictions[0].t == pytest.approx(latency)
    assert mpc.predictions[0].measured_t == 0.
    assert outcome.status == 'fault' and outcome.reason == 'lease_expired' and outcome.t == pytest.approx(.1)
    assert not plant.applied and all(tick.plan is None for tick in ticks)
    assert len(solves) == 1 and solves[0].plan.t0 == pytest.approx(latency)
    assert solves[0].request_time_s == 0.


@pytest.mark.parametrize('real', [False, True])
def test_late_results_keep_predicted_origin_and_execute_aged_knots(real, wall):
    config = tm.LatencyCompensation(latency_s=.002)
    timing = (tm.RealTimeSettings(.002, .02) if real else tm.VirtualTimeSettings(.002, .02, .006))
    mpc = Mpc(dt=.002, delay=2)
    ticks, _ = run(timing, mpc, duration=.01, compensation=config)
    activated = next(tick for tick in ticks if tick.plan_activated)
    assert activated.t == pytest.approx(.006)
    assert activated.plan.t0 == pytest.approx(.002)
    assert activated.plan.measured_t == 0.
    assert activated.command[0] == 3.
    assert mpc.predictions[0].measured_t == 0. and mpc.predictions[0].t == .002
    assert ticks[0].state.t == 0.


def test_real_activation_publication_timestamp_cannot_precede_result(wall):
    class SlowPoll(Mpc):
        def result(self, block=False):
            if self.waiting is not None:
                wall.now += .0003
            result = super().result(block)
            if result is not None:
                self.result_at = wall.now
            return result

    timing = tm.RealTimeSettings(.002, .02)
    plant, mpc, solves = Plant(), SlowPoll(dt=.002), []
    ticks, _ = run(timing, mpc, plant=plant, duration=.008,
                   compensation=tm.LatencyCompensation(latency_s=.002),
                   solve_completed_callbacks=[solves.append])
    assert plant.applied[0][0] >= mpc.result_at
    assert plant.applied[0][0] == pytest.approx(.0023)
    assert next(tick for tick in ticks if tick.plan_activated).plan is solves[0].plan
    assert solves[0].request_time_s == 0.


def test_previous_delay_forecast_never_uses_current_virtual_solve_time():
    class Variable(Mpc):
        def request(self, state, reference, updates=None):
            self.wall = [.006, .002, .008][len(self.requests)]
            super().request(state, reference, updates)

    timing = tm.VirtualTimeSettings(.002, 0., 'measured')
    mpc = Variable(dt=.002)
    ticks, outcome = run(timing, mpc, duration=.02,
                         compensation=tm.default_compensation(timing))
    assert outcome.status == 'timeout'
    assert [p.t for p in mpc.predictions] == pytest.approx([.002, .012, .014])
    activated = [tick for tick in ticks if tick.plan_activated]
    assert [tick.t for tick in activated] == pytest.approx([.006, .012, .02])
    assert all(set(tick.plan.stats) == {'wall_s'} for tick in activated)


def test_superseded_early_plan_does_not_misassociate_recorded_request(tmp_path):
    timing = tm.VirtualTimeSettings(.002, 0., 0.)
    mpc, solves = Mpc(dt=.002), []
    record = tm.RunRecord(tmp_path/'superseded', settings=timing)
    ticks, outcome = run(timing, mpc, duration=.01, callbacks=[record], diagnostic_callbacks=[record.diagnostic],
                         solve_completed_callbacks=[record.solve_completed, solves.append],
                         compensation=tm.LatencyCompensation(initial_latency_s=.005))
    assert outcome.status == 'timeout'
    assert [tick.t for tick in ticks if tick.update] == pytest.approx([0., .006, .008])
    assert [tick.t for tick in ticks if tick.plan_activated] == pytest.approx([.006, .008])
    record.save()
    assert record.plans['update_t'] == pytest.approx([0., .006, .008])
    assert [solved.request_time_s for solved in solves] == pytest.approx([0., .006, .008])
    assert all(tick.plan is not solves[0].plan for tick in ticks)
    assert next(tick for tick in ticks if tick.plan_activated).plan is solves[1].plan
    np.testing.assert_array_equal(record.ticks['plan'], [-1, -1, -1, 1, 2, 2])
    assert set(record.plans) == {'t0', 'measured_t', 'update_t', 'states', 'torques', 'wall_s'}
    loaded = tm.load_run(record.out)
    np.testing.assert_array_equal(loaded.ticks['plan'], record.ticks['plan'])
    np.testing.assert_array_equal(loaded.plans['update_t'], record.plans['update_t'])


def test_an_unapplied_over_limit_knot_does_not_reject_a_simulated_plan(tmp_path):
    timing = tm.VirtualTimeSettings(.002, 0., 0.)
    record = tm.RunRecord(tmp_path/'faulted', settings=timing)
    ticks, outcome = run(timing, Mpc(dt=.002, over=2), duration=.01, callbacks=[record], diagnostic_callbacks=[record.diagnostic],
                         solve_completed_callbacks=[record.solve_completed],
                         compensation=tm.LatencyCompensation(initial_latency_s=.005))
    assert outcome.status == 'timeout' and outcome.reason is None
    assert any(tick.command is not None for tick in ticks)
    record.save()
    assert record.plans['update_t'][1] == pytest.approx(.006)
    assert record.plans['torques'][1, -1, 6] == pytest.approx(26.1)


def test_stale_measured_state_stays_raw_for_requests_feedback_and_recording(tmp_path):
    config = tm.LatencyCompensation(latency_s=.004, armature=tm.ARMATURE_KG_M2)
    timing = tm.VirtualTimeSettings(.002, .02, .004)
    plant, mpc = Plant(measure=lambda t: t-.0007), Mpc(dt=.002)
    record = tm.RunRecord(tmp_path/'run', settings=timing)
    ticks, outcome = run(timing, mpc, duration=.026, plant=plant, callbacks=[record],
                         solve_completed_callbacks=[record.solve_completed],
                         diagnostic_callbacks=[record.diagnostic], compensation=config)
    assert outcome.status == 'timeout'
    assert [p.measured_t for p in mpc.predictions] == pytest.approx([-.0007, .0193])
    assert [p.t for p in mpc.predictions] == pytest.approx([.004, .024])
    record.save()
    assert record.ticks['state_t'][0] == 0.
    assert record.plans['measured_t'] == pytest.approx([0., .02])
    assert record.plans['update_t'] == pytest.approx([.0007, .0207])
    assert [tick.state.t for tick in ticks if tick.plan_activated] == pytest.approx([.0033, .0233])
    assert tm.load_run(record.out).config['settings'] == timing


@pytest.mark.parametrize('timing', [tm.RealTimeSettings(), tm.VirtualTimeSettings(.002, .02, 'measured')])
def test_nested_compensation_config_round_trips_and_run_preserves_its_dataclass(timing, wall):
    config = tm.LatencyCompensation(friction=tm.urdf_friction(), friction_law=tm.Tanh(.03),
                                    armature=tm.ARMATURE_KG_M2, latency_s=.004)
    restored = tm.from_dict(json.loads(json.dumps(tm.to_dict(config))))
    assert restored == config and isinstance(restored, tm.LatencyCompensation)
    ticks, outcome = run(timing, Mpc(dt=.002, wall=.004), duration=.012, compensation=restored)
    assert outcome.status == 'timeout' and any(tick.command is not None for tick in ticks)
