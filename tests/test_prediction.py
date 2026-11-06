"""Latency forecasts, physical propagation and both loop clocks; CPU-only known answers."""
from dataclasses import replace
import json

import numpy as np
import pinocchio as pin
import pytest

import tiago_mpc as tm
from tiago_mpc import prediction as module
from tiago_mpc.prediction import _Predictor
from test_loops import Mpc, Plant, Task, commanded, wall


def predictor(**settings):
    return _Predictor(tm.LatencyCompensation(**settings), tm.loops.LEASE_S)


def plan(t0=0., measured_t=0., dt=.002, n=40):
    return tm.Plan(t0, dt, np.zeros((n, 14)), np.ones((n - 1, 7)), measured_t=measured_t)


def test_off_grid_history_and_future_ticks_use_partial_intervals_and_held_commands(monkeypatch):
    calls, ticks = [], []

    def step(model, data, q, qd, u, h, friction, law):
        calls.append((float(u[0]), h))
        return q + u*h, qd.copy()

    monkeypatch.setattr(module, 'plant_euler', step)
    p = predictor()
    p.sent(.0001, np.ones(7))
    p.sent(.0017, np.full(7, 2.))
    state = tm.State(.0008, np.zeros(7), np.zeros(7))

    def law(plan, t, state):
        ticks.append(t)
        return np.full(7, 3. if t < .005 else 4.)

    result = p.predict(state, .0033, .0071, plan(), law, np.full(7, 2.), .004, .002)
    expected = 1*(.0017-.0008) + 2*(.004-.0017) + 3*(.006-.004) + 4*(.0071-.006)
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
    result = p.predict(state, 0., .0023, None, tm.SampleKnot(), u, .004, .002)
    np.testing.assert_allclose(result.q, expected_q, atol=1e-13)
    np.testing.assert_allclose(result.qd, expected_v, atol=1e-13)


def test_composed_ff_pd_replays_total_command_then_forecasts_future_feedback(monkeypatch):
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
        seen.append((t, state, command))
        return command

    command = law(active, .002, raw)
    p = predictor(friction=f, armature=tm.ARMATURE_KG_M2)
    p.sent(0., np.ones(7))
    p.sent(.002, command)
    result = p.predict(raw, .0033, .0061, active, law, command, .004, .002)
    assert result is not None
    assert [t for t, _, _ in seen] == pytest.approx([.002, .004, .006])
    assert seen[0][1] is raw
    q_at_first_future_tick = raw.q + .0013 + command*.002
    qd_at_first_future_tick = raw.qd + .0013 + command*.002
    np.testing.assert_allclose(seen[1][1].q, q_at_first_future_tick, atol=1e-14)
    np.testing.assert_allclose(seen[1][1].qd, qd_at_first_future_tick, atol=1e-14)
    assert np.max(np.abs(seen[1][2] - command)) > 1e-5
    np.testing.assert_array_equal(raw.q, np.full(7, .1))
    np.testing.assert_array_equal(raw.qd, np.full(7, -.2))


def test_initial_no_command_means_onboard_hold_instead_of_zero_effort():
    p = predictor()
    measured = tm.State(0., tm.Q_START.copy(), np.zeros(7))
    result = p.predict(measured, .003, .009, None, tm.SampleKnot(), None, .004, .002)
    np.testing.assert_array_equal(result.q, measured.q)
    np.testing.assert_array_equal(result.qd, measured.qd)


def test_history_is_copied_bounded_and_retains_cutoff_predecessor():
    p = predictor()
    u = np.ones(7)
    p.sent(0., u)
    u[:] = 100.
    np.testing.assert_array_equal(p.history[-1][1], np.ones(7))
    assert not p.history[-1][1].flags.writeable
    for i in range(1, 2001):
        p.sent(i*.002, np.ones(7))
    assert len(p.history) <= 52
    assert p.history[0][0] <= 3.9 < p.history[1][0]
    assert p.predict(tm.State(0., np.zeros(7), np.zeros(7)), 4., 4.002, None,
                     tm.SampleKnot(), u, 4.002, .002) is None


def test_forecast_cannot_cross_off_grid_old_plan_expiry_or_unsafe_future_command():
    p = predictor()
    state = tm.State(0., np.zeros(7), np.zeros(7))
    short = plan(dt=.0015, n=4)  # lease/horizon ends at 4.5 ms, between command ticks
    assert p.predict(state, 0., .006, short, tm.SampleKnot(), np.ones(7), .002, .002) is None
    assert p.predict(state, 0., .0045, short, tm.SampleKnot(), np.ones(7), .002, .002) is not None
    unsafe = lambda plan, t, state: np.full(7, 1000.)
    assert p.predict(state, 0., .006, plan(), unsafe, np.ones(7), .002, .002) is None


@pytest.mark.parametrize('field,value', [('t', np.nan), ('q', np.zeros(6)), ('qd', np.full(7, np.inf))])
def test_prediction_rejects_invalid_initial_conditions(field, value):
    fields = dict(t=1., q=np.zeros(7), qd=np.zeros(7), measured_t=0.)
    fields[field] = value
    with pytest.raises(ValueError):
        tm.Prediction(**fields)


def test_predicted_origin_does_not_renew_observation_lease_or_plan_horizon():
    future = plan(t0=.08, measured_t=0.)
    assert tm.lease_fault(future, .099) is None
    assert tm.lease_fault(future, .1) == 'lease_expired'
    short = plan(t0=.01, measured_t=0., dt=.002, n=4)
    assert tm.lease_fault(short, .016) == 'lease_expired'
    with pytest.raises(ValueError, match='measured_t'):
        plan(t0=.01, measured_t=.02)


def run(timing, mpc, duration=.024, plant=None, callbacks=()):
    ticks = []
    result = timing.run(plant or Plant(), mpc, Task(), tm.SampleKnot(), [ticks.append, *callbacks],
                        duration=duration)
    return ticks, result


@pytest.mark.parametrize('real', [False, True])
def test_early_results_wait_until_predicted_origin_and_block_next_request(real, wall):
    config = tm.LatencyCompensation(latency_s=.006)
    timing = (tm.RealTime(.002, 0., config) if real else tm.VirtualTime(.002, 0., .002, config))
    mpc = Mpc(dt=.002)
    ticks, outcome = run(timing, mpc, duration=.016)
    assert outcome.status == 'timeout'
    assert [tick.t for tick in ticks if tick.new] == pytest.approx([.006, .012])
    assert [tick.t for tick in ticks if tick.update] == pytest.approx([0., .006, .012])
    assert all(tick.command is None for tick in ticks if tick.t < .006)
    assert all(tick.plan.t0 <= tick.t + 1e-12 for tick in ticks if tick.plan is not None)


@pytest.mark.parametrize('real', [False, True])
def test_late_results_keep_predicted_origin_and_execute_aged_knots(real, wall):
    config = tm.LatencyCompensation(latency_s=.002)
    timing = (tm.RealTime(.002, .02, config) if real else tm.VirtualTime(.002, .02, .006, config))
    mpc = Mpc(dt=.002, delay=2)
    ticks, _ = run(timing, mpc, duration=.01)
    activated = next(tick for tick in ticks if tick.new)
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

    timing = tm.RealTime(.002, .02, tm.LatencyCompensation(latency_s=.002))
    plant, mpc = Plant(), SlowPoll(dt=.002)
    ticks, _ = run(timing, mpc, plant=plant, duration=.008)
    assert plant.applied[0][0] >= mpc.result_at
    assert plant.applied[0][0] == pytest.approx(.0023)
    assert next(tick for tick in ticks if tick.new).plan_request_t == 0.


def test_previous_delay_forecast_never_uses_current_virtual_solve_time():
    class Variable(Mpc):
        def request(self, state, reference, updates=None):
            self.wall = [.006, .002, .008][len(self.requests)]
            super().request(state, reference, updates)

    timing = tm.VirtualTime(.002, 0., 'measured', tm.LatencyCompensation(initial_latency_s=.002))
    mpc = Variable(dt=.002)
    ticks, outcome = run(timing, mpc, duration=.02)
    assert outcome.status == 'timeout'
    assert [p.t for p in mpc.predictions] == pytest.approx([.002, .012, .014])
    activated = [tick for tick in ticks if tick.new]
    assert [tick.t for tick in activated] == pytest.approx([.006, .012, .02])
    assert all(set(tick.plan.stats) == {'wall_s'} for tick in activated)


def test_superseded_early_plan_does_not_misassociate_recorded_request(tmp_path):
    timing = tm.VirtualTime(.002, 0., 0., tm.LatencyCompensation(initial_latency_s=.005))
    mpc = Mpc(dt=.002)
    record = tm.RunRecord(tmp_path/'superseded', timing=timing)
    ticks, outcome = run(timing, mpc, duration=.01, callbacks=[record])
    assert outcome.status == 'timeout'
    assert [tick.t for tick in ticks if tick.update] == pytest.approx([0., .006, .008])
    assert [tick.t for tick in ticks if tick.new] == pytest.approx([.006, .008])
    record.save()
    assert record.plans['update_t'] == pytest.approx([.006, .008])
    assert [tick.plan_request_t for tick in ticks if tick.new] == pytest.approx([.006, .008])
    assert set(record.plans) == {'t0', 'measured_t', 'update_t', 'states', 'torques', 'wall_s'}


def test_an_unapplied_over_limit_knot_does_not_reject_a_simulated_plan(tmp_path):
    timing = tm.VirtualTime(.002, 0., 0., tm.LatencyCompensation(initial_latency_s=.005))
    record = tm.RunRecord(tmp_path/'faulted', timing=timing)
    ticks, outcome = run(timing, Mpc(dt=.002, over=2), duration=.01, callbacks=[record])
    assert outcome.status == 'timeout' and outcome.reason is None
    assert any(tick.command is not None for tick in ticks)
    assert all(tick.faulted is None for tick in ticks)
    record.save()
    assert record.plans['update_t'][0] == pytest.approx(.006)
    assert record.plans['torques'][0, -1, 6] == pytest.approx(26.1)


def test_stale_measured_state_stays_raw_for_requests_feedback_and_recording(tmp_path):
    config = tm.LatencyCompensation(latency_s=.004, armature=tm.ARMATURE_KG_M2)
    timing = tm.VirtualTime(.002, .02, .004, config)
    plant, mpc = Plant(measure=lambda t: t-.0007), Mpc(dt=.002)
    record = tm.RunRecord(tmp_path/'run', timing=timing)
    ticks, outcome = run(timing, mpc, duration=.026, plant=plant, callbacks=[record])
    assert outcome.status == 'timeout'
    assert [p.measured_t for p in mpc.predictions] == pytest.approx([-.0007, .0193])
    assert [p.t for p in mpc.predictions] == pytest.approx([.004, .024])
    record.save()
    assert record.ticks['state_t'][0] == 0.
    assert record.plans['measured_t'] == pytest.approx([0., .02])
    assert record.plans['update_t'] == pytest.approx([.0007, .0207])
    assert [tick.state.t for tick in ticks if tick.new] == pytest.approx([.0033, .0233])
    assert tm.load_run(record.out).config['timing'] == timing


@pytest.mark.parametrize('timing', [tm.RealTime(), tm.VirtualTime(.002, .02, 'measured')])
def test_nested_compensation_config_round_trips_and_run_preserves_its_dataclass(timing, wall):
    config = tm.LatencyCompensation(friction=tm.urdf_friction(), friction_law=tm.Tanh(.03),
                                    armature=tm.ARMATURE_KG_M2, latency_s=.004)
    timing = replace(timing, compensation=config)
    restored = tm.from_dict(json.loads(json.dumps(tm.to_dict(timing))))
    assert restored == timing and isinstance(restored.compensation, tm.LatencyCompensation)
    ticks, outcome = run(restored, Mpc(dt=.002, wall=.004), duration=.012)
    assert outcome.status == 'timeout' and any(tick.command is not None for tick in ticks)
