"""Physical examples for stalls/chatter; CPU only, no numerical solver."""
from dataclasses import replace
import json

import numpy as np
import pytest

from experiments.motion_metrics import motion_metrics, planned_velocity, high_frequency_rms


def test_stationary_joint_is_not_a_stall_without_a_motion_request():
    t = np.arange(101) * .002
    zero = np.zeros((len(t), 7))
    result = motion_metrics(t, zero, zero, zero)
    assert result['stall']['fraction'] is None
    assert result['stall']['count'] == 0
    assert result['stall']['p95_s'] is None
    assert result['stall']['max_s'] is None
    assert result['torque_variation_nm_per_s'] == [0.] * 7
    json.dumps(result, allow_nan=False)


def test_short_zero_crossings_do_not_count_but_a_sustained_pause_does():
    t = np.arange(201) * .002
    velocity = np.full((len(t), 7), .1)
    intended = np.zeros_like(velocity)
    intended[:, 2] = .03
    velocity[(t >= .04) & (t < .07), 2] = 0.  # 30 ms: below the fixed duration threshold
    velocity[(t >= .12) & (t < .22), 2] = 0.  # 100 ms pause
    command = np.zeros_like(velocity)
    command[:3] = np.nan  # no commanded motion before plan activation
    command[-1] = np.nan  # final endpoint is not a sent command
    result = motion_metrics(t, velocity, intended, command)
    stall = result['stall']
    assert stall['count'] == 1
    assert stall['eligible_joint_s'] == pytest.approx(.394)
    assert stall['stalled_joint_s'] == pytest.approx(.1)
    assert stall['fraction'] == pytest.approx(.1 / .394)
    assert stall['median_s'] == pytest.approx(.1)
    assert stall['per_joint'][0]['stalled_fraction'] is None
    assert stall['per_joint'][2]['episodes'][0]['left_censored'] is False


def test_durations_use_actual_intervals_and_flag_a_window_cut():
    t = np.array([0., .01, .04, .10, .16])
    qd = np.zeros((5, 7))
    intended = np.full((5, 7), .03)
    result = motion_metrics(t, qd, intended, qd, start_s=.03, ripple_cutoff_hz=1.)
    assert result['stall']['per_joint'][0]['stalled_s'] == pytest.approx(.13)
    assert result['stall']['per_joint'][0]['episodes'][0]['left_censored']
    assert result['stall']['per_joint'][0]['episodes'][0]['right_censored']


def test_torque_variation_counts_sent_adjacent_commands_without_bridging_a_gap():
    t = np.arange(8) * .01
    command = np.repeat(np.array([np.nan, 0., 2., np.nan, 100., 101., 103., np.nan])[:, None], 7, axis=1)
    result = motion_metrics(t, np.zeros((8, 7)), np.ones((8, 7)), command)
    assert result['torque_total_variation_nm'] == [5.] * 7
    assert result['command_pair_duration_s'] == pytest.approx(.03)
    np.testing.assert_allclose(result['torque_variation_nm_per_s'], np.full(7, 5 / .03))


def test_velocity_ripple_distinguishes_fast_oscillation_from_slow_motion():
    t = np.arange(501) * .002
    fast = np.repeat(np.sin(2*np.pi*20*t)[:, None], 7, axis=1)
    slow = np.repeat(np.sin(2*np.pi*t)[:, None], 7, axis=1)
    assert high_frequency_rms(fast, .002)[0] == pytest.approx(1/np.sqrt(2), rel=.03)
    assert high_frequency_rms(slow, .002)[0] < .001


def test_planned_velocity_uses_active_plan_and_elapsed_origin_not_newest_request():
    ticks = dict(t=np.array([0., .006, .012, .02, .024]), plan=np.array([-1, 0, 0, 1, 1]))
    states = np.zeros((2, 3, 14))
    states[0, :, 7:] = np.array([0., 1., 2.])[:, None]
    states[1, :, 7:] = np.array([10., 20., 30.])[:, None]
    result = planned_velocity(ticks, dict(t0=np.array([0., .01]), states=states), .01)
    assert np.isnan(result[0]).all()
    np.testing.assert_allclose(result[1:, 0], [.6, 1.2, 20., 24.])


def test_no_friction_removes_only_passive_losses_and_internal_mpc_has_no_external_ff():
    import tiago_mpc as tm
    from experiments.controller_tuning import scenario, laws, TUNED_PD
    from experiments.friction_compensation import case_params

    nominal, free = scenario('nominal'), scenario('no_friction')
    assert free.physical.friction is None
    assert replace(free.physical, friction=nominal.physical.friction) == nominal.physical
    assert free.feedback_period_s == nominal.feedback_period_s
    choices = laws()
    assert type(choices['mpc_friction']) is tm.SampleKnot
    assert choices['mpc_friction_pd'] == TUNED_PD
    assert case_params('default').friction is None
    assert case_params('mpc_friction').friction_law.width_rad_s == .2
    assert isinstance(choices['ff_prior01'], tm.FrictionFeedforward)


def test_nonuniform_spectral_data_is_not_silently_treated_as_uniform():
    qd = np.zeros((4, 7))
    with pytest.raises(ValueError, match='uniform'):
        motion_metrics(np.array([0., .002, .009, .02]), qd, qd, qd)


def test_missing_true_velocity_censors_both_sides_and_exact_zero_stays_separate():
    t = np.arange(201) * .002
    qd = np.zeros((len(t), 7))
    qd[100] = np.nan
    intended, command = np.full_like(qd, .03), np.zeros_like(qd)
    result = motion_metrics(t, qd, intended, command)
    first, second = result['stall']['per_joint'][0]['episodes']
    assert first['right_censored'] and second['left_censored']
    assert result['exact_zero_eligible_fraction'] == pytest.approx(1.)
    qd[:] = .003  # near-rest is not an exact complementarity zero
    result = motion_metrics(t, qd, intended, command)
    assert result['stall']['fraction'] == pytest.approx(1.)
    assert result['exact_zero_eligible_fraction'] == 0.
