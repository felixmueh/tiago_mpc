"""Causal actuation delays without ROS, CUDA, or a simulation clock double."""
import math

import numpy as np
import pytest

from tiago_mpc.playback import Plan, PlanEnded, knot_torques
from tiago_mpc.simulation import control_intervals, control_segments


def test_fixed_latency_is_independent_of_solve_wall_time():
    instant = control_intervals(.01, .04)
    delayed = control_intervals(.01, .04, fixed_s=.003)
    assert (instant.previous_s, instant.new_s) == pytest.approx((0, .01))
    assert (delayed.previous_s, delayed.new_s) == pytest.approx((.003, .007))
    # For xdot=u, old u=2 and new u=5 produce different physical motion.
    assert 2 * delayed.previous_s + 5 * delayed.new_s == pytest.approx(.041)
    assert delayed.elapsed_s == pytest.approx(.01)


def test_measured_latency_delays_new_actuation_and_allows_overrun():
    normal = control_intervals(.01, .004, mode="measured")
    overrun = control_intervals(.01, .025, mode="measured")
    assert (normal.previous_s, normal.new_s) == pytest.approx((.004, .006))
    assert (overrun.previous_s, overrun.new_s) == pytest.approx((.025, 0))
    assert overrun.elapsed_s == pytest.approx(.025)
    assert overrun.charged_s == pytest.approx(.025)
    zero = control_intervals(.01, 0, mode="measured")
    assert (zero.previous_s, zero.new_s) == pytest.approx((0, .01))


def test_simulation_end_clips_old_or_new_control_without_overshoot():
    during_solve = control_intervals(.01, .02, mode="measured", remaining_s=.003)
    after_solve = control_intervals(.01, .002, mode="measured", remaining_s=.006)
    finished = control_intervals(.01, .02, mode="measured", remaining_s=0)
    assert during_solve.charged_s == pytest.approx(.02)
    assert (during_solve.previous_s, during_solve.new_s) == pytest.approx((.003, 0))
    assert (after_solve.previous_s, after_solve.new_s) == pytest.approx((.002, .004))
    assert finished.elapsed_s == 0


@pytest.mark.parametrize("kwargs", [
    {"fixed_s": -.001}, {"fixed_s": math.nan}, {"fixed_s": math.inf},
    {"solve_wall_s": -.001}, {"solve_wall_s": math.nan},
    {"period_s": 0}, {"period_s": math.inf},
    {"remaining_s": -.001}, {"remaining_s": math.nan}, {"mode": "adaptive"},
])
def test_invalid_timing_is_rejected(kwargs):
    args = {"period_s": .01, "solve_wall_s": .002, **kwargs}
    with pytest.raises(ValueError):
        control_intervals(**args)


def plan(start, dt=.002, knots=5, mode="playback"):
    return Plan(start, dt, 1. + np.arange(knots)[:, None] * np.ones(2), mode)


def test_playback_indexes_knots_from_the_state_time_with_a_zero_order_hold():
    item = plan(10.)
    assert item.end_s == pytest.approx(10.01)
    assert item.command(10.)[0] == 0
    assert item.command(10.0019)[0] == 0
    # 10 + 3 * .002 rounds below the boundary; it still selects knot 3.
    assert item.command(10. + 3 * .002)[0] == 3
    np.testing.assert_array_equal(item.command(10.0099)[1], [5., 5.])
    with pytest.raises(PlanEnded):
        item.command(10.01)
    with pytest.raises(ValueError):
        item.command(9.999)
    knot, torque = plan(10., mode="first").command(10.009)
    assert knot == 0 and np.array_equal(torque, item.torques[0])


def test_segments_split_a_control_period_at_knot_boundaries():
    # 10 ms control period, 2 ms knots, from offset 3 ms.
    segments = plan(0.).segments(.003, .006)
    assert [k for k, *_ in segments] == [1, 2, 3, 4]
    assert [d for _, _, d, _ in segments] == pytest.approx([.001, .002, .002, .001])
    assert sum(d for _, _, d, _ in segments) == pytest.approx(.006)
    assert not any(past for *_, past in segments)
    [(knot, torque, duration, past)] = plan(0., mode="first").segments(.003, .006)
    assert (knot, duration, past) == (0, .006, False) and np.array_equal(torque, [1., 1.])
    assert plan(0.).segments(.003, 0.) == []


def test_period_equal_to_knot_interval_applies_knot_zero_for_exactly_dt():
    item = plan(.03, dt=.01)
    [(knot, torque, duration, past)] = item.segments(0., .01)
    assert (knot, duration, past) == (0, .01, False)
    np.testing.assert_array_equal(torque, item.torques[0])


def test_past_the_last_knot_the_simulation_holds_it_and_flags_it():
    item = plan(0., knots=3)  # 6 ms plan
    segments = item.segments(.004, .006)
    assert [(k, past) for k, _, _, past in segments] == [(2, False), (2, True)]
    assert [d for _, _, d, _ in segments] == pytest.approx([.002, .004])
    np.testing.assert_array_equal(segments[-1][1], item.torques[-1])
    # Starting past the end, the whole span holds the final knot.
    assert [(k, past) for k, _, _, past in item.segments(.008, .002)] == [(2, True)]


def test_control_segments_continue_the_previous_plan_while_solving():
    previous, new = plan(0.), plan(.01, knots=7)
    intervals = control_intervals(.01, 0., fixed_s=.003)
    segments = control_segments(previous, new, intervals)
    assert [(source is new, k, past) for source, k, _, _, past in segments] == [
        (False, 4, True), (True, 1, False), (True, 2, False), (True, 3, False), (True, 4, False)]
    assert sum(d for *_, d, _ in segments) == pytest.approx(.01)
    first = control_segments(None, new, intervals)
    assert first[0][:2] == (None, -1) and not np.any(first[0][2]) and first[0][3] == .003
    # Without latency the previous plan is not applied at all.
    assert [k for _, k, *_ in control_segments(previous, new, control_intervals(.01, 0.))] == [0, 1, 2, 3, 4]


def test_knot_torques_take_every_knot_and_cross_the_sign_boundary():
    signs = np.array([1., -1., 1., 1., 1., 1., -1.])
    knots = np.arange(21.).reshape(3, 7)
    xu = np.concatenate([np.r_[np.full(14, 99.), u] for u in knots] + [np.full(14, 99.)])
    np.testing.assert_array_equal(knot_torques(xu, signs), knots * signs)
    with pytest.raises(ValueError):
        knot_torques(xu[:-1], signs)
    with pytest.raises(ValueError):
        knot_torques(np.zeros(14), signs)


@pytest.mark.parametrize("kwargs", [
    {"dt_s": 0.}, {"dt_s": math.nan}, {"start_s": math.inf}, {"mode": "smooth"},
    {"torques": np.zeros((0, 7))}, {"torques": np.full((2, 7), np.nan)}, {"torques": np.zeros(7)},
])
def test_invalid_plans_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Plan(**{"start_s": 0., "dt_s": .01, "torques": np.zeros((2, 7)), **kwargs})
