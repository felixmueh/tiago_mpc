"""Causal actuation delays without ROS, CUDA, or a simulation clock double."""
import math

import numpy as np
import pytest

from tiago_mpc.plan import Plan, PlanEnded
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
    {"update_period_s": 0}, {"update_period_s": math.inf},
    {"remaining_s": -.001}, {"remaining_s": math.nan}, {"mode": "adaptive"},
])
def test_invalid_timing_is_rejected(kwargs):
    args = {"update_period_s": .01, "solve_wall_s": .002, **kwargs}
    with pytest.raises(ValueError):
        control_intervals(**args)


def plan(start, dt=.002, knots=5):
    return Plan(start, dt, np.zeros((knots + 1, 4)), 1. + np.arange(knots)[:, None] * np.ones(2))


def test_controls_over_split_a_control_update_at_knot_boundaries():
    # 10 ms control update, 2 ms knots, from offset 3 ms.
    pieces = plan(0.).controls_over(.003, .006)
    assert [torque[0] for torque, _ in pieces] == [2., 3., 4., 5.]
    assert [d for _, d in pieces] == pytest.approx([.001, .002, .002, .001])
    assert sum(d for _, d in pieces) == pytest.approx(.006)
    assert plan(0.).controls_over(.003, 0.) == []


def test_update_equal_to_knot_interval_applies_knot_zero_for_exactly_dt():
    item = plan(.03, dt=.01)
    [(torque, duration)] = item.controls_over(0., .01)
    assert duration == .01
    np.testing.assert_array_equal(torque, item.torques[0])


def test_a_span_past_the_last_knot_raises():
    item = plan(0., knots=3)  # 6 ms plan
    assert sum(d for _, d in item.controls_over(.004, .002)) == pytest.approx(.002)
    with pytest.raises(PlanEnded):
        item.controls_over(.004, .006)
    with pytest.raises(PlanEnded):
        item.controls_over(.008, .002)


def test_control_segments_continue_the_previous_plan_while_solving():
    previous, new = plan(0.), plan(.01, knots=7)
    intervals = control_intervals(.01, 0., fixed_s=.003)
    with pytest.raises(PlanEnded):  # the charged 3 ms start at 10 ms, past the 10 ms plan
        control_segments(previous, new, intervals)
    previous = plan(0., knots=7)
    segments = control_segments(previous, new, intervals)
    assert [torque[0] for torque, _ in segments] == [6., 7., 2., 3., 4., 5.]
    assert [d for _, d in segments] == pytest.approx([.002, .001, .001, .002, .002, .002])
    assert sum(d for _, d in segments) == pytest.approx(.01)
    first = control_segments(None, new, intervals)
    assert not np.any(first[0][0]) and first[0][1] == .003
    # Without latency the previous plan is not applied at all.
    assert [t[0] for t, _ in control_segments(previous, new, control_intervals(.01, 0.))] == [1., 2., 3., 4., 5.]


@pytest.mark.parametrize("kwargs", [
    {"dt": 0.}, {"dt": math.nan}, {"t0": math.inf},
    {"torques": np.zeros((0, 7))}, {"torques": np.full((2, 7), np.nan)}, {"torques": np.zeros(7)},
    {"states": np.zeros((2, 14))}, {"states": np.full((3, 14), np.inf)},
])
def test_invalid_plans_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Plan(**{"t0": 0., "dt": .01, "states": np.zeros((3, 14)), "torques": np.zeros((2, 7)), **kwargs})
