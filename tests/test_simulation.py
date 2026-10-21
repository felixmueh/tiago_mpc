"""Causal actuation delays without ROS, CUDA, or a simulation clock double."""
import math

import pytest

from tiago_mpc.simulation import control_intervals


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
