"""Control laws on a plan of 2 ms knots."""
import numpy as np
import pytest

from tiago_mpc.laws import HoldFirst, SampleKnot
from tiago_mpc.plan import Plan

sample_knot, hold_first = SampleKnot(), HoldFirst()
PLAN = Plan(10., .002, np.zeros((6, 4)), 1. + np.arange(5)[:, None] * np.ones(2))


def test_sample_knot_holds_each_knot_from_the_state_time():
    assert sample_knot(PLAN, 10., None)[0] == 1.
    assert sample_knot(PLAN, 10.0019, None)[0] == 1.
    # 10 + 3 * .002 rounds below the boundary; it still selects knot 3.
    assert sample_knot(PLAN, 10. + 3 * .002, None)[0] == 4.
    np.testing.assert_array_equal(sample_knot(PLAN, 10.0099, None), [5., 5.])
    with pytest.raises(RuntimeError, match="last knot"):  # beyond the lease: a broken invariant
        sample_knot(PLAN, 10.01, None)
    with pytest.raises(ValueError):
        sample_knot(PLAN, 9.999, None)


def test_hold_first_ignores_the_time():
    for t in (10., 10.005, 11.):
        np.testing.assert_array_equal(hold_first(PLAN, t, None), [1., 1.])
