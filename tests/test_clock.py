"""VirtualClock and WallClock: the two kinds of time and their rules."""
import math

import pytest

from tiago_mpc import clock as clock_module
from tiago_mpc.clock import VirtualClock, WallClock


def test_virtual_time_only_moves_forward_and_stays_finite():
    clock = VirtualClock(2.)
    clock.reach(2.5)
    clock.reach(2.5)
    assert clock.now() == 2.5
    for t in (2.4, math.nan, math.inf):
        with pytest.raises(ValueError, match="backward"):
            clock.reach(t)
    assert clock.now() == 2.5 and VirtualClock().now() == 0. and VirtualClock(-1.).now() == -1.
    with pytest.raises(ValueError):
        VirtualClock(math.nan)


def test_wall_time_sleeps_until_it_reaches_t_and_returns_at_once_when_late(monkeypatch):
    now, slept = [42.], []

    class Time:
        monotonic = staticmethod(lambda: now[0])
        sleep = staticmethod(slept.append)

    monkeypatch.setattr(clock_module, "time", Time)
    clock = WallClock()
    assert clock.now() == 42.
    clock.reach(42.01)
    clock.reach(41.)
    assert slept == [pytest.approx(.01), 0.]
