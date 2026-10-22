"""Clock: the three domains and their mutation rules."""
import math

import pytest

from tiago_mpc import clock as clock_module
from tiago_mpc.clock import Clock


def test_a_simulation_clock_only_advances():
    clock = Clock.simulation(start=2.)
    clock.advance(.5)
    clock.advance(0.)
    assert clock.now() == 2.5
    for dt in (-.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="backward"):
            clock.advance(dt)
    with pytest.raises(RuntimeError, match="cannot be updated"):
        clock.update(3.)
    with pytest.raises(ValueError):
        Clock.simulation(start=math.nan)
    assert clock.now() == 2.5


def test_an_external_clock_only_takes_timestamps_and_may_pause():
    clock = Clock.external()
    with pytest.raises(RuntimeError, match="no time yet"):
        clock.now()
    clock.update(5.)
    clock.update(5.)  # a paused simulation repeats its time
    assert clock.now() == 5.
    for t in (4.9, math.nan, math.inf):
        with pytest.raises(ValueError, match="backward"):
            clock.update(t)
    with pytest.raises(RuntimeError, match="cannot be advanced"):
        clock.advance(.1)
    assert clock.now() == 5.


def test_a_wall_clock_can_be_neither_advanced_nor_updated(monkeypatch):
    monkeypatch.setattr(clock_module.time, "monotonic", lambda: 42.)
    clock = Clock.wall()
    with pytest.raises(RuntimeError, match="cannot be advanced"):
        clock.advance(1.)
    with pytest.raises(RuntimeError, match="cannot be updated"):
        clock.update(50.)
    assert clock.now() == clock.wall_time() == Clock.simulation().wall_time() == 42.
