"""SimPlant: today's RK4 integration, held commands, and the simulation and wall clocks."""
from gato.common import rk4
import numpy as np
import pytest

from tiago_mpc import sim
from tiago_mpc.clock import Clock
from tiago_mpc.dynamics import plant_rk4
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import urdf_friction

U = np.linspace(-1., 1., 7)


@pytest.mark.parametrize("friction,substep", [(None, .003), (urdf_friction(), 1e-4)])
def test_advance_holds_the_command_over_substeps_up_to_the_tick(friction, substep):
    plant = SimPlant(substep=substep, friction=friction)
    q, qd, t = plant.q.copy(), plant.qd.copy(), 0.
    plant.start()
    plant.apply(U)
    plant.advance(.01)
    while t < .01 - 1e-12:  # e.g. 3, 3, 3 and 1 ms
        h = min(substep, .01 - t)
        q, qd = (rk4(plant.model, plant.data, q, qd, U, h) if friction is None
                 else plant_rk4(plant.model, plant.data, q, qd, U, h, friction))
        t += h
    s = plant.state()
    assert s.t == plant.now() == .01 and plant.settings["clock"] == "simulation"
    np.testing.assert_array_equal(s.q, q)
    np.testing.assert_array_equal(s.qd, qd)
    plant.apply(-U)  # the state is at the simulation clock's time: nothing to integrate
    assert plant.state().t == .01
    np.testing.assert_array_equal(plant.q, q)
    plant.stop()
    np.testing.assert_array_equal(plant.command, np.zeros(7))


def test_on_the_wall_clock_advance_integrates_then_waits_for_it(monkeypatch):
    wall, slept = [100.], []
    monkeypatch.setattr(sim.time, "monotonic", lambda: wall[0])  # also the Clock's
    monkeypatch.setattr(sim.time, "sleep", slept.append)
    plant = SimPlant(substep=.01, clock=Clock.wall())
    plant.start()
    assert plant.state().t == 100.
    wall[0] += .004
    plant.advance(100.01)
    assert slept == [pytest.approx(.006)] and plant.state().t == 100.01
    wall[0] += .03  # late: integrates to the deadline, no wait
    plant.advance(100.02)
    assert slept[1] == 0. and plant.state().t == 100.02
    assert plant.settings["clock"] == "wall"


def test_on_the_wall_clock_a_command_holds_from_when_it_is_applied(monkeypatch):
    """Torque 1 until 50 ms, torque 2 until 60 ms, integrated to 10 ms before the second
    is applied: the input integrates to 0.07 (rk4 stubbed to integrate the torque alone)."""
    wall = [100.]
    monkeypatch.setattr(sim.time, "monotonic", lambda: wall[0])
    monkeypatch.setattr(sim.time, "sleep", lambda s: None)
    monkeypatch.setattr(sim, "rk4", lambda model, data, q, qd, u, h: (q + u * h, qd))
    plant = SimPlant(substep=.003, clock=Clock.wall())
    q0 = plant.q.copy()
    plant.start()
    plant.apply(np.ones(7))
    plant.advance(100.01)
    wall[0] = 100.05
    plant.apply(2 * np.ones(7))
    plant.advance(100.06)
    np.testing.assert_allclose(plant.q - q0, .07)


def test_run_robot_on_a_wall_clock_simplant():
    """The robot's loop on the wall clock, with gravity-holding plans from a fake Mpc."""
    import time
    from dataclasses import replace
    import pinocchio as pin
    from test_experiments import Mpc
    from test_loops import Task
    from tiago_mpc.loops import run_robot
    from tiago_mpc.mpc import TIAGO

    plant, ticks = SimPlant(substep=.003, clock=Clock.wall()), []
    # Held until the first command, as the robot's onboard controllers hold the arm.
    plant.command = pin.rnea(plant.model, plant.data, plant.q, np.zeros(7), np.zeros(7))
    q0, start = plant.q.copy(), time.monotonic()
    assert not run_robot(plant, Mpc(replace(TIAGO, N=8)), Task(), callbacks=[ticks.append],
                         command_period=.01, update_period=.01, duration=.2)
    assert time.monotonic() - start >= .2 and len(ticks) == 21
    assert all(k.command is not None and k.new for k in ticks[:-1])
    np.testing.assert_allclose(plant.q, q0, atol=1e-3)
