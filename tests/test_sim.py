"""SimPlant: today's RK4 integration, held commands and the real-time clock."""
from gato.common import rk4
import numpy as np
import pytest

from tiago_mpc import sim
from tiago_mpc.joint_dynamics import plant_rk4, urdf_friction
from tiago_mpc.sim import SimPlant

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
    assert s.t == .01
    np.testing.assert_array_equal(s.q, q)
    np.testing.assert_array_equal(s.qd, qd)
    plant.stop()
    np.testing.assert_array_equal(plant.command, np.zeros(7))


def test_realtime_waits_for_the_wall_clock(monkeypatch):
    clock, slept = [100.], []
    monkeypatch.setattr(sim.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sim.time, "sleep", slept.append)
    plant = SimPlant(substep=.01, realtime=True)
    plant.start()
    clock[0] += .004
    plant.advance(.01)
    assert slept == [pytest.approx(.006)]
    assert plant.settings["clock"] == "wall"
