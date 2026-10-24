"""SimPlant: held commands on virtual and wall time, and pacing."""
from dataclasses import asdict

from gato.common import rk4
import numpy as np
import pytest

from tiago_mpc import clock as clock_module, sim
from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.dynamics import Sign, StickSlip, Tanh, plant_euler, plant_rk4
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import urdf_friction

U = np.linspace(-1., 1., 7)


@pytest.mark.parametrize("friction,substep,integrator", [(None, .003, "rk4"), (None, .001, "euler"),
                                                         (urdf_friction(Tanh(.03)), .001, "euler"),
                                                         (urdf_friction(Sign()), .001, "euler"),
                                                         (urdf_friction(), .001, "euler"),
                                                         (urdf_friction(Tanh(.03)), 1e-4, "rk4"),
                                                         (urdf_friction(Sign()), 1e-4, "rk4")])
def test_advance_holds_the_command_over_substeps_up_to_the_tick(friction, substep, integrator):
    default = "rk4" if friction is None else "euler"
    plant = SimPlant(substep=substep, friction=friction, integrator=None if integrator == default else integrator)
    q, qd, t = plant.q.copy(), plant.qd.copy(), 0.
    plant.start(VirtualClock())
    plant.apply(U)
    plant.advance(.01)
    while t < .01 - 1e-12:  # e.g. 3, 3, 3 and 1 ms
        h = min(substep, .01 - t)
        q, qd = (plant_euler(plant.model, plant.data, q, qd, U, h, friction) if integrator == "euler"
                 else rk4(plant.model, plant.data, q, qd, U, h) if friction is None
                 else plant_rk4(plant.model, plant.data, q, qd, U, h, friction))
        t += h
    s = plant.state()
    assert s.t == plant.clock.now() == .01 and plant.settings["time"] == "virtual"
    assert plant.settings["friction"] == (None if friction is None else asdict(friction))
    assert plant.settings["integrator"] == integrator
    np.testing.assert_array_equal(s.q, q)
    np.testing.assert_array_equal(s.qd, qd)
    plant.apply(-U)  # the state is at the virtual time: nothing to integrate
    assert plant.state().t == .01
    np.testing.assert_array_equal(plant.q, q)
    plant.stop()
    np.testing.assert_array_equal(plant.command, np.zeros(7))


@pytest.mark.parametrize("settings", [dict(substep=0), dict(substep=-.001), dict(substep=np.nan),
                                     dict(integrator="other"), dict(integrator="rk4", friction=urdf_friction()),
                                     dict(integrator="rk4", friction=urdf_friction(StickSlip()))])
def test_invalid_integration_settings_fail_before_a_run(settings):
    with pytest.raises(ValueError):
        SimPlant(**settings)


class FakeTime:
    def __init__(self, now=100.):
        self.now, self.slept = now, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)


@pytest.fixture
def wall(monkeypatch):
    """WallClock's and the pacing's time; sleeping records the wait without moving time on."""
    fake = FakeTime()
    monkeypatch.setattr(clock_module, "time", fake)
    monkeypatch.setattr(sim, "time", fake)
    return fake


def test_on_wall_time_advance_integrates_then_waits_for_it(wall):
    plant = SimPlant(substep=.01)
    plant.start(WallClock())
    assert plant.state().t == 100.
    wall.now += .004
    plant.advance(100.01)
    wall.now = 100.01
    assert wall.slept == [pytest.approx(.006)] and plant.state().t == 100.01
    wall.now += .03  # late: integrates to the deadline, no wait
    plant.advance(100.02)
    assert wall.slept[1] == 0. and plant.t == 100.02
    assert plant.state().t == 100.04  # a state is read at the clock's time
    assert plant.settings["time"] == "wall"


def test_on_wall_time_the_start_state_holds_until_the_first_command(wall):
    """Held through a tick and a 50 ms first solve (or warm-up): read then, the held state
    has the clock's time; the command integrates from when it is applied."""
    plant = SimPlant(substep=.01)
    q0 = plant.q.copy()
    plant.start(WallClock())
    plant.advance(100.01)
    wall.now = 100.05
    s = plant.state()
    assert s.t == 100.05
    np.testing.assert_array_equal(s.q, q0)
    np.testing.assert_array_equal(s.qd, np.zeros(7))
    plant.apply(U)
    plant.advance(100.06)
    q, qd = rk4(plant.model, plant.data, q0, np.zeros(7), U, .01)
    assert plant.state().t == 100.06
    np.testing.assert_allclose(plant.q, q)
    np.testing.assert_allclose(plant.qd, qd)


def test_on_wall_time_a_command_holds_from_when_it_is_applied(wall, monkeypatch):
    """Torque 1 until 50 ms, torque 2 until 60 ms, integrated to 10 ms before the second
    is applied: the input integrates to 0.07 (rk4 stubbed to integrate the torque alone)."""
    monkeypatch.setattr(sim, "rk4", lambda model, data, q, qd, u, h: (q + u * h, qd))
    plant = SimPlant(substep=.003)
    q0 = plant.q.copy()
    plant.start(WallClock())
    plant.apply(np.ones(7))
    plant.advance(100.01)
    wall.now = 100.05
    plant.apply(2 * np.ones(7))
    plant.advance(100.06)
    np.testing.assert_allclose(plant.q - q0, .07)


def test_sync_real_time_waits_for_the_wall_clock_warns_once_behind_and_keeps_the_results(wall):
    """Counted from the first tick, so the warm-up before it neither warns nor runs the start fast."""
    paced, plain = SimPlant(substep=.003, sync_real_time=True), SimPlant(substep=.003)
    paced.start(VirtualClock(5.))
    plain.start(VirtualClock(5.))
    for plant in (paced, plain):
        plant.apply(U)
    wall.now += .15  # the warm-up and the first tick's work
    first = wall.now
    paced.advance(5.01)
    assert wall.slept == [pytest.approx(.01)]
    wall.now = first + .014  # the wait, then 4 ms of computing: wait 6 ms for the tick at 20 ms
    paced.advance(5.02)
    assert wall.slept[1] == pytest.approx(.006)
    wall.now = first + .045  # 15 ms behind at 30 ms: no wait, one warning
    with pytest.warns(UserWarning, match="15.0 ms behind the wall clock"):
        paced.advance(5.03)
    wall.now += .02
    paced.advance(5.04)  # still behind: no second warning
    assert wall.slept[2:] == [0., 0.] and paced.settings["sync_real_time"]
    for t in (5.01, 5.02, 5.03, 5.04):
        plain.advance(t)
    np.testing.assert_array_equal(paced.q, plain.q)
    np.testing.assert_array_equal(paced.qd, plain.qd)


def test_sync_real_time_refuses_wall_time(wall):
    with pytest.raises(ValueError, match="sync_real_time"):
        SimPlant(sync_real_time=True).start(WallClock())


@pytest.mark.filterwarnings("ignore:a plan took effect")  # 10 ms ticks: each plan from the tick after its state
def test_run_real_time_on_a_simplant():
    """The rehearsal: the robot's loop on wall time, with gravity-holding plans from a fake Mpc."""
    import time
    from dataclasses import replace
    from test_experiments import Mpc
    from test_loops import Task
    from tiago_mpc.loops import run_real_time
    from tiago_mpc.mpc import TIAGO

    plant, ticks = SimPlant(substep=.003), []
    q0, start = plant.q.copy(), time.monotonic()
    outcome = run_real_time(plant, Mpc(replace(TIAGO, N=8)), Task(), callbacks=[ticks.append],
                            command_period=.01, update_period=.01, duration=.2)
    assert not outcome.done and time.monotonic() - start >= .2 and len(ticks) == 21
    # Each plan is used from the tick after its request, which makes the next request.
    assert ticks[0].command is None and all(k.command is not None and k.new for k in ticks[1:-1])
    np.testing.assert_allclose(plant.q, q0, atol=1e-3)
