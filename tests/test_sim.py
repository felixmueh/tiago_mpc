"""SimConfig and SimPlant: held commands on virtual and wall time, and pacing."""
from dataclasses import replace

from gato.common import rk4
import numpy as np
import pytest

from tiago_mpc import clock as clock_module, sim
from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.dynamics import Sign, StickSlip, Tanh, plant_euler, plant_rk4
from tiago_mpc.record import from_dict, to_dict
from tiago_mpc.sim import SimConfig, SimPlant
from tiago_mpc.tiago import Q_START, urdf_friction

U = np.linspace(-1., 1., 7)


@pytest.mark.parametrize("friction,law,substep,integrator", [(None, StickSlip(), .003, "rk4"),
                                                            (None, StickSlip(), .001, "euler"),
                                                            (urdf_friction(), Tanh(.03), .001, "euler"),
                                                            (urdf_friction(), Sign(), .001, "euler"),
                                                            (urdf_friction(), StickSlip(), .001, "euler"),
                                                            (urdf_friction(), Tanh(.03), 1e-4, "rk4"),
                                                            (urdf_friction(), Sign(), 1e-4, "rk4")])
def test_advance_holds_the_command_over_substeps_up_to_the_tick(friction, law, substep, integrator):
    plant = SimConfig(substep=substep, friction=friction, friction_law=law, integrator=integrator).build()
    q, qd, t = plant.q.copy(), plant.qd.copy(), 0.
    plant.start(VirtualClock())
    plant.apply(U)
    plant.advance(.01)
    while t < .01 - 1e-12:  # e.g. 3, 3, 3 and 1 ms
        h = min(substep, .01 - t)
        q, qd = (plant_euler(plant.model, plant.data, q, qd, U, h, friction, law) if integrator == "euler"
                 else rk4(plant.model, plant.data, q, qd, U, h) if friction is None
                 else plant_rk4(plant.model, plant.data, q, qd, U, h, friction, law))
        t += h
    s = plant.state()
    assert s.t == plant.clock.now() == .01 and plant.run_state == {} and plant.take_events() == []
    np.testing.assert_array_equal(s.q, q)
    np.testing.assert_array_equal(s.qd, qd)
    plant.apply(-U)  # the state is at the virtual time: nothing to integrate
    assert plant.state().t == .01
    np.testing.assert_array_equal(plant.q, q)
    plant.stop()
    np.testing.assert_array_equal(plant.command, np.zeros(7))


@pytest.mark.parametrize("settings", [dict(substep=0), dict(substep=-.001), dict(substep=np.nan),
                                     dict(integrator="other"), dict(friction=urdf_friction()),  # rk4, the default
                                     dict(integrator="rk4", friction=urdf_friction(), friction_law=StickSlip())])
def test_invalid_integration_settings_fail_when_the_plant_is_built(settings):
    config = SimConfig(**settings)  # a value; SimPlant checks it
    with pytest.raises(ValueError):
        config.build()


def test_a_sim_config_is_a_value_of_tuples_that_round_trips():
    config = SimConfig(q=np.asarray(Q_START), qd=[0] * 7, friction=urdf_friction(),
                       friction_law=Tanh(.03), integrator="euler")
    assert config.q == tuple(Q_START) and config.qd == (0.,) * 7 and hash(config) == hash(replace(config))
    assert from_dict(to_dict(config)) == config and SimConfig().integrator == "rk4"
    plant = config.build()
    assert plant.config is config and plant.q.dtype == float
    np.testing.assert_array_equal(plant.q, Q_START)


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
    plant = SimConfig(substep=.01).build()
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


def test_on_wall_time_the_start_state_holds_until_the_first_command(wall):
    """Held through a tick and a 50 ms first solve (or warm-up): read then, the held state
    has the clock's time; the command integrates from when it is applied."""
    plant = SimConfig(substep=.01).build()
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
    plant = SimConfig(substep=.003).build()
    q0 = plant.q.copy()
    plant.start(WallClock())
    plant.apply(np.ones(7))
    plant.advance(100.01)
    wall.now = 100.05
    plant.apply(2 * np.ones(7))
    plant.advance(100.06)
    np.testing.assert_allclose(plant.q - q0, .07)


def test_sync_real_time_waits_for_the_wall_clock_has_an_event_behind_and_keeps_the_results(wall):
    """Counted from the first tick, so the warm-up before it neither counts nor runs the start fast."""
    paced, plain = SimConfig(substep=.003, sync_real_time=True).build(), SimConfig(substep=.003).build()
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
    assert paced.take_events() == []
    wall.now = first + .045  # 15 ms behind at 30 ms: no wait, one event
    paced.advance(5.03)
    behind, = paced.take_events()
    assert behind.kind == "pacing_behind" and behind.t == 5.03 and behind.data["behind_s"] == pytest.approx(.015)
    assert "15.0 ms behind the wall clock" in behind.message
    wall.now += .02
    paced.advance(5.04)  # still behind: the same occurrence
    assert wall.slept[2:] == [0., 0.] and paced.take_events() == []
    wall.now = first + .04
    paced.advance(5.05)  # caught up: it waits again
    wall.now = first + .07
    paced.advance(5.06)  # behind again: a new occurrence
    assert [e.kind for e in paced.take_events()] == ["pacing_behind"]
    for t in (5.01, 5.02, 5.03, 5.04, 5.05, 5.06):
        plain.advance(t)
    np.testing.assert_array_equal(paced.q, plain.q)
    np.testing.assert_array_equal(paced.qd, plain.qd)


def test_sync_real_time_refuses_wall_time(wall):
    with pytest.raises(ValueError, match="sync_real_time"):
        SimConfig(sync_real_time=True).build().start(WallClock())


def test_run_real_time_on_a_simplant():
    """The rehearsal: the robot's loop on wall time, with gravity-holding plans from a fake Mpc."""
    import time
    from dataclasses import replace
    from test_experiments import Mpc
    from test_loops import Task
    from tiago_mpc.loops import RealTimeSettings, run_real_time
    from tiago_mpc.mpc import TIAGO
    from tiago_mpc.costs import default_cost

    plant, ticks = SimConfig(substep=.003).build(), []
    q0, start = plant.q.copy(), time.monotonic()
    outcome = run_real_time(plant, Mpc(replace(TIAGO, N=8, cost=default_cost(8))), Task(), tick_callbacks=[ticks.append],
                            settings=RealTimeSettings(.01, .01), compensation=None, duration=.2)
    assert outcome.status == "timeout" and time.monotonic() - start >= .2 and len(ticks) == 21
    # Each plan is used from the tick after its request, which makes the next request.
    assert ticks[0].command is None and all(k.command is not None and k.new for k in ticks[1:-1])
    np.testing.assert_allclose(plant.q, q0, atol=1e-3)
