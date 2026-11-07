"""The feedforward torque uses the intended motion and composes with feedback."""
import numpy as np
import pytest

from tiago_mpc.laws import FrictionFeedforward, PlanFeedback
from tiago_mpc.util import inertia_gains
from tiago_mpc.dynamics import Friction, StickSlip, Tanh
from tiago_mpc.interfaces import Plan, State


def test_pd_initialization_accounts_for_reflected_armature_once():
    armature = np.array([.7, .8, .2, .6, .3, .35, .27])
    old_kp, old_kd = inertia_gains(20., 1.4)
    kp, kd = inertia_gains(20., 1.4, armature=armature)
    np.testing.assert_allclose(np.subtract(kp, old_kp), armature*20.**2)
    np.testing.assert_allclose(np.subtract(kd, old_kd), armature*2*1.4*20.)
    with pytest.raises(ValueError, match='armature'):
        inertia_gains(20., armature=(-1.,)*7)


def test_friction_feedforward_uses_planned_velocity_and_native_signs():
    v0 = np.array([.1, -.1, .2, -.2, .3, -.3, 0.])
    v1 = 2*v0
    states = np.array([np.r_[np.zeros(7), v] for v in (v0, v1)])
    torque = np.arange(7.)[None]
    plan = Plan(1., .01, states, torque, measured_t=1.)
    measured = State(1.005, np.zeros(7), -v0)
    friction = Friction((1.,)*7, (.5,)*7)
    expected = torque[0] + np.tanh(1.5*v0/.03) + .5*1.5*v0
    np.testing.assert_allclose(FrictionFeedforward(friction)(plan, 1.005, measured).applied, expected)
    pd = PlanFeedback(kp=0., kd=2.)
    np.testing.assert_allclose(FrictionFeedforward(friction, pd)(plan, 1.005, measured).applied,
                               expected + 2*(1.5*v0 - measured.qd))
    with pytest.raises(RuntimeError, match='last knot'):
        FrictionFeedforward(friction)(plan, 1.02, measured)


def test_smooth_feedforward_has_no_breakaway_kick_at_rest():
    plan = Plan(0., .01, np.zeros((2, 14)), np.ones((1, 7)), measured_t=0.)
    law = FrictionFeedforward(Friction((3.,)*7, (2.,)*7))
    np.testing.assert_array_equal(law(plan, .004, State(.004, np.zeros(7), np.zeros(7))).applied, np.ones(7))
    with pytest.raises(ValueError, match='explicit'):
        FrictionFeedforward(Friction((1.,)*7, (0.,)*7), evaluation=StickSlip())


class ContinuousPlant:
    """A known linear trajectory isolates timing from integration and optimization."""
    def __init__(self):
        self.run_state, self.visited, self.commands = {}, [], []

    def start(self, clock):
        self.clock = clock
        self.t = clock.now()

    def state(self):
        import tiago_mpc as tm
        return State(self.t, tm.Q_START + .01*self.t, np.full(7, .01))

    def advance(self, t):
        self.t = t
        self.visited.append(t)
        self.clock.reach(t)

    def apply(self, command):
        self.commands.append((self.t, command.copy()))

    def take_diagnostics(self):
        return []

    def stop(self):
        pass

    def close(self):
        pass


def test_feedback_samples_between_command_ticks_and_holds_original_timestamps(monkeypatch):
    import tiago_mpc as tm
    from experiments.friction_compensation import SampledSimConfig

    physical = ContinuousPlant()
    monkeypatch.setattr(tm.SimConfig, 'build', lambda self: physical)
    plant = SampledSimConfig(tm.SimConfig(), .02).build()
    plant.start(tm.VirtualClock())
    plant.state()  # warmup
    plant.state()  # first tick: no duplicate truth event
    first, = plant.take_diagnostics()
    assert first.t == 0 and first.kind == 'plant_state'
    saved = plant.state()
    for t in (.008, .016):
        plant.advance(t)
        assert plant.state().t == 0
    plant.advance(.024)
    observed = plant.state()
    assert observed.t == .02 and .02 in physical.visited
    np.testing.assert_allclose(observed.q, tm.Q_START + .0002)
    np.testing.assert_array_equal(saved.q, tm.Q_START)  # cached data never mutates
    plant.apply(np.ones(7))
    assert physical.commands[-1][0] == .024
    plant.advance(.04 - 1e-16)  # coincident event with a rounded command deadline
    assert plant.state().t == pytest.approx(.04)
    assert plant.clock.now() == .04 - 1e-16  # no advance beyond deadline/backwards clock
    plant.start(tm.VirtualClock())  # reset clears the old sampled state and truth events
    assert plant.state().t == 0 and len(plant.take_diagnostics()) == 1
    with pytest.raises(ValueError, match='virtual time'):
        plant.start(tm.WallClock())


def test_saved_metrics_and_portable_motion_use_truth_separately_from_feedback(tmp_path, monkeypatch):
    from dataclasses import replace
    import json
    from test_loops import Mpc
    import tiago_mpc as tm
    from experiments.common import run_experiment
    from experiments.figure8 import Figure8
    from experiments.friction_compensation import (PARAMS, SampledSimConfig,
                                                  export_and_measure)

    monkeypatch.setattr(tm.SimConfig, 'build', lambda self: ContinuousPlant())
    config = SampledSimConfig(tm.SimConfig(), .02)
    timing = tm.VirtualTimeSettings(.008, .008, .0055)
    mpc = Mpc(N=3, dt=.03, wall=.004)
    mpc.params = replace(PARAMS, N=3, cost=tm.default_cost(3, velocity=.01, torque=1e-4, terminal_position=80.))
    record = run_experiment(config, mpc, Figure8(period_s=.048), tm.SampleKnot(), timing, out=tmp_path/'run')
    row = export_and_measure(record, tmp_path/'executed.npz', 'default')
    assert row['status'] == 'done' and row['samples'] == 7
    np.testing.assert_allclose(record.ticks['state_t'], [0, 0, 0, .02, .02, .04, .04])
    np.testing.assert_allclose(record.plans['update_t'][:3], [0, .008, .016])
    np.testing.assert_allclose(record.plans['t0'][:3], [0, 0, 0])
    assert row['observation_age_max_s'] == pytest.approx(.016)
    with np.load(tmp_path/'executed.npz') as data:
        np.testing.assert_allclose(data['q'], tm.Q_START + .01*data['time_s'][:, None])
        assert np.any(data['q'] != record.ticks['q'])  # held feedback isn't relabelled as executed motion
        meta = json.loads(str(data['metadata_json']))
        assert meta['time_basis'] == 'simulation' and 'true simulated state' in meta['provenance']['motion']
    assert (record.out/'held-feedback-replay.npz').exists()
    loaded = tm.load_run(record.out, classes=[Figure8, SampledSimConfig])
    assert loaded.config['plant'] == config and loaded.config['settings'] == timing
    np.testing.assert_array_equal(loaded.ticks['plant_q'], record.ticks['plant_q'])
    assert loaded.metadata['results']['plant_metrics_full_cycle'] != loaded.metadata['results']['metrics_full_cycle']


def test_feedback_phase_stays_correct_over_a_full_cycle_of_noncommensurate_ticks(monkeypatch):
    import tiago_mpc as tm
    from experiments.friction_compensation import SampledSimConfig

    monkeypatch.setattr(tm.SimConfig, 'build', lambda self: ContinuousPlant())
    plant = SampledSimConfig(tm.SimConfig(), .02).build()
    plant.start(tm.VirtualClock())
    t = 0.
    for tick in range(3001):
        # 8 ms commands and 20 ms feedback coincide every fifth command tick.
        expected_sample = (tick*8 // 20)*.02
        assert plant.state().t == pytest.approx(expected_sample, abs=1e-9)
        assert 0 <= t-plant.state().t <= .016 + 1e-9
        t += .008
        plant.advance(t)


def test_six_cases_share_armature_and_internal_friction_has_no_external_feedforward():
    import tiago_mpc as tm
    from experiments.friction_compensation import (ARMATURE_KG_M2, PARAMS, cases, case_params, plant_config)

    laws = cases()
    assert len(laws) == 6
    assert plant_config().physical.armature == PARAMS.armature == ARMATURE_KG_M2
    for key, law in laws.items():
        params = case_params(key)
        assert params.armature == ARMATURE_KG_M2
        if key.startswith('mpc_friction'):
            assert isinstance(params.friction_law, tm.Tanh)
            assert not isinstance(law, FrictionFeedforward)
        else:
            assert params.friction is None
