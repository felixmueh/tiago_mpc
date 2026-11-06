"""The run format (tiago_mpc/record.py): configurations to and from dicts, and a run written
by RunRecord and read back by load_run (CPU only)."""
from dataclasses import MISSING, dataclass, fields, is_dataclass, replace
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.figure8 import Figure8
from experiments.friction_compensation import SampledSimConfig
from experiments.latency_compensation import Compensated
from experiments.random100.run import Case, Random100Config, Random100Study
from experiments.reach import Reach
from gato.cost_menu import CostMenu, LinearTerm, Term
from test_loops import Mpc, Plant, Task
from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh
from tiago_mpc.laws import EffortClip, FrictionFeedforward, HoldFirst, PlanFeedback, RejectOnEffort, SampleKnot
from tiago_mpc.loops import RealTime, VirtualTime, run_virtual_time
from tiago_mpc.mpc import TIAGO, AdmmLimits, MpcParams, SolverSettings
from tiago_mpc.prediction import LatencyCompensation
from tiago_mpc.record import CLASSES, FORMAT, Event, RunRecord, from_dict, load_run, to_dict
from tiago_mpc.robot import Checks, RobotConfig
from tiago_mpc.safety import CollisionSafetySettings
from tiago_mpc.sim import SimConfig
from tiago_mpc.tiago import urdf_friction


def test_a_run_is_written_and_read_back(tmp_path):
    """A fake loop's run, its states 2 ms old, with the plant's events, an experiment column,
    a stats column and results: the loaded run has the record's read interface and values."""
    class Emitting(Plant):
        def advance(self, t):
            super().advance(t)
            self.events.append(Event(t, "pacing_behind", "behind", dict(behind_s=.001)))

    plant, mpc, task, timing = Emitting(measure=lambda t: t - .002), Mpc(wall=.004), Task(), VirtualTime(.01, .02, .01)
    record = RunRecord(tmp_path / "r", task=task, params=TIAGO,
                       plant=SimConfig(friction=urdf_friction(), friction_law=Tanh(.03)),
                       law=SampleKnot(), timing=timing, duration=.05, case="first")
    written = json.loads((tmp_path / "r" / "summary.json").read_text())  # at once
    assert written["format"] == FORMAT and written["metadata"]["status"] == "running"
    assert written["config"]["timing"] == dict(type="VirtualTime", command_period=.01, update_period=.02,
                                               solve_latency_s=.01, compensation=None, max_plan_age_s=.1,
                                               effort_handling=to_dict(EffortClip()))
    outcome = timing.run(plant, mpc, task, SampleKnot(), [record], duration=.05)
    record.add_events([Event(plant.clock.now(), "handback", "handed back", dict(restored=True, errors=[]))])
    record.ticks["distance_m"] = np.arange(len(record.ticks["t"])) * .1
    record.plans["label"] = np.array([f"plan {i}" for i in range(len(record.plans["t0"]))])
    record.metadata.update(status=outcome.status, end_t=outcome.t, results=dict(final_distance_m=.5))
    record.save()

    ticks, plans = record.ticks, record.plans  # times from the first state's, -2 ms
    np.testing.assert_allclose(ticks["t"], np.arange(6) * .01 + .002, atol=1e-12)
    np.testing.assert_allclose(ticks["state_t"], np.arange(6) * .01, atol=1e-12)
    np.testing.assert_array_equal(ticks["plan"], [-1, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(ticks["update"], [0, -1, 1, -1, 2, -1])
    np.testing.assert_allclose(plans["update_t"], [.002, .022, .042], atol=1e-12)  # the requests' ticks
    np.testing.assert_allclose(plans["t0"], [0., .02, .04], atol=1e-12)  # their states
    assert set(plans) == {"t0", "measured_t", "update_t", "states", "torques", "wall_s", "label"}
    assert [(e.kind, round(e.t, 6)) for e in record.events] == [
        ("pacing_behind", .012), ("late_plan", .012), ("pacing_behind", .022), ("pacing_behind", .032),
        ("late_plan", .032), ("pacing_behind", .042), ("pacing_behind", .052), ("handback", .052)]

    run = load_run(tmp_path / "r", classes=[Task])
    assert run.out == record.out and run.config == record.config and run.config["task"] == task
    assert run.config["case"] == "first" and run.provenance == record.provenance and run.metadata == record.metadata
    assert run.events == record.events
    for loaded, saved in ((run.ticks, ticks), (run.plans, plans)):
        assert set(loaded) == set(saved)
        for key, value in saved.items():
            np.testing.assert_array_equal(loaded[key], value)
            assert loaded[key].dtype == value.dtype
    assert load_run(tmp_path / "r").config["task"] == dict(type="Task", done_at=float("inf"))  # its class not given
    run.ticks["speed"] = np.zeros(len(run.ticks["t"]))  # a metric added later, written back
    run.ticks["distance_m"] = run.ticks["distance_m"] * 2  # recomputed: an experiment column, not a standard one
    run.metadata["results"]["later"] = 1.
    run.save()
    again = load_run(tmp_path / "r")
    assert "speed" in again.ticks and again.metadata["results"] == dict(final_distance_m=.5, later=1.)


def test_save_refuses_a_column_of_the_wrong_length(tmp_path):
    record = RunRecord(tmp_path / "r", task=Task())
    run_virtual_time(Plant(), Mpc(), Task(), tick_callbacks=[record], command_period=.01, update_period=.01,
                     duration=.03, solve_latency_s=0.)
    record.ticks["short"] = np.zeros(2)
    with pytest.raises(ValueError, match="rows"):
        record.save()


@pytest.mark.parametrize("old_format", [1, 2])
def test_load_run_refuses_a_run_of_another_format(tmp_path, old_format):
    record = RunRecord(tmp_path / "r", task=Task())
    record.save()
    summary = json.loads((tmp_path / "r" / "summary.json").read_text())
    (tmp_path / "r" / "summary.json").write_text(json.dumps({**summary, "format": old_format}))
    with pytest.raises(ValueError, match=f"format {old_format}, not {FORMAT}: load it with the code that wrote it"):
        load_run(tmp_path / "r")


@dataclass(frozen=True)
class Target:
    position_m: tuple[float, float, float] = (0., 0., 0.)


def test_from_dict_rebuilds_known_types_keeps_unknown_ones_and_refuses_two_classes_of_one_name():
    data = json.loads(json.dumps(dict(task=to_dict(Target((1., 2., 3.))), plain=[1, [2, 3]], other={"type": "Other"})))
    assert from_dict(data, [Target]) == dict(task=Target((1., 2., 3.)), plain=(1, (2, 3)), other=dict(type="Other"))
    assert from_dict(data)["task"] == dict(type="Target", position_m=(1., 2., 3.))

    @dataclass(frozen=True)
    class SimConfig:  # an experiment's class with a package class's name
        pass

    with pytest.raises(ValueError, match="two classes named SimConfig"):
        from_dict(data, [SimConfig])
    with pytest.raises(ValueError, match="two classes named Target"):
        from_dict(data, [Target, type("Target", (), {})])


@pytest.mark.parametrize("config", [RobotConfig(execute=True, sea_absolute=True),
                                    CollisionSafetySettings(urdf_path=Path("/tmp/arm.urdf"))])
def test_the_robot_configs_round_trip_with_their_paths(config):
    """from_dict knows no Path: CollisionSafetySettings converts its strings."""
    rebuilt = from_dict(json.loads(json.dumps(to_dict(config))))
    assert rebuilt == config
    safety = rebuilt.safety if isinstance(rebuilt, RobotConfig) else rebuilt
    assert isinstance(safety.blacklist_path, Path)


def test_a_native_cost_menu_survives_a_saved_run(tmp_path):
    """Native terms, per-knot arrays, rotations, gravity targets and infinite bounds reload."""
    n = 4
    menu = CostMenu(
        frame_pos=Term(weight=np.ones((n, 3), dtype=np.float32), target=np.arange(n * 3).reshape(n, 3)),
        state=Term(weight=np.ones(14), activation=("quadratic_barrier",) * n,
                   lower=np.full(14, -np.inf), upper=np.ones(14)),
        input=Term(weight=.1, target="gravity"), q_ref=np.full((n, 7), .2),
        orientation=Term(weight=np.full((n, 3), .5), target=np.repeat(np.eye(3)[None], n, axis=0)),
        linear_u=LinearTerm(weight=.2, target=np.zeros((n, 7)), A=np.eye(7), b=np.arange(7),
                            activation="pseudo_huber", eps=np.full(n, .03)))
    params = replace(TIAGO, N=n, cost=menu)
    record = RunRecord(tmp_path / "menu", params=params)
    written = json.loads((record.out / "summary.json").read_text())
    assert written["config"]["params"]["cost"] == to_dict(menu)
    record.save()

    run = load_run(record.out)
    rebuilt = run.config["params"]
    assert isinstance(rebuilt, MpcParams) and isinstance(rebuilt.cost, CostMenu)
    assert isinstance(rebuilt.cost.frame_pos, Term)
    assert isinstance(rebuilt.cost.linear_u, LinearTerm)
    assert rebuilt.cost.input.target == "gravity"
    assert to_dict(rebuilt) == to_dict(params)
    np.testing.assert_array_equal(rebuilt.cost.state.lower, np.full(14, -np.inf))
    np.testing.assert_array_equal(rebuilt.cost.orientation.target, menu.orientation.target)
    run.save()
    assert to_dict(load_run(record.out).config["params"]) == to_dict(params)


# One instance of each class the record rebuilds, with every field away from its default. A
# field that the serializer drops or rebuilds wrongly then fails the round trip below.
FRICTION = Friction([1.] * 7, [.5] * 7)
LATENCY = LatencyCompensation(latency_s=.03, initial_latency_s=.01, friction=FRICTION, friction_law=Tanh(.05),
                              armature=(.1,) * 7, substep=.002)
TERM = Term(weight=(1., 2., 3.), target=(.1, .2, .3), activation="smooth_hinge", eps=.03,
            lower=(-1.,) * 3, upper=(1.,) * 3)
LINEAR = LinearTerm(weight=.2, target=(.1,) * 7, activation="pseudo_huber", eps=.05,
                    lower=(-2.,) * 7, upper=(2.,) * 7,
                    A=tuple(tuple(row) for row in np.eye(7)), b=(.2,) * 7)
COST_MENU = CostMenu(
    frame_pos=TERM, state=Term(weight=(0.,) * 7 + (.1,) * 7), input=Term(weight=.01, target="gravity"),
    linear_x=LinearTerm(weight=.3, A=tuple(tuple(row) for row in np.eye(14))), linear_u=LINEAR,
    orientation=Term(weight=.4, target=tuple(tuple(row) for row in np.eye(3))),
    velocity=Term(weight=.2, target=(.1,) * 6), frame_pos_2=TERM, q_ref=((.1,) * 7,) * 8)
SAMPLES = [
    SolverSettings(max_sqp_iters=3, max_pcg_iters=50, pcg_tol=1e-5, solve_ratio=.5, mu=2., rho=.02,
                 adapt_rho=False, linsys="bdsv",
                 q_lim_cost=.1, vel_lim_cost=.1, ctrl_lim_cost=.1, exact_hessian=True,
                 cost_curvature=True, defect_telemetry=True, defect_exit=.1),
    AdmmLimits(iters=4, eps_abs=1e-4, eps_rel=1e-3, check_every=2, rho_q=.2, rho_qd=.04, rho_u=2e-5),
    MpcParams(N=8, dt_s=.02, solver=SolverSettings(max_sqp_iters=2), warm_start="hold",
              reset_rho_each_step=False, admm_linsys="auto", admm_threshold=.2, admm=AdmmLimits(iters=4),
              linsys="bdsv", bdsv_threshold=.3, reseed_threshold=.5, joint_margin_rad=.05,
              velocity_fraction=.5, effort_fraction=.8, friction=FRICTION, friction_law=Tanh(.05),
              armature=(.1,) * 7, max_sqp_time_s=None, warmup=False, cost=COST_MENU),
    COST_MENU, TERM, LINEAR,
    FRICTION, Tanh(.05), Sign(), StickSlip(), SampleKnot(), HoldFirst(),
    PlanFeedback(kp=(1.,) * 7, kd=(2.,) * 7),
    FrictionFeedforward(friction=FRICTION, base_law=HoldFirst(), evaluation=Tanh(.05)),
    EffortClip(effort=(2.,) * 7), RejectOnEffort(effort=(3.,) * 7),
    VirtualTime(.01, .02, .01, compensation=LATENCY, max_plan_age_s=.2, effort_handling=RejectOnEffort()),
    RealTime(command_period=.004, update_period=.04, compensation=None),
    LATENCY,
    SimConfig(q=(.1,) * 7, qd=(.2,) * 7, substep=.002, friction=FRICTION, friction_law=Sign(),
              integrator="euler", sync_real_time=True, armature=(.1,) * 7),
    RobotConfig(execute=True, sea_absolute=True,
                checks=Checks(state_age_s=.2, first_state_s=10., service_s=1., start_rad=.1,
                              start_speed_l1_rad_s=2., body_drift_rad=.03, torso_drift_m=.01),
                safety=CollisionSafetySettings(enabled=False)),
    Checks(state_age_s=.2, first_state_s=10., service_s=1., start_rad=.1, start_speed_l1_rad_s=2.,
           body_drift_rad=.03, torso_drift_m=.01),
    CollisionSafetySettings(enabled=False, urdf_path=Path("/tmp/arm.urdf"), blacklist_path=Path("/tmp/b.yaml"),
                            controlled_joint_names=("a", "b"), min_distance_m=.05, check_timeout_sec=.1,
                            max_monitored_geometry_speed_m_s=2., joint_position_margin_rad=.01,
                            joint_velocity_scale=.5),
]

SAMPLE = {type(sample).__name__: sample for sample in SAMPLES}

# The experiments' classes that their records hold. load_run rebuilds them when given them.
REACH = Reach(target_m=(.4, -.2, .9), position_tol_m=.03, speed_tol_m_s=.01, timeout_s=5.,
              orientation_rad=(.1, .2, .3))
CASE = Case(name="c", initial_state=(.1,) * 14, target_m=(.4, -.2, .9))
RANDOM100 = Random100Config(ids=(1, 2), inputs_sha256="ab", params=SAMPLE["MpcParams"], plant=SAMPLE["SimConfig"],
                            timing=SAMPLE["VirtualTime"])
EXPERIMENT_SAMPLES = [
    REACH, Figure8(start_m=(.4, -.2, .9), x_span_m=.2, y_amplitude_m=.05, edge_clearance_m=.01, period_s=12.,
                   orientation_rad=(.2, .1, -.2)),
    Compensated(solve_latency_s=.03, command_period=.004, update_period=.02, compensate=False,
                prediction_substep_s=.001, predictor="gpu"),
    SampledSimConfig(physical=SAMPLE["SimConfig"], feedback_period_s=.005),
    CASE, RANDOM100, Random100Study(config=RANDOM100, cases=(CASE,)),
]
EXPERIMENT_CLASSES = [type(sample) for sample in EXPERIMENT_SAMPLES]


def nested(obj):
    """obj and every dataclass in its fields and their tuples, at any depth."""
    yield obj
    for field in fields(obj):
        value = getattr(obj, field.name)
        for part in value if isinstance(value, tuple) else (value,):
            if is_dataclass(part):
                yield from nested(part)


def test_the_samples_cover_every_recorded_class_and_field():
    """A class added to CLASSES, or a field added to a sampled class, needs a sample value here."""
    assert sorted(type(sample).__name__ for sample in SAMPLES) == sorted(cls.__name__ for cls in CLASSES)
    for sample in SAMPLES + EXPERIMENT_SAMPLES:
        for field in fields(sample):
            default = (field.default if field.default is not MISSING
                       else field.default_factory() if field.default_factory is not MISSING else MISSING)
            assert getattr(sample, field.name) != default, f"{type(sample).__name__}.{field.name} is its default"


@pytest.mark.parametrize("sample", SAMPLES + EXPERIMENT_SAMPLES, ids=lambda sample: type(sample).__name__)
def test_every_field_of_every_recorded_class_survives_the_record(sample):
    """Every field survives the record.

    to_dict writes every field, from_dict rebuilds an equal object through JSON, and every
    dataclass inside is one that from_dict knows."""
    data = to_dict(sample)
    assert set(data) == {"type"} | {field.name for field in fields(sample)}
    assert from_dict(json.loads(json.dumps(data)), EXPERIMENT_CLASSES) == sample
    assert {type(part) for part in nested(sample)} <= {*CLASSES, *EXPERIMENT_CLASSES}
