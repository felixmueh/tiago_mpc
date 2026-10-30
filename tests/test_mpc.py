"""MpcParams, plant names, loading GATO's module, the settings GATO receives, and both Mpc
twins on fakes (CPU only)."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import queue
import sys
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import gato
from gato import _cache
import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc import mpc
from experiments.figure8 import FIGURE8
from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh
from tiago_mpc.mpc import TIAGO, Mpc, MpcParams, MpcProcess, compile_binary, find_binary
from tiago_mpc.interfaces import Plan, Prediction, State
from tiago_mpc.record import from_dict, to_dict
from tiago_mpc.tiago import ARMATURE_KG_M2, GRASP_FRAME, Q_START, arm_urdf, fk, limits, urdf_friction

URDF_FRICTION = urdf_friction()  # the coefficients come from the URDF only
LOSSES = Friction([1.] * 7, [.5] * 7)


@pytest.mark.parametrize("params", [TIAGO, FIGURE8, replace(TIAGO, N=8, admm=replace(TIAGO.admm, iters=5))])
def test_to_dict_round_trips_also_through_json_and_pickle(params):
    assert from_dict(to_dict(params)) == params
    assert from_dict(json.loads(json.dumps(to_dict(params)))) == params
    assert pickle.loads(pickle.dumps(params)) == params


def test_the_admm_limits_are_gatos_defaults_except_rho_u():
    ours, gatos = asdict(TIAGO.admm), gato.MPCController.LIMIT_DEFAULTS["admm"]
    assert {k: v for k, v in ours.items() if k != "rho_u"} == {k: v for k, v in gatos.items() if k != "rho_u"}
    assert ours["rho_u"] == 5e-4


@pytest.fixture
def no_compile(monkeypatch, tmp_path):
    """A fresh GATO cache in which compiling or a solver raises."""
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    def compiled(*args, **kwargs):
        raise AssertionError("compiled")
    for name in ("ensure_module", "load_module"):
        monkeypatch.setattr(_cache, name, compiled)
    monkeypatch.setattr(gato, "BSQP", compiled)


def fake_binary(directory, params, plant=None):
    """An importable Python file named as GATO's module for params."""
    path = Path(directory) / f"bsqpN{params.N}_{plant or mpc._plant(params)[2]}.py"
    path.write_text("")
    return path


@pytest.mark.parametrize("params", [TIAGO, FIGURE8, replace(FIGURE8, friction_law=Tanh(.05))])
def test_load_controller_imports_the_binary_and_passes_the_params_to_gato(monkeypatch, no_compile, tmp_path, params):
    seen = {}
    binary = fake_binary(tmp_path, params)
    name = "gato." + binary.stem

    class Solver:
        def __init__(self, urdf, *, batch_size, N, dt, plant_type, params):
            seen.update(N=N, dt=dt, solver=params.asdict(), module=sys.modules[name].__file__)

        def set_row_group_bounds(self, group, lo, hi):
            seen.setdefault("bounds", []).append(group)

    monkeypatch.setattr(gato, "register_robot", lambda name, path, **kwargs: seen.update(register=kwargs))
    monkeypatch.setattr(gato, "BSQP", Solver)
    monkeypatch.setattr(gato, "MPCController", lambda solver, **kwargs: seen.update(controller=kwargs),
                        raising=False)
    monkeypatch.setattr(mpc, "check_model", lambda *args: None)
    try:
        mpc.load_controller(params, binary)
    finally:
        sys.modules.pop(name, None)
    assert (seen["N"], seen["dt"], seen["solver"]) == (params.N, params.dt_s, params.solver.asdict())
    assert seen["controller"]["limits_kw"] == asdict(params.admm) and seen["bounds"] == [0, 1, 2]
    assert seen["module"] == str(binary)
    # Smooth predictor friction is registered, with its width, only when the params have one; the
    # supplied armature is the default.
    friction = {} if params.friction is None else dict(joint_dynamics=True,
                                                         friction_width=params.friction_law.width_rad_s)
    assert seen["register"] == dict(N=[params.N], ee_frame=GRASP_FRAME, joint_armature=ARMATURE_KG_M2, **friction)


def test_load_controller_registers_only_a_plant_new_to_the_cache(monkeypatch, no_compile, tmp_path):
    params, seen = replace(TIAGO, N=3), []
    binary = fake_binary(tmp_path, params)

    def register(name, path, **kwargs):  # what GATO's registration leaves in the cache
        cache, key = _cache.cache_dir(), "key"
        (cache / "robots" / key / name).mkdir(parents=True)
        for header in ("grid.cuh", "limits.cuh"):
            (cache / "robots" / key / name / header).write_text("")
        _cache._save_manifest(cache, {"schema": _cache.MANIFEST_SCHEMA,
                                      "robots": {name: {"name": name, "codegen_key": key}}})
        seen.append(name)

    def refuse(*args, **kwargs):
        raise AssertionError("registered or locked again")

    monkeypatch.setattr(gato, "register_robot", register)
    monkeypatch.setattr(gato, "BSQP", lambda *args, **kwargs: NS(set_row_group_bounds=lambda *a: None))
    monkeypatch.setattr(gato, "MPCController", lambda solver, **kwargs: None, raising=False)
    monkeypatch.setattr(mpc, "check_model", lambda *args: None)
    try:
        mpc.load_controller(params, binary)  # missing entry: registers
        assert len(seen) == 1
        monkeypatch.setattr(gato, "register_robot", refuse)
        monkeypatch.setattr(_cache, "_locked", refuse)
        mpc.load_controller(params, binary)  # registered: neither registers nor locks
        (_cache.cache_dir() / "robots" / "key" / seen[0] / "limits.cuh").unlink()
        monkeypatch.setattr(gato, "register_robot", lambda name, path, **kwargs: seen.append(name))
        mpc.load_controller(params, binary)  # headers missing: registers again
    finally:
        sys.modules.pop("gato." + binary.stem, None)
    assert len(seen) == 2


def test_load_controller_refuses_a_missing_or_foreign_binary_without_compiling(monkeypatch, no_compile, tmp_path):
    params = replace(TIAGO, N=3)
    with pytest.raises(FileNotFoundError, match="compile_binary"):
        Mpc(params, None)
    with pytest.raises(FileNotFoundError):
        Mpc(params, tmp_path / "missing.so")
    for other in (replace(params, N=4), replace(params, friction=LOSSES)):
        with pytest.raises(ValueError, match="is not the GATO module"):
            Mpc(params, fake_binary(tmp_path, other))
    binary = fake_binary(tmp_path, params)
    binary.write_text("raise OSError('not a module')")
    with pytest.raises(ImportError, match="not a module"):
        Mpc(params, binary)
    assert "gato." + binary.stem not in sys.modules
    (tmp_path / "other").mkdir()
    elsewhere = fake_binary(tmp_path / "other", params)
    monkeypatch.setitem(sys.modules, "gato." + binary.stem, NS(__file__=str(elsewhere)))
    with pytest.raises(RuntimeError, match="already loaded"):
        Mpc(params, binary)


def test_find_binary_finds_nothing_for_an_unregistered_plant(no_compile):
    assert find_binary(replace(TIAGO, N=3)) is None


@pytest.mark.parametrize("use_cache", [True, False])
def test_compile_binary_registers_and_rebuilds_without_overwriting(monkeypatch, tmp_path, use_cache):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    old, seen = tmp_path / "old.so", {}
    old.write_bytes(b"mapped")
    monkeypatch.setattr(gato, "register_robot", lambda name, path, **kwargs: seen.update(register=name))
    monkeypatch.setattr(_cache, "find_module", lambda name, N: old)
    def ensure(name, N, jobs):
        seen.update(ensure=(name, N, jobs), old=old.exists())
        return old
    monkeypatch.setattr(_cache, "ensure_module", ensure)
    assert compile_binary(replace(TIAGO, N=3), use_cache, jobs=5) == old
    assert seen["ensure"] == (seen["register"], 3, 5) and seen["old"] == use_cache


def test_same_plant_inputs_same_name(tmp_path):
    (tmp_path / "arm.urdf").write_text("<robot/>")
    a = mpc.plant_name(tmp_path / "arm.urdf", LOSSES)
    assert a == mpc.plant_name(tmp_path / "arm.urdf", Friction([1.] * 7, [.5] * 7), friction_law=Tanh(.03))
    assert len(a) == len("tiago_right_") + 8 and a.startswith("tiago_right_")


@pytest.mark.parametrize("friction,armature,width,expected", [
    (None, (0.,)*7, .03, "tiago_right_8dbe11dc"),
    (None, (0.,)*7, .2, "tiago_right_8dbe11dc"),
    (Friction((0.,)*7, (0.,)*7), (0.,)*7, .03, "tiago_right_dcad8b05"),
    (LOSSES, (0.,)*7, .03, "tiago_right_8a7f89af"),
    (LOSSES, (0.,)*7, .05, "tiago_right_9265f19b"),
    (LOSSES, (1.,)*7, .03, "tiago_right_dfdac0d5"),
    (LOSSES, (1.,)*7, .05, "tiago_right_49c9e22a"),
])
def test_separate_law_preserves_pre_refactor_compiled_names(monkeypatch, tmp_path, friction, armature,
                                                           width, expected):
    # Names captured from f66172f's coefficient-plus-law representation, before this refactor.
    monkeypatch.setattr(mpc.importlib.metadata, "version", lambda name: "0.1.5")
    path = tmp_path / "arm.urdf"
    path.write_text("<robot/>")
    assert mpc.plant_name(path, friction, armature, Tanh(width)) == expected


def test_changed_plant_identity_changes_name(tmp_path):
    (tmp_path / "arm.urdf").write_text("<robot/>")
    (tmp_path / "other.urdf").write_text("<robot name='x'/>")
    base = mpc.plant_name(tmp_path / "arm.urdf", LOSSES)
    names = {
        mpc.plant_name(tmp_path / "other.urdf", LOSSES),
        mpc.plant_name(tmp_path / "arm.urdf", None),
        mpc.plant_name(tmp_path / "arm.urdf", LOSSES, friction_law=Tanh(.05)),
        mpc.plant_name(tmp_path / "arm.urdf", replace(LOSSES, coulomb_nm=[2.] * 7)),
        mpc.plant_name(tmp_path / "arm.urdf", replace(LOSSES, viscous_nm_s_per_rad=[.6] * 7)),
    }
    assert base not in names and len(names) == 5


def test_another_gato_version_changes_name(monkeypatch, tmp_path):
    (tmp_path / "arm.urdf").write_text("<robot/>")
    base = mpc.plant_name(tmp_path / "arm.urdf", LOSSES)
    monkeypatch.setattr(mpc.importlib.metadata, "version", lambda name: "0.0.0")
    assert mpc.plant_name(tmp_path / "arm.urdf", LOSSES) != base


def test_smooth_model_check_uses_the_grasp_frame():
    model = pin.buildModelFromUrdf(str(arm_urdf()))
    # Stop after FK: this regression tests the frame, not CUDA dynamics.
    reached_dynamics = RuntimeError("FK check passed")
    solver = Mock()
    solver.ee_pos.side_effect = lambda q: fk(q, GRASP_FRAME)
    solver.sim_forward.side_effect = reached_dynamics
    with pytest.raises(RuntimeError, match="FK check passed"):
        mpc.check_model(solver, model, model.createData(), Friction([1.] * 7, [1.] * 7), Tanh(.03))
    solver.sim_forward.assert_called_once()


def test_selected_current_bounds_map_native_joint_box_and_limit_fractions():
    rows = {}
    solver = NS(set_row_group_bounds=lambda group, lo, hi: rows.update({group: (lo, hi)}))
    model = NS(lowerPositionLimit=np.full(7, -2.), upperPositionLimit=np.full(7, 3.),
               velocityLimit=np.arange(7) + 1., effortLimit=np.arange(7) + 10.)
    mpc.set_bounds(solver, model, replace(TIAGO, joint_margin_rad=.1, velocity_fraction=.8, effort_fraction=.85))
    np.testing.assert_array_equal(rows[0][0], np.full(7, -1.9))
    np.testing.assert_array_equal(rows[0][1], np.full(7, 2.9))
    np.testing.assert_allclose(rows[1][1], .8 * model.velocityLimit)
    np.testing.assert_allclose(rows[2][1], .85 * model.effortLimit)


def test_the_start_pose_lies_inside_the_planning_box():
    """Outside it, every solve worked at the ADMM cap once latency skipped knot 0."""
    raw, margin = limits(), TIAGO.joint_margin_rad
    assert np.all(raw.lower + margin < Q_START) and np.all(Q_START < raw.upper - margin)


@pytest.mark.parametrize("change", [*(dict(max_sqp_time_s=b) for b in (0., -1., np.nan, np.inf)),
                                    *(dict(solver=TIAGO.solver.replace(max_sqp_iters=n)) for n in (0, 2.5)),
                                    *(dict(friction_law=law) for law in (Sign(), StickSlip()))])
def test_invalid_params_are_refused(change):
    with pytest.raises(ValueError):
        replace(TIAGO, **change)


# Both Mpc twins around a fake GATO controller: N = 3 knots, 10 ms.
N, DT = 3, .01
PARAMS = replace(TIAGO, N=N, dt_s=DT, max_sqp_time_s=.004)
BINARY = Path("bsqpN3_fake.so")  # load_controller is faked
REFERENCE = np.tile([.3, -.2, .1, 0., 0., 0.], (N, 1)).astype(np.float32)


class Controller:
    """Plans x_k = (k + 1) x and u_k = torque + k."""

    def __init__(self):
        self.events, self.torque, self.gate = [], np.arange(7, dtype=float), None
        self.solver = NS(**{name: (lambda *a, name=name, **k: self.events.append((name, a, k)))
                            for name in ("set_cost_weights", "set_cost_weights_per_knot",
                                         "set_q_pos_cost", "set_q_nom", "set_f_ext_B")})

    def reset(self, x):
        self.events.append(("reset", x.copy()))

    def warmup(self, x, reference):
        self.events.append(("warmup", x.copy()))

    def step(self, x, reference, *, dt_elapsed, budget_s):
        self.events.append(("step", x.copy(), reference, dt_elapsed, budget_s))
        if self.gate is not None:
            self.gate.wait()
        if isinstance(self.torque, BaseException):
            raise self.torque
        xu = np.concatenate([np.r_[(k + 1) * x, self.torque + k] for k in range(N - 1)] + [N * x])
        raw = NS(sqp_iters=np.array([2]), pcg_iters=np.array([[3], [0]]),
                 admm_iters=np.array([5]), admm_linsys_iters=np.array([20]))
        return NS(xu_best=xu, solve=NS(stats=raw, solve_time_us=1234.), wall_s=.002, iterations=2,
                  budget_hit=False, row_viol=np.array([.01, .02]), admm_linsys="pcg")


@pytest.fixture
def controller(monkeypatch):
    """Every Mpc built in this process (also MpcProcess on threads) gets this controller."""
    controller = Controller()
    monkeypatch.setattr(mpc, "load_controller", lambda params, binary: controller)
    return controller


def state(t=1., scale=1.):
    return State(t, scale * np.arange(7) * .1, scale * np.arange(7) * -.02)


def test_the_solver_gets_and_returns_native_coordinates(controller):
    item = state()
    m = Mpc(PARAMS, BINARY)
    assert m.step is None
    m.request(item, REFERENCE)
    plan = m.result()
    assert m.step.iterations == 2  # the raw GATO step result
    x = np.r_[item.q, item.qd]
    np.testing.assert_array_equal(controller.events[-1][1], x)
    assert plan.t0 == item.t and plan.dt == DT
    np.testing.assert_array_equal(plan.states, [(k + 1) * x for k in range(N)])
    np.testing.assert_array_equal(plan.torques, [np.arange(7.) + k for k in range(N - 1)])
    assert plan.stats["sqp"] == 2 and plan.stats["pcg"] == 3 and plan.stats["zero_pcg_rows"] == 1
    assert m.result() is None  # one plan per request


@pytest.mark.parametrize('worker', [False, True])
def test_prediction_is_separate_from_measured_state_and_crosses_worker_interface(controller, process, worker):
    m = process() if worker else Mpc(PARAMS, BINARY)
    measured = state(1.)
    predicted = Prediction(1.006, measured.q + .01, measured.qd - .02, measured_t=measured.t)
    q, qd = measured.q.copy(), measured.qd.copy()
    m.request(predicted, REFERENCE)
    plan = m.result(block=True)
    assert plan.t0 == 1.006 and plan.measured_t == 1.
    assert measured.t == 1.
    np.testing.assert_array_equal(measured.q, q)
    np.testing.assert_array_equal(measured.qd, qd)
    np.testing.assert_array_equal(plan.states[0], np.r_[predicted.q, predicted.qd])
    assert 'predicted_initial_state' not in plan.stats
    with pytest.raises(ValueError, match='precede'):
        m.request(Prediction(.99, q, qd, measured_t=measured.t), REFERENCE)


def test_after_a_warmup_requests_never_reset_and_dt_elapsed_follows_state_times(controller):
    m = Mpc(PARAMS, BINARY)
    m.warmup(state(.5, scale=2.), REFERENCE)
    for t in (1., 1.013, 1.02):
        m.request(state(t, scale=t), REFERENCE)
        m.result()
    kinds = [event[0] for event in controller.events]
    assert kinds == ["reset", "warmup", "step", "step", "step"]
    for event in controller.events[:2]:
        np.testing.assert_array_equal(event[1][:7], state(scale=2.).q)
    steps = [event for event in controller.events if event[0] == "step"]
    assert [event[3] for event in steps] == [None, pytest.approx(.013), pytest.approx(.007)]
    assert all(event[4] == .004 for event in steps)  # max_sqp_time_s


def test_a_disabled_warmup_does_nothing_and_the_first_request_resets(controller):
    m = Mpc(replace(PARAMS, warmup=False), BINARY)
    m.warmup(state(.5, scale=2.), REFERENCE)
    assert not controller.events
    m.request(state(), REFERENCE)
    assert [event[0] for event in controller.events] == ["reset", "step"]
    np.testing.assert_array_equal(controller.events[0][1][:7], state().q)


def test_updates_are_applied_before_the_solve_and_unknown_names_raise(controller):
    m = Mpc(PARAMS, BINARY)
    q_nom = np.arange(7.)
    m.request(state(), REFERENCE, {"cost_weights": dict(q_cost=2.), "posture": (.5, q_nom)})
    names = [event[0] for event in controller.events]
    assert names == ["set_cost_weights", "set_q_pos_cost", "set_q_nom", "reset", "step"]
    assert controller.events[0][2] == dict(q_cost=2.)
    np.testing.assert_array_equal(controller.events[2][1][0], q_nom)
    with pytest.raises(KeyError, match="posture_cost"):
        m.request(state(2.), REFERENCE, {"posture_cost": 1.})
    assert len(controller.events) == 5


def test_a_request_raises_while_the_previous_plan_is_unread(controller):
    m = Mpc(PARAMS, BINARY)
    m.request(state(), REFERENCE)
    with pytest.raises(RuntimeError, match="pending"):
        m.request(state(2.), REFERENCE)
    assert m.result().t0 == 1. and m.result() is None
    m.request(state(2.), REFERENCE)
    assert [event[0] for event in controller.events] == ["reset", "step", "step"]


def test_a_plan_above_the_raw_effort_limit_is_returned(controller):
    controller.torque = np.full(7, 100.)  # the execution side checks effort (loops.checked)
    m = Mpc(PARAMS, BINARY)
    m.request(state(), REFERENCE)
    assert np.all(np.abs(m.result().torques) > 26.)


@pytest.mark.parametrize("bad", ["t0", "dt", "shape", "nonfinite"])
def test_every_knot_is_checked(controller, bad):
    m = Mpc(PARAMS, BINARY)
    fields = dict(t0=1., dt=DT, states=np.zeros((N, 14)), torques=np.ones((N - 1, 7)), measured_t=1.)
    assert mpc._checked(m, Plan(**fields), 1., 1.)
    if bad == "shape":
        fields.update(states=np.zeros((N + 1, 14)), torques=np.ones((N, 7)))
    elif bad in ("t0", "dt"):
        fields[bad] += .001
    plan = Plan(**fields)
    if bad == "nonfinite":  # only an unpickled plan skips Plan's own check
        object.__setattr__(plan, "torques", np.full((N - 1, 7), np.nan))
    with pytest.raises(RuntimeError):
        mpc._checked(m, plan, 1., 1.)


class Channel(queue.Queue):
    def cancel_join_thread(self):
        pass

    def close(self):
        pass


class Thread(threading.Thread):
    """multiprocessing.Process on a thread, so the fake controller is shared."""
    exitcode = None

    def __init__(self, target, args, daemon):
        super().__init__(target=target, args=args, daemon=daemon)

    def run(self):
        try:
            super().run()
        except SystemExit:  # a process that dies without replying
            pass
        finally:
            self.exitcode = 0

    def terminate(self):
        pass


@pytest.fixture
def process(controller, monkeypatch):
    """make() is an MpcProcess on a thread; make.signals: the handlers its _serve set (a
    thread cannot set them)."""
    monkeypatch.setattr(mpc, "_SPAWN", NS(Queue=Channel, Process=Thread))
    processes, signals = [], []
    monkeypatch.setattr(mpc.signal, "signal", lambda *args: signals.append(args))
    def make(params=PARAMS):
        processes.append(MpcProcess(params, BINARY))
        return processes[-1]
    make.signals = signals
    yield make
    for item in processes:
        item.close()


def test_process_result_waits_for_its_reply(process, controller):
    p = process()
    assert (p.N, p.dt) == (N, DT)
    assert process.signals == [(mpc.signal.SIGINT, mpc.signal.SIG_IGN)]  # Ctrl-C is the parent's
    assert p.result() is None  # nothing requested
    controller.gate = threading.Event()
    p.request(state(), REFERENCE)
    assert p.result() is None  # solving
    with pytest.raises(RuntimeError, match="pending"):
        p.request(state(2.), REFERENCE)
    controller.gate.set()
    plan = p.result(block=True)
    assert plan.t0 == 1. and plan.torques.shape == (N - 1, 7)


def test_process_warmup_runs_in_the_worker_before_it_returns(process, controller):
    p = process()
    p.warmup(state(.5), REFERENCE)
    assert [event[0] for event in controller.events] == ["reset", "warmup"]
    p.request(state(), REFERENCE)
    with pytest.raises(RuntimeError, match="pending"):
        p.warmup(state(), REFERENCE)
    assert p.result(block=True).t0 == 1.
    assert [event[0] for event in controller.events] == ["reset", "warmup", "step"]
    off = process(replace(PARAMS, warmup=False))
    off.warmup(state(), REFERENCE)
    assert off._requests.empty() and len(controller.events) == 3


def test_process_rejects_a_reply_for_another_state(process, controller):
    p = process()
    controller.gate = threading.Event()
    p.request(state(), REFERENCE)
    p._replies.put(Plan(.99, DT, np.zeros((N, 14)), np.ones((N - 1, 7)), measured_t=.99))
    with pytest.raises(RuntimeError, match="does not match"):
        p.result()
    controller.gate.set()


def test_an_error_reply_raises_and_ends_the_process(process, controller):
    p = process()
    controller.torque = FloatingPointError("nonfinite")
    p.request(state(), REFERENCE)
    with pytest.raises(RuntimeError, match="FloatingPointError: nonfinite"):
        p.result(block=True)
    p._process.join(1.)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        p.result()


def test_a_dead_process_raises_at_the_next_call(process, controller):
    p = process()
    controller.torque = SystemExit(9)  # ends the process without a reply
    p.request(state(), REFERENCE)
    p._process.join(1.)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        p.result()


def test_unknown_update_raises_in_the_callers_process(process, controller):
    p = process()
    with pytest.raises(KeyError):
        p.request(state(), REFERENCE, {"gravity": 0.})
    assert p._requests.empty() and p.result() is None


def test_a_spawned_process_reports_a_failed_load(tmp_path):
    # A real spawned process: loading fails on the missing binary before any GPU work.
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        MpcProcess(TIAGO, tmp_path / "missing.so")


def test_armature_has_its_own_model_identity_and_is_forwarded(monkeypatch, tmp_path):
    values = (.746, .882, .211, .648, .326, .343, .27)
    params = replace(TIAGO, armature=values)
    assert params == TIAGO  # the supplied armature is the default
    model, path, name = mpc._plant(params)
    np.testing.assert_array_equal(model.armature, values)
    assert name != mpc._plant(replace(TIAGO, armature=(0.,) * 7))[2]
    assert mpc.plant_name(path, None, (0.,)*7) == mpc.plant_name(path)
    seen = {}
    monkeypatch.setattr(gato, 'register_robot', lambda name, path, **kw: seen.update(kw))
    mpc._register(params, path, name)
    assert seen['joint_armature'] == values
    assert from_dict(to_dict(params)) == params


@pytest.mark.parametrize('values', [(0.,)*6, (-1.,)*7, (float('nan'),)*7, (float('inf'),)*7])
def test_invalid_armature_is_rejected_before_model_generation(values):
    from tiago_mpc.sim import SimConfig
    for cls in (MpcParams, SimConfig):
        with pytest.raises(ValueError, match='armature'):
            cls(armature=values)
