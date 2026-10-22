"""MpcParams, plant names, the settings GATO receives, and both Mpc twins on fakes (CPU only)."""
from dataclasses import asdict, replace
import json
import pickle
import queue
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import gato
import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc import mpc
from experiments.figure8 import FIGURE8
from tiago_mpc.dynamics import Friction
from tiago_mpc.mpc import TIAGO, Mpc, MpcParams, MpcProcess
from tiago_mpc.plan import Plan, State
from tiago_mpc.tiago import ARM_URDF, FLANGE_FRAME, Q_START, fk, urdf_friction

URDF_FRICTION = urdf_friction()  # the coefficients come from the URDF only
LOSSES = Friction([1.] * 7, [.5] * 7, .03)


@pytest.mark.parametrize("params", [TIAGO, FIGURE8, replace(TIAGO, N=8, admm=replace(TIAGO.admm, iters=5))])
def test_asdict_round_trips_also_through_json_and_pickle(params):
    assert MpcParams.from_dict(asdict(params)) == params
    assert MpcParams.from_dict(json.loads(json.dumps(asdict(params)))) == params
    assert pickle.loads(pickle.dumps(params)) == params


@pytest.mark.parametrize("params", [TIAGO, FIGURE8])
def test_build_controller_passes_the_params_to_gato(monkeypatch, params):
    seen = {}

    class Solver:
        def __init__(self, urdf, *, batch_size, N, dt, plant_type, params):
            seen.update(N=N, dt=dt, solver=params.asdict())

        def set_row_group_bounds(self, group, lo, hi):
            seen.setdefault("bounds", []).append(group)

    monkeypatch.setattr(gato, "register_robot", lambda name, path, **kwargs: seen.update(register=kwargs))
    monkeypatch.setattr(gato, "BSQP", Solver)
    monkeypatch.setattr(gato, "MPCController", lambda solver, **kwargs: seen.update(controller=kwargs),
                        raising=False)
    monkeypatch.setattr(mpc, "check_model", lambda *args: None)
    mpc.build_controller(params)
    assert (seen["N"], seen["dt"], seen["solver"]) == (params.N, params.dt_s, params.solver.asdict())
    assert seen["controller"]["limits_kw"] == asdict(params.admm) and seen["bounds"] == [0, 1, 2]
    # Smooth predictor friction is registered only when the params have one.
    assert seen["register"].get("joint_friction_smoothing") == (params.friction and params.friction.width_rad_s)


def test_same_plant_inputs_same_name(tmp_path):
    (tmp_path / "arm.urdf").write_text("<robot/>")
    a = mpc.plant_name(tmp_path / "arm.urdf", "ee", LOSSES)
    assert a == mpc.plant_name(tmp_path / "arm.urdf", "ee", Friction([1.] * 7, [.5] * 7, .03))
    assert len(a) == len("tiago_right_") + 8 and a.startswith("tiago_right_")


def test_changed_plant_identity_changes_name(tmp_path):
    (tmp_path / "arm.urdf").write_text("<robot/>")
    (tmp_path / "other.urdf").write_text("<robot name='x'/>")
    base = mpc.plant_name(tmp_path / "arm.urdf", "ee", LOSSES)
    names = {
        mpc.plant_name(tmp_path / "other.urdf", "ee", LOSSES),
        mpc.plant_name(tmp_path / "arm.urdf", "other_ee", LOSSES),
        mpc.plant_name(tmp_path / "arm.urdf", "ee", None),
        mpc.plant_name(tmp_path / "arm.urdf", "ee", replace(LOSSES, width_rad_s=.05)),
        mpc.plant_name(tmp_path / "arm.urdf", "ee", replace(LOSSES, coulomb_nm=[2.] * 7)),
        mpc.plant_name(tmp_path / "arm.urdf", "ee", replace(LOSSES, viscous_nm_s_per_rad=[.6] * 7)),
    }
    assert base not in names and len(names) == 6


def test_smooth_model_check_uses_requested_flange_frame():
    model = pin.buildModelFromUrdf(str(ARM_URDF))
    # Stop after FK: this regression tests frame selection, not CUDA dynamics.
    reached_dynamics = RuntimeError("FK check passed")
    solver = Mock()
    solver.ee_pos.side_effect = lambda q: fk(q, FLANGE_FRAME)  # signs are ones
    solver.sim_forward.side_effect = reached_dynamics
    with pytest.raises(RuntimeError, match="FK check passed"):
        mpc.check_model(solver, model, model.createData(), FLANGE_FRAME, np.ones(7),
                        Friction([1.] * 7, [1.] * 7, .03))
    solver.sim_forward.assert_called_once()


def test_selected_current_bounds_map_native_joint_box_and_limit_fractions():
    rows = {}
    solver = NS(set_row_group_bounds=lambda group, lo, hi: rows.update({group: (lo, hi)}))
    model = NS(lowerPositionLimit=np.full(7, -2.), upperPositionLimit=np.full(7, 3.),
               velocityLimit=np.arange(7) + 1., effortLimit=np.arange(7) + 10.)
    mpc.set_bounds(solver, model, SIGNS, replace(TIAGO, joint_margin_rad=.1, velocity_fraction=.8, effort_fraction=.85))
    assert rows[0][0][5] == -2.9 and rows[0][1][5] == 1.9
    assert rows[0][0][0] == -1.9 and rows[0][1][0] == 2.9
    np.testing.assert_allclose(rows[1][1], .8 * model.velocityLimit)
    np.testing.assert_allclose(rows[2][1], .85 * model.effortLimit)


@pytest.mark.parametrize("change", [*(dict(budget_s=b) for b in (0., -1., np.nan, np.inf)),
                                    dict(friction=URDF_FRICTION)])  # hard sign: plant only
def test_invalid_params_are_refused(change):
    with pytest.raises(ValueError):
        replace(TIAGO, **change)


# Both Mpc twins around a fake GATO controller: N = 3 knots, 10 ms.
SIGNS = np.array([1., 1., 1., 1., 1., -1., 1.])
N, DT = 3, .01
PARAMS = replace(TIAGO, N=N, dt_s=DT, budget_s=.004)
REFERENCE = np.tile([.3, -.2, .1, 0., 0., 0.], (N, 1)).astype(np.float32)


class Controller:
    """Plans x_k = (k + 1) x and u_k = torque + k, in GPU coordinates."""

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
    monkeypatch.setattr(mpc, "build_controller", lambda params: (controller, SIGNS))
    return controller


def state(t=1., scale=1.):
    return State(t, scale * np.arange(7) * .1, scale * np.arange(7) * -.02)


def test_native_coordinates_cross_only_the_gpu_boundary(controller):
    item = state()
    m = Mpc(PARAMS)
    m.request(item, REFERENCE)
    plan = m.result()
    x = np.r_[item.q, item.qd]
    gpu = x * np.tile(SIGNS, 2)
    np.testing.assert_array_equal(controller.events[-1][1], gpu)
    assert plan.t0 == item.t and plan.dt == DT
    np.testing.assert_array_equal(plan.states, [(k + 1) * x for k in range(N)])
    np.testing.assert_array_equal(plan.torques, [(np.arange(7.) + k) * SIGNS for k in range(N - 1)])
    assert plan.stats["sqp"] == 2 and plan.stats["pcg"] == 3 and plan.stats["zero_pcg_rows"] == 1
    assert m.result() is None  # one plan per request


def test_first_request_resets_and_warms_up_once_and_dt_elapsed_follows_state_times(controller):
    m = Mpc(replace(PARAMS, warmup=2))
    for t in (1., 1.013, 1.02):
        m.request(state(t, scale=t), REFERENCE)
        m.result()
    kinds = [event[0] for event in controller.events]
    assert kinds == ["reset", "warmup", "warmup", "step", "step", "step"]
    np.testing.assert_array_equal(controller.events[0][1][:7], state().q * SIGNS)
    steps = [event for event in controller.events if event[0] == "step"]
    assert [event[3] for event in steps] == [None, pytest.approx(.013), pytest.approx(.007)]
    assert all(event[4] == .004 for event in steps)


def test_updates_are_applied_before_the_solve_and_unknown_names_raise(controller):
    m = Mpc(PARAMS)
    q_nom = np.arange(7.)
    m.request(state(), REFERENCE, {"cost_weights": dict(q_cost=2.), "posture": (.5, q_nom)})
    names = [event[0] for event in controller.events]
    assert names == ["set_cost_weights", "set_q_pos_cost", "set_q_nom", "reset", "step"]
    assert controller.events[0][2] == dict(q_cost=2.)
    np.testing.assert_array_equal(controller.events[2][1][0], q_nom * SIGNS)
    with pytest.raises(KeyError, match="posture_cost"):
        m.request(state(2.), REFERENCE, {"posture_cost": 1.})
    assert len(controller.events) == 5


def test_a_request_raises_while_the_previous_plan_is_unread(controller):
    m = Mpc(PARAMS)
    m.request(state(), REFERENCE)
    with pytest.raises(RuntimeError, match="pending"):
        m.request(state(2.), REFERENCE)
    assert m.result().t0 == 1. and m.result() is None
    m.request(state(2.), REFERENCE)
    assert [event[0] for event in controller.events] == ["reset", "step", "step"]


@pytest.mark.parametrize("bad", ["t0", "dt", "shape", "nonfinite", "effort"])
def test_every_knot_is_checked(controller, bad):
    m = Mpc(PARAMS)
    fields = dict(t0=1., dt=DT, states=np.zeros((N, 14)), torques=np.ones((N - 1, 7)))
    assert mpc._checked(m, Plan(**fields), 1.)
    if bad == "shape":
        fields.update(states=np.zeros((N + 1, 14)), torques=np.ones((N, 7)))
    elif bad in ("t0", "dt"):
        fields[bad] += .001
    elif bad == "effort":
        fields["torques"][-1, 3] = 26.1  # only the last knot exceeds joint 4's raw effort limit
    plan = Plan(**fields)
    if bad == "nonfinite":  # only an unpickled plan skips Plan's own check
        object.__setattr__(plan, "torques", np.full((N - 1, 7), np.nan))
    with pytest.raises(RuntimeError):
        mpc._checked(m, plan, 1.)


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
    monkeypatch.setattr(mpc, "_SPAWN", NS(Queue=Channel, Process=Thread))
    processes = []
    def make(params=PARAMS):
        processes.append(MpcProcess(params))
        return processes[-1]
    yield make
    for item in processes:
        item.close()


def test_process_result_waits_for_its_reply(process, controller):
    p = process()
    assert (p.N, p.dt) == (N, DT)
    assert p.result() is None  # nothing requested
    controller.gate = threading.Event()
    p.request(state(), REFERENCE)
    assert p.result() is None  # solving
    with pytest.raises(RuntimeError, match="pending"):
        p.request(state(2.), REFERENCE)
    controller.gate.set()
    plan = p.result(block=True)
    assert plan.t0 == 1. and plan.torques.shape == (N - 1, 7)


def test_process_rejects_a_reply_for_another_state(process, controller):
    p = process()
    controller.gate = threading.Event()
    p.request(state(), REFERENCE)
    p._replies.put(Plan(.99, DT, np.zeros((N, 14)), np.ones((N - 1, 7))))
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


def test_a_spawned_process_reports_a_failed_build():
    # A real spawned process: the build fails on the frame before any GPU work.
    with pytest.raises(RuntimeError, match="Missing end-effector frame"):
        MpcProcess(replace(TIAGO, ee_frame="no_such_frame"))
