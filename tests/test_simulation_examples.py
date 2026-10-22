"""Exercise example timing and saved telemetry with a deterministic controller."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc.joint_dynamics import urdf_friction
from tiago_mpc.model_runtime import URDF, EE_FRAME
from tiago_mpc.mpc import FIGURE8, TIAGO

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def output_root():
    parent = ROOT / "_expr_data" / "test-simulation-examples"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        yield Path(directory)


def knot_offset(k):
    """Stub plans give knot k the solve's u_0 plus k/4, so each knot is identifiable."""
    return .25 * k


def run_example(example, mode, fixed, wall, output_root, monkeypatch, *, duration=.03,
                period=.01, with_losses=False, n=2, control_update_period=None, distinct_knots=False,
                name="run"):
    """Run an example with a deterministic controller; return what it saved and saw."""
    spec = importlib.util.spec_from_file_location(example, ROOT / "examples" / f"{example}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracking = example == "track_figure8"
    params = replace(FIGURE8 if tracking else TIAGO, N=n, dt_s=period)
    plant = None
    if with_losses:  # opt-in reaching friction, with the figure eight's predictor
        params, plant = replace(params, friction=FIGURE8.friction), urdf_friction()
    if tracking:
        monkeypatch.setattr(module, "SIM_DT_S", min(.003, period, control_update_period or period))
        monkeypatch.setattr(module, "SECONDARY_METRICS_START_S", .01)
        monkeypatch.setattr(module, "TASK", replace(module.TASK, period_s=duration))
        plant = replace(module.PLANT_FRICTION, coulomb_nm=(1.234,) + module.PLANT_FRICTION.coulomb_nm[1:])
    out = output_root / name
    argv = ["--output", str(out), "--solve-time-mode", mode, "--solve-time-s", str(fixed)]
    argv += ["--losses", "plant-only"] if tracking else ["--hold", "--timeout", str(duration)]
    if control_update_period is not None:
        argv += ["--control-update-period-s", str(control_update_period)]
    monkeypatch.setattr(module, "git", lambda *args, **kwargs: "test")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")

    model = pin.buildModelFromUrdf(str(URDF))
    calls, applied = [], []
    controller = NS(reset=lambda state: None, solver=object(), reset_rho_each_step=False,
                    LIMIT_DEFAULTS={"admm": {}}, admm_linsys="pcg")

    def step(state, reference, *, dt_elapsed):
        calls.append((state.copy(), reference.copy(), dt_elapsed))
        stats = NS(**{key: np.ones(1, dtype=int) for key in
                      ("sqp_iters", "pcg_iters", "admm_iters", "admm_linsys_iters")})
        u = np.full(7, float(min(len(calls), 3)))
        plan = np.concatenate([np.r_[np.zeros(14), u + knot_offset(k) * distinct_knots] for k in range(n - 1)]
                              + [np.zeros(14)])
        return NS(u=u, xu_best=plan, wall_s=wall, admm_linsys="pcg",
                  row_viol=np.zeros(1), solve=NS(stats=stats, solve_time_us=123.,
                                               xu=plan[None].copy()))

    def integrate(model, data, q, v, u, h, *losses):
        applied.append((u.copy(), h))
        if plant is None:
            assert losses == ()
        else:
            assert losses == (plant,)
        return q + .001 * h * u, v.copy()

    controller.step = step
    setup = NS(controller=controller, model=model, data=model.createData(),
               frame_id=model.getFrameId(EE_FRAME), signs=np.ones(7),
               urdf_path=URDF, gpu_urdf_path=URDF,
               details=dict(model_step_max_abs_errors=[], generator_adapter="stub", plant_name="stub"))
    def build(built):
        assert built == (params if with_losses else replace(params, friction=None))
        return setup
    monkeypatch.setattr(module, "build_controller", build)
    monkeypatch.setattr(module, "plant_rk4", integrate)
    if not tracking:
        monkeypatch.setattr(module, "rk4", integrate)
    if tracking:
        monkeypatch.setattr(module, "model_receipt", lambda *args: {})
        prepare = module.runtime_urdf
        monkeypatch.setattr(module, "runtime_urdf", lambda cache, losses: prepare(output_root / "plant", losses))
    assert module.main(argv, params=params, plant_friction=plant) == 0
    return NS(out=out, calls=calls, applied=applied, plant=plant, module=module, tracking=tracking)


def run_example_check(example, mode, fixed, wall, starts, output_root, monkeypatch,
                      *, duration=.03, period=.01, with_losses=False, n=4):
    # Plans cover the charged time of the previous plan (up to 14 ms after its 10 ms update).
    run = run_example(example, mode, fixed, wall, output_root, monkeypatch,
                      duration=duration, period=period, with_losses=with_losses, n=n)
    out, calls, applied, module, tracking = run.out, run.calls, run.applied, run.module, run.tracking
    summary = json.loads((out / "summary.json").read_text())
    assert summary["plant_friction"] == json.loads(json.dumps(run.plant and asdict(run.plant)))
    assert (summary["params"]["friction"] is None) == (not with_losses)
    if tracking:
        joint = ET.parse(out / "plant.urdf").find(".//joint[@name='arm_right_1_joint']/dynamics")
        assert float(joint.get("friction")) == 1.234

    with np.load(out / "trajectory.npz") as trajectory:
        np.testing.assert_allclose(trajectory["solve_start_s"], starts)
        assert len(calls) == len(starts)
        assert trajectory["time_s"][-1] == pytest.approx(duration)
        assert np.sum([h for _, h in applied]) == pytest.approx(duration)
        latency = fixed if mode == "fixed" else wall
        np.testing.assert_allclose(trajectory["charged_solve_time_s"], latency)
        np.testing.assert_allclose(trajectory["controller_wall_s"], wall)
        # Every plant substep uses the control available at its start timestamp.
        # The final solve's control may never become active before the episode ends.
        sample_starts = trajectory["time_s"][:-1]
        expected_u = np.minimum(3, np.searchsorted(np.asarray(starts) + latency,
                                                  sample_starts + 1e-12, side="right"))
        torque_key = "applied_actuator_torque_nm" if tracking else "applied_torque_nm"
        np.testing.assert_allclose(trajectory[torque_key], np.repeat(expected_u[:, None], 7, axis=1))
        assert calls[0][2] is None
        np.testing.assert_allclose([call[2] for call in calls[1:]], np.diff(starts))
        if tracking:
            start_xyz = trajectory["ee_xyz_m"][0]
            for call, start in zip(calls, starts):
                expected = module.reference(start + np.arange(n) * period, start_xyz, module.TASK)
                np.testing.assert_allclose(call[1].reshape(n, 6)[:, :3], expected, atol=1e-7)
            np.testing.assert_allclose(trajectory["reference_xyz_m"],
                module.reference(trajectory["time_s"], start_xyz, module.TASK))


@pytest.mark.parametrize("example", ["reach_sim", "track_figure8"])
@pytest.mark.parametrize("mode,fixed,wall,starts", [
    ("fixed", 0., .014, [0., .01, .02]),
    ("fixed", .004, .014, [0., .01, .02]),
    ("measured", 0., .004, [0., .01, .02]),
    ("measured", 0., .014, [0., .014, .028]),
])
def test_example_delays_actuation_and_advances_reference(
        example, mode, fixed, wall, starts, output_root, monkeypatch):
    run_example_check(example, mode, fixed, wall, starts, output_root, monkeypatch)


@pytest.mark.parametrize("period", [.01, .002])  # the 10 ms default and the earlier 2 ms knots
def test_default_tracking_clock_does_not_add_a_roundoff_solve(output_root, monkeypatch, period):
    assert FIGURE8.dt_s == .01
    run_example_check("track_figure8", "fixed", 0., .001, np.arange(round(24. / period)) * period,
                      output_root, monkeypatch, duration=24., period=period)


def test_opt_in_reaching_losses_reach_plant_and_predictor(output_root, monkeypatch):
    run_example_check("reach_sim", "fixed", 0., .001, [0., .01, .02],
                      output_root, monkeypatch, with_losses=True)


def saved(out):
    with np.load(out / "trajectory.npz") as trajectory:
        return {key: trajectory[key] for key in trajectory.files}


@pytest.mark.parametrize("example", ["reach_sim", "track_figure8"])
def test_update_period_equal_to_dt_applies_each_plans_first_torque(
        example, output_root, monkeypatch):
    # Control update period = dt and zero latency: each plan's knot 0 spans the
    # whole interval, so the plant applies exactly the solve's u_0 (step.u).
    run = run_example(example, "fixed", 0., .001, output_root, monkeypatch, n=4, distinct_knots=True)
    trajectory = saved(run.out)
    torque_key = "applied_actuator_torque_nm" if run.tracking else "applied_torque_nm"
    starts = trajectory["time_s"][:-1]
    expected = [np.full(7, float(min(int(np.floor(t / .01 + 1e-9)) + 1, 3))) for t in starts]
    np.testing.assert_array_equal(trajectory[torque_key][:len(expected)], expected)
    assert set(np.unique(trajectory[torque_key])) <= {1., 2., 3.}


@pytest.mark.parametrize("example", ["reach_sim", "track_figure8"])
def test_update_period_longer_than_knots_applies_each_knot_for_its_interval(
        example, output_root, monkeypatch):
    # 10 ms control update period, 2 ms knots, 4 ms charged latency, N=8 (7 knots, 14 ms plans).
    run = run_example(example, "fixed", .004, .001, output_root, monkeypatch, n=8,
                      period=.002, control_update_period=.01, distinct_knots=True)
    trajectory = saved(run.out)
    torque_key = "applied_actuator_torque_nm" if run.tracking else "applied_torque_nm"
    np.testing.assert_allclose(trajectory["solve_start_s"], [0., .01, .02])
    np.testing.assert_allclose([call[2] for call in run.calls[1:]], [.01, .01])
    starts = trajectory["time_s"][:-1]
    knots = set()
    for t, torque in zip(starts, trajectory[torque_key]):
        solve = int(np.floor((t - .004) / .01 + 1e-9))  # latest plan already applied
        if solve < 0:
            assert not np.any(torque)
            continue
        knot = int(np.floor((t - .01 * solve) / .002 + 1e-9))
        knots.add(knot)
        np.testing.assert_array_equal(torque, np.full(7, min(solve + 1, 3) + knot_offset(knot)))
    # Knots 2..6 are applied: the plan is sampled, not only u_0, and never past its end.
    assert knots == {2, 3, 4, 5, 6}
