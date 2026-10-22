"""MpcParams: round trip, and the solver settings GATO receives (CPU only)."""
from dataclasses import asdict, replace
import json
import pickle

import gato
import numpy as np
import pytest

from tiago_mpc import mpc
from tiago_mpc.mpc import FIGURE8, TIAGO, MpcParams

URDF_COULOMB = [1.66, 1.15, 1.16, 0.912, 1.25, 0.473, 0.56]
URDF_VISCOUS = [4.7, 9.69, 5.3, 4.2, 3.66, 2.95, 2.95]
SOLVER = dict(max_sqp_iters=5, max_pcg_iters=120, pcg_tol=.001, solve_ratio=1., mu=10., rho=.01,
              adapt_rho=True, linsys=None, q_cost=4., qd_cost=.05, u_cost=1e-4, N_cost=8.,
              q_lim_cost=0., vel_lim_cost=0., ctrl_lim_cost=0., exact_hessian=False)
CONTROLLER = dict(limits="admm", linsys="pcg", warm_start="shift", reset_rho_each_step=True,
                  admm_linsys="pcg", limits_kw=dict(rho=.01, iters=100, eps_rel=.001,
                                                    rho_q=1., rho_qd=.3, rho_u=5e-4))
# The effective values of configs/reaching.json and configs/tracking.json at 2eb9544.
EXPECTED = {
    "reaching": dict(N=32, dt=.01, solver=SOLVER, controller=CONTROLLER, friction=None),
    "figure8": dict(
        N=32, dt=.01, solver=dict(SOLVER, mu=1., qd_cost=.01, u_cost=1e-6),
        controller=dict(CONTROLLER, limits_kw=dict(CONTROLLER["limits_kw"], rho_q=.1, rho_qd=.03)),
        friction=dict(coulomb_nm=URDF_COULOMB, viscous_nm_s_per_rad=URDF_VISCOUS, width=.03)),
}


@pytest.mark.parametrize("params", [TIAGO, FIGURE8, replace(TIAGO, N=8, admm=replace(TIAGO.admm, iters=5))])
def test_asdict_round_trips_also_through_json_and_pickle(params):
    assert MpcParams.from_dict(asdict(params)) == params
    assert MpcParams.from_dict(json.loads(json.dumps(asdict(params)))) == params
    assert pickle.loads(pickle.dumps(params)) == params


@pytest.mark.parametrize("name,params", [("reaching", TIAGO), ("figure8", FIGURE8)])
def test_effective_solver_settings_are_unchanged(monkeypatch, name, params):
    seen = {}

    class Solver:
        joint_friction_smoothing = None

        def __init__(self, urdf, *, batch_size, N, dt, plant_type, params):
            seen.update(N=N, dt=dt, solver=params.asdict())

        def set_row_group_bounds(self, group, lo, hi):
            seen.setdefault("bounds", {})[group] = (lo, hi)

    def controller(solver, **kwargs):
        seen["controller"] = kwargs

    def register(name, path, *, n_points, ee_frame, friction):
        seen["friction"] = friction and dict(coulomb_nm=list(friction.coulomb_nm),
                                             viscous_nm_s_per_rad=list(friction.viscous_nm_s_per_rad),
                                             width=friction.width_rad_s)

    monkeypatch.setattr(gato, "BSQP", Solver)
    monkeypatch.setattr(gato, "MPCController", controller, raising=False)
    monkeypatch.setattr(mpc, "register_robot", register)
    monkeypatch.setattr(mpc, "check_model", lambda *args: [])
    setup = mpc.build_controller(params)
    bounds, model, signs = seen.pop("bounds"), setup.model, setup.signs
    assert seen == EXPECTED[name]
    lo = (model.lowerPositionLimit + .1) * signs
    hi = (model.upperPositionLimit - .1) * signs
    np.testing.assert_array_equal(bounds[0][0], np.minimum(lo, hi))
    np.testing.assert_array_equal(bounds[0][1], np.maximum(lo, hi))
    np.testing.assert_array_equal(bounds[1][1], .8 * model.velocityLimit)
    np.testing.assert_array_equal(bounds[2][1], .85 * model.effortLimit)
    assert setup.details["params"] == asdict(params)
