"""Shared MPC configuration; explicit historical profiles keep their base."""
from __future__ import annotations

import json
import math
from pathlib import Path

import gato
from gato.linsys_autotune import resolve_admm_linsys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/reaching.json"
# Historical profiles are partial overrides. Keep this base frozen so promoting
# a new default cannot silently change a recorded experimental protocol.
LEGACY_PARAMS = gato.SolverParams(
    max_sqp_iters=5, max_pcg_iters=120, pcg_tol=1e-3,
    mu=10.0, rho=0.01, q_cost=4.0, N_cost=80.0,
    qd_cost=0.05, u_cost=0.0001,
    q_lim_cost=0.02125, vel_lim_cost=1.275, ctrl_lim_cost=0.02,
)
LEGACY_N, LEGACY_DT_S = 32, 0.030


def load_configuration(path=None):
    """Return selected path, original overrides and effective runtime settings.

    No path selects the complete current default. An explicit path always uses
    the historical partial-override semantics, including an empty JSON object.
    This is CPU-only: no model registration, CUDA build or solver is created.
    """
    selected = Path(path) if path is not None else DEFAULT_CONFIG
    settings = json.loads(selected.read_text())
    tracking_fields = {"sim_dt_s", "duration_s", "secondary_metrics_start_s", "figure8",
                       "joint_dynamics", "predictor_smoothing_rad_s", "armature_runtime"}
    effective = resolve_configuration(
        {key: value for key, value in settings.items() if key not in tracking_fields})
    from .joint_dynamics import validate
    losses = validate(settings.get("joint_dynamics"))
    if losses is not None:
        losses = validate(dict(losses, model="coulomb_tanh_viscous",
                               smoothing_rad_s=settings["predictor_smoothing_rad_s"]))
    elif "predictor_smoothing_rad_s" in settings:
        raise ValueError("predictor_smoothing_rad_s requires joint_dynamics")
    effective["prediction_joint_dynamics"] = losses
    return selected, settings, effective


def resolve_configuration(settings):
    """Resolve the shared solver/controller fields without changing legacy defaults."""
    unknown = set(settings)-{"solver_params", "controller", "limit_bounds", "save_plan_steps", "description", "N", "dt_s"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    params = LEGACY_PARAMS.replace(**settings.get("solver_params", {}))
    n = int(settings.get("N", LEGACY_N))
    dt = float(settings.get("dt_s", LEGACY_DT_S))
    if n < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("N must be >=2 and dt_s finite and positive")
    controller = settings.get("controller", {})
    if set(controller)-{"limits_kw", "admm_linsys", "reset_rho_each_step", "warm_start"}:
        raise ValueError("Unknown controller configuration")
    bounds = settings.get("limit_bounds", {})
    if set(bounds)-{"joint_margin_rad", "velocity_fraction", "effort_fraction"}:
        raise ValueError("Unknown limit bound configuration")
    # Unspecified speed/effort rows use compiled absolute offsets, not 100%
    # of the raw URDF limits. Null fractions distinguish that legacy behavior.
    effective_bounds = {"joint_margin_rad": 0.1, "velocity_fraction": None,
                        "effort_fraction": None, **bounds}
    if "velocity_fraction" not in bounds:
        effective_bounds["compiled_velocity_offset_rad_s"] = 0.1
    if "effort_fraction" not in bounds:
        effective_bounds["compiled_effort_offset_nm"] = 0.1
    admm_policy, _ = resolve_admm_linsys(controller.get("admm_linsys"), N=n)
    effective = dict(
        N=n, dt_s=dt, limits="admm", linsys="pcg", solver_params=params.asdict(),
        controller=dict(warm_start=controller.get("warm_start", "shift"),
                        reset_rho_each_step=controller.get("reset_rho_each_step", True),
                        admm_linsys=admm_policy,
                        limits_kw={**gato.MPCController.LIMIT_DEFAULTS["admm"],
                                   **controller.get("limits_kw", {})}),
        limit_bounds=effective_bounds,
        save_plan_steps=int(settings.get("save_plan_steps", 0)))
    return effective
