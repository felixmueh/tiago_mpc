"""A GATO update changes no setting silently.

gato_defaults.json is a snapshot of GATO's defaults, and DEVIATIONS lists where the presets
differ from them. When the snapshot test fails, check each changed default: a followed one
changes the runs, and an overridden one may make its deviation unnecessary. Then rewrite the
snapshot with `python tests/test_gato_defaults.py` and adjust DEVIATIONS.
"""
from dataclasses import asdict, fields, replace
import json
from pathlib import Path

import inspect
from typing import get_type_hints

from gato import MPCController, SolverParams
from gato.interface import BSQP
from gato.linsys_autotune import resolve_admm_linsys, resolve_linsys
import pytest

from experiments.figure8 import FIGURE8
from tiago_mpc.mpc import CONTROLLER_SETTINGS, TIAGO, MpcParams, SolverSettings

SNAPSHOT = Path(__file__).with_name("gato_defaults.json")
PRESETS = {"TIAGO": TIAGO, "FIGURE8": FIGURE8}
# Where each preset differs from GATO's defaults; the reasons are at the values.
DEVIATIONS = {
    "TIAGO": {"solver.max_sqp_iters", "solver.max_pcg_iters", "solver.pcg_tol", "solver.q_lim_cost", "admm.rho_u"},
    "FIGURE8": {"solver.max_sqp_iters", "solver.max_pcg_iters", "solver.pcg_tol", "solver.q_lim_cost", "solver.mu"},
}


def gato_defaults() -> dict:
    """The installed GATO's defaults for MpcParams' settings, as JSON values."""
    linsys, tau = resolve_linsys(False)
    enable = {name: p.default for name, p in inspect.signature(BSQP.enable_limit_admm).parameters.items()
              if p.default is not inspect.Parameter.empty}
    return json.loads(json.dumps(dict(
        solver=asdict(SolverParams()), admm=MPCController.LIMIT_DEFAULTS["admm"], admm_enable=enable,
        controller=MPCController.__init__.__kwdefaults__, fixed_base_linsys=[linsys, tau],
        admm_linsys_by_N={str(N): resolve_admm_linsys(None, None, fallback_tau=tau, N=N)
                          for N in sorted({params.N for params in PRESETS.values()})})))


def deviations(params: MpcParams) -> set[str]:
    """The settings in which params does not follow GATO.

    These are the SolverParams fields that differ from GATO's defaults and the AdmmLimits
    fields and controller settings that are set."""
    solver = {f"solver.{key}" for key, value in asdict(params.solver).items() if value != asdict(SolverParams())[key]}
    admm = {f"admm.{key}" for key, value in asdict(params.admm).items() if value is not None}
    return solver | admm | {name for name in CONTROLLER_SETTINGS if getattr(params, name) is not None}


def changes(old: dict, new: dict, path: str = ""):
    """Each setting that differs between two snapshots, and which presets override it."""
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key, "absent"), new.get(key, "absent")
        if isinstance(a, dict) and isinstance(b, dict):
            yield from changes(a, b, f"{path}{key}.")
        elif a != b:
            name = f"{path}{key}".removeprefix("controller.").replace("admm_enable.", "admm.")
            presets = [preset for preset, found in DEVIATIONS.items() if name in found]
            yield f"{path}{key}: {a!r} -> {b!r} ({'overridden by ' + ', '.join(presets) if presets else 'followed'})"


def test_gatos_defaults_are_the_snapshot():
    found = list(changes(json.loads(SNAPSHOT.read_text()), gato_defaults()))
    assert not found, ("GATO's defaults changed: " + "; ".join(found)
                       + ". Check them, then run python tests/test_gato_defaults.py")


def test_solver_settings_follow_every_native_field_default_and_type_except_tracking_weights():
    # Upstream field, default or type changes must fail here so the thin wrapper cannot
    # silently stop following GATO; only these four weights belong to the menu instead.
    omitted = {"q_cost", "N_cost", "qd_cost", "u_cost"}
    native, ours = asdict(SolverParams()), asdict(SolverSettings())
    assert set(native) - set(ours) == omitted
    assert set(ours) == {field.name for field in fields(SolverParams)} - omitted
    assert ours == {name: value for name, value in native.items() if name not in omitted}
    native_types, our_types = get_type_hints(SolverParams), get_type_hints(SolverSettings)
    assert our_types == {name: value for name, value in native_types.items() if name not in omitted}
    assert SolverSettings().native() == SolverParams()


def test_solver_settings_native_conversion_and_resolved_readback_preserve_every_exposed_field():
    settings = SolverSettings(max_sqp_iters=7, max_pcg_iters=83, pcg_tol=.03, solve_ratio=.4, mu=3., rho=.2,
                              adapt_rho=False, linsys="pcg", q_lim_cost=.9, vel_lim_cost=.8, ctrl_lim_cost=.7,
                              exact_hessian=True, cost_curvature=True, defect_telemetry=True, defect_exit=.6)
    assert SolverSettings.resolved(settings.native()) == settings
    assert replace(settings, mu=4.) == SolverSettings.resolved(replace(settings.native(), mu=4.))


@pytest.mark.parametrize("name", PRESETS)
def test_the_presets_deviate_from_gatos_defaults_only_where_listed(name):
    assert deviations(PRESETS[name]) == DEVIATIONS[name]


if __name__ == "__main__":
    current = gato_defaults()
    for change in changes(json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}, current):
        print(change)
    SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    print("wrote", SNAPSHOT)
