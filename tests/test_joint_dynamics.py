"""Passive-model registration and receipts without constructing a CUDA solver."""
from pathlib import Path
from types import SimpleNamespace
import json

import gato
import numpy as np
import pytest

from tiago_mpc.configuration import load_configuration
from tiago_mpc.joint_dynamics import model_receipt, register_robot

ROOT = Path(__file__).resolve().parents[1]


def test_both_task_defaults_register_smooth_losses(monkeypatch):
    calls = []
    monkeypatch.setattr(gato, "register_robot", lambda *args, **kwargs: calls.append((args, kwargs)))
    for config_path in (None, ROOT / "configs/tracking.json"):
        _, _, effective = load_configuration(config_path)
        register_robot("test", "arm.urdf", n_points=effective["N"], ee_frame="tcp",
                       config=effective["prediction_joint_dynamics"])
    assert calls[0] == (("test", "arm.urdf"),
                        dict(N=[32], ee_frame="tcp", jobs=1, joint_friction_smoothing=.03))
    assert calls[1] == (("test", "arm.urdf"),
                        dict(N=[32], ee_frame="tcp", jobs=1, joint_friction_smoothing=.03))


@pytest.mark.parametrize("widths", [None, np.linspace(.02, .08, 7, dtype=np.float32)])
def test_receipt_records_effective_instance_widths_not_only_compiled_identity(tmp_path, monkeypatch, widths):
    from gato import _cache
    directory = tmp_path / "arm"
    directory.mkdir()
    for name in ("grid.cuh", "limits.cuh", "module.so", "arm.urdf"):
        (directory / name).write_text(name)
    monkeypatch.setattr(_cache, "resolve", lambda name: ({"codegen_key": "compiled"}, tmp_path))
    solver = SimpleNamespace(joint_friction_smoothing=widths,
        lib=SimpleNamespace(__file__=str(directory / "module.so"), NATIVE_SOURCE_ID="source"))
    first = model_receipt(solver, "arm", directory / "arm.urdf")
    assert json.loads(json.dumps(first))["joint_friction_smoothing_rad_s"] == (
        None if widths is None else widths.tolist())
    if widths is not None:
        original = widths.copy()
        widths[:] = .1  # A later solver update must not mutate an earlier receipt.
        second = model_receipt(solver, "arm", directory / "arm.urdf")
        assert first["joint_friction_smoothing_rad_s"] == original.tolist()
        assert second["joint_friction_smoothing_rad_s"] == widths.tolist()
        assert first["module_sha256"] == second["module_sha256"]
