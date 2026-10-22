"""Plant names follow model identity; all callers share one GATO cache (CPU only)."""
from dataclasses import replace
import os
from pathlib import Path

from tiago_mpc import plant_cache
from tiago_mpc.joint_dynamics import Friction
from tiago_mpc.model_runtime import runtime_urdf

LOSSES = Friction([1.] * 7, [.5] * 7, .03)


def urdf(tmp_path, text="<robot/>"):
    path = tmp_path / "arm.urdf"
    path.write_text(text)
    return path


def test_same_inputs_same_name(tmp_path):
    a = plant_cache.plant_name(urdf(tmp_path), "ee", LOSSES)
    assert a == plant_cache.plant_name(urdf(tmp_path), "ee", Friction([1.] * 7, [.5] * 7, .03))
    assert len(a) == len("tiago_right_") + 8 and a.startswith("tiago_right_")


def test_changed_identity_changes_name(tmp_path):
    base = plant_cache.plant_name(urdf(tmp_path), "ee", LOSSES)
    names = {
        plant_cache.plant_name(urdf(tmp_path, "<robot name='x'/>"), "ee", LOSSES),
        plant_cache.plant_name(urdf(tmp_path), "other_ee", LOSSES),
        plant_cache.plant_name(urdf(tmp_path), "ee", None),
        plant_cache.plant_name(urdf(tmp_path), "ee", replace(LOSSES, width_rad_s=.05)),
        plant_cache.plant_name(urdf(tmp_path), "ee", replace(LOSSES, coulomb_nm=[2.] * 7)),
        plant_cache.plant_name(urdf(tmp_path), "ee", replace(LOSSES, viscous_nm_s_per_rad=[.6] * 7)),
    }
    assert base not in names and len(names) == 6


def test_cache_root_respects_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "shared"))
    assert plant_cache.cache_root() == (tmp_path / "shared").resolve()
    assert (tmp_path / "shared").is_dir()
    monkeypatch.delenv("GATO_CACHE_DIR")
    assert plant_cache.cache_root() == plant_cache.ROOT / ".cache/gato"
    assert Path(os.environ["GATO_CACHE_DIR"]) == plant_cache.ROOT / ".cache/gato"


def test_generated_models_are_named_by_identity(tmp_path):
    plain = runtime_urdf(tmp_path)[1]
    lossy = runtime_urdf(tmp_path, LOSSES)[1]
    assert plain.parent != lossy.parent and plain.parent.parent == tmp_path / "models"
    assert runtime_urdf(tmp_path)[1] == plain
