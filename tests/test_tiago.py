"""The derived URDFs: content-addressed in the cache; the GPU model's URDF keeps the
native model, joint 6's -z axis included (CPU only)."""
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc import tiago, util
from tiago_mpc.dynamics import Friction


def test_gpu_urdf_keeps_the_native_axes_and_limits_and_only_the_collision_meshes(monkeypatch, tmp_path):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path))
    gpu = tiago.runtime_urdf()
    assert gpu.is_relative_to(tmp_path)
    original = pin.buildModelFromUrdf(str(tiago.arm_urdf()))
    gpu_model = pin.buildModelFromUrdf(str(gpu))
    for limit in ("lowerPositionLimit", "upperPositionLimit", "velocityLimit", "effortLimit"):
        np.testing.assert_array_equal(getattr(gpu_model, limit), getattr(original, limit))
    tree = ET.parse(gpu)
    axes = [joint.find("axis").get("xyz") for joint in tree.iter("joint") if joint.get("type") != "fixed"]
    assert axes[5] == "0 0 -1" and axes == [joint.find("axis").get("xyz") for joint in ET.parse(
        tiago.arm_urdf()).iter("joint") if joint.get("type") != "fixed"]
    assert not tree.findall("link/visual") and tree.findall("link/collision/geometry/mesh")
    for mesh in tree.iter("mesh"):  # cached copies, relative to the URDF
        assert (gpu.parent / mesh.get("filename")).is_file()
    assert str(tiago.ROOT) not in gpu.read_text()  # the bytes do not depend on the checkout


def test_mesh_copies_reject_git_lfs_pointers(monkeypatch, tmp_path):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "pointer.stl").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        util.cache_meshes(ET.fromstring('<robot><mesh filename="pointer.stl"/></robot>'),
                          lambda filename: tmp_path / filename)


def test_derived_files_are_written_once_by_content(monkeypatch, tmp_path):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path))
    arm, gpu = tiago.arm_urdf(), tiago.runtime_urdf()
    assert arm.parent.name == util.digest(arm.read_bytes())
    before = arm.stat().st_mtime_ns, gpu.stat().st_mtime_ns
    assert (tiago.arm_urdf(), tiago.runtime_urdf()) == (arm, gpu)
    assert (arm.stat().st_mtime_ns, gpu.stat().st_mtime_ns) == before
    friction = Friction((1.,) * 7, (2.,) * 7)
    assert tiago.runtime_urdf(friction) != gpu  # other coefficients, another path


def test_worktrees_share_the_main_checkouts_cache(tmp_path):
    main, worktree = tmp_path / "main", tmp_path / "worktree"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(main)], check=True)
    subprocess.run([*git, "-C", str(main), "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", str(worktree)], check=True)
    assert util.main_checkout(worktree) == util.main_checkout(main) == main.resolve()
    (tmp_path / "plain").mkdir()  # without git: the directory itself
    assert util.main_checkout(tmp_path / "plain") == tmp_path / "plain"


@pytest.mark.parametrize("q", [tiago.Q_START, tiago.Q_REACH])
def test_the_gpu_model_places_the_grasp_frame_like_the_native_one(monkeypatch, tmp_path, q):
    """The derived GPU model has the native grasp-frame pose in native joint coordinates."""
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path))
    gpu = tiago.runtime_urdf()
    model = pin.buildModelFromUrdf(str(gpu))
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    placement = data.oMf[model.getFrameId(tiago.GRASP_FRAME)]
    np.testing.assert_allclose(placement.translation, tiago.ee_position(q), atol=1e-12)
    np.testing.assert_allclose(pin.log3(placement.rotation), tiago.ee_orientation(q), atol=1e-12)


def test_orientation_error_is_the_angle_to_the_target():
    target = tiago.ee_orientation(tiago.Q_START)
    assert tiago.orientation_error(tiago.Q_START, target) == pytest.approx(0., abs=1e-9)
    turned = pin.log3(pin.exp3(np.array([0., 0., .3])) @ pin.exp3(target))
    assert tiago.orientation_error(tiago.Q_START, turned) == pytest.approx(.3)
    assert tiago.orientation_error(tiago.Q_REACH, tiago.ee_orientation(tiago.Q_START)) > .1


@pytest.mark.parametrize("q", [tiago.Q_START, tiago.Q_REACH])
def test_ee_velocity_matches_finite_difference(q):
    qd = np.linspace(.1, -.3, 7)
    h = 1e-6
    numeric = (tiago.ee_position(q + h * qd) - tiago.ee_position(q - h * qd)) / (2 * h)
    assert tiago.ee_velocity(q, qd) == pytest.approx(numeric, abs=1e-6)
