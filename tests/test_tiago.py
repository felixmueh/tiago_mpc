"""The derived URDFs: content-addressed in the cache; the GPU model's URDF keeps the
native model and maps joint 6's sign (CPU only)."""
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc import tiago
from tiago_mpc.dynamics import Friction


def test_gpu_urdf_maps_joint_6s_sign_and_keeps_only_the_collision_meshes(monkeypatch, tmp_path):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path))
    gpu, signs = tiago.runtime_urdf()
    assert gpu.is_relative_to(tmp_path)
    np.testing.assert_array_equal(signs, [1, 1, 1, 1, 1, -1, 1])
    original = pin.buildModelFromUrdf(str(tiago.arm_urdf()))
    gpu_model = pin.buildModelFromUrdf(str(gpu))
    np.testing.assert_array_equal(gpu_model.lowerPositionLimit,
                                  np.minimum(original.lowerPositionLimit * signs,
                                             original.upperPositionLimit * signs))
    tree = ET.parse(gpu)
    assert not tree.findall("link/visual") and tree.findall("link/collision/geometry/mesh")
    for mesh in tree.iter("mesh"):  # cached copies, relative to the URDF
        assert (gpu.parent / mesh.get("filename")).is_file()
    assert str(tiago.ROOT) not in gpu.read_text()  # the bytes do not depend on the checkout


def test_mesh_copies_reject_git_lfs_pointers(monkeypatch, tmp_path):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "pointer.stl").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        tiago.cache_meshes(ET.fromstring('<robot><mesh filename="pointer.stl"/></robot>'),
                           lambda filename: tmp_path / filename)


def test_derived_files_are_written_once_by_content(monkeypatch, tmp_path):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path))
    arm, gpu = tiago.arm_urdf(), tiago.runtime_urdf()[0]
    assert arm.parent.name == tiago.digest(arm.read_bytes())
    before = arm.stat().st_mtime_ns, gpu.stat().st_mtime_ns
    assert (tiago.arm_urdf(), tiago.runtime_urdf()[0]) == (arm, gpu)
    assert (arm.stat().st_mtime_ns, gpu.stat().st_mtime_ns) == before
    friction = Friction((1.,) * 7, (2.,) * 7)
    assert tiago.runtime_urdf(friction)[0] != gpu  # other coefficients, another path
