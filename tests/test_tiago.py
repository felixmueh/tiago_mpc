"""The GPU model's URDF keeps the native model and maps joint 6's sign (CPU only)."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from tiago_mpc import tiago


def test_gpu_urdf_maps_joint_6s_sign_and_resolves_the_meshes(tmp_path):
    gpu, signs = tiago.runtime_urdf(tmp_path)
    np.testing.assert_array_equal(signs, [1, 1, 1, 1, 1, -1, 1])
    original = pin.buildModelFromUrdf(str(tiago.ARM_URDF))
    gpu_model = pin.buildModelFromUrdf(str(gpu))
    np.testing.assert_array_equal(gpu_model.lowerPositionLimit,
                                  np.minimum(original.lowerPositionLimit * signs,
                                             original.upperPositionLimit * signs))
    for mesh in ET.parse(gpu).iter("mesh"):
        assert Path(mesh.get("filename")).is_file()
    before = gpu.stat().st_mtime_ns
    assert tiago.runtime_urdf(tmp_path)[0] == gpu
    assert gpu.stat().st_mtime_ns == before
