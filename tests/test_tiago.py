"""The GPU model's URDF keeps the native model and maps joint 6's sign (CPU only)."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from tiago_mpc import tiago


def test_runtime_copies_preserve_native_model_and_gpu_sign_mapping():
    output = tiago.ROOT / "_expr_data/model-runtime-tests"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        native, gpu, signs = tiago.runtime_urdf(tmp)
        np.testing.assert_array_equal(signs, [1, 1, 1, 1, 1, -1, 1])
        original = pin.buildModelFromUrdf(str(tiago.ARM_URDF))
        native_model = pin.buildModelFromUrdf(str(native))
        np.testing.assert_array_equal(native_model.lowerPositionLimit, original.lowerPositionLimit)
        np.testing.assert_array_equal(native_model.lowerPositionLimit, tiago.limits().lower)
        gpu_model = pin.buildModelFromUrdf(str(gpu))
        np.testing.assert_array_equal(gpu_model.lowerPositionLimit,
                                      np.minimum(original.lowerPositionLimit * signs,
                                                 original.upperPositionLimit * signs))
        for mesh in ET.parse(native).iter("mesh"):
            assert Path(mesh.get("filename")).is_file()
        before = [path.stat().st_mtime_ns for path in (native, gpu)]
        tiago.runtime_urdf(tmp)
        assert [path.stat().st_mtime_ns for path in (native, gpu)] == before
