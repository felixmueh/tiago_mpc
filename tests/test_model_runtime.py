"""Shared model setup preserves native coordinates and the requested TCP."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from tiago_mpc import model_runtime as runtime
from tiago_mpc.joint_dynamics import Friction


class ModelRuntimeTests(unittest.TestCase):
    def test_runtime_copies_preserve_native_model_and_gpu_sign_mapping(self):
        output = runtime.ROOT / '_expr_data/model-runtime-tests'
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            native, gpu, signs, names, root = runtime.runtime_urdf(tmp)
            self.assertEqual(names, tuple(f'arm_right_{i}_joint' for i in range(1, 8)))
            self.assertEqual(root, 'torso_lift_link')
            np.testing.assert_array_equal(signs, [1, 1, 1, 1, 1, -1, 1])
            original = pin.buildModelFromUrdf(str(runtime.URDF))
            native_model = pin.buildModelFromUrdf(str(native))
            np.testing.assert_array_equal(native_model.lowerPositionLimit, original.lowerPositionLimit)
            gpu_model = pin.buildModelFromUrdf(str(gpu))
            np.testing.assert_array_equal(gpu_model.lowerPositionLimit,
                                         np.minimum(original.lowerPositionLimit * signs,
                                                    original.upperPositionLimit * signs))
            for mesh in ET.parse(native).iter('mesh'):
                self.assertTrue(Path(mesh.get('filename')).is_file())
            before = [path.stat().st_mtime_ns for path in (native, gpu)]
            runtime.runtime_urdf(tmp)
            self.assertEqual([path.stat().st_mtime_ns for path in (native, gpu)], before)

    def test_smooth_model_check_uses_requested_flange_frame(self):
        model = pin.buildModelFromUrdf(str(runtime.URDF))
        data = model.createData()
        flange = model.getFrameId('arm_right_tool_joint')
        expected = runtime.ee_position(model, data, runtime.Q_CHECK, flange)
        # Stop after FK: this regression tests frame selection, not CUDA dynamics.
        reached_dynamics = RuntimeError('FK check passed')
        solver = Mock()
        solver.ee_pos.return_value = expected
        solver.sim_forward.side_effect = reached_dynamics
        losses = Friction([1.] * 7, [1.] * 7, .03)
        with self.assertRaisesRegex(RuntimeError, 'FK check passed'):
            runtime.check_model(solver, model, data, flange, np.ones(7), losses)
        solver.sim_forward.assert_called_once()


if __name__ == '__main__':
    unittest.main()
