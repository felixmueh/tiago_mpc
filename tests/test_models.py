"""CPU checks of the full model, its meshes and the reduced arm derived from it."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from tiago_mpc.tiago import FULL_URDF, MESHES, Q_START, RIGHT_ARM_JOINTS, ROOT_FRAME, arm_urdf


class ModelTests(unittest.TestCase):
    def test_all_mesh_references_resolve_to_included_assets(self):
        missing = sorted({Path(e.get('filename')).name for e in ET.parse(FULL_URDF).iter('mesh')
                          if not (MESHES / Path(e.get('filename')).name).is_file()})
        self.assertEqual(missing, [], f'missing original meshes {missing}')
        # The arm's meshes are cached copies of included ones, named by their contents.
        included = {hashlib.sha256(path.read_bytes()).hexdigest() for path in MESHES.glob('*.stl')}
        arm = arm_urdf()
        for mesh in ET.parse(arm).iter('mesh'):
            copy = arm.parent / mesh.get('filename')
            self.assertEqual(copy.stem, hashlib.sha256(copy.read_bytes()).hexdigest())
            self.assertIn(copy.stem, included)

    def test_included_mesh_geometry_is_finite(self):
        import hppfcl
        loader = hppfcl.MeshLoader()
        meshes = sorted(MESHES.glob('*.stl'))
        self.assertEqual(len(meshes), 35)
        for path in meshes:
            with self.subTest(mesh=path.name):
                mesh = loader.load(str(path))
                self.assertGreater(mesh.num_vertices, 0)
                self.assertGreater(mesh.num_tris, 0)
                self.assertTrue(np.isfinite(mesh.vertices()).all())

    def test_full_visual_and_collision_geometry_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = ET.parse(FULL_URDF)
            for mesh in tree.iter('mesh'):
                mesh.set('filename', str(MESHES / Path(mesh.get('filename')).name))
            resolved = Path(tmp) / 'resolved-full.urdf'
            tree.write(resolved)
            model = pin.buildModelFromUrdf(str(resolved))
            for kind, geometry_type in (('visual', pin.GeometryType.VISUAL),
                                        ('collision', pin.GeometryType.COLLISION)):
                geometry = pin.buildGeomFromUrdf(model, str(resolved), geometry_type)
                expected = len(tree.findall(f'link/{kind}'))
                self.assertEqual(len(geometry.geometryObjects), expected)
                for obj in geometry.geometryObjects:
                    obj.geometry.computeLocalAABB()
                    self.assertTrue(np.isfinite(obj.geometry.aabb_local.min_).all())
                    self.assertTrue(np.isfinite(obj.geometry.aabb_local.max_).all())

    def test_full_and_reduced_fk_mass_gravity_and_joint_parameters(self):
        # Pinocchio's own reduction of the full model, gripper locked at neutral as in arm_urdf.
        full = pin.buildModelFromUrdf(str(FULL_URDF))
        locked = [i for i in range(1, full.njoints) if full.names[i] not in RIGHT_ARM_JOINTS]
        reference = pin.buildReducedModel(full, locked, pin.neutral(full))
        arm = pin.buildModelFromUrdf(str(arm_urdf()))
        self.assertEqual(list(reference.names)[1:], list(RIGHT_ARM_JOINTS))
        self.assertEqual(list(arm.names)[1:], list(RIGHT_ARM_JOINTS))
        for name in ('lowerPositionLimit', 'upperPositionLimit', 'velocityLimit', 'effortLimit',
                     'damping', 'friction'):
            np.testing.assert_array_equal(getattr(arm, name), getattr(reference, name), name)
        rd, ad = reference.createData(), arm.createData()
        pin.framesForwardKinematics(reference, rd, pin.neutral(reference))
        arm.gravity.linear = rd.oMf[reference.getFrameId(ROOT_FRAME)].rotation.T @ reference.gravity.linear
        lower, upper = arm.lowerPositionLimit, arm.upperPositionLimit
        for q in (Q_START, lower + .33*(upper-lower), lower + .71*(upper-lower)):
            with self.subTest(q=q):
                np.testing.assert_allclose(pin.crba(arm, ad, q), pin.crba(reference, rd, q), atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(pin.computeGeneralizedGravity(arm, ad, q),
                                           pin.computeGeneralizedGravity(reference, rd, q), atol=1e-8, rtol=1e-8)
                pin.framesForwardKinematics(reference, rd, q)
                pin.framesForwardKinematics(arm, ad, q)
                poses = []
                for model, data in ((reference, rd), (arm, ad)):
                    base = data.oMf[model.getFrameId(ROOT_FRAME)].inverse()
                    poses.append((base * data.oMf[model.getFrameId('arm_right_tool_link')]).homogeneous)
                np.testing.assert_allclose(poses[0], poses[1], atol=1e-8, rtol=1e-8)

    def test_grasp_frame_matches_neutral_finger_pad_midpoint(self):
        full, arm = (pin.buildModelFromUrdf(str(p)) for p in (FULL_URDF, arm_urdf()))
        fd, ad = full.createData(), arm.createData()
        # Center of the inward flat face in fingertip.stl: x=-11.5 mm,
        # y in [7,47] mm, z in [-10,10] mm. Fingers stay at zero as in the
        # inertia reduction; the fixed TCP is nominal, not calibrated.
        pad_center = np.array([-.0115, .027, 0.])
        arm_indices = [full.joints[full.getJointId(n)].idx_q for n in RIGHT_ARM_JOINTS]
        for q in (np.zeros(7), Q_START, arm.lowerPositionLimit + .6*(arm.upperPositionLimit-arm.lowerPositionLimit)):
            q_full = pin.neutral(full)
            q_full[arm_indices] = q
            pin.framesForwardKinematics(full, fd, q_full)
            pin.framesForwardKinematics(arm, ad, q)
            pads = [fd.oMf[full.getFrameId(f'gripper_right_fingertip_{side}_link')].act(pad_center)
                    for side in ('left', 'right')]
            torso = fd.oMf[full.getFrameId(ROOT_FRAME)]
            expected = torso.actInv(np.mean(pads, axis=0))
            actual = ad.oMf[arm.getFrameId('arm_right_grasp_joint')].translation
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
        self.assertEqual(arm.nq, 7)
        self.assertEqual(len(ET.parse(arm_urdf()).findall('link[@name="arm_right_grasp_link"]/inertial')), 0)


if __name__ == '__main__':
    unittest.main()
