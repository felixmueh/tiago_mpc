"""CPU checks for the canonical simulator model and both-arm parameter patch."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'
RAW = MODELS / 'tiago_pro/tiago_pro_sim_source.urdf'
FULL = MODELS / 'tiago_pro/tiago_pro.urdf'
ARM = MODELS / 'tiago_right/tiago_right_arm.urdf'
NAMES = [f'arm_right_{i}_joint' for i in range(1, 8)]
ACTUAL = {
    'left': ([2.47, 3.52, 1.04, 1.35, .779, .773, .945],
             [7.42, 8.71, 4.75, 3.65, 3.65, 2.82, 1.79],
             [.511, 1.19, .472, .296, .424, .151, .14]),
    'right': ([1.66, 1.15, 1.16, .912, 1.25, .473, .56],
              [4.7, 9.69, 5.3, 4.2, 3.66, 2.95, 2.95],
              [.746, .882, .211, .648, .326, .343, .27]),
}


def structure(node, ignore_dynamics=False):
    return (node.tag, dict(node.attrib), (node.text or '').strip(), [
        structure(child, ignore_dynamics) for child in node
        if not (ignore_dynamics and child.tag in ('dynamics', 'tiago_mpc'))])


class ModelTests(unittest.TestCase):
    def test_raw_source_preserved_exactly(self):
        self.assertEqual(hashlib.sha256(RAW.read_bytes()).hexdigest(), 'c8d51a1e8cf363740a394ea272b9961a122a7bcfc418a6186088552cd907d3e6')
        original = Path('/workspace/GATO/tiago_tools/sim_tiago_urdf.xml')
        if original.is_file():
            self.assertEqual(RAW.read_bytes(), original.read_bytes())

    def test_only_arm_parameters_change_in_full_model(self):
        raw, full = ET.parse(RAW).getroot(), ET.parse(FULL).getroot()
        names = {f'arm_{side}_{i}_joint' for side in ACTUAL for i in range(1, 8)}
        for tree in (raw, full):
            for joint in tree.findall('joint'):
                if joint.get('name') in names:
                    for child in list(joint):
                        if child.tag in ('dynamics', 'tiago_mpc'):
                            joint.remove(child)
                    # Supplied effort values equal the original source numerically.
                    limit = joint.find('limit')
                    limit.set('effort', str(float(limit.get('effort'))))
        self.assertEqual(structure(full), structure(raw))

    def test_actual_coefficients_and_metadata(self):
        for path, sides in ((FULL, ACTUAL), (ARM, ('right',))):
            tree = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
            for side in sides:
                c, b, armature = ACTUAL[side]
                for i in range(7):
                    name = f'arm_{side}_{i+1}_joint'
                    joint = next(j for j in tree.findall('joint') if j.get('name') == name)
                    dynamics = joint.find('dynamics')
                    self.assertEqual(float(dynamics.get('friction')), c[i])
                    self.assertEqual(float(dynamics.get('damping')), b[i])
                    self.assertEqual(joint.find('tiago_mpc').attrib, {'armature': str(armature[i])})
                    self.assertEqual(float(joint.find('limit').get('effort')), 43 if i < 2 else 26)
                    comments = ' '.join(child.text or '' for child in joint if child.tag is ET.Comment)
                    self.assertIn('values supplied by Felix 2026-10-21', comments)
                    self.assertIn('QUESTIONABLE', comments)
                    self.assertIn('custom metadata', comments)
                    self.assertNotIn('ASSUMED', comments)

    def test_armature_is_metadata_only(self):
        model = pin.buildModelFromUrdf(str(ARM))
        np.testing.assert_array_equal(model.armature, np.zeros(7))

    def test_retained_arm_kinematics_and_inertias_match_simulator(self):
        raw, arm = ET.parse(RAW).getroot(), ET.parse(ARM).getroot()
        raw_links = {e.get('name'): e for e in raw.findall('link')}
        raw_joints = {e.get('name'): e for e in raw.findall('joint')}
        for link in arm.findall('link'):
            if link.get('name') == 'arm_right_grasp_link':
                continue  # Additional massless application frame, checked below.
            if link.get('name') == 'arm_right_tool_link':
                continue  # Composite gripper inertia is checked numerically below.
            self.assertEqual(structure(link), structure(raw_links[link.get('name')]))
        for joint in arm.findall('joint'):
            if joint.get('name') == 'arm_right_grasp_joint':
                continue
            original = raw_joints[joint.get('name')]
            for node in (joint, original):
                limit = node.find('limit')
                if limit is not None:
                    limit.set('effort', str(float(limit.get('effort'))))
            self.assertEqual(structure(joint, True), structure(original, True))

    def test_all_mesh_references_resolve_to_included_assets(self):
        for path in (RAW, FULL, ARM):
            missing = sorted({Path(e.get('filename')).name for e in ET.parse(path).iter('mesh')
                              if not (MODELS / 'tiago_pro/meshes' / Path(e.get('filename')).name).is_file()})
            self.assertEqual(missing, [], f'{path.name}: missing original meshes {missing}')

    def test_included_mesh_geometry_is_finite(self):
        import hppfcl
        loader = hppfcl.MeshLoader()
        meshes = sorted((MODELS / 'tiago_pro/meshes').glob('*.stl'))
        self.assertEqual(len(meshes), 35)
        for path in meshes:
            with self.subTest(mesh=path.name):
                mesh = loader.load(str(path))
                self.assertGreater(mesh.num_vertices, 0)
                self.assertGreater(mesh.num_tris, 0)
                self.assertTrue(np.isfinite(mesh.vertices()).all())

    def test_full_visual_and_collision_geometry_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = ET.parse(FULL)
            for mesh in tree.iter('mesh'):
                mesh.set('filename', str(MODELS / 'tiago_pro/meshes' / Path(mesh.get('filename')).name))
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

    def test_full_and_reduced_fk_mass_and_gravity(self):
        full = pin.buildModelFromUrdf(str(FULL))
        locked = [i for i in range(1, full.njoints) if full.names[i] not in NAMES]
        reference = pin.buildReducedModel(full, locked, pin.neutral(full))
        arm = pin.buildModelFromUrdf(str(ARM))
        self.assertEqual(list(reference.names)[1:], NAMES)
        self.assertEqual(list(arm.names)[1:], NAMES)
        rd, ad = reference.createData(), arm.createData()
        pin.framesForwardKinematics(reference, rd, pin.neutral(reference))
        arm.gravity.linear = rd.oMf[reference.getFrameId('torso_lift_link')].rotation.T @ reference.gravity.linear
        lower, upper = arm.lowerPositionLimit, arm.upperPositionLimit
        states = [np.array([-.39, -1.73, -.38, -2.35, 0., -1.21, .04]),
                  lower + .33*(upper-lower), lower + .71*(upper-lower)]
        for q in states:
            with self.subTest(q=q):
                np.testing.assert_allclose(pin.crba(arm, ad, q), pin.crba(reference, rd, q), atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(pin.computeGeneralizedGravity(arm, ad, q),
                                           pin.computeGeneralizedGravity(reference, rd, q), atol=1e-8, rtol=1e-8)
                pin.framesForwardKinematics(reference, rd, q)
                pin.framesForwardKinematics(arm, ad, q)
                poses = []
                for model, data in ((reference, rd), (arm, ad)):
                    base = data.oMf[model.getFrameId('torso_lift_link')].inverse()
                    poses.append((base * data.oMf[model.getFrameId('arm_right_tool_link')]).homogeneous)
                np.testing.assert_allclose(poses[0], poses[1], atol=1e-8, rtol=1e-8)

    def test_grasp_frame_matches_neutral_finger_pad_midpoint(self):
        full, arm = (pin.buildModelFromUrdf(str(p)) for p in (FULL, ARM))
        fd, ad = full.createData(), arm.createData()
        # Center of the inward flat face in fingertip.stl: x=-11.5 mm,
        # y in [7,47] mm, z in [-10,10] mm. Fingers stay at zero as in the
        # retained inertia reduction; the fixed TCP is nominal, not calibrated.
        pad_center = np.array([-.0115, .027, 0.])
        arm_indices = [full.joints[full.getJointId(n)].idx_q for n in NAMES]
        for q in (np.zeros(7), np.array([-.39, -1.73, -.38, -2.35, 0., -1.21, .04]),
                  arm.lowerPositionLimit + .6*(arm.upperPositionLimit-arm.lowerPositionLimit)):
            q_full = pin.neutral(full)
            q_full[arm_indices] = q
            pin.framesForwardKinematics(full, fd, q_full)
            pin.framesForwardKinematics(arm, ad, q)
            pads = [fd.oMf[full.getFrameId(f'gripper_right_fingertip_{side}_link')].act(pad_center)
                    for side in ('left', 'right')]
            torso = fd.oMf[full.getFrameId('torso_lift_link')]
            expected = torso.actInv(np.mean(pads, axis=0))
            actual = ad.oMf[arm.getFrameId('arm_right_grasp_joint')].translation
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
        self.assertEqual(arm.nq, 7)
        self.assertEqual(len(ET.parse(ARM).findall('link[@name="arm_right_grasp_link"]/inertial')), 0)

    def test_preparation_is_reproducible(self):
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_models.py'), '--check'], check=True)


if __name__ == '__main__':
    unittest.main()
