"""CPU checks for the canonical simulator model and right-arm metadata patch."""
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
C = [2.1, 2.6, 2., 1.95, .8, 1.1, .3]
B = [3.35, 4.15, 2.45, 2.25, .5, .3, .15]


def structure(node, ignore_dynamics=False):
    return (node.tag, dict(node.attrib), (node.text or '').strip(), [
        structure(child, ignore_dynamics) for child in node
        if not (ignore_dynamics and child.tag == 'dynamics')])


class ModelTests(unittest.TestCase):
    def test_raw_source_preserved_exactly(self):
        self.assertEqual(hashlib.sha256(RAW.read_bytes()).hexdigest(), 'c8d51a1e8cf363740a394ea272b9961a122a7bcfc418a6186088552cd907d3e6')
        original = Path('/workspace/GATO/tiago_tools/sim_tiago_urdf.xml')
        if original.is_file():
            self.assertEqual(RAW.read_bytes(), original.read_bytes())

    def test_only_right_arm_dynamics_change_in_full_model(self):
        raw, full = ET.parse(RAW).getroot(), ET.parse(FULL).getroot()
        # Everything outside the seven right-arm dynamics is canonical, including
        # left-arm metadata, all other geometry and the ten mimic followers.
        for tree in (raw, full):
            for joint in tree.findall('joint'):
                if joint.get('name') in NAMES:
                    joint.remove(joint.find('dynamics'))
        self.assertEqual(structure(full), structure(raw))

    def test_seven_coefficients_and_assumption_comments(self):
        for path in (FULL, ARM):
            tree = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
            for i, name in enumerate(NAMES):
                joint = next(j for j in tree.findall('joint') if j.get('name') == name)
                dynamics = joint.find('dynamics')
                self.assertEqual(float(dynamics.get('friction')), C[i])
                self.assertEqual(float(dynamics.get('damping')), B[i])
                comments = ' '.join(child.text or '' for child in joint if child.tag is ET.Comment)
                self.assertIn('TIAGO_MPC dynamics:', comments)
                self.assertIn('Nm s/rad', comments)
                self.assertIn('ASSUMED' if i >= 4 else 'User', comments)
                if i == 5:
                    self.assertIn('User c approximately 1.1', comments)
                if i == 6:
                    self.assertIn('both unmeasured', comments)

    def test_retained_arm_kinematics_and_inertias_match_simulator(self):
        raw, arm = ET.parse(RAW).getroot(), ET.parse(ARM).getroot()
        raw_links = {e.get('name'): e for e in raw.findall('link')}
        raw_joints = {e.get('name'): e for e in raw.findall('joint')}
        for link in arm.findall('link'):
            if link.get('name') == 'arm_right_tool_link':
                continue  # Composite gripper inertia is checked numerically below.
            self.assertEqual(structure(link), structure(raw_links[link.get('name')]))
        for joint in arm.findall('joint'):
            self.assertEqual(structure(joint, True), structure(raw_joints[joint.get('name')], True))

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
        data = ROOT / '_expr_data/canonical-model-correction-20261020'
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data) as tmp:
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

    def test_preparation_is_reproducible(self):
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_models.py'), '--check'], check=True)


if __name__ == '__main__':
    unittest.main()
