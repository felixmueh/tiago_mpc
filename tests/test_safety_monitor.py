"""ROS-free port regressions; run with Pinocchio using unittest or pytest.

The synthetic posture is intentionally NOT a physical qualification. Its default
4 cm rejection is retained. A separate dummy checker exercises healthy transport.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import queue
from types import SimpleNamespace
import os
import signal
import tempfile
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from tiago_mpc import safety as s
from tiago_mpc.tiago import FULL_URDF, Q_START, RIGHT_ARM_JOINTS
from tiago_mpc.util import cache_root


# The reviewed collision source, blacklist and alpha-wrap receipt, byte for byte as in
# GATO fcc9621 tiago_configs/ (models/README.md).
RETAINED_SHA256 = {
    s.COLLISION_OVERLAY_PATH: '98d18490cdf1ec9e9f3f749f43d96d0203d598344f47425e0287d1029b3894fd',
    s.DEFAULT_COLLISION_BLACKLIST_PATH: 'af70b26bd0ab6936a1d928b8e580d805097d914129f349dd351cb8d77b663e75',
    s.COLLISION_DIRECTORY / 'alpha_wrap_results.json':
        'a9ba92bbd9e78fdd513f3de41b7394e9f400ffb6261a318a812f3e35840225cf',
}


def xml_tree(path):
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()


def structure(node, *, ignore_collisions=False):
    return (node.tag, node.attrib, (node.text or '').strip(), [
        structure(child, ignore_collisions=ignore_collisions) for child in node
        if not (ignore_collisions and child.tag == 'collision')])


def make_state(model, seq=1):
    positions = {name: 0. for name in model.state_updated_joint_names}
    positions.update(dict(zip(RIGHT_ARM_JOINTS, map(float, Q_START))))
    return s.SafetyCheckState(
        q=Q_START.copy(), qd=np.zeros(7), stamp_sec=1.,
        received_monotonic_sec=time.monotonic(), seq=seq,
        joint_positions=positions,
        joint_velocities={name: 0. for name in positions},
    )


def mechanics_settings(**kwargs):
    return s.CollisionSafetySettings(check_timeout_sec=.5, **kwargs)


class DummyChecker:
    """Transport-only test fixture; never used by production or geometry tests."""
    def __init__(self, **kwargs):
        pass

    def check(self, state):
        pass


def dummy_worker_main(**kwargs):
    s._SafetyChecker = DummyChecker
    s._safety_worker_main(**kwargs)


class CollisionPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = s.build_tiago_collision_model(blacklist_path=s.DEFAULT_COLLISION_BLACKLIST_PATH)

    def named(self):
        state = make_state(self.model)
        return s.NamedJointState(state.joint_positions, state.joint_velocities)

    def test_default_policy_preserved(self):
        settings = s.CollisionSafetySettings()
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.controlled_joint_names, RIGHT_ARM_JOINTS)
        self.assertEqual(settings.min_distance_m, .04)
        self.assertEqual(settings.check_timeout_sec, .05)
        self.assertEqual(settings.max_monitored_geometry_speed_m_s, 1.)
        self.assertEqual(settings.joint_position_margin_rad, 0.)
        self.assertEqual(settings.joint_velocity_scale, 1.)
        for name in ('min_distance_m', 'joint_position_margin_rad',
                     'check_timeout_sec', 'max_monitored_geometry_speed_m_s',
                     'joint_velocity_scale'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                s.CollisionSafetySettings(**{name: float('nan')})

    def test_assets_resolve_outside_repository_working_directory(self):
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                urdf = s._resolved_collision_urdf()
                self.assertTrue(urdf.is_relative_to(cache_root()))
                for mesh in ET.parse(urdf).findall('.//mesh'):
                    self.assertFalse(Path(mesh.get('filename')).is_absolute())
                    path = (urdf.parent / mesh.get('filename')).resolve()
                    self.assertTrue(path.is_file())
                    self.assertTrue(path.is_relative_to(cache_root() / 'models' / 'meshes'))
            finally:
                os.chdir(cwd)

    def test_reduced_state_boundary_and_monitored_geometry(self):
        m = self.model
        self.assertEqual(m.model.nq, 17)
        self.assertEqual(len(m.geometry_model.geometryObjects), 51)
        self.assertEqual(len(m.geometry_model.collisionPairs), 569)
        self.assertFalse(set(s.TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES)
                         & set(m.state_updated_joint_names))
        self.assertIn('arm_left_1_joint', m.state_updated_joint_names)
        monitored = set(m.monitored_geometry_indices)
        self.assertTrue(any(m.geometry_model.geometryObjects[i].name.startswith('gripper_right_')
                            for i in monitored))
        self.assertNotIn('arm_left_1_joint', m.monitored_geometry_parent_joint_names)
        for pair in m.geometry_model.collisionPairs:
            self.assertTrue(pair.first in monitored or pair.second in monitored)
        self.assertTrue(np.all(m.geometry_radii_m > 0))

    def test_overlay_preserves_all_canonical_noncollision_metadata(self):
        active, canonical = s.collision_urdf_tree(), ET.parse(FULL_URDF).getroot()
        self.assertEqual(len(active.findall('link')), 70)
        self.assertEqual(len(active.findall('joint')), 69)
        self.assertEqual(len(active.findall('joint/mimic')), 10)
        # Includes all joint origins/axes/limits/dynamics/mimics, link inertias,
        # native sensor frames, visuals and plugins.
        self.assertEqual(structure(active, ignore_collisions=True),
                         structure(canonical, ignore_collisions=True))

    def test_collision_overlay_and_retained_sources(self):
        # The reviewed source, blacklist and generation receipt are historical evidence.
        for path, expected in RETAINED_SHA256.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, path.name)
        active = {link.get('name'): link for link in s.collision_urdf_tree().findall('link')}
        reviewed = {link.get('name'): link for link in xml_tree(s.COLLISION_OVERLAY_PATH).findall('link')}
        canonical = {link.get('name'): link for link in xml_tree(FULL_URDF).findall('link')}
        for name, link in active.items():
            expected_link = reviewed.get(name, canonical[name])
            self.assertEqual([structure(c) for c in link.findall('collision')],
                             [structure(c) for c in expected_link.findall('collision')], name)

    def test_new_fixed_sensor_obstacles_are_present_and_paired_with_arm(self):
        expected = {'base_antenna_left_link_0', 'base_antenna_right_link_0',
                    'base_front_laser_link_0', 'base_rear_laser_link_0'}
        geometry = self.model.geometry_model
        indices = {index for index, obj in enumerate(geometry.geometryObjects) if obj.name in expected}
        self.assertEqual(len(indices), 4)
        monitored = set(self.model.monitored_geometry_indices)
        for index in indices:
            pairs = [pair for pair in geometry.collisionPairs if index in (pair.first, pair.second)]
            self.assertEqual(len(pairs), len(monitored))
            self.assertTrue(all(pair.first in monitored or pair.second in monitored for pair in pairs))
        canonical = xml_tree(FULL_URDF)
        new_links = {name.removesuffix('_0') for name in expected}
        joints = [joint for joint in canonical.findall('joint')
                  if joint.find('child').get('link') in new_links]
        self.assertEqual(len(joints), 4)
        self.assertTrue(all(joint.get('type') == 'fixed' for joint in joints))

    def test_safety_mesh_geometry_is_finite(self):  # the full model's meshes: test_models.py
        import hppfcl
        loader = hppfcl.MeshLoader()
        meshes = sorted(s.COLLISION_DIRECTORY.glob('meshes/*.stl'))
        self.assertEqual(len(meshes), 18)
        for path in meshes:
            with self.subTest(mesh=str(path.relative_to(s.REPO_ROOT))):
                mesh = loader.load(str(path))
                self.assertGreater(mesh.num_vertices, 0)
                self.assertGreater(mesh.num_tris, 0)
                self.assertTrue(np.isfinite(mesh.vertices()).all())

    def test_all_named_positions_and_velocities_required_and_finite(self):
        for field in ('position', 'velocity'):
            for value in ('missing', float('nan'), float('inf')):
                with self.subTest(field=field, value=value):
                    named = self.named()
                    mapping = getattr(named, field)
                    if value == 'missing':
                        del mapping['arm_left_1_joint']
                    else:
                        mapping['arm_left_1_joint'] = value
                    with self.assertRaisesRegex(ValueError, 'arm_left_1_joint'):
                        s.state_to_qv(self.model.model, named)

    def test_position_velocity_limits_and_controlled_joint_filter(self):
        state = self.named()
        name = RIGHT_ARM_JOINTS[0]
        joint = self.model.model.joints[self.model.model.getJointId(name)]
        upper = self.model.model.upperPositionLimit[joint.idx_q]
        state.position[name] = upper + .01
        state.velocity[name] = self.model.model.velocityLimit[joint.idx_v] + .01
        violations = s.check_joint_limits(self.model.model, state, joint_names=RIGHT_ARM_JOINTS)
        self.assertEqual([v.kind for v in violations], ['position_upper', 'velocity'])
        state.position[name] = upper - .005
        state.velocity[name] = 0.
        self.assertTrue(s.check_joint_limits(self.model.model, state,
                        position_margin=.01, joint_names=RIGHT_ARM_JOINTS))
        self.assertFalse(s.check_joint_limits(self.model.model, state,
                         joint_names=['arm_left_1_joint']))

    def test_speed_bound_contains_finite_difference_point_velocity(self):
        state = self.named()
        state.velocity.update(dict(zip(RIGHT_ARM_JOINTS, [.1, -.2, .3, -.1, .1, .1, -.2])))
        q, v = s.state_to_qv(self.model.model, state)
        reports = s.compute_collision_body_speeds(self.model, q, v)
        dt = 1e-7
        def placements(q):
            data, geometry_data = self.model.make_data()
            pin.updateGeometryPlacements(self.model.model, data, self.model.geometry_model,
                                         geometry_data, q)
            return list(geometry_data.oMg)
        before, after = placements(q), placements(pin.integrate(self.model.model, q, dt * v))
        for report, index in zip(reports, self.model.monitored_geometry_indices):
            radius = self.model.geometry_radii_m[index]
            for local in (np.zeros(3), np.asarray([radius, 0., 0.])):
                old = before[index].translation + before[index].rotation @ local
                new = after[index].translation + after[index].rotation @ local
                self.assertLessEqual(np.linalg.norm((new-old)/dt), report.speed_bound_m_s + 2e-4)

    def test_default_high_clearance_arm_is_rejected_with_zero_other_joints(self):
        # Right-arm configuration alone cannot qualify a full-body initial state.
        state = make_state(self.model)
        checker = s._SafetyChecker(settings=s.CollisionSafetySettings(), initial_state=state)
        with self.assertRaises(s._SafetyFault) as caught:
            checker.check(state)
        self.assertEqual(caught.exception.fault_kind, 'collision_margin')
        q, _ = s.state_to_qv(self.model.model, self.named())
        violations = s.compute_collision_margin_violations(self.model, q, margin_m=.04)
        self.assertTrue(any(v.geometry_a == 'torso_base_link_0' and
                            v.geometry_b == 'arm_right_4_link_0' for v in violations))

    def test_body_speed_fault_and_joint_limit_fault(self):
        state = make_state(self.model)
        checker = s._SafetyChecker(settings=mechanics_settings(
            max_monitored_geometry_speed_m_s=1e-9), initial_state=state)
        state.joint_velocities[RIGHT_ARM_JOINTS[0]] = .1
        # Isolate classification/order from a separate collision-margin failure.
        with patch.object(s, 'compute_collision_margin_violation', return_value=None):
            with self.assertRaises(s._SafetyFault) as caught:
                checker.check(state)
        self.assertEqual(caught.exception.fault_kind, 'collision_body_speed')
        state.joint_positions[RIGHT_ARM_JOINTS[0]] = 100.
        with self.assertRaises(s._SafetyFault) as caught:
            checker.check(state)
        self.assertEqual(caught.exception.fault_kind, 'joint_limit')

    def test_missing_assets_and_blacklist_fail_closed(self):
        missing = s.REPO_ROOT / 'models' / 'not-a-safety-asset'
        with self.assertRaises(FileNotFoundError):
            s.build_tiago_collision_model(urdf_path=missing)
        with self.assertRaises(FileNotFoundError):
            s._SafetyChecker(settings=replace(mechanics_settings(), blacklist_path=missing),
                             initial_state=make_state(self.model))
        with self.assertRaises(FileNotFoundError):
            s.load_ignored_geometry_pairs(missing)


class SpawnedSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = s.build_tiago_collision_model(blacklist_path=s.DEFAULT_COLLISION_BLACKLIST_PATH)

    def test_spawned_transport_checks_new_sequences_and_closes(self):
        state = make_state(self.model)
        # Replace only the child checker; retain real spawn/queues/worker loop.
        context = s.mp.get_context('spawn')
        process_constructor = context.Process
        def dummy_process(*args, **kwargs):
            kwargs['target'] = dummy_worker_main
            return process_constructor(*args, **kwargs)
        with patch.object(s.mp, 'get_context', return_value=context):
            with patch.object(context, 'Process', side_effect=dummy_process):
                monitor = s.AsyncSafetyMonitor(settings=mechanics_settings(), initial_state=state)
        try:
            self.assertTrue(monitor._process.is_alive())
            monitor.wait_until_checked(state)
            self.assertEqual(monitor.last_checked_seq, 1)
            monitor.wait_until_checked(replace(state, seq=2))
            self.assertEqual(monitor.safety_status, 'ok')
            self.assertEqual(monitor.last_checked_seq, 2)
        finally:
            monitor.close()
            monitor.close()
        self.assertFalse(monitor._process.is_alive())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            monitor.check(state)

    def test_the_spawned_worker_ignores_sigint(self):
        """Ctrl-C reaches the worker too; the parent ends the run and closes it."""
        state = make_state(self.model)
        context = s.mp.get_context('spawn')
        process_constructor = context.Process
        def dummy_process(*args, **kwargs):
            kwargs['target'] = dummy_worker_main
            return process_constructor(*args, **kwargs)
        with patch.object(s.mp, 'get_context', return_value=context):
            with patch.object(context, 'Process', side_effect=dummy_process):
                monitor = s.AsyncSafetyMonitor(settings=mechanics_settings(), initial_state=state)
        try:
            monitor.wait_until_checked(state)
            os.kill(monitor._process.pid, signal.SIGINT)
            time.sleep(.2)
            self.assertTrue(monitor._process.is_alive())
            monitor.wait_until_checked(replace(state, seq=2))
            self.assertEqual(monitor.last_checked_seq, 2)
        finally:
            monitor.close()
        self.assertFalse(monitor._process.is_alive())

    def test_worker_classifies_invalid_full_state_and_latches_fault(self):
        state = make_state(self.model)
        monitor = s.AsyncSafetyMonitor(settings=mechanics_settings(), initial_state=state)
        try:
            del state.joint_velocities['head_2_joint']
            with self.assertRaisesRegex(RuntimeError, 'head_2_joint'):
                monitor.wait_until_checked(state)
            self.assertEqual(monitor.safety_fault, 'state_invalid')
            with self.assertRaisesRegex(RuntimeError, 'head_2_joint'):
                monitor.check(make_state(self.model, seq=2))
        finally:
            monitor.close()

    def test_spawned_default_margin_fault(self):
        state = make_state(self.model)
        settings = replace(s.CollisionSafetySettings(), check_timeout_sec=.5)
        monitor = s.AsyncSafetyMonitor(settings=settings, initial_state=state)
        try:
            with self.assertRaisesRegex(RuntimeError, 'margin violation'):
                monitor.wait_until_checked(state)
            self.assertEqual(monitor.safety_fault, 'collision_margin')
        finally:
            monitor.close()

    def test_worker_death_is_fault(self):
        state = make_state(self.model)
        monitor = s.AsyncSafetyMonitor(settings=mechanics_settings(), initial_state=state)
        monitor._process.terminate()
        monitor._process.join(timeout=2.)
        try:
            with self.assertRaisesRegex(RuntimeError, 'worker exited'):
                monitor.check(state)
            self.assertEqual(monitor.safety_fault, 'worker_error')
        finally:
            monitor.close()

    def test_alive_but_unresponsive_worker_times_out(self):
        state = make_state(self.model)
        settings = replace(mechanics_settings(), check_timeout_sec=.03)
        monitor = s.AsyncSafetyMonitor(settings=settings, initial_state=state)
        os.kill(monitor._process.pid, signal.SIGSTOP)
        try:
            self.assertTrue(monitor._process.is_alive())
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                monitor.wait_until_checked(state)
            self.assertEqual(monitor.safety_status, 'stale')
            self.assertEqual(monitor.safety_fault, 'worker_stale')
        finally:
            os.kill(monitor._process.pid, signal.SIGCONT)
            monitor.close()


class SafetyDeadlineTests(unittest.TestCase):
    def test_late_success_cannot_clear_pending_deadline_and_fault_latches(self):
        monitor = s.AsyncSafetyMonitor.__new__(s.AsyncSafetyMonitor)
        monitor.timeout_sec = .05
        monitor._closed = False
        monitor.safety_fault = monitor.safety_message = ''
        monitor.safety_status = 'ok'
        monitor._process = SimpleNamespace(exitcode=None)
        monitor._last_checked_seq = 0
        monitor._pending_seq = 1
        monitor._pending_since_monotonic_sec = 10.
        monitor._request_q = queue.Queue(maxsize=1)
        monitor._response_q = queue.Queue()
        monitor._response_q.put(s._SafetyCheckResult(seq=1, ok=True, elapsed_sec=.06))
        state = s.SafetyCheckState(np.zeros(7), np.zeros(7), 100.06, 10.06, 2, {}, {})
        with patch.object(s.time, 'monotonic', return_value=10.06):
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                monitor.check(state)
            self.assertEqual(monitor.safety_fault, 'worker_stale')
            self.assertEqual(monitor.last_checked_seq, 0)
            self.assertTrue(monitor._request_q.empty())
            # A queued success cannot recover a previously detected fault.
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                monitor.check(state)


if __name__ == '__main__':
    unittest.main()
