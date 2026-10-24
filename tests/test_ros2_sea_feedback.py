# Adapted from GATO ros-tiago c4cf5cc tests; no ROS runtime is needed.
"""SEA source selection and estimator safety, without ROS, GPU, or robot access."""
import builtins
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tiago_mpc import ros2 as ros
from tiago_mpc.tiago import RIGHT_ARM_JOINTS


def message(stamp, q=0., *, names=RIGHT_ARM_JOINTS):
    q = np.broadcast_to(q, (len(names),)).astype(float)
    sec = int(stamp)
    return NS(name=list(names), absolute_position=q.tolist(), position=(q + .2).tolist(),
              velocity=np.zeros(len(names)).tolist(),
              header=NS(stamp=NS(sec=sec, nanosec=round((stamp - sec) * 1e9))))


@pytest.fixture
def client(monkeypatch):
    clock = NS(now=1.)
    monkeypatch.setattr(ros.time, "monotonic", lambda: clock.now)
    subscriptions, publications = [], []
    node = NS(create_subscription=lambda *args: subscriptions.append(args),
              create_publisher=lambda *args: publications.append(args), destroy_node=lambda: None)
    fake_ros = dict(rclpy=NS(ok=lambda: True, create_node=lambda name: node),
                    JointState=object, Float64MultiArray=object)
    monkeypatch.setattr(ros, "_ros_imports", lambda: fake_ros)
    monkeypatch.setattr(ros, "_sea_state_type", lambda: NS())
    monkeypatch.setitem(sys.modules, "rclpy.qos", NS(
        QoSProfile=lambda **kwargs: NS(**kwargs), ReliabilityPolicy=NS(BEST_EFFORT="best_effort")))
    arm = ros.TiagoRightArmClient(sea_absolute=True)
    return NS(arm=arm, clock=clock, subscriptions=subscriptions, publications=publications)


def test_irregular_source_dt_name_order_and_unfiltered_position():
    estimator = ros.SeaAbsoluteVelocityEstimator(tau_sec=.01)
    names = tuple(reversed(RIGHT_ARM_JOINTS))
    base = np.arange(7, dtype=float)
    speed = np.arange(1, 8, dtype=float)
    assert estimator.update(message(10., base, names=names), 1.) is None
    for stamp, received in [(10.007, 1.02), (10.026, 1.03), (10.030, 1.08)]:
        raw = base + speed * (stamp - 10.)
        q, qd, returned_stamp = estimator.update(message(stamp, raw, names=names), received)
        np.testing.assert_allclose(q, raw[::-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(qd, speed[::-1], rtol=0, atol=1e-9)
        assert returned_stamp == stamp


def test_time_based_filter_only_changes_velocity():
    estimator = ros.SeaAbsoluteVelocityEstimator(tau_sec=.01)
    estimator.update(message(1., 0.), 1.)
    estimator.update(message(1.01, .01), 1.01)
    q, qd, _ = estimator.update(message(1.03, .07), 1.03)
    np.testing.assert_allclose(q, .07)
    np.testing.assert_allclose(qd, 1. + (1 - np.exp(-2.)) * 2.)


def test_duplicate_reversed_and_invalid_samples_do_not_refresh_estimator():
    estimator = ros.SeaAbsoluteVelocityEstimator()
    estimator.update(message(1., 0.), 1.)
    estimator.update(message(1.01, .01), 1.01)
    for stamp in [1.01, 1.005]:
        assert estimator.update(message(stamp, 100.), 1.05) is None
    q, qd, _ = estimator.update(message(1.02, .02), 1.06)
    np.testing.assert_allclose(q, .02)
    np.testing.assert_allclose(qd, 1.)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "absolute_short", "motor_short", "absolute_nan", "motor_nan", "stamp_nan"])
def test_reject_malformed_sea_samples(client, fault):
    arm = client.arm
    arm._on_joint_state(message(1.))
    arm._on_sea_state(message(1.))
    bad = message(1.01)
    if fault == "missing": bad.name[0] = "other_joint"
    elif fault == "duplicate": bad.name[0] = bad.name[1]
    elif fault == "absolute_short": bad.absolute_position.pop()
    elif fault == "motor_short": bad.position.pop()
    elif fault == "absolute_nan": bad.absolute_position[0] = float("nan")
    elif fault == "motor_nan": bad.position[0] = float("nan")
    elif fault == "stamp_nan": bad.header.stamp.sec = float("nan")
    arm._on_sea_state(bad)
    assert arm.latest_state() is None
    client.clock.now = 1.02
    arm._on_sea_state(message(1.02, .02))
    np.testing.assert_allclose(arm.latest_state().qd, 1.)


@pytest.mark.parametrize("gap_kind", ["source", "receive"])
def test_gap_requires_two_new_samples(client, gap_kind):
    arm = client.arm
    arm._on_joint_state(message(1.))
    arm._on_sea_state(message(1.))
    client.clock.now = 1.01
    arm._on_sea_state(message(1.01, .01))
    assert arm.latest_state() is not None
    client.clock.now = 1.2
    stamp = 1.2 if gap_kind == "source" else 1.02
    arm._on_joint_state(message(1.2))
    arm._on_sea_state(message(stamp, 1.))
    assert arm.latest_state() is None
    client.clock.now += .01
    arm._on_sea_state(message(stamp + .01, 1.02))
    np.testing.assert_allclose(arm.latest_state().qd, 2.)


def test_full_body_merge(client):
    arm = client.arm
    body_names = (*RIGHT_ARM_JOINTS, "torso_lift_joint", "wheel_right_joint")
    body = message(2., .4, names=body_names)
    body.velocity[-1] = float("nan")  # Normal unused wheel state on this robot.
    arm._on_joint_state(body)
    arm._on_sea_state(message(3., 1.))
    assert arm.latest_state() is None  # No fabricated startup velocity.
    client.clock.now = 1.02
    arm._on_sea_state(message(3.02, 1.04, names=tuple(reversed(RIGHT_ARM_JOINTS))))
    state = arm.latest_state()
    np.testing.assert_allclose(state.q, 1.04)
    np.testing.assert_allclose(state.qd, 2.)
    assert state.stamp_sec == 3.02
    assert state.received_monotonic_sec == 1.
    assert state.joint_positions["torso_lift_joint"] == pytest.approx(.6)
    assert np.isnan(state.joint_velocities["wheel_right_joint"])
    assert all(state.joint_positions[name] == 1.04 for name in RIGHT_ARM_JOINTS)
    assert client.subscriptions[0][3].depth == 1
    assert client.subscriptions[1][1] == ros.SEA_STATE_TOPIC


@pytest.mark.parametrize("frozen_source", ["body", "arm"])
def test_fresh_other_source_cannot_mask_stalled_stamp(client, frozen_source):
    arm = client.arm
    arm._on_joint_state(message(1.))
    arm._on_sea_state(message(1.))
    client.clock.now = 1.01
    arm._on_sea_state(message(1.01))
    assert arm.latest_state() is not None
    for now in [1.04, 1.08, 1.12]:
        client.clock.now = now
        arm._on_joint_state(message(1. if frozen_source == "body" else now))
        arm._on_sea_state(message(1.01 if frozen_source == "arm" else now))
    assert arm.latest_state() is None


def test_default_does_not_resolve_sea_dependency_or_change_jointstate_path(client, monkeypatch):
    monkeypatch.setattr(ros, "_sea_state_type", lambda: pytest.fail("optional SEA import"))
    arm = ros.TiagoRightArmClient()
    msg = message(5., .4)
    msg.velocity = [.3] * 7
    arm._on_joint_state(msg)
    state = arm.latest_state()
    np.testing.assert_allclose(state.q, .6)
    np.testing.assert_allclose(state.qd, .3)
    assert client.subscriptions[-1][3] == 10


def test_missing_optional_interface_gives_clear_error(monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "sea_state_msgs.msg":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="sea_state_msgs/msg/SeaState"):
        ros._sea_state_type()


@pytest.mark.parametrize("tau", [-1., float("nan"), float("inf")])
def test_bad_tau_rejected(tau):
    with pytest.raises(ValueError, match="finite and non-negative"):
        ros.SeaAbsoluteVelocityEstimator(tau_sec=tau)


def test_abs_position_is_sufficient_without_optional_motor_channel(client):
    arm = client.arm
    arm._on_joint_state(message(1.))
    first, second = message(1.), message(1.01, .02)
    first.position = second.position = []
    arm._on_sea_state(first)
    client.clock.now = 1.01
    arm._on_sea_state(second)
    np.testing.assert_allclose(arm.latest_state().qd, 2.)


def test_body_only_update_gets_new_safety_identity_without_new_arm_sample(client, monkeypatch):
    import queue
    from tiago_mpc import robot
    from tiago_mpc.safety import AsyncSafetyMonitor
    monkeypatch.setattr(robot, "TiagoRightArmClient", lambda **kw: client.arm)
    monkeypatch.setattr(robot.time, "monotonic", lambda: client.clock.now)
    safety_state = robot.safety_state
    arm = client.arm
    names = (*RIGHT_ARM_JOINTS, "torso_lift_joint")
    arm._on_joint_state(message(1., 0., names=names))
    arm._on_sea_state(message(1.))
    client.clock.now = 1.01
    arm._on_sea_state(message(1.01))
    first = arm.latest_state()
    plant = robot.RosPlant(robot.RobotConfig())
    plant._initial = first
    plant._check(first)
    client.clock.now = 1.02
    body = message(1.02, 0., names=names)
    body.position[-1] = .201  # New body configuration, below runtime drift cutoff.
    arm._on_joint_state(body)
    merged = arm.latest_state()
    assert merged.seq == first.seq and merged.stamp_sec == first.stamp_sec
    assert merged.received_monotonic_sec == 1.01  # the arm sample's receipt, older than the body's
    assert safety_state(merged).seq > safety_state(first).seq
    assert safety_state(merged).joint_positions["torso_lift_joint"] == .201
    plant._check(merged)  # Equal arm stamp remains valid for this arm seq.
    monitor = AsyncSafetyMonitor.__new__(AsyncSafetyMonitor)
    monitor.timeout_sec, monitor._closed = .05, False
    monitor.safety_fault = monitor.safety_message = ""
    monitor._process = NS(exitcode=None)
    monitor._last_checked_seq = safety_state(first).seq
    monitor._pending_seq = monitor._pending_since_monotonic_sec = None
    monitor._request_q, monitor._response_q = queue.Queue(), queue.Queue()
    monitor.check(safety_state(merged))
    submitted = monitor._request_q.get_nowait()
    assert submitted.joint_positions["torso_lift_joint"] == .201
    assert submitted.seq == safety_state(merged).seq

    # Body-only messages cannot make a stalled SEA arm source fresh.
    client.clock.now = 1.12
    arm._on_joint_state(message(1.12, 0., names=names))
    assert arm.latest_state() is None
