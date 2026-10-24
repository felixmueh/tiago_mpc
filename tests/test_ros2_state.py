"""Exercise the transport boundary with a deterministic ROS executor double."""
from collections import deque
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from tiago_mpc import ros2 as ros
from tiago_mpc.tiago import RIGHT_ARM_JOINTS


def message(stamp=1., *, names=RIGHT_ARM_JOINTS, offset=0.):
    n = len(names)
    values = (np.arange(n) + offset).tolist()
    return NS(name=list(names), position=values, velocity=[-.1] * n,
              absolute_position=values, header=NS(stamp=NS(sec=int(stamp),
              nanosec=round((stamp - int(stamp)) * 1e9))))


@pytest.fixture
def transport(monkeypatch):
    clock = NS(now=100.)
    monkeypatch.setattr(ros.time, 'monotonic', lambda: clock.now)
    callbacks, subscriptions, publications, timeouts = deque(), [], [], []
    def publisher(*args):
        sent = []
        publications.append((args, sent))
        return NS(publish=sent.append)
    node = NS(create_subscription=lambda *args: subscriptions.append(args),
              create_publisher=publisher, destroy_node=lambda: None)
    def spin_once(node, timeout_sec):
        timeouts.append(timeout_sec)
        if callbacks:
            callbacks.popleft()()
        else:
            clock.now += timeout_sec
    fake = dict(rclpy=NS(ok=lambda: True, create_node=lambda name: node,
                         spin_once=spin_once), JointState=object,
                Float64MultiArray=NS, Duration=NS)
    monkeypatch.setattr(ros, '_ros_imports', lambda: fake)
    monkeypatch.setattr(ros, '_sea_state_type', lambda: NS())
    monkeypatch.setitem(sys.modules, 'rclpy.qos', NS(
        QoSProfile=lambda **kwargs: NS(**kwargs),
        ReliabilityPolicy=NS(BEST_EFFORT='best_effort')))
    return NS(new=ros.TiagoRightArmClient, callbacks=callbacks,
              publications=publications, subscriptions=subscriptions,
              timeouts=timeouts, clock=clock)


def test_two_feedback_subscriptions_are_serviced_in_one_command_tick(transport):
    """Legacy one spin_once consumed only body; SEA estimates lagged behind."""
    arm = transport.new(sea_absolute=True)
    arm._on_sea_state(message())  # Prior encoder sample; no fabricated velocity.
    transport.callbacks.extend([
        lambda: arm._on_joint_state(message()),
        lambda: arm._on_sea_state(message(1.01, offset=.01)),
    ])
    count = arm.spin_once(timeout_sec=.1)
    assert arm.latest_state() is not None
    assert arm.latest_state().stamp_sec == 1.01
    assert count == 2
    np.testing.assert_allclose(arm.latest_state().qd, 1.)
    assert transport.timeouts == [.1, 0., 0.]
    assert all(args[3].depth == 1 for args in transport.subscriptions)


def test_callback_burst_is_bounded_under_continuous_input(transport):
    arm = transport.new()
    for i in range(100):
        transport.callbacks.append(lambda i=i: arm._on_joint_state(message(1. + .001*i)))
    assert arm.spin_once(max_callbacks=4) == 4
    assert len(transport.callbacks) == 96
    assert arm.latest_state().seq == 4


def test_drain_budget_limits_extra_calls_but_does_not_interrupt_callback(transport):
    arm = transport.new()
    def expensive_callback():
        arm._on_joint_state(message())
        transport.clock.now += .01
    transport.callbacks.extend([expensive_callback] * 4)
    # Initial callback may wait/work. The drain then stops after one slow callback.
    assert arm.spin_once(drain_budget_sec=.002) == 2
    assert transport.clock.now == pytest.approx(100.02)
    assert len(transport.callbacks) == 2


def test_read_after_seq_does_not_accept_a_cached_measurement(transport):
    arm = transport.new()
    arm._on_joint_state(message())
    old = arm.latest_state()
    assert arm.read_state(timeout_sec=0) is old
    with pytest.raises(TimeoutError, match='after_seq=1'):
        arm.read_state(timeout_sec=.01, after_seq=old.seq)
    transport.callbacks.append(lambda: arm._on_joint_state(message(1.02)))
    # Same q/v values still constitute a distinct observation at an advancing stamp.
    assert arm.read_state(timeout_sec=0, after_seq=old.seq).seq == 2


def test_qualifying_cached_read_does_not_add_a_50ms_wait(transport):
    arm = transport.new()
    arm._on_joint_state(message())
    before = transport.clock.now
    state = arm.read_state(timeout_sec=2.)
    assert state.seq == 1
    assert transport.clock.now == before
    assert transport.timeouts == [0.]


def test_no_new_state_or_no_fresh_state_times_out(transport):
    arm = transport.new()
    with pytest.raises(TimeoutError):
        arm.read_state(timeout_sec=0)
    arm._on_joint_state(message())
    transport.clock.now += 1.
    with pytest.raises(TimeoutError):
        arm.read_state(timeout_sec=0, max_age_sec=.1)


def test_replayed_joint_message_does_not_refresh_age_or_sequence(transport):
    arm = transport.new()
    arm._on_joint_state(message(2.))
    state = arm.latest_state()
    transport.clock.now += .1
    arm._on_joint_state(message(2.))
    arm._on_joint_state(message(1.))
    assert arm.latest_state() is state
    assert transport.clock.now - state.received_monotonic_sec == pytest.approx(.1)
    with pytest.raises(TimeoutError):
        arm.read_state(timeout_sec=0, after_seq=state.seq)


def test_named_order_native_coordinates_and_full_body_are_retained(transport):
    arm = transport.new()
    names = ('torso_lift_joint', *reversed(RIGHT_ARM_JOINTS), 'wheel_right_joint')
    msg = message(names=names)
    msg.velocity[-1] = float('nan')
    arm._on_joint_state(msg)
    state = arm.latest_state()
    np.testing.assert_array_equal(state.q, np.arange(7, 0, -1))
    np.testing.assert_allclose(state.qd, -.1)
    assert state.q[5] == 2.  # Native joint6 sign: no GPU-model reflection here.
    assert state.joint_positions['torso_lift_joint'] == 0.
    assert np.isnan(state.joint_velocities['wheel_right_joint'])
    assert transport.subscriptions[0][3] == 10  # Preserve standard reliable QoS.


@pytest.mark.parametrize('fault', ['missing_joint', 'duplicate_joint', 'short_position',
                                  'no_velocity', 'q_nan', 'qd_nan'])
def test_malformed_controlled_state_does_not_become_fresh(transport, fault):
    arm = transport.new()
    msg = message()
    if fault == 'missing_joint': msg.name[0] = 'other'
    if fault == 'duplicate_joint': msg.name[0] = msg.name[1]
    if fault == 'short_position': msg.position.pop()
    if fault == 'no_velocity': msg.velocity = []
    if fault == 'q_nan': msg.position[0] = float('nan')
    if fault == 'qd_nan': msg.velocity[0] = float('nan')
    arm._on_joint_state(msg)
    assert arm.latest_state() is None


def test_effort_preserves_joint_order_and_native_signs(transport):
    arm = transport.new()
    values = [-1., 2., -3., 4., -5., -6., 7.]
    arm.publish_effort(values)
    assert transport.publications[0][1][-1].data == values


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_bad_durations_and_nonfinite_commands_are_rejected(transport, bad):
    arm = transport.new()
    with pytest.raises(ValueError):
        ros._duration_msg(arm.ros, bad)
    with pytest.raises(ValueError):
        arm.publish_effort([0.]*6 + [float('nan')])
    assert not transport.publications[0][1]


def test_importing_the_package_loads_no_ros_and_no_compiled_gato():
    # Importing may load Pinocchio and gato's Python package (MpcParams is gato's SolverParams);
    # only RosPlant() loads ROS, and only the MPC process loads the compiled CUDA module.
    import subprocess
    from pathlib import Path
    result = subprocess.run([sys.executable, '-c',
        'import sys\n'
        'import tiago_mpc, tiago_mpc.sim, tiago_mpc.robot, experiments.replay\n'
        'assert "rclpy" not in sys.modules\n'
        'compiled = [m for m, v in list(sys.modules.items())\n'
        '            if str(getattr(v, "__file__", "")).endswith(".so") and m.split(".")[0] == "gato"]\n'
        'assert not compiled, compiled'],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_ros_raises_source_hint(monkeypatch):
    import builtins
    real = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == 'rclpy':
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fail)
    with pytest.raises(RuntimeError, match='Source ROS 2 in the shell'):
        ros._ros_imports()
