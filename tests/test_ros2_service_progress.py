"""Supervision continues during synchronous service waits; no live ROS is used."""
from types import SimpleNamespace as NS

import pytest

from tiago_mpc import ros2
from tiago_mpc.tiago import RIGHT_ARM_JOINTS


class GuardFailure(RuntimeError):
    pass


@pytest.fixture
def rpc(monkeypatch):
    clock = NS(now=10.)
    monkeypatch.setattr(ros2.time, 'monotonic', lambda: clock.now)
    record = NS(discovery=[], spins=[], events=[], destroyed=[], sent=[],
                available_after=0, complete_after=2, result=NS(ok=True),
                error=None, cancelled=0)
    future = NS(complete=False)
    future.done = lambda: future.complete
    def result():
        record.events.append('result')
        if record.error is not None:
            raise record.error
        return record.result
    future.result = result
    def cancel():
        record.cancelled += 1
    future.cancel = cancel
    def wait_for_service(timeout_sec):
        record.events.append('discovery')
        record.discovery.append(timeout_sec)
        available = len(record.discovery) > record.available_after
        if not available:
            clock.now += timeout_sec
        return available
    def call_async(request):
        record.events.append('dispatch')
        record.sent.append(request)
        future.complete = record.complete_after == 0
        return future
    client = NS(wait_for_service=wait_for_service, call_async=call_async)
    node = NS(create_client=lambda kind, name: client,
              destroy_client=record.destroyed.append)
    def spin_until_future_complete(node, pending, timeout_sec):
        record.events.append('spin')
        record.spins.append(timeout_sec)
        clock.now += timeout_sec
        pending.complete = (record.complete_after is not None
                            and len(record.spins) >= record.complete_after)
    arm = ros2.TiagoRightArmClient.__new__(ros2.TiagoRightArmClient)
    arm.node, arm.service_progress = node, None
    arm.rclpy = NS(spin_until_future_complete=spin_until_future_complete)
    return NS(arm=arm, clock=clock, record=record, client=client, future=future)


def call(rpc, timeout=.5):
    return rpc.arm._service_call('/controller_manager/switch_controller', object, 'request', timeout)


def test_success_without_hook(rpc):
    assert call(rpc) is rpc.record.result
    assert rpc.record.sent == ['request']
    assert rpc.record.destroyed == [rpc.client]
    assert all(0 < value <= .01 for value in rpc.record.spins)
    assert all(0 <= value <= .01 for value in rpc.record.discovery)


def test_hook_covers_discovery_dispatch_and_future_completion(rpc):
    rpc.record.available_after = 2
    def progress():
        rpc.record.events.append('guard')
    rpc.arm.service_progress = progress
    assert call(rpc) is rpc.record.result
    # A guard surrounds every blocking discovery/spin chunk. A completed RPC
    # still gets its final expiry check before returning a successful response.
    assert rpc.record.events == [
        'guard', 'discovery', 'guard', 'discovery', 'guard', 'discovery', 'guard',
        'dispatch', 'guard', 'spin', 'guard', 'spin', 'guard', 'result',
    ]
    assert rpc.record.destroyed == [rpc.client]


def test_guard_can_stop_during_service_discovery_before_dispatch(rpc):
    rpc.record.available_after = 100
    started = rpc.clock.now
    def progress():
        if rpc.clock.now - started >= .015:
            raise GuardFailure('state stale during discovery')
    rpc.arm.service_progress = progress
    with pytest.raises(GuardFailure, match='stale during discovery'):
        call(rpc)
    assert len(rpc.record.discovery) == 2
    assert rpc.record.sent == []
    assert rpc.record.destroyed == [rpc.client]
    assert rpc.record.cancelled == 0


def test_guard_can_stop_pending_rpc_without_claiming_cancellation(rpc):
    rpc.record.complete_after = None
    started = rpc.clock.now
    def progress():
        if rpc.clock.now - started >= .015:
            raise GuardFailure('command expired during switch')
    rpc.arm.service_progress = progress
    with pytest.raises(GuardFailure, match='expired during switch'):
        call(rpc)
    assert rpc.record.sent == ['request']
    assert not rpc.future.done()
    assert rpc.record.destroyed == [rpc.client]
    assert rpc.record.cancelled == 0
    assert 'result' not in rpc.record.events


def test_completion_does_not_skip_expiry_guard(rpc):
    rpc.record.complete_after = 1
    def progress():
        if rpc.future.done():
            raise GuardFailure('activation may have happened; restore separately')
    rpc.arm.service_progress = progress
    with pytest.raises(GuardFailure, match='restore separately'):
        call(rpc)
    assert rpc.future.done()
    assert rpc.record.destroyed == [rpc.client]
    assert rpc.record.cancelled == 0
    assert 'result' not in rpc.record.events


@pytest.mark.parametrize('stage', ['discovery', 'response'])
def test_timeout_budgets_remain_bounded_and_destroy_client(rpc, stage):
    if stage == 'discovery':
        rpc.record.available_after = 100
    else:
        rpc.record.complete_after = None
    started = rpc.clock.now
    with pytest.raises(TimeoutError, match='not available' if stage == 'discovery' else 'timed out'):
        call(rpc, timeout=.025)
    assert rpc.clock.now - started == pytest.approx(.025)
    waits = rpc.record.discovery if stage == 'discovery' else rpc.record.spins
    assert max(waits) <= .01
    assert waits[-1] == pytest.approx(.005)
    assert rpc.record.destroyed == [rpc.client]
    assert rpc.record.cancelled == 0


@pytest.mark.parametrize('failure', ['exception', 'no_result'])
def test_future_failure_is_preserved_and_client_destroyed(rpc, failure):
    if failure == 'exception':
        rpc.record.error = LookupError('server future failed')
        expected, match = LookupError, 'server future failed'
    else:
        rpc.record.result = None
        expected, match = RuntimeError, 'returned no result'
    with pytest.raises(expected, match=match):
        call(rpc)
    assert rpc.record.destroyed == [rpc.client]


def test_hook_failure_at_entry_still_destroys_created_client(rpc):
    def progress():
        raise GuardFailure('entry guard')
    rpc.arm.service_progress = progress
    with pytest.raises(GuardFailure, match='entry guard'):
        call(rpc)
    assert rpc.record.discovery == []
    assert rpc.record.destroyed == [rpc.client]


def test_topic_discovery_uses_hook_and_small_poll_slices(rpc):
    rpc.arm._topic_has_subscription = lambda topic: len(rpc.record.spins) >= 2
    def spin_once(timeout_sec):
        rpc.record.spins.append(timeout_sec)
        rpc.clock.now += timeout_sec
    rpc.arm.spin_once = spin_once
    rpc.arm.service_progress = lambda: rpc.record.events.append('guard')
    rpc.arm._wait_for_topic_subscription('/effort/commands', .5)
    assert rpc.record.spins == [.01, .01]
    assert rpc.record.events == ['guard', 'guard', 'guard']


def test_topic_wait_propagates_guard_failure(rpc):
    rpc.arm._topic_has_subscription = lambda topic: False
    def spin_once(timeout_sec):
        rpc.record.spins.append(timeout_sec)
        rpc.clock.now += timeout_sec
    rpc.arm.spin_once = spin_once
    def progress():
        if rpc.record.spins:
            raise GuardFailure('topic discovery expired')
    rpc.arm.service_progress = progress
    with pytest.raises(GuardFailure, match='topic discovery expired'):
        rpc.arm._wait_for_topic_subscription('/effort/commands', .5)
    assert rpc.record.spins == [.01]


def test_public_active_owner_inspection_uses_existing_validation_without_switch(rpc):
    position = NS(state='active', claimed_interfaces=[
        f'{joint}/position' for joint in RIGHT_ARM_JOINTS])
    rpc.arm.joint_names = RIGHT_ARM_JOINTS
    rpc.arm.effort_controller = ros2.EFFORT_CONTROLLER
    calls = []
    def inventory(timeout):
        calls.append(timeout)
        return {'arm_right_controller': position}
    rpc.arm._list_controllers = inventory
    assert rpc.arm.active_arm_controller(timeout_sec=.25) == 'arm_right_controller'
    assert calls == [.25]
    position.claimed_interfaces = []
    with pytest.raises(RuntimeError, match='claimed interfaces'):
        rpc.arm.active_arm_controller()
    assert rpc.record.sent == []
