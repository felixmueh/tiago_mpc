"""Process-boundary and fault regressions; never connect to ROS or a robot."""
from dataclasses import dataclass, replace
import json
import queue
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tiago_mpc import runner as r
from tiago_mpc.mpc_worker import WorkerConfig
from tiago_mpc.protocol import SolveReply, SolveRequest, TaskReference, WorkerReady
from tiago_mpc.ros2 import ArmState, RIGHT_ARM_JOINTS


LIMITS = dict(q_lower=[-3.] * 7, q_upper=[3.] * 7,
              velocity_limits=[2.5] * 7, effort_limits=[26.] * 7, dt_s=.01, N=32)


def knot_plan():
    """Distinct torques per knot, so tests can tell which knot was published."""
    return 1. + .1 * np.arange(LIMITS["N"] - 1)[:, None] * np.ones(7)


@dataclass
class SafetySettings:
    enabled: bool = True
    controlled_joint_names: tuple = RIGHT_ARM_JOINTS


class Clock:
    now = 10.
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def state(clock, seq=1, **kwargs):
    q = np.asarray(r.COMFORTABLE_Q)
    return ArmState(q=q, qd=np.zeros(7), stamp_sec=1000 + clock(),
                    received_monotonic_sec=clock(), seq=seq,
                    joint_positions={**dict(zip(RIGHT_ARM_JOINTS, q)), "torso_lift_joint": .2},
                    joint_velocities={name: 0. for name in (*RIGHT_ARM_JOINTS, "torso_lift_joint")},
                    **kwargs)


class Channel(queue.Queue):
    on_put = None
    closed = False
    def put_nowait(self, value):
        if self.on_put:
            self.on_put(value)
        else:
            super().put_nowait(value)
    def cancel_join_thread(self):
        pass
    def close(self):
        self.closed = True


@pytest.fixture
def rig(monkeypatch):
    return make_rig(monkeypatch)


def make_rig(monkeypatch, config=r.RunnerConfig()):
    clock = Clock()
    faults = NS(start=False, hang=False, mismatched=False, switch=False,
                switch_death=False, monitor=False, restore=False, join=False,
                never_arrive=False, already_arrived=False, moving_source=False)
    publications, requests = [], []
    ctx = NS(queues=[], process=None)

    class Arm:
        seq = 0
        owner = "arm_right_controller"
        closed = restored = False
        service_progress = None
        def spin_once(self, timeout_sec=0):
            clock.sleep(.00001)  # each emitted source observation has a new stamp
            self.seq += 1
            self.latest = state(clock, self.seq)
            if faults.moving_source and self.seq == 5:
                self.latest = replace(self.latest, qd=np.full(7, .2))
        def read_state(self, **kwargs):
            self.spin_once()
            return self.latest
        def latest_state(self):
            return self.latest
        def active_arm_controller(self, **kwargs):
            return self.owner
        def configure_runtime_effort_controller(self, **kwargs):
            assert self.owner == "arm_right_controller"
        def publish_effort(self, torque):
            publications.append((clock(), np.asarray(torque).copy()))
        def switch_to_effort_control(self, **kwargs):
            if faults.switch:
                clock.sleep(.2)
            if faults.switch_death:
                ctx.process.exitcode = 9
            if self.service_progress:
                self.service_progress()
            self.owner = "effort"
        def switch_to_default_control(self, **kwargs):
            if faults.restore:
                raise RuntimeError("handback rejected")
            self.restored = True
            self.owner = "arm_right_controller"
        def close(self):
            self.closed = True
    arm = Arm()

    class Monitor:
        closed = False
        last_checked_seq = 0
        def __init__(self, **kwargs):
            pass
        def check(self, snapshot):
            if faults.monitor and publications:
                raise RuntimeError("collision margin fault")
            self.last_checked_seq = snapshot.seq
        def wait_until_checked(self, snapshot):
            self.check(snapshot)
        def close(self):
            self.closed = True

    monitor = Monitor()
    def channel(maxsize):
        value = Channel(maxsize)
        ctx.queues.append(value)
        return value
    ctx.Queue = channel
    ctx.Event = lambda: NS(set=lambda: None)
    def process(target, kwargs):
        class Process:
            exitcode = None
            started = False
            def start(self):
                if faults.start:
                    raise RuntimeError("spawn failed")
                self.started = True
                kwargs["response_q"].put_nowait(WorkerReady(True, details=LIMITS))
                def solve(request):
                    requests.append(request)
                    if faults.hang:
                        return
                    kwargs["response_q"].put_nowait(SolveReply(
                        request.request_id, request.task.generation + int(faults.mismatched),
                        request.state.seq, request.state.received_monotonic_sec,
                        clock(), request.deadline_monotonic_sec, knot_plan(), LIMITS["dt_s"],
                        dict(position_error_m=(0. if faults.already_arrived or
                              (len(requests) > 1 and not faults.never_arrive) else .1))))
                kwargs["request_q"].on_put = solve
            def join(self, timeout):
                assert self.started, "unstarted process must not be joined"
                if faults.join:
                    raise RuntimeError("join failed")
                self.exitcode = 0
            def is_alive(self):
                return self.exitcode is None
            def terminate(self):
                self.exitcode = -15
        ctx.process = Process()
        return ctx.process
    ctx.Process = process
    # Geometry and real process checks have their own tests. This fixture
    # deliberately replaces only the boundary conversion to avoid Pinocchio.
    monkeypatch.setattr(r, "safety_state", lambda snapshot: snapshot)
    runner = r.RosMpcRunner(config, TaskReference((.6, .1, .2)), WorkerConfig(),
        SafetySettings(), arm_factory=lambda: arm, monitor_factory=lambda **kw: monitor,
        context=ctx, clock=clock, sleep=clock.sleep)
    return NS(runner=runner, clock=clock, arm=arm, monitor=monitor, ctx=ctx,
              faults=faults, publications=publications, requests=requests)


def test_check_only_has_no_worker_commands_or_controller_mutation(rig, tmp_path):
    result = rig.runner.run(tmp_path / "check")
    assert result["status"] == "checked"
    assert rig.ctx.process is None and not rig.publications
    assert rig.arm.owner == "arm_right_controller"
    assert rig.arm.closed and rig.monitor.closed


def test_normal_run_hands_back_and_records_original_command_lease(rig, tmp_path):
    result = rig.runner.run(tmp_path / "run", execute=True)
    assert result["status"] == "arrived" and result["restored"] is True
    assert result["accepted_commands"] == 1 and len(rig.requests) == 2
    assert rig.arm.restored and rig.arm.closed and rig.monitor.closed
    assert all(c.closed for c in rig.ctx.queues)
    assert np.all(rig.publications[-1][1] == 0)
    saved = json.loads((tmp_path / "run/summary.json").read_text())
    assert saved["status"] == "arrived"
    assert "ros_distro" in saved



def test_request_deadline_ends_no_later_than_the_plan(rig, tmp_path, monkeypatch):
    monkeypatch.setitem(LIMITS, "dt_s", .002)  # 2 ms knots: 31 x 2 ms = 62 ms plan
    rig.runner.run(tmp_path / "short-plan", execute=True)
    request = rig.requests[0]
    assert request.deadline_monotonic_sec - request.state.received_monotonic_sec == pytest.approx(.062)

@pytest.mark.parametrize("fault,text,uncertain", [
    ("hang", "deadline", False), ("mismatched", "mismatched", False),
    ("switch", "expired", True), ("switch_death", "exited", True),
    ("monitor", "collision margin", True)])
def test_fault_stops_and_partial_activation_never_claims_restoration(rig, tmp_path, fault, text, uncertain):
    setattr(rig.faults, fault, True)
    result = rig.runner.run(tmp_path / fault, execute=True)
    assert result["status"] == "fault" and text in result["error"]
    assert rig.arm.closed and rig.monitor.closed
    if uncertain:
        assert rig.arm.restored and result["restored"] is False
        assert "uncertain" in " ".join(result["cleanup_errors"])
        assert np.all(rig.publications[-1][1] == 0)
    else:
        assert result["restored"] is None
        assert not rig.publications


def test_existing_effort_mode_is_rejected_before_any_solve_or_command(rig, tmp_path):
    rig.arm.owner = "effort"
    result = rig.runner.run(tmp_path / "existing-effort", execute=True)
    assert result["status"] == "fault" and "existing effort" in result["error"]
    assert rig.ctx.process is None and not rig.publications


def test_goal_already_met_never_activates_or_publishes_effort(rig, tmp_path):
    rig.faults.already_arrived = True
    result = rig.runner.run(tmp_path / "already-arrived", execute=True)
    assert result["status"] == "arrived" and result["accepted_commands"] == 0
    assert not rig.publications and result["restored"] is None


def test_arrival_uses_source_velocity_even_if_latest_velocity_is_zero(rig, tmp_path):
    rig.faults.already_arrived = rig.faults.moving_source = True
    result = rig.runner.run(tmp_path / "moving-source", execute=True)
    assert result["status"] == "arrived"
    assert len(rig.requests) == 2 and result["accepted_commands"] == 1


def test_slow_logging_cannot_cause_a_late_nonzero_republication(rig, tmp_path, monkeypatch):
    rig.faults.never_arrive = True
    original = r._log_values
    def delayed(value):
        if isinstance(value, dict) and value.get("event") == "request" and value.get("request_id") == 2:
            rig.clock.sleep(.2)
        return original(value)
    monkeypatch.setattr(r, "_log_values", delayed)
    result = rig.runner.run(tmp_path / "slow-log", execute=True)
    assert result["status"] == "fault" and rig.arm.restored
    assert all(t < rig.requests[0].deadline_monotonic_sec
               for t, u in rig.publications if np.any(u))


@pytest.mark.parametrize("fault", ["start", "join"])
def test_process_cleanup_failure_does_not_skip_other_resources_or_summary(rig, tmp_path, fault):
    setattr(rig.faults, fault, True)
    result = rig.runner.run(tmp_path / fault, execute=True)
    assert result["status"] == "fault"
    assert rig.arm.closed and rig.monitor.closed and all(c.closed for c in rig.ctx.queues)
    assert (tmp_path / fault / "summary.json").exists()


def test_handback_failure_is_reported(rig, tmp_path):
    rig.faults.restore = True
    result = rig.runner.run(tmp_path / "restore-failure", execute=True)
    assert result["status"] == "fault" and result["restored"] is False
    assert "handback rejected" in " ".join(result["cleanup_errors"])


def test_safety_cannot_be_disabled_or_monitor_a_different_arm():
    for safety in (SafetySettings(False), SafetySettings(True, ("arm_left_1_joint",))):
        with pytest.raises(ValueError):
            r.RosMpcRunner(r.RunnerConfig(), TaskReference((0., 0., 0.)), WorkerConfig(), safety)


def test_guards_check_source_velocity_expiry_torque_and_body_motion(rig):
    snapshot = state(rig.clock)
    guard = r.RuntimeGuard(r.RunnerConfig(), snapshot, LIMITS)
    guard.state(snapshot, rig.clock()); guard.start(snapshot)
    request = SolveRequest(1, snapshot, rig.runner.task, 0., rig.clock() + .1)
    reply = SolveReply(1, 1, snapshot.seq, snapshot.received_monotonic_sec,
                       rig.clock(), request.deadline_monotonic_sec, knot_plan(), .01)
    plan = guard.reply(reply, request, rig.clock())
    np.testing.assert_array_equal(plan.torques, knot_plan())
    assert plan.start_s == snapshot.received_monotonic_sec and plan.dt_s == .01
    late_knot = knot_plan()
    late_knot[-1, 3] = 27.  # only the last knot exceeds the raw effort limit
    for bad in (replace(reply, generation=2), replace(reply, expires_monotonic_sec=rig.clock()),
                replace(reply, expires_monotonic_sec=rig.clock()+1),
                replace(reply, torques_native=np.ones(7)), replace(reply, torques_native=late_knot),
                replace(reply, torques_native=np.where(late_knot > 26, np.nan, late_knot)),
                replace(reply, dt_s=.002)):
        with pytest.raises(RuntimeError):
            guard.reply(bad, request, rig.clock())
    for bad in (replace(snapshot, qd=np.ones(7)*3),
                replace(snapshot, received_monotonic_sec=rig.clock()-1),
                replace(snapshot, joint_positions={**snapshot.joint_positions, "torso_lift_joint": .3})):
        with pytest.raises(RuntimeError):
            guard.state(bad, rig.clock())


def test_uncontrolled_nonfinite_metadata_can_be_logged_without_relaxing_control_checks():
    assert r._log_values({"wheel": float("nan")}) == {"wheel": None}
    json.dumps(r._log_values({"wheel": np.float32("nan")}), allow_nan=False)


def test_reference_phase_is_time_based_and_orientation_unconstrained():
    ref = TaskReference((.5, .1, .2), kind="figure8")
    # Default shape matches configs/tracking.json: 2 cm edge clearance, 28 cm x span.
    assert ref.horizon(0., 2, .01)[0, 0] == pytest.approx(.52)
    assert ref.horizon(6., 2, .01)[0, 0] == pytest.approx(.66)
    assert np.all(ref.horizon(6., 2, .01)[:, 3:] == 0)


@pytest.mark.parametrize("io_s,dt_s", [(.002, .01), (.01, .002)])
def test_io_ticks_publish_the_due_knot_of_plans_whose_interval_differs(monkeypatch, tmp_path, io_s, dt_s):
    monkeypatch.setitem(LIMITS, "dt_s", dt_s)
    rig = make_rig(monkeypatch, r.RunnerConfig(io_period_sec=io_s, duration_sec=.2))
    rig.faults.never_arrive = True
    result = rig.runner.run(tmp_path / "playback", execute=True)
    assert result["status"] == "timeout" and result["restored"] is True
    events = [json.loads(line) for line in (tmp_path / "playback/events.jsonl").read_text().splitlines()]
    published = [e for e in events if e["event"] == "effort_publish" and "knot" in e]
    assert len(published) == sum(np.any(u) for _, u in rig.publications)
    plan = knot_plan()
    for event in published:
        offset = event["evaluated_monotonic_sec"] - event["plan_start_monotonic_sec"]
        assert event["knot"] == int(np.floor(offset / dt_s + 1e-9))
        np.testing.assert_array_equal(event["torque_nm"], plan[event["knot"]])
        assert event["evaluated_monotonic_sec"] < event["expires_monotonic_sec"]
    # Plans are played: later knots are published, not only u_0.
    assert {e["knot"] for e in published} - {0}
    # One solve per 10 ms in both cases: per knot with fast I/O, per tick with slow I/O.
    receipts = np.array([q.state.received_monotonic_sec for q in rig.requests])
    np.testing.assert_allclose(np.diff(receipts), .01, atol=1e-4)
    # dt_elapsed is the time between consecutive plans' states, as in simulation.
    assert rig.requests[0].dt_elapsed_s is None
    np.testing.assert_allclose([q.dt_elapsed_s for q in rig.requests[1:]], np.diff(receipts))
    # Every nonzero publication ends no later than its plan's lease.
    leases = {e["plan_start_monotonic_sec"]: e["expires_monotonic_sec"] for e in published}
    assert all(lease <= start + (LIMITS["N"] - 1) * dt_s for start, lease in leases.items())


def test_first_mode_keeps_publishing_the_first_knot(monkeypatch, tmp_path):
    monkeypatch.setitem(LIMITS, "dt_s", .002)
    rig = make_rig(monkeypatch, r.RunnerConfig(duration_sec=.1, command_mode="first"))
    rig.faults.never_arrive = True
    assert rig.runner.run(tmp_path / "first", execute=True)["status"] == "timeout"
    nonzero = [u for _, u in rig.publications if np.any(u)]
    assert nonzero and all(np.array_equal(u, knot_plan()[0]) for u in nonzero)


def test_guard_never_plays_a_plan_past_its_end(rig, monkeypatch):
    monkeypatch.setitem(LIMITS, "dt_s", .002)
    snapshot = state(rig.clock)
    guard = r.RuntimeGuard(r.RunnerConfig(), snapshot, LIMITS)
    start = snapshot.received_monotonic_sec
    request = SolveRequest(1, snapshot, rig.runner.task, 0., start + .1)
    reply = SolveReply(1, 1, snapshot.seq, start, rig.clock(), start + .1, knot_plan(), .002)
    with pytest.raises(RuntimeError, match="outlives"):
        guard.reply(reply, request, rig.clock())  # 100 ms lease, 62 ms plan
    plan = guard.reply(replace(reply, expires_monotonic_sec=start + .062), request, rig.clock())
    assert guard.command(plan, start + .0611)[0] == 30
    with pytest.raises(RuntimeError):
        guard.command(plan, start + .062)


def test_runner_rejects_unknown_command_mode():
    with pytest.raises(ValueError):
        r.RunnerConfig(command_mode="smooth")
