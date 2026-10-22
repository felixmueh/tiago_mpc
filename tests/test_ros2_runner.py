"""Process-boundary and fault regressions; never connect to ROS or a robot."""
from dataclasses import dataclass, replace
import json
import queue
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tiago_mpc import runner as r
from tiago_mpc.mpc import TIAGO
from tiago_mpc.plan import Plan
from tiago_mpc.ros2 import ArmState, RIGHT_ARM_JOINTS


LIMITS = dict(q_lower=[-3.] * 7, q_upper=[3.] * 7,
              velocity_limits=[2.5] * 7, effort_limits=[26.] * 7, dt_s=.01, N=32)


def knot_plan():
    """Distinct torques per knot, so tests can tell which knot was published."""
    return 1. + .1 * np.arange(LIMITS["N"] - 1)[:, None] * np.ones(7)


def plan_from(t0, **stats):
    return Plan(t0, LIMITS["dt_s"], np.zeros((LIMITS["N"], 14)), knot_plan(), stats)


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


@pytest.fixture
def rig(monkeypatch):
    return make_rig(monkeypatch)


def make_rig(monkeypatch, config=r.RunnerConfig()):
    clock = Clock()
    faults = NS(start=False, hang=False, switch=False, error=False,
                switch_death=False, monitor=False, restore=False, close=False,
                never_arrive=False, already_arrived=False, moving_source=False)
    publications, requests = [], []
    ctx = NS(mpc=None)

    class Arm:
        seq = 0
        owner = "arm_right_controller"
        closed = restored = False
        service_progress = None
        def spin_once(self, timeout_sec=0):
            clock.sleep(.00001)  # each emitted source observation has a new stamp
            self.seq += 1
            self.latest = state(clock, self.seq)
            if faults.moving_source and self.seq == 4:  # the first requested state
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
                ctx.mpc.dead = True
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

    class Mpc:
        """MpcWorker's interface; the worker's own checks are tested in test_mpc."""
        N, dt, details = LIMITS["N"], LIMITS["dt_s"], LIMITS
        dead = closed = False
        def __init__(self, params):
            if faults.start:
                raise RuntimeError("spawn failed")
            ctx.mpc, self.reply = self, None
        def request(self, state, reference, updates=None):
            assert self.reply is None, "one request at a time"
            requests.append(state)
            self.reply = None if faults.hang else plan_from(state.t)
        def result(self, block=False):
            if self.dead:
                raise RuntimeError("MPC worker exited unexpectedly: 9")
            if faults.error and self.reply is not None:
                raise RuntimeError("MPC worker: FloatingPointError")
            reply, self.reply = self.reply, None
            return reply
        def close(self):
            self.closed = True
            if faults.close:
                raise RuntimeError("close failed")

    @dataclass(frozen=True)
    class Task:
        """A reach that arrives with the second plan (the first if already arrived)."""
        def reference(self, t, n, dt):
            return np.zeros((n, 6), dtype=np.float32)

        def done(self, s, t):
            arrived = faults.already_arrived or (len(requests) > 1 and not faults.never_arrive)
            return arrived and np.abs(s.qd).sum() <= 1.

    # Geometry and real process checks have their own tests. This fixture
    # deliberately replaces only the boundary conversion to avoid Pinocchio.
    monkeypatch.setattr(r, "safety_state", lambda snapshot: snapshot)
    runner = r.RosMpcRunner(config, Task(), TIAGO,
        SafetySettings(), arm_factory=lambda: arm, monitor_factory=lambda **kw: monitor,
        mpc_factory=Mpc, clock=clock, sleep=clock.sleep)
    return NS(runner=runner, clock=clock, arm=arm, monitor=monitor, ctx=ctx,
              faults=faults, publications=publications, requests=requests)


def test_check_only_has_no_worker_commands_or_controller_mutation(rig, tmp_path):
    result = rig.runner.run(tmp_path / "check")
    assert result["status"] == "checked"
    assert rig.ctx.mpc is None and not rig.publications
    assert rig.arm.owner == "arm_right_controller"
    assert rig.arm.closed and rig.monitor.closed


def test_normal_run_hands_back_and_records_original_command_lease(rig, tmp_path):
    result = rig.runner.run(tmp_path / "run", execute=True)
    assert result["status"] == "arrived" and result["restored"] is True
    assert result["accepted_commands"] == 1 and len(rig.requests) == 2
    assert rig.arm.restored and rig.arm.closed and rig.monitor.closed and rig.ctx.mpc.closed
    assert np.all(rig.publications[-1][1] == 0)
    saved = json.loads((tmp_path / "run/summary.json").read_text())
    assert saved["status"] == "arrived"
    assert "ros_distro" in saved


@pytest.mark.parametrize("fault,text,uncertain", [
    ("hang", "deadline", False), ("error", "FloatingPointError", False),
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
    assert rig.ctx.mpc is None and not rig.publications


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
    assert all(t < rig.requests[0].t + .1 for t, u in rig.publications if np.any(u))


@pytest.mark.parametrize("fault", ["start", "close"])
def test_process_cleanup_failure_does_not_skip_other_resources_or_summary(rig, tmp_path, fault):
    setattr(rig.faults, fault, True)
    result = rig.runner.run(tmp_path / fault, execute=True)
    assert result["status"] == "fault"
    assert rig.arm.closed and rig.monitor.closed
    assert (tmp_path / fault / "summary.json").exists()


def test_handback_failure_is_reported(rig, tmp_path):
    rig.faults.restore = True
    result = rig.runner.run(tmp_path / "restore-failure", execute=True)
    assert result["status"] == "fault" and result["restored"] is False
    assert "handback rejected" in " ".join(result["cleanup_errors"])


def test_safety_cannot_be_disabled_or_monitor_a_different_arm():
    for safety in (SafetySettings(False), SafetySettings(True, ("arm_left_1_joint",))):
        with pytest.raises(ValueError):
            r.RosMpcRunner(r.RunnerConfig(), None, TIAGO, safety)


def test_guards_check_source_velocity_expiry_torque_and_body_motion(rig):
    snapshot = state(rig.clock)
    guard = r.RuntimeGuard(r.RunnerConfig(), snapshot, LIMITS)
    guard.state(snapshot, rig.clock()); guard.start(snapshot)
    plan = plan_from(snapshot.received_monotonic_sec)
    np.testing.assert_array_equal(guard.command(plan, rig.clock() + .015), knot_plan()[1])
    late_knot = knot_plan()
    late_knot[1, 3] = 27.  # this knot exceeds the raw effort limit
    for bad, now in ((plan, rig.clock() + .1),
                     (Plan(plan.t0, plan.dt, plan.states, late_knot), rig.clock() + .015)):
        with pytest.raises(RuntimeError):
            guard.command(bad, now)
    for bad in (replace(snapshot, qd=np.ones(7)*3),
                replace(snapshot, received_monotonic_sec=rig.clock()-1),
                replace(snapshot, joint_positions={**snapshot.joint_positions, "torso_lift_joint": .3})):
        with pytest.raises(RuntimeError):
            guard.state(bad, rig.clock())


def test_uncontrolled_nonfinite_metadata_can_be_logged_without_relaxing_control_checks():
    assert r._log_values({"wheel": float("nan")}) == {"wheel": None}
    json.dumps(r._log_values({"wheel": np.float32("nan")}), allow_nan=False)


@pytest.mark.parametrize("command_s,dt_s", [(.002, .01), (.01, .002)])
def test_command_ticks_publish_the_due_knot_of_plans_whose_interval_differs(monkeypatch, tmp_path, command_s, dt_s):
    monkeypatch.setitem(LIMITS, "dt_s", dt_s)
    rig = make_rig(monkeypatch, r.RunnerConfig(command_period_sec=command_s, duration_sec=.2))
    rig.faults.never_arrive = True
    result = rig.runner.run(tmp_path / "ticks", execute=True)
    assert result["status"] == "timeout" and result["restored"] is True
    events = [json.loads(line) for line in (tmp_path / "ticks/events.jsonl").read_text().splitlines()]
    published = [e for e in events if e["event"] == "effort_publish" and "plan_start_monotonic_sec" in e]
    commands = {e["plan_start_monotonic_sec"]: e for e in events if e["event"] == "command"}
    assert len(published) == sum(np.any(u) for _, u in rig.publications)
    plan, knots = knot_plan(), set()
    for event in published:
        command = commands[event["plan_start_monotonic_sec"]]
        offset = event["monotonic_sec"] - command["plan_start_monotonic_sec"]
        knot = int(np.floor(offset / dt_s + 1e-9))
        np.testing.assert_array_equal(event["torque_nm"], plan[knot])
        assert event["monotonic_sec"] < command["expiry"]
        knots.add(knot)
    # Plans are sampled: later knots are published, not only u_0.
    assert knots - {0}
    # One solve per 10 ms in both cases: per knot with fast I/O, per tick with slow I/O.
    receipts = np.array([item.t for item in rig.requests])
    np.testing.assert_allclose(np.diff(receipts), .01, atol=1e-4)
    # Every nonzero publication ends no later than its plan's lease.
    assert all(c["expiry"] <= c["plan_start_monotonic_sec"] + (LIMITS["N"] - 1) * dt_s
               for c in commands.values())


def test_lease_ends_no_later_than_the_plan(rig, monkeypatch):
    monkeypatch.setitem(LIMITS, "dt_s", .002)  # 2 ms knots: 31 x 2 ms = 62 ms plan, 100 ms timeout
    snapshot = state(rig.clock)
    guard = r.RuntimeGuard(r.RunnerConfig(), snapshot, LIMITS)
    plan = plan_from(snapshot.received_monotonic_sec)
    assert guard.lease_end(plan.t0) - plan.t0 == pytest.approx(.062)
    np.testing.assert_array_equal(guard.command(plan, plan.t0 + .0611), knot_plan()[30])
    with pytest.raises(RuntimeError, match="past the plan's last knot|expired"):
        guard.command(plan, plan.t0 + .062)
