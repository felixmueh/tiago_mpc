"""RosPlant and the robot mains on a fake ROS client, safety worker and clock (CPU only)."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments import common, figure8 as f8, reach as rc
from test_loops import Mpc, Task
import tiago_mpc as tm
from tiago_mpc import clock as clock_module, robot
from tiago_mpc.loops import run_robot
from tiago_mpc.ros2 import ArmState
from tiago_mpc.safety import CollisionSafetySettings
from tiago_mpc.tiago import Q_START, RIGHT_ARM_JOINTS

BODY = ("torso_lift_joint", "head_1_joint", "wheel_front_left_joint")


class Clock:
    def __init__(self):
        self.now = 10.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


class Arm:
    """TiagoRightArmClient's interface: a fresh state at every spin; change alters each one."""

    def __init__(self, clock):
        self.clock, self.seq, self.owner, self.change = clock, 0, "arm_right_controller", None
        self.calls, self.published, self.service_progress = [], [], None
        self.slow_switch = self.reject_switch = self.reject_handback = self.closed = False

    def spin_once(self, timeout_sec=0.):
        self.clock.now += 1e-5
        self.seq += 1
        q = np.array(Q_START, dtype=np.float32)
        positions = {**dict(zip(RIGHT_ARM_JOINTS, map(float, q))), "torso_lift_joint": .2,
                     "head_1_joint": 0., "wheel_front_left_joint": float("nan")}
        s = ArmState(q=q, qd=np.zeros(7, dtype=np.float32), stamp_sec=1000 + self.clock.now,
                     received_monotonic_sec=self.clock.now, seq=self.seq, joint_positions=positions,
                     joint_velocities={name: 0. for name in positions})
        self.latest = self.change(s) if self.change else s

    def read_state(self, **kwargs):
        self.spin_once()
        return self.latest

    def latest_state(self):
        return self.latest

    def active_arm_controller(self, **kwargs):
        self.calls.append("list")
        return self.owner

    def configure_runtime_effort_controller(self, **kwargs):
        self.calls.append("configure")

    def publish_effort(self, torque):
        self.published.append(np.array(torque, dtype=float))

    def switch_to_effort_control(self, **kwargs):
        self.calls.append("switch")
        for _ in range(3):  # service chunks
            self.clock.now += .07 if self.slow_switch else .005
            self.service_progress()
        if self.reject_switch:
            raise RuntimeError("switch rejected")
        self.owner = "effort"

    def switch_to_default_control(self, **kwargs):
        self.calls.append("handback")
        if self.reject_handback:
            raise RuntimeError("handback rejected")
        self.owner = "arm_right_controller"

    def close(self):
        self.closed = True


class Monitor:
    """AsyncSafetyMonitor's interface; its own checks are tested in test_safety_monitor."""

    def __init__(self, settings, initial_state):
        self.checked, self.waited, self.fault, self.closed = [], [], False, False

    def check(self, s):
        if self.fault:
            raise RuntimeError("collision margin fault")
        self.checked.append(s.seq)

    def wait_until_checked(self, s):
        self.check(s)
        self.waited.append(s.seq)

    def close(self):
        self.closed = True


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    arm = Arm(clock)
    monkeypatch.setattr(robot, "time", clock)
    monkeypatch.setattr(clock_module, "time", clock)  # the plant's Clock.wall()
    monkeypatch.setattr(robot, "TiagoRightArmClient", lambda **kwargs: arm)
    monkeypatch.setattr(robot, "AsyncSafetyMonitor", Monitor)
    return NS(clock=clock, arm=arm)


def test_check_only_never_loads_a_controller_switches_or_publishes(rig):
    plant = robot.RosPlant()
    plant.start()
    plant.state()
    with pytest.raises(RuntimeError, match="check-only"):
        plant.apply(np.ones(7))
    plant.stop()
    plant.close()
    assert rig.arm.calls == ["list"] and not rig.arm.published
    assert plant.monitor.waited and plant.monitor.closed and rig.arm.closed
    assert plant.settings["restored"] is None


def test_the_first_apply_activates_and_stop_hands_back(rig):
    plant = robot.RosPlant(execute=True)
    plant.start()
    assert rig.arm.calls == ["list", "configure"]
    plant.state()
    plant.apply(np.ones(7))
    # Primed, republished at each service chunk, then published once switched.
    assert rig.arm.calls[-1] == "switch" and len(rig.arm.published) == 5
    plant.state()
    plant.apply(2 * np.ones(7))
    assert rig.arm.calls.count("switch") == 1 and len(rig.arm.published) == 6
    plant.stop()
    assert rig.arm.calls[-1] == "handback" and rig.arm.owner == "arm_right_controller"
    assert all(not u.any() for u in rig.arm.published[-5:])
    assert plant.settings["activated"] and plant.settings["restored"] is True


def test_stop_before_activation_publishes_nothing_and_keeps_the_controller(rig):
    plant = robot.RosPlant(execute=True)
    plant.start()
    plant.stop()
    assert "handback" not in rig.arm.calls and not rig.arm.published


@pytest.mark.parametrize("change,text", [
    (lambda s: replace(s, received_monotonic_sec=s.received_monotonic_sec - .2), "stale"),
    (lambda s: replace(s, stamp_sec=float("nan")), "stamp"),
    (lambda s: replace(s, q=np.r_[s.q[:6], np.nan]), "feedback"),
    (lambda s: replace(s, q=np.r_[s.q[:6], 10.]), "position"),
    (lambda s: replace(s, qd=np.full(7, 10.)), "velocity"),
    (lambda s: replace(s, seq=1), "reversed"),
    (lambda s: replace(s, stamp_sec=1.), "clock reset"),
    (lambda s: replace(s, joint_positions=None), "full named body"),
    (lambda s: replace(s, joint_positions={**s.joint_positions, "torso_lift_joint": .21}), "torso_lift_joint"),
    (lambda s: replace(s, joint_positions={**s.joint_positions, "head_1_joint": .03}), "head_1_joint"),
])
def test_each_state_check_faults(rig, change, text):
    plant = robot.RosPlant()
    plant.start()
    plant.state()
    rig.arm.change = change
    with pytest.raises(RuntimeError, match=text):
        plant.state()


def test_unobserved_wheels_and_small_drift_pass(rig):
    plant = robot.RosPlant()
    plant.start()
    rig.arm.change = lambda s: replace(s, joint_positions={**s.joint_positions, "torso_lift_joint": .204})
    assert plant.state().t == rig.clock.now


@pytest.mark.parametrize("change,text", [
    (lambda s: replace(s, q=s.q + .1), "comfortable_high_clearance"),
    (lambda s: replace(s, qd=np.full(7, .2, dtype=np.float32)), "moving"),
])
def test_start_needs_the_comfortable_pose_at_rest(rig, change, text):
    rig.arm.change = change
    with pytest.raises(RuntimeError, match=text):
        robot.RosPlant(execute=True).start()
    assert "configure" in rig.arm.calls and "switch" not in rig.arm.calls


def test_existing_effort_control_is_not_adopted(rig):
    rig.arm.owner = "effort"
    with pytest.raises(RuntimeError, match="existing effort"):
        robot.RosPlant(execute=True).start()
    assert rig.arm.calls == ["list"]


def test_safety_cannot_be_disabled_or_monitor_another_arm(rig):
    for safety in (replace(CollisionSafetySettings(), enabled=False),
                   replace(CollisionSafetySettings(), controlled_joint_names=("arm_left_1_joint",))):
        with pytest.raises(ValueError, match="safety worker"):
            robot.RosPlant(safety=safety)


def test_a_safety_fault_ends_state(rig):
    plant = robot.RosPlant()
    plant.start()
    plant.monitor.fault = True
    with pytest.raises(RuntimeError, match="collision"):
        plant.state()


@pytest.mark.parametrize("fault,text", [("slow_switch", "lease"), ("reject_switch", "rejected")])
def test_a_failed_activation_hands_back_but_never_claims_restoration(rig, fault, text):
    setattr(rig.arm, fault, True)
    plant = robot.RosPlant(execute=True)
    plant.start()
    plant.state()
    with pytest.raises(RuntimeError, match=text):
        plant.apply(np.ones(7))
    with pytest.raises(RuntimeError, match="uncertain"):
        plant.stop()
    assert rig.arm.calls[-1] == "handback" and plant.settings["restored"] is False
    assert not rig.arm.published[-1].any()


def test_a_failed_handback_is_reported(rig):
    rig.arm.reject_handback = True
    plant = robot.RosPlant(execute=True)
    plant.start()
    plant.state()
    plant.apply(np.ones(7))
    with pytest.raises(RuntimeError, match="handback rejected"):
        plant.stop()
    assert plant.settings["restored"] is False


def test_advance_waits_until_its_clock_reaches_the_deadline(rig):
    plant = robot.RosPlant()
    plant.start()
    deadline = plant.now() + .01
    plant.advance(deadline)
    assert plant.now() == pytest.approx(deadline, abs=1e-12)
    rig.clock.now += .02  # late: at once
    plant.advance(deadline + .01)
    assert plant.now() == pytest.approx(deadline + .02, abs=1e-12)


def test_run_robot_activates_once_and_hands_back(rig):
    plant = robot.RosPlant(execute=True)
    assert not run_robot(plant, Mpc(N=32), Task(), command_period=.01, update_period=.01, duration=.1)
    assert rig.arm.calls.count("switch") == 1 and rig.arm.calls[-1] == "handback"
    assert plant.settings["restored"] is True and not rig.arm.published[-1].any()
    assert sum(u.any() for u in rig.arm.published) >= 10


def test_run_robot_never_activates_for_a_goal_already_met(rig):
    plant = robot.RosPlant(execute=True)
    assert run_robot(plant, Mpc(), Task(done_at=0.), command_period=.01, update_period=.01, duration=1.)
    assert "switch" not in rig.arm.calls and not rig.arm.published


def test_a_dead_mpc_process_on_the_robot_stops_and_hands_back(rig):
    mpc, plant = Mpc(N=32), robot.RosPlant(execute=True)

    def die(tick):
        mpc.dead = tick.command is not None

    with pytest.raises(RuntimeError, match="exited"):
        run_robot(plant, mpc, Task(), callbacks=[die], command_period=.01, update_period=.01, duration=1.)
    assert rig.arm.calls[-1] == "handback" and not rig.arm.published[-1].any()


@pytest.mark.parametrize("module", [rc, f8])
def test_robot_main_checks_only_by_default(rig, monkeypatch, module):
    monkeypatch.setattr(tm, "MpcProcess", NS(compile=lambda params: pytest.fail("check-only builds no MPC")))
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("check-only must not prompt"))
    monkeypatch.setattr(common.signal, "signal", lambda *args: None)
    assert module.main(["--plant", "robot"]) == 0
    assert rig.arm.calls == ["list"] and not rig.arm.published and rig.arm.closed


def test_cancelled_execution_creates_no_ros_client(rig, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    monkeypatch.setattr(tm, "RosPlant", lambda **kwargs: pytest.fail("must cancel before the plant"))
    with pytest.raises(SystemExit) as exit:
        rc.main(["--plant", "robot", "--execute", "--target", ".5", ".1", ".2"])
    assert exit.value.code == 2


@pytest.mark.parametrize("args", [["--plant", "robot", "--execute"], ["--execute", "--target", "1", "1", "1"],
                                  ["--plant", "robot", "--target", "nan", "0", "0"]])
def test_invalid_robot_launch_fails_before_ros(rig, monkeypatch, args):
    monkeypatch.setattr(tm, "RosPlant", lambda **kwargs: pytest.fail("must fail before the plant"))
    with pytest.raises(SystemExit) as exit:
        rc.main(args)
    assert exit.value.code == 2


@pytest.mark.parametrize("module", [rc, f8])
def test_confirmed_execution_runs_the_experiment_on_the_robot_and_closes_both(rig, monkeypatch, module):
    runs, closed = [], []

    def experiment(plant, mpc, run, task, **kw):
        runs.append((plant, mpc, run, task))
        return NS(summary=dict(reached=True, limits_pass=True), out=None)

    name = "reach" if module is rc else "figure8"
    monkeypatch.setattr("builtins.input", lambda prompt: "EXECUTE")
    monkeypatch.setattr(common.signal, "signal", lambda *args: None)
    monkeypatch.setattr(tm, "MpcProcess",
                        NS(compile=lambda params: NS(params=params, close=lambda: closed.append("mpc"))))
    monkeypatch.setattr(module, name, experiment)
    assert module.main(["--plant", "robot", "--execute", "--target", ".5", ".1", ".2"]) == 0
    (plant, mpc, run, task), = runs
    assert type(plant) is robot.RosPlant and plant.execute and run is run_robot
    assert mpc.params is (rc.tm.TIAGO if module is rc else f8.FIGURE8)
    assert (task.target_m if module is rc else task.start_m) == (.5, .1, .2)
    assert closed == ["mpc"] and rig.arm.closed
