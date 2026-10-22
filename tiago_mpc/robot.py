"""RosPlant: the TIAGo right arm (robot or Gazebo) through ROS 2, with the robot's checks.

Check-only is the default: start() runs the startup checks, and stop() has nothing
to hand back. With execute=True the first apply() switches to effort control, and
stop() sends zero torque and hands the arm back to PAL's position controller.
The collision-safety worker (safety.AsyncSafetyMonitor) checks every state; the
plant does not run without it. Importing this module does not load ROS.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import time

import numpy as np

from .loops import LEASE_S
from .plan import State
from .ros2 import TiagoRightArmClient
from .safety import AsyncSafetyMonitor, CollisionSafetySettings, SafetyCheckState
from .tiago import Q_START, RIGHT_ARM_JOINTS, limits


@dataclass(frozen=True)
class Checks:
    """Tolerances of the robot checks."""
    state_age_s: float = .1  # every state is at most this old
    first_state_s: float = 180.  # wait for the first state
    service_s: float = .5  # each controller-manager service call
    start_rad: float = .05  # start this close to the comfortable pose
    start_speed_l1_rad_s: float = 1.
    body_drift_rad: float = .02  # the other body joints stay where they started
    torso_drift_m: float = .005

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def safety_state(s):
    """The safety worker's view of an ArmState; SEA body-only updates have their own seq."""
    if not s.joint_positions or not s.joint_velocities:
        raise RuntimeError("full named body positions and velocities are required")
    return SafetyCheckState(
        q=s.q.copy(), qd=s.qd.copy(), stamp_sec=s.stamp_sec,
        received_monotonic_sec=s.received_monotonic_sec,
        seq=s.seq if s.safety_seq is None else s.safety_seq,
        joint_positions=dict(s.joint_positions), joint_velocities=dict(s.joint_velocities))


class RosPlant:
    """The plant interface on ROS 2; client is the ros2.TiagoRightArmClient.

    settings records the run's ROS side: execute, the checks, the safety settings,
    ROS_DISTRO, and after stop() whether the arm was handed back (restored).
    """

    def __init__(self, execute=False, *, checks=Checks(), safety=CollisionSafetySettings(),
                 sea_absolute=False):
        if not safety.enabled or tuple(safety.controlled_joint_names) != RIGHT_ARM_JOINTS:
            raise ValueError("the robot needs the safety worker on the controlled right-arm joints")
        self.execute, self.checks, self.safety = execute, checks, safety
        self.settings = dict(clock="wall", execute=execute, sea_absolute=sea_absolute,
                             checks=asdict(checks), safety=asdict(safety),
                             ros_distro=os.environ.get("ROS_DISTRO"),
                             activated=False, restored=None, cleanup_errors=[])
        self.monitor = self._initial = self._last = None
        self._seq, self._stamp, self._tick = -1, None, None
        self._activated = self._uncertain = False
        self.client = TiagoRightArmClient(sea_absolute=sea_absolute)

    def start(self):
        """Startup checks on a fresh state: the safety worker's initial check, PAL's
        position controller in charge, the arm at rest at the comfortable pose.
        execute=True also loads and configures the effort controller (no switch)."""
        c, client = self.checks, self.client
        first = client.read_state(timeout_sec=c.first_state_s, max_age_sec=c.state_age_s)
        self.monitor = AsyncSafetyMonitor(settings=self.safety, initial_state=safety_state(first))
        if client.active_arm_controller(timeout_sec=c.service_s) != "arm_right_controller":
            raise RuntimeError("start in PAL's position controller; existing effort control is not adopted")
        if self.execute:
            client.configure_runtime_effort_controller(timeout_sec=c.service_s)
        # The safety model and the configuration take time: check a state read after them.
        self._initial = client.read_state(timeout_sec=c.state_age_s, after_seq=first.seq,
                                          max_age_sec=c.state_age_s)
        self._ready(self._check(self._initial))
        self._last, self._tick = self._initial, time.monotonic()

    def state(self):
        """The checked state, t its monotonic receipt time."""
        self._last = s = self._fresh()
        # Gazebo may need the ROS stamp instead (its real-time factor can differ from 1); open.
        return State(s.received_monotonic_sec, s.q, s.qd, s.joint_positions, s.joint_velocities)

    def apply(self, command):
        if self._activated:
            self.client.publish_effort(command)
        else:
            self._activate(command)

    def advance(self, t):
        """Wait one command period (t minus this tick's state time) after the previous tick,
        or after now when late. Receipt times lag the wall clock, so t itself may be past."""
        period, now = t - self._last.received_monotonic_sec, time.monotonic()
        self._tick += period
        if self._tick <= now:
            self._tick = now + period
        time.sleep(self._tick - now)

    def stop(self):
        """Zero torque and handback to PAL's position controller; nothing before activation.

        Raises if a step failed or the handback is uncertain (settings: cleanup_errors)."""
        if not self._activated:
            return
        self._activated, errors = False, self.settings["cleanup_errors"]
        self.client.service_progress = None
        for _ in range(5):  # attempt the handback even if zero publication fails
            try:
                self.client.publish_effort(np.zeros(7))
                self.client.spin_once(timeout_sec=0.)
            except BaseException as exc:
                errors.append(f"zero effort: {exc}")
            time.sleep(.01)
        try:
            self.client.switch_to_default_control(timeout_sec=self.checks.service_s)
            self.settings["restored"] = not self._uncertain
            if self._uncertain:
                errors.append("restoration uncertain: the interrupted activation RPC may still "
                              "execute; manual controller reconciliation required")
        except BaseException as exc:
            self.settings["restored"] = False
            errors.append(f"controller handback: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self):
        try:
            if self.monitor is not None:
                self.monitor.close()
        finally:
            self.client.close()

    def _fresh(self):
        """Read and check the latest state; the safety worker checks it asynchronously."""
        self.client.spin_once(timeout_sec=0.)
        s = self._check(self.client.latest_state())
        self.monitor.check(safety_state(s))
        return s

    def _check(self, s):
        """Fresh, finite, within the limits, in order, and the rest of the body still."""
        if s is None:
            raise RuntimeError("joint state unavailable")
        age = time.monotonic() - s.received_monotonic_sec
        if not math.isfinite(age) or age < 0 or age > self.checks.state_age_s:
            raise RuntimeError("joint state stale or invalid receipt time")
        if not math.isfinite(s.stamp_sec):
            raise RuntimeError("invalid ROS state stamp")
        q, qd, raw = np.asarray(s.q), np.asarray(s.qd), limits()
        if q.shape != (7,) or qd.shape != (7,) or not np.isfinite(np.r_[q, qd]).all():
            raise RuntimeError("invalid controlled joint feedback")
        if np.any(q < raw.lower) or np.any(q > raw.upper):
            raise RuntimeError("measured joint position exceeds physical limits")
        if np.any(np.abs(qd) > raw.velocity):
            raise RuntimeError("measured joint velocity exceeds physical limits")
        if s.seq < self._seq:
            raise RuntimeError("joint-state sequence reversed")
        if s.seq != self._seq:
            if self._stamp is not None and s.stamp_sec <= self._stamp:
                raise RuntimeError("ROS clock reset or state stamp failed to advance")
            self._seq, self._stamp = s.seq, s.stamp_sec
        if not s.joint_positions or not s.joint_velocities:
            raise RuntimeError("full named body feedback required")
        # The arm predictor has a fixed torso root, so the other independently updated
        # body joints must stay at their initial measured positions.
        for name, initial in self._initial.joint_positions.items():
            if name in RIGHT_ARM_JOINTS:
                continue
            if name.startswith(("wheel_", "gripper_")) and not math.isfinite(initial):
                continue  # unobserved fixed joints remain an explicit limitation
            value = s.joint_positions.get(name)
            tolerance = self.checks.torso_drift_m if name == "torso_lift_joint" else self.checks.body_drift_rad
            if value is None or not math.isfinite(value) or abs(value - initial) > tolerance:
                raise RuntimeError(f"non-controlled body joint moved or became invalid: {name}")
        return s

    def _ready(self, s):
        """At rest at the comfortable pose, and this state checked by the safety worker."""
        if np.max(np.abs(np.asarray(s.q) - Q_START)) > self.checks.start_rad:
            raise RuntimeError("position the arm at comfortable_high_clearance first; there is no automatic reset")
        if np.sum(np.abs(s.qd)) > self.checks.start_speed_l1_rad_s:
            raise RuntimeError("arm is moving too quickly to start effort control")
        self.monitor.wait_until_checked(safety_state(s))

    def _activate(self, command):
        """Switch to effort control (blocking service calls), republishing command meanwhile.

        Each republication first reads and checks a fresh state; the command is held at
        most LEASE_S after the state it was computed from, or the activation faults."""
        if not self.execute:
            raise RuntimeError("check-only: effort control needs execute=True")
        s = self._last
        self._ready(s)  # recheck right before activation
        self.client.publish_effort(command)  # prime the inactive receiver
        self._activated = self.settings["activated"] = True  # a failed switch may still activate
        hold_until = s.received_monotonic_sec + LEASE_S

        def progress():
            self._fresh()
            if time.monotonic() >= hold_until:
                raise RuntimeError("effort activation outlasted the command lease")
            self.client.publish_effort(command)

        self.client.service_progress = progress
        try:
            self.client.switch_to_effort_control(timeout_sec=self.checks.service_s)
        except BaseException:
            self._uncertain = True  # the server may still apply a dispatched switch
            raise
        finally:
            self.client.service_progress = None
        progress()
