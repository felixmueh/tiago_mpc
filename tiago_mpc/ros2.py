"""Optional ROS 2 I/O for the TIAGo right arm, in native joint coordinates.

Ported from GATO ros-tiago c4cf5cc466bc576fd7f5a86bc02d6690c248044a.
See docs/robot.md for provenance, validation and deployment limits.
This module provides transport and controller handover, not a motion supervisor.

The module avoids importing ROS at import time so non-ROS tests and experiments
can still import :mod:`tiago_mpc.ros2`. ROS packages are loaded lazily when a
client is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

import numpy as np

from .tiago import RIGHT_ARM_JOINTS

DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_SEA_STATE_TOPIC = "/arm_right_sea_state_broadcaster_controller/sea_joint_states"
DEFAULT_SEA_VELOCITY_FILTER_TAU_SEC = 0.01
DEFAULT_TRAJECTORY_TOPIC = "/arm_right_controller/joint_trajectory"
DEFAULT_CONTROLLER_MANAGER = "/controller_manager"
DEFAULT_EFFORT_CONTROLLER = "gato_arm_effort_forward_runtime"


@dataclass(frozen=True)
class ArmState:
    """Latest joint state for the right arm."""

    q: np.ndarray
    qd: np.ndarray
    stamp_sec: float
    received_monotonic_sec: float
    seq: int = 0
    joint_positions: dict[str, float] | None = None
    joint_velocities: dict[str, float] | None = None
    body_stamp_sec: float | None = None
    arm_received_monotonic_sec: float | None = None
    body_received_monotonic_sec: float | None = None
    feedback_metadata: dict[str, Any] | None = None
    # SEA body updates require collision checks without inventing an arm sample.
    safety_seq: int | None = None

    @property
    def age_sec(self) -> float:
        return time.monotonic() - self.received_monotonic_sec

def validate_controller_manager_interfaces() -> None:
    """Reject the known incompatible Humble layout before sending any RPCs.

    Discovery can succeed with these mismatched types while service results are
    malformed. This local check targets the PAL 4.32.2 API used by both peers;
    it does not establish compatibility with an arbitrary remote ROS system.
    """
    from controller_manager_msgs.msg import HardwareInterface, ControllerState
    from controller_manager_msgs.srv import SwitchController

    expected = (
        (HardwareInterface, ("name", "data_type", "is_available", "is_claimed")),
        (ControllerState, ("name", "state", "type", "is_async", "update_rate",
                           "claimed_interfaces", "required_command_interfaces",
                           "required_state_interfaces", "is_chainable", "is_chained",
                           "exported_state_interfaces", "reference_interfaces", "chain_connections")),
        (SwitchController.Request, ("activate_controllers", "deactivate_controllers",
                                    "strictness", "activate_asap", "timeout")),
        (SwitchController.Response, ("ok", "message")),
    )
    for message_type, fields in expected:
        if tuple(message_type.get_fields_and_field_types()) != fields:
            raise RuntimeError(
                "Incompatible controller_manager_msgs: PAL 4.32.2 interfaces required; "
                "source a compatible PAL interface overlay in a fresh shell "
                "before constructing the client (see docs/robot.md)."
            )


def _ros_imports() -> dict[str, Any]:
    try:
        import rclpy
        from builtin_interfaces.msg import Duration
        from controller_manager_msgs.srv import (
            ConfigureController,
            LoadController,
            ListControllers,
            SwitchController,
        )
        from rcl_interfaces.msg import Parameter as RosParameter
        from rcl_interfaces.msg import ParameterType, ParameterValue
        from rcl_interfaces.srv import GetParameters, SetParameters
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        validate_controller_manager_interfaces()
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are not importable. Source ROS 2 in the shell that "
            "launches this program (for example `source /opt/ros/humble/setup.bash`, "
            "or PAL's setup on the robot); tiago_mpc does not load it automatically."
        ) from exc

    return locals()


def _duration_msg(ros: dict[str, Any], seconds: float) -> Any:
    if not np.isfinite(seconds) or seconds < 0.0:
        raise ValueError("duration must be finite and non-negative")
    duration = ros["Duration"]()
    duration.sec = int(seconds)
    duration.nanosec = int(round((seconds - int(seconds)) * 1e9))
    if duration.nanosec >= 1_000_000_000:
        duration.sec += 1
        duration.nanosec -= 1_000_000_000
    return duration


def _vector(values: Sequence[float], *, size: int, name: str) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must contain {size} values, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return [float(value) for value in arr]


class SeaAbsoluteVelocityEstimator:
    """Differentiate advancing output-encoder samples, filtering velocity only."""

    def __init__(self, joint_names=RIGHT_ARM_JOINTS, *, tau_sec=0.01, stale_timeout_sec=0.1):
        self.joint_names = tuple(joint_names)
        self.tau_sec = float(tau_sec)
        self.stale_timeout_sec = float(stale_timeout_sec)
        if not np.isfinite(self.tau_sec) or self.tau_sec < 0:
            raise ValueError("sea_velocity_filter_tau_sec must be finite and non-negative")
        if not np.isfinite(self.stale_timeout_sec) or self.stale_timeout_sec <= 0:
            raise ValueError("stale_timeout_sec must be finite and positive")
        self._previous = None
        self._velocity = None

    def update(self, msg: Any, received_sec: float):
        try:
            names = tuple(msg.name)
            values = np.asarray(msg.absolute_position, dtype=np.float64)
            motor = np.asarray(msg.position, dtype=np.float64)
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            if (values.shape != (len(names),) or motor.shape not in ((0,), values.shape)
                    or not np.isfinite(motor).all() or len(set(names)) != len(names)
                    or not np.isfinite(stamp) or not np.isfinite(values).all()
                    or not all(name in names for name in self.joint_names)):
                return None
            q = values[[names.index(name) for name in self.joint_names]]
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        previous = self._previous
        if previous is not None and stamp <= previous[0]:
            return None  # Replayed/duplicate samples never refresh freshness or the filter.
        self._previous = (stamp, q.copy(), received_sec)
        if (previous is None or stamp - previous[0] > self.stale_timeout_sec
                or received_sec - previous[2] > self.stale_timeout_sec):
            self._velocity = None
            return None  # Two distinct, recent samples are needed after startup/gaps.
        dt = stamp - previous[0]
        derivative = (q - previous[1]) / dt
        if not np.isfinite(derivative).all():
            self._velocity = None
            return None
        alpha = 1.0 if self.tau_sec == 0 else -np.expm1(-dt / self.tau_sec)
        self._velocity = (derivative if self._velocity is None else
                          self._velocity + alpha * (derivative - self._velocity))
        return q.copy(), self._velocity.copy(), stamp


def _sea_state_type():
    try:
        from sea_state_msgs.msg import SeaState
    except ImportError as exc:
        raise RuntimeError(
            "SEA absolute feedback requires sea_state_msgs/msg/SeaState; "
            "install/source that interface package before selecting sea_absolute=True."
        ) from exc
    return SeaState


class TiagoRightArmClient:
    """ROS 2 client for TIAGo right-arm state, position, and effort commands."""

    def __init__(
        self,
        *,
        node_name: str = "tiago_mpc_right_arm_client",
        joint_names: Sequence[str] = RIGHT_ARM_JOINTS,
        joint_states_topic: str = DEFAULT_JOINT_STATES_TOPIC,
        sea_absolute: bool = False,
        sea_state_topic: str = DEFAULT_SEA_STATE_TOPIC,
        sea_velocity_filter_tau_sec: float = DEFAULT_SEA_VELOCITY_FILTER_TAU_SEC,
        stale_timeout_sec: float = 0.1,
        trajectory_topic: str = DEFAULT_TRAJECTORY_TOPIC,
        controller_manager: str = DEFAULT_CONTROLLER_MANAGER,
        effort_controller: str = DEFAULT_EFFORT_CONTROLLER,
    ) -> None:
        self.ros = _ros_imports()
        # Resolve the optional dependency before creating a node or any controller RPC.
        sea_type = _sea_state_type() if sea_absolute else None
        self.sea_absolute = bool(sea_absolute)
        self.sea_state_topic = str(sea_state_topic).strip()
        self._sea_estimator = SeaAbsoluteVelocityEstimator(
            joint_names, tau_sec=sea_velocity_filter_tau_sec, stale_timeout_sec=stale_timeout_sec,
        ) if self.sea_absolute else None
        if self.sea_absolute and not self.sea_state_topic:
            raise ValueError("sea_state_topic must be nonempty")
        self._stale_timeout_sec = float(stale_timeout_sec)
        if not np.isfinite(self._stale_timeout_sec) or self._stale_timeout_sec <= 0:
            raise ValueError("stale_timeout_sec must be finite and positive")
        self._body_state = None
        self._sea_state = None
        self.rclpy = self.ros["rclpy"]
        self._owns_rclpy = False
        if not self.rclpy.ok():
            self.rclpy.init()
            self._owns_rclpy = True

        self.joint_names = tuple(joint_names)
        self.joint_states_topic = joint_states_topic
        self.trajectory_topic = trajectory_topic
        self.controller_manager = controller_manager.rstrip("/")
        self.effort_controller = effort_controller
        self.effort_command_topic = f"/{effort_controller}/commands"
        self._latest_state: ArmState | None = None
        self._state_seq = 0
        self._safety_seq = 0
        self._state_callback_count = 0
        self._last_joint_stamp_sec: float | None = None
        # Optional synchronous supervisor; it may raise to abort our wait.
        self.service_progress: Callable[[], None] | None = None

        self.node = self.rclpy.create_node(node_name)
        state_qos = 10
        if self.sea_absolute:
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(
            self.ros["JointState"],
            self.joint_states_topic,
            self._on_joint_state,
            state_qos,
        )
        if self.sea_absolute:
            self.node.create_subscription(sea_type, self.sea_state_topic, self._on_sea_state, state_qos)
        self._trajectory_pub = self.node.create_publisher(
            self.ros["JointTrajectory"],
            self.trajectory_topic,
            10,
        )
        self._effort_pub = self.node.create_publisher(
            self.ros["Float64MultiArray"],
            self.effort_command_topic,
            10,
        )

    def close(self) -> None:
        self.node.destroy_node()
        if self._owns_rclpy:
            self.rclpy.shutdown()

    def __enter__(self) -> "TiagoRightArmClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _on_joint_state(self, msg: Any) -> None:
        self._state_callback_count += 1
        if self.sea_absolute:
            self._on_body_joint_state(msg)
            return
        try:
            names = tuple(msg.name)
            if (len(names) != len(msg.position) or len(names) != len(msg.velocity)
                    or len(set(names)) != len(names)):
                return
            positions = dict(zip(names, map(float, msg.position)))
            velocities = dict(zip(names, map(float, msg.velocity)))
            if not all(name in positions for name in self.joint_names):
                return
            q = np.asarray([positions[name] for name in self.joint_names], dtype=np.float32)
            qd = np.asarray([velocities[name] for name in self.joint_names], dtype=np.float32)
            stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            if (not np.isfinite(q).all() or not np.isfinite(qd).all()
                    or not np.isfinite(stamp_sec)
                    or (self._last_joint_stamp_sec is not None
                        and stamp_sec <= self._last_joint_stamp_sec)):
                return
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return
        # A duplicate source message must not refresh age or satisfy after_seq.
        # Missing velocity is invalid rather than a fabricated stationary arm.
        self._last_joint_stamp_sec = stamp_sec
        self._state_seq += 1
        self._latest_state = ArmState(
            q=q,
            qd=qd,
            stamp_sec=stamp_sec,
            received_monotonic_sec=time.monotonic(),
            seq=self._state_seq,
            joint_positions=positions,
            joint_velocities=velocities,
        )

    def _on_body_joint_state(self, msg: Any) -> None:
        try:
            names = tuple(msg.name)
            if len(names) != len(msg.position) or len(set(names)) != len(names):
                return
            positions = dict(zip(names, map(float, msg.position)))
            velocities = dict(zip(names, map(float, msg.velocity)))
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            if (not np.isfinite(stamp) or not all(name in positions for name in self.joint_names)
                    or (self._body_state is not None and stamp <= self._body_state[0])):
                return
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        # Retain the full body for the application's separate safety monitor.
        # Wheel velocity NaNs on the robot are unrelated to the controlled arm.
        self._body_state = stamp, time.monotonic(), positions, velocities
        self._merge_sea_state()

    def _on_sea_state(self, msg: Any) -> None:
        self._state_callback_count += 1
        received = time.monotonic()
        sample = self._sea_estimator.update(msg, received)
        if sample is None:
            # A gap invalidates velocity readiness; ignored replay/invalid data ages out.
            if self._sea_estimator._velocity is None:
                self._sea_state = None
                self._latest_state = None
            return
        self._state_seq += 1
        motor = dict(zip(msg.name, map(float, msg.position)))
        self._sea_state = (*sample, received, self._state_seq,
                           [motor[name] for name in self.joint_names] if motor else None)
        self._merge_sea_state()

    def _merge_sea_state(self) -> None:
        if self._body_state is None or self._sea_state is None:
            return
        body_stamp, body_received, positions, velocities = self._body_state
        q, qd, stamp, arm_received, seq, motor = self._sea_state
        positions, velocities = positions.copy(), velocities.copy()
        self._safety_seq += 1
        positions.update(zip(self.joint_names, map(float, q)))
        velocities.update(zip(self.joint_names, map(float, qd)))
        self._latest_state = ArmState(
            q=q.copy(), qd=qd.copy(), stamp_sec=stamp,
            received_monotonic_sec=min(arm_received, body_received), seq=seq,
            joint_positions=positions, joint_velocities=velocities,
            body_stamp_sec=body_stamp, arm_received_monotonic_sec=arm_received,
            body_received_monotonic_sec=body_received, safety_seq=self._safety_seq,
            feedback_metadata={
                "arm_source": "sea_state_msgs/msg/SeaState.absolute_position",
                "arm_topic": self.sea_state_topic,
                "body_topic": self.joint_states_topic,
                "body_stamp_sec": body_stamp,
                "body_received_monotonic_sec": body_received,
                "arm_received_monotonic_sec": arm_received,
                "joint_names": list(self.joint_names),
                "sea_motor_position": motor,
                "sea_absolute_position": q.tolist(),
                "velocity_filter_tau_sec": self._sea_estimator.tau_sec,
            },
        )

    def spin_once(
        self, timeout_sec: float = 0.0, *, max_callbacks: int = 64,
        drain_budget_sec: float = 0.002,
    ) -> int:
        """Service a bounded burst of pending state callbacks; return their count.

        Only the first executor call may wait. Extra calls are nonblocking, so
        body and arm feedback do not share a one-callback-per-command budget.
        Stop after an executor call without a state callback, the callback cap,
        or the drain budget. A callback itself can exceed the budget; this is
        not a hard deadline or a background executor. Call from one owner only.
        """
        if not np.isfinite(timeout_sec) or timeout_sec < 0:
            raise ValueError("timeout_sec must be finite and non-negative")
        if not isinstance(max_callbacks, int) or max_callbacks < 1:
            raise ValueError("max_callbacks must be a positive integer")
        if not np.isfinite(drain_budget_sec) or drain_budget_sec < 0:
            raise ValueError("drain_budget_sec must be finite and non-negative")
        start_count = self._state_callback_count
        previous_count = start_count
        self.rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        drain_started = time.monotonic()
        while (self._state_callback_count != previous_count
               and self._state_callback_count - start_count < max_callbacks
               and time.monotonic() - drain_started < drain_budget_sec):
            previous_count = self._state_callback_count
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
        return self._state_callback_count - start_count

    def read_state(
        self, timeout_sec: float = 2.0, *, after_seq: int | None = None,
        max_age_sec: float | None = None,
    ) -> ArmState:
        """Read cached state, or require a newer sequence and/or bounded age.

        ``after_seq=previous.seq`` waits for a distinct arm observation, even
        while the robot stands still. Timeouts/ages use the monotonic clock;
        ROS source timestamps remain separately available on the result.
        A zero timeout performs one nonblocking poll.
        """
        if not np.isfinite(timeout_sec) or timeout_sec < 0:
            raise ValueError("timeout_sec must be finite and non-negative")
        if max_age_sec is not None and (not np.isfinite(max_age_sec) or max_age_sec < 0):
            raise ValueError("max_age_sec must be finite and non-negative")
        deadline = time.monotonic() + timeout_sec
        first_poll = True
        while self.rclpy.ok():
            remaining = max(0.0, deadline - time.monotonic())
            self.spin_once(timeout_sec=0.0 if first_poll else min(0.05, remaining))
            first_poll = False
            state = self.latest_state()
            if (state is not None and (after_seq is None or state.seq > after_seq)
                    and (max_age_sec is None or state.age_sec <= max_age_sec)):
                return state
            if time.monotonic() >= deadline:
                break
        raise TimeoutError(
            f"did not receive qualifying {len(self.joint_names)}-joint arm state on "
            f"{self.joint_states_topic}"
            + (f" and fresh SEA absolute position/estimated velocity on {self.sea_state_topic}"
               if self.sea_absolute else "")
            + f" within {timeout_sec:.1f}s (after_seq={after_seq}, max_age_sec={max_age_sec})"
        )

    def latest_state(self) -> ArmState | None:
        if (self.sea_absolute and self._latest_state is not None
                and self._latest_state.age_sec > self._stale_timeout_sec):
            return None
        return self._latest_state

    def publish_position_trajectory(
        self,
        positions: Sequence[float],
        *,
        duration_sec: float = 2.0,
        velocities: Sequence[float] | None = None,
    ) -> None:
        point = self.ros["JointTrajectoryPoint"]()
        point.positions = _vector(positions, size=len(self.joint_names), name="positions")
        if velocities is not None:
            point.velocities = _vector(
                velocities,
                size=len(self.joint_names),
                name="velocities",
            )
        point.time_from_start = _duration_msg(self.ros, duration_sec)

        msg = self.ros["JointTrajectory"]()
        msg.joint_names = list(self.joint_names)
        msg.points = [point]
        self._trajectory_pub.publish(msg)

    def publish_position_trajectory_points(
        self,
        positions: Sequence[Sequence[float]],
        *,
        dt: float,
    ) -> None:
        """Publish a multi-point joint position trajectory.

        The first row is sent at ``dt`` seconds, not zero. This avoids asking
        the controller to jump if the first IK point is slightly different from
        the current measured state.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        rows = np.asarray(positions, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.joint_names):
            raise ValueError(
                f"positions must have shape (N, {len(self.joint_names)}), "
                f"got {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("positions contains non-finite values")

        msg = self.ros["JointTrajectory"]()
        msg.joint_names = list(self.joint_names)
        for idx, row in enumerate(rows):
            point = self.ros["JointTrajectoryPoint"]()
            point.positions = [float(value) for value in row]
            point.time_from_start = _duration_msg(self.ros, (idx + 1) * dt)
            msg.points.append(point)
        self._trajectory_pub.publish(msg)

    def publish_effort(self, torques: Sequence[float]) -> None:
        msg = self.ros["Float64MultiArray"]()
        msg.data = _vector(torques, size=len(self.joint_names), name="torques")
        self._effort_pub.publish(msg)

    def configure_runtime_effort_controller(self, timeout_sec: float = 5.0) -> None:
        """Load and configure a forward effort controller through public ROS APIs."""
        # TODO: Does the installed PAL effort-command receiver expire commands
        # when the external publisher stops, and what fallback does it perform?
        # Verify the installed implementation/configuration with PAL; the upstream
        # Humble forward controller has no command-age check. The service timeout
        # below only bounds our wait for a response, not the lifetime of an effort.
        controllers = self._list_controllers(timeout_sec)
        controller = controllers.get(self.effort_controller)
        expected_type = "forward_command_controller/ForwardCommandController"
        if controller is not None:
            if controller.type != expected_type:
                raise RuntimeError(f"unexpected effort controller type: {controller.type}")
            if controller.state in {"inactive", "active"}:
                self._validate_effort_controller(controller, timeout_sec)
                return
            if controller.state != "unconfigured":
                raise RuntimeError(f"unexpected effort controller state: {controller.state}")
        else:
            self._set_controller_manager_type(self.effort_controller, expected_type, timeout_sec)
            self._load_controller(self.effort_controller, timeout_sec)
        self._set_remote_parameters(
            f"/{self.effort_controller}",
            {"joints": list(self.joint_names), "interface_name": "effort"},
            timeout_sec,
        )
        self._configure_controller(self.effort_controller, timeout_sec)
        self._validate_effort_controller(
            self._list_controllers(timeout_sec).get(self.effort_controller), timeout_sec,
        )

    def _validate_effort_controller(self, controller: Any, timeout_sec: float) -> None:
        """Verify the ordered vector contract; claimed resources are only a set."""
        if (controller is None or controller.state not in {"inactive", "active"}
                or controller.type != "forward_command_controller/ForwardCommandController"):
            raise RuntimeError("effort controller must be a configured ForwardCommandController")
        request = self.ros["GetParameters"].Request()
        request.names = ["joints", "interface_name"]
        response = self._service_call(
            f"/{self.effort_controller}/get_parameters",
            self.ros["GetParameters"], request, timeout_sec,
        )
        values = response.values
        types = self.ros["ParameterType"]
        if (len(values) != 2 or values[0].type != types.PARAMETER_STRING_ARRAY
                or tuple(values[0].string_array_value) != self.joint_names):
            raise RuntimeError("effort controller joint order must exactly match commanded joints")
        if values[1].type != types.PARAMETER_STRING or values[1].string_value != "effort":
            raise RuntimeError("effort controller interface_name must be effort")

    def active_arm_controller(self, timeout_sec: float = 5.0) -> str | None:
        """Read the unique verified arm owner without changing controller mode."""
        return self._active_arm_controller(self._list_controllers(timeout_sec))

    def switch_to_effort_control(self, timeout_sec: float = 5.0) -> None:
        self._switch_arm_mode(self.effort_controller, timeout_sec)
        self._wait_for_topic_subscription(self.effort_command_topic, timeout_sec)

    def switch_to_default_control(self, timeout_sec: float = 5.0) -> None:
        self._switch_arm_mode("arm_right_controller", timeout_sec)

    def _list_controllers(self, timeout_sec: float) -> dict[str, Any]:
        response = self._service_call(
            f"{self.controller_manager}/list_controllers",
            self.ros["ListControllers"], self.ros["ListControllers"].Request(),
            timeout_sec,
        )
        controllers = {c.name: c for c in response.controller}
        if "" in controllers or len(controllers) != len(response.controller):
            raise RuntimeError("invalid controller inventory: blank or duplicate names")
        return controllers

    def _active_arm_controller(self, controllers: dict[str, Any]) -> str | None:
        """Reject mixed modes before any mutation, including during cleanup.

        Position and effort are distinct ROS resources but must not be treated
        as independent physical arm modes. Topic discovery cannot establish this.
        """
        known = {"arm_right_controller", "arm_right_gravity_compensation_controller",
                 self.effort_controller}
        active = []
        for name, controller in controllers.items():
            owns_arm = any(interface.split("/")[0] in self.joint_names
                           for interface in controller.claimed_interfaces)
            if name not in known and owns_arm:
                raise RuntimeError(f"unexpected right-arm interface owner: {name}")
            if name in known and controller.state == "active":
                active.append(name)
        if len(active) > 1:
            raise RuntimeError(
                f"multiple right-arm controllers active: {active}; "
                "restore a known working PAL control mode before running TIAGo MPC; "
                "automatic mixed-mode recovery is disabled"
            )
        if active:
            name = active[0]
            interface = "position" if name == "arm_right_controller" else "effort"
            expected = {f"{joint}/{interface}" for joint in self.joint_names}
            if set(controllers[name].claimed_interfaces) != expected:
                raise RuntimeError(f"unexpected claimed interfaces for {name}")
            return name
        return None

    def _switch_arm_mode(self, destination: str, timeout_sec: float) -> None:
        controllers = self._list_controllers(timeout_sec)
        source = self._active_arm_controller(controllers)
        if destination not in controllers:
            raise RuntimeError(f"required controller is not loaded: {destination}")
        if destination == self.effort_controller:
            self._validate_effort_controller(controllers[destination], timeout_sec)
        if source == destination:
            return
        if controllers[destination].state != "inactive":
            raise RuntimeError(f"controller is not ready to activate: {destination}")
        if destination == "arm_right_controller":
            # Reactivation must hold measured position, not an old trajectory.
            self._set_remote_parameters(
                "/arm_right_controller",
                {"set_last_command_interface_value_as_state_on_activation": False},
                timeout_sec,
            )
        # One request performs the handover; never enable gravity compensation
        # as an additional step after enabling position control.
        self._switch_controllers(
            activate=[destination], deactivate=[source] if source else [],
            strictness=2, timeout_sec=timeout_sec,
        )
        actual = self._active_arm_controller(self._list_controllers(timeout_sec))
        if actual != destination:
            raise RuntimeError(f"controller switch not established: wanted {destination}, got {actual}")

    def _service_progress_tick(self) -> None:
        callback = getattr(self, "service_progress", None)
        if callback is not None:
            callback()

    def _service_call(
        self,
        service_name: str,
        service_type: type,
        request: Any,
        timeout_sec: float,
    ) -> Any:
        """Wait in <=10 ms slices, allowing a synchronous supervisor between them.

        ``service_progress`` runs before dispatch and between blocking chunks,
        including after a chunk completes the future. It may poll state and
        raise on expiry/safety failure. It must not recursively issue services.
        A running ROS callback or guard can overrun a slice; no hard deadline is
        implied. Aborting this wait does NOT cancel an already dispatched RPC.
        Discovery and response retain separate timeout budgets, as before.
        """
        if not np.isfinite(timeout_sec) or timeout_sec < 0:
            raise ValueError("timeout_sec must be finite and non-negative")
        client = self.node.create_client(service_type, service_name)
        try:
            self._service_progress_tick()
            discovery_deadline = time.monotonic() + timeout_sec
            while True:
                remaining = max(0.0, discovery_deadline - time.monotonic())
                available = client.wait_for_service(timeout_sec=min(0.01, remaining))
                self._service_progress_tick()
                if available:
                    break
                if time.monotonic() >= discovery_deadline:
                    raise TimeoutError(f"service not available: {service_name}")

            response_deadline = time.monotonic() + timeout_sec
            future = client.call_async(request)
            self._service_progress_tick()
            while not future.done():
                remaining = response_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"service call timed out: {service_name}")
                self.rclpy.spin_until_future_complete(
                    self.node, future, timeout_sec=min(0.01, remaining),
                )
                self._service_progress_tick()
            result = future.result()
            if result is None:
                raise RuntimeError(f"service call returned no result: {service_name}")
            return result
        finally:
            self.node.destroy_client(client)

    def _set_controller_manager_type(
        self,
        controller_name: str,
        controller_type: str,
        timeout_sec: float,
    ) -> None:
        self._set_remote_parameters(
            self.controller_manager,
            {f"{controller_name}.type": controller_type},
            timeout_sec,
        )

    def _set_remote_parameters(
        self,
        node_name: str,
        values: dict[str, str | bool | list[str]],
        timeout_sec: float,
    ) -> None:
        request = self.ros["SetParameters"].Request()
        for name, value in values.items():
            request.parameters.append(self._parameter_msg(name, value))
        response = self._service_call(
            f"{node_name}/set_parameters",
            self.ros["SetParameters"],
            request,
            timeout_sec,
        )
        failed = [result.reason for result in response.results if not result.successful]
        if failed:
            raise RuntimeError(f"failed setting parameters on {node_name}: {failed}")

    def _parameter_msg(self, name: str, value: str | bool | list[str]) -> Any:
        msg = self.ros["RosParameter"]()
        msg.name = name
        msg.value = self.ros["ParameterValue"]()
        if isinstance(value, bool):
            msg.value.type = self.ros["ParameterType"].PARAMETER_BOOL
            msg.value.bool_value = value
        elif isinstance(value, str):
            msg.value.type = self.ros["ParameterType"].PARAMETER_STRING
            msg.value.string_value = value
        else:
            msg.value.type = self.ros["ParameterType"].PARAMETER_STRING_ARRAY
            msg.value.string_array_value = list(value)
        return msg

    def _load_controller(self, name: str, timeout_sec: float) -> None:
        request = self.ros["LoadController"].Request()
        request.name = name
        response = self._service_call(
            f"{self.controller_manager}/load_controller",
            self.ros["LoadController"],
            request,
            timeout_sec,
        )
        if not response.ok and not self._node_exists(f"/{name}"):
            raise RuntimeError(f"failed to load controller: {name}")

    def _configure_controller(self, name: str, timeout_sec: float) -> None:
        request = self.ros["ConfigureController"].Request()
        request.name = name
        response = self._service_call(
            f"{self.controller_manager}/configure_controller",
            self.ros["ConfigureController"],
            request,
            timeout_sec,
        )
        if not response.ok and not self._node_exists(f"/{name}"):
            raise RuntimeError(f"failed to configure controller: {name}")

    def _switch_controllers(
        self,
        *,
        activate: Sequence[str],
        deactivate: Sequence[str],
        strictness: int,
        timeout_sec: float,
    ) -> None:
        request = self.ros["SwitchController"].Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = strictness
        request.activate_asap = True
        request.timeout = _duration_msg(self.ros, timeout_sec)
        response = self._service_call(
            f"{self.controller_manager}/switch_controller",
            self.ros["SwitchController"],
            request,
            timeout_sec,
        )
        if not response.ok:
            raise RuntimeError(
                "failed to switch controllers: "
                f"activate={list(activate)} deactivate={list(deactivate)}; "
                f"reason={response.message}"
            )

    def _node_exists(self, full_name: str) -> bool:
        for name, namespace in self.node.get_node_names_and_namespaces():
            candidate = f"{namespace.rstrip('/')}/{name}" if namespace != "/" else f"/{name}"
            if candidate == full_name:
                return True
        return False

    def _topic_has_subscription(self, topic_name: str) -> bool:
        return bool(self.node.get_subscriptions_info_by_topic(topic_name))

    def _wait_for_topic_subscription(self, topic_name: str, timeout_sec: float) -> None:
        if not np.isfinite(timeout_sec) or timeout_sec < 0:
            raise ValueError("timeout_sec must be finite and non-negative")
        deadline = time.monotonic() + timeout_sec
        self._service_progress_tick()
        while True:
            if self._topic_has_subscription(topic_name):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no subscriber appeared on {topic_name}")
            self.spin_once(timeout_sec=min(0.01, remaining))
            self._service_progress_tick()
