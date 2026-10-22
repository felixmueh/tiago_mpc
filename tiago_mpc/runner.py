"""ROS I/O owner with separate MPC and collision-safety processes.

Importing this module does not initialize ROS, CUDA or a collision model.
The default run checks feedback/safety only; effort execution is explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .laws import sample_knot
from .loops import guard
from .mpc import MpcWorker
from .plan import State
from .ros2 import RIGHT_ARM_JOINTS, TiagoRightArmClient

COMFORTABLE_Q = (-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04)


@dataclass(frozen=True)
class RunnerConfig:
    command_period_sec: float = 0.01
    state_timeout_sec: float = 0.1
    command_timeout_sec: float = 0.1
    startup_timeout_sec: float = 180.0
    service_timeout_sec: float = 0.5
    duration_sec: float = 8.0
    speed_l1_tolerance_rad_s: float = 1.0
    start_tolerance_rad: float = 0.05
    body_drift_tolerance_rad: float = 0.02
    torso_drift_tolerance_m: float = 0.005

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def safety_state(state):
    from .safety import SafetyCheckState
    if not state.joint_positions or not state.joint_velocities:
        raise RuntimeError("full named body positions and velocities are required")
    return SafetyCheckState(
        q=state.q.copy(), qd=state.qd.copy(), stamp_sec=state.stamp_sec,
        received_monotonic_sec=state.received_monotonic_sec,
        seq=state.seq if state.safety_seq is None else state.safety_seq,
        joint_positions=dict(state.joint_positions),
        joint_velocities=dict(state.joint_velocities))


class RuntimeGuard:
    """Fast native-coordinate checks; the independent geometry worker adds more."""

    def __init__(self, config, initial_state, limits):
        self.config, self.initial = config, initial_state
        self.lower = np.asarray(limits["q_lower"], dtype=float)
        self.upper = np.asarray(limits["q_upper"], dtype=float)
        self.velocity = np.asarray(limits["velocity_limits"], dtype=float)
        self.effort = np.asarray(limits["effort_limits"], dtype=float)
        if (any(v.shape != (7,) or not np.isfinite(v).all()
                for v in (self.lower, self.upper, self.velocity, self.effort))
                or np.any(self.lower >= self.upper) or np.any(self.velocity <= 0)
                or np.any(self.effort <= 0)):
            raise ValueError("invalid native physical limits from MPC worker")
        self.knots, self.dt_s = int(limits["N"]) - 1, float(limits["dt_s"])
        if self.knots < 1 or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("invalid MPC horizon from MPC worker")
        self.last_seq, self.last_stamp = -1, None

    def state(self, state, now):
        if state is None:
            raise RuntimeError("joint state unavailable")
        age = now - state.received_monotonic_sec
        if not math.isfinite(age) or age < 0 or age > self.config.state_timeout_sec:
            raise RuntimeError("joint state stale or invalid receipt time")
        if not math.isfinite(state.stamp_sec):
            raise RuntimeError("invalid ROS state stamp")
        q, v = np.asarray(state.q), np.asarray(state.qd)
        if q.shape != (7,) or v.shape != (7,) or not np.isfinite(np.r_[q, v]).all():
            raise RuntimeError("invalid controlled joint feedback")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise RuntimeError("measured joint position exceeds physical limits")
        if np.any(np.abs(v) > self.velocity):
            raise RuntimeError("measured joint velocity exceeds physical limits")
        if state.seq < self.last_seq:
            raise RuntimeError("joint-state sequence reversed")
        if state.seq != self.last_seq:
            if self.last_stamp is not None and state.stamp_sec <= self.last_stamp:
                raise RuntimeError("ROS clock reset or state stamp failed to advance")
            self.last_seq, self.last_stamp = state.seq, state.stamp_sec
        if not state.joint_positions or not state.joint_velocities:
            raise RuntimeError("full named body feedback required")
        # The arm predictor has a fixed torso root. Other independently updated
        # body joints must remain at their initial measured positions.
        for name, initial in (self.initial.joint_positions or {}).items():
            if name in RIGHT_ARM_JOINTS:
                continue
            if name.startswith(("wheel_", "gripper_")) and not math.isfinite(initial):
                continue  # unobserved fixed joints remain an explicit limitation
            value = state.joint_positions.get(name)
            tolerance = (self.config.torso_drift_tolerance_m if name == "torso_lift_joint"
                         else self.config.body_drift_tolerance_rad)
            if value is None or not math.isfinite(value) or abs(value - initial) > tolerance:
                raise RuntimeError(f"non-controlled body joint moved or became invalid: {name}")

    def start(self, state):
        if np.max(np.abs(state.q - COMFORTABLE_Q)) > self.config.start_tolerance_rad:
            raise RuntimeError("position the arm at comfortable_high_clearance before execution; no automatic reset")
        if np.sum(np.abs(state.qd)) > self.config.speed_l1_tolerance_rad_s:
            raise RuntimeError("arm is moving too quickly to start effort control")

    def lease_end(self, t0):
        """A plan from a state at t0 may be commanded until then; it ends no later than
        the plan: 31 x 2 ms knots span only 62 ms, less than the 100 ms command timeout."""
        return t0 + min(self.config.command_timeout_sec, self.knots * self.dt_s)

    def command(self, plan, now):
        """Torque due at ``now`` while the plan's lease lasts (loops.guard)."""
        return guard(sample_knot(plan, now, None), plan, now, self.effort).copy()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _log_values(value):
    """Keep diagnostic metadata JSON-valid, including unobserved wheel NaNs."""
    if isinstance(value, np.ndarray):
        return _log_values(value.tolist())
    if isinstance(value, dict):
        return {k: _log_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_log_values(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


class RosMpcRunner:
    """Caller process owns I/O; spawned children own CUDA and geometry checks."""

    def __init__(self, config, task, params, safety_settings, *,
                 arm_factory=TiagoRightArmClient, monitor_factory=None,
                 mpc_factory=MpcWorker, clock=time.monotonic, sleep=time.sleep):
        self.config, self.task, self.params = config, task, params
        self.safety_settings = safety_settings
        if not safety_settings.enabled:
            raise ValueError("the ROS runner requires the safety worker")
        if tuple(safety_settings.controlled_joint_names) != RIGHT_ARM_JOINTS:
            raise ValueError("the safety worker must monitor the controlled right-arm joints")
        self.arm_factory, self.monitor_factory, self.mpc_factory = arm_factory, monitor_factory, mpc_factory
        self.clock, self.sleep = clock, sleep

    def run(self, output: Path, *, execute=False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        summary = {"status": "starting", "execute": execute,
                   "config": asdict(self.config), "task": asdict(self.task),
                   "safety": asdict(self.safety_settings),
                   "params": asdict(self.params),
                   "restored": None, "cleanup_errors": [], "accepted_commands": 0,
                   "ros_distro": os.environ.get("ROS_DISTRO")}
        root = Path(__file__).resolve().parents[1]
        summary["upstream"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root / "external/gato", text=True).strip()
        summary["application_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        summary["source_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip())
        source = output / "source"
        source.mkdir()
        for name in ("runner.py", "loops.py", "plan.py", "laws.py", "ros2.py", "model_runtime.py", "mpc.py", "joint_dynamics.py", "safety.py"):
            shutil.copyfile(root / "tiago_mpc" / name, source / name)
        arm = monitor = mpc = None
        mode_attempted = False
        mode_uncertain = False
        log = (output / "events.jsonl").open("w", buffering=1)

        def record(event, **fields):
            log.write(json.dumps(_log_values(dict(event=event, monotonic_sec=self.clock(), **fields)),
                                 default=_json, allow_nan=False) + "\n")

        try:
            arm = self.arm_factory()
            initial = arm.read_state(timeout_sec=self.config.startup_timeout_sec,
                                     max_age_sec=self.config.state_timeout_sec)
            if self.monitor_factory is None:
                from .safety import AsyncSafetyMonitor
                monitor_factory = AsyncSafetyMonitor
            else:
                monitor_factory = self.monitor_factory
            monitor = monitor_factory(settings=self.safety_settings,
                                      initial_state=safety_state(initial))
            # Model construction can take time. Check a fresh observation after
            # it finishes, rather than accepting an old startup snapshot.
            initial = arm.read_state(timeout_sec=self.config.state_timeout_sec,
                                     after_seq=initial.seq, max_age_sec=self.config.state_timeout_sec)
            monitor.wait_until_checked(safety_state(initial))
            record("initial_safety_checked", seq=initial.seq)
            if not execute:
                summary["status"] = "checked"
                return summary

            if arm.active_arm_controller(timeout_sec=self.config.service_timeout_sec) != "arm_right_controller":
                raise RuntimeError("start in the default PAL position controller; existing effort control is not adopted")

            # Returns once the worker is ready; the arm stays in PAL position
            # control meanwhile, and a fresh state is checked below.
            mpc = self.mpc_factory(self.params)
            summary["worker"] = mpc.details
            guard = RuntimeGuard(self.config, initial, mpc.details)
            # Configure while still in the existing PAL control mode. Service
            # waits are bounded, and must finish before the first command solve.
            arm.configure_runtime_effort_controller(timeout_sec=self.config.service_timeout_sec)
            state = arm.read_state(timeout_sec=self.config.state_timeout_sec,
                                   after_seq=initial.seq, max_age_sec=self.config.state_timeout_sec)
            guard.state(state, self.clock()); guard.start(state)
            monitor.wait_until_checked(safety_state(state))
            epoch_stamp, task_start = state.stamp_sec, self.clock()
            next_solve = task_start
            next_io = task_start
            pending = current = None  # the requested state (ArmState) and the plan in use
            pending_elapsed = None  # the task time of the requested state
            request_id = 0
            last_request_seq = -1
            stopping = False

            # ==========================================================================
            # ==========================================================================
            # TODO(torque command, Felix 2026-10-22): STILL NOT THE FINAL COMMAND
            # TYPE FOR REAL-TIME CONTROL OF THE ROBOT.
            # Done: at each command tick the execution controller sends the knot
            # of the latest plan that is due, u_k for t0 + k dt <= now < t0 + (k+1) dt,
            # where t0 is the receipt time of the plan's state (tiago_mpc/plan.py,
            # shared with the simulation). The lease ends no later than the plan,
            # so a plan is never sampled past its last knot: expiry faults first.
            # Remaining: the robot's onboard controllers hold each torque until the
            # next tick, so with a command period longer than the knots the plan is
            # only sampled once per tick, and the torque still steps at every tick
            # and every control update. GATO shifts its warm start by one knot per
            # solve whatever dt_elapsed says, so solves more than one knot apart
            # start from a misaligned guess.
            # Deferred (thesis workspace research/tiago-mpc-simplification-proposals.md,
            # "Torque commands"): smoothing or local feedback; feedforward torque
            # plus feedback in a PAL real-time controller; possibly separate
            # control modes for simulation and the robot.
            # ==========================================================================
            def publish_checked(plan):
                # This guard is adjacent to every nonzero publication, including
                # repeats and service-progress publication. Slow logging or a
                # service response must not renew the original command lease.
                current_state = arm.latest_state()
                guard.state(current_state, self.clock())
                monitor.check(safety_state(current_state))
                torque = guard.command(plan, self.clock())
                arm.publish_effort(torque)
                record("effort_publish", plan_start_monotonic_sec=plan.t0, torque_nm=torque)

            while not stopping:
                arm.spin_once(timeout_sec=0.0)
                now = self.clock()
                state = arm.latest_state()
                guard.state(state, now)
                monitor.check(safety_state(state))
                if pending is not None and now >= guard.lease_end(pending.received_monotonic_sec):
                    raise RuntimeError("pending MPC solve exceeded the command deadline")
                plan = mpc.result()  # raises on an error reply or a dead worker
                if plan is not None:
                    arrived = (self.task.done(pending, pending_elapsed)
                               and np.sum(np.abs(state.qd)) <= self.config.speed_l1_tolerance_rad_s)
                    if arrived:
                        summary["status"] = "arrived"
                        record("arrival", source_seq=pending.seq, stats=plan.stats)
                        break  # do not enable/publish effort for an already-met goal
                    if not mode_attempted:
                        # Recheck current full-body state immediately before
                        # activation, not the older model-build snapshot.
                        guard.start(state)
                        monitor.wait_until_checked(safety_state(state))
                        publish_checked(plan)  # prime the inactive receiver
                        mode_attempted = True  # failed RPC may have activated it

                        def activation_progress():
                            arm.spin_once(timeout_sec=0.0)
                            if mpc.result() is not None:  # nothing pending: raises if the worker died
                                raise RuntimeError("unexpected MPC result")
                            publish_checked(plan)

                        arm.service_progress = activation_progress
                        try:
                            arm.switch_to_effort_control(timeout_sec=self.config.service_timeout_sec)
                        except BaseException:
                            # The server may still apply a dispatched switch.
                            # A subsequent position-mode inventory is not proof
                            # that this pending activation has been cancelled.
                            mode_uncertain = True
                            raise
                        finally:
                            arm.service_progress = None
                        guard.state(arm.latest_state(), self.clock())
                        monitor.check(safety_state(arm.latest_state()))
                    publish_checked(plan)
                    summary["accepted_commands"] += 1
                    record("command", source_seq=pending.seq, plan_torques_nm=plan.torques,
                           plan_dt_s=plan.dt, plan_start_monotonic_sec=plan.t0,
                           expiry=guard.lease_end(plan.t0), stats=plan.stats)
                    current, pending = plan, None
                elapsed = state.stamp_sec - epoch_stamp
                if elapsed < 0:
                    raise RuntimeError("ROS task clock reset")
                if now - task_start >= self.config.duration_sec:
                    summary["status"] = "completed" if self.task.done(state, elapsed) else "timeout"
                    stopping = True
                # One solve per knot, at the command tick nearest each knot boundary,
                # and at most one per tick (command period longer than the knots: every tick).
                # Missed slots are skipped. GATO shifts its warm start by exactly one
                # knot per solve, so this keeps it aligned whenever ticks allow it.
                half_tick = .5 * self.config.command_period_sec
                if (not stopping and pending is None and now >= next_solve - half_tick
                        and state.seq != last_request_seq):
                    request_id += 1
                    # The plan starts at the state's receipt time.
                    mpc.request(State(state.received_monotonic_sec, state.q, state.qd,
                                      state.joint_positions, state.joint_velocities),
                                self.task.reference(elapsed, mpc.N, mpc.dt))
                    pending, pending_elapsed, last_request_seq = state, elapsed, state.seq
                    record("request", request_id=request_id, source_seq=state.seq,
                           source_stamp_sec=state.stamp_sec, q=state.q, qd=state.qd,
                           task_elapsed_sec=elapsed, deadline=guard.lease_end(state.received_monotonic_sec))
                    next_solve += (math.floor((now + half_tick - next_solve) / mpc.dt) + 1) * mpc.dt
                if mode_attempted and current is not None and not stopping:
                    # The receiver holds each torque until the next tick; publish the
                    # current plan's due knot only while its lease is valid.
                    publish_checked(current)
                record("state", seq=state.seq,
                       safety_seq=state.seq if state.safety_seq is None else state.safety_seq,
                       stamp_sec=state.stamp_sec,
                       receipt_sec=state.received_monotonic_sec, q=state.q, qd=state.qd,
                       named_positions=state.joint_positions,
                       named_velocities=state.joint_velocities,
                       safety_checked_seq=monitor.last_checked_seq)
                next_io += self.config.command_period_sec
                if next_io <= self.clock():
                    next_io = self.clock() + self.config.command_period_sec
                self.sleep(max(0.0, next_io - self.clock()))
            return summary
        except BaseException as exc:
            summary.update(status="fault", error=f"{type(exc).__name__}: {exc}")
            return summary
        finally:
            if arm is not None and mode_attempted:
                arm.service_progress = None
                # Attempt handback even if zero publication fails; report every
                # cleanup error and never claim an unconfirmed restoration.
                for _ in range(5):
                    try:
                        arm.publish_effort(np.zeros(7))
                        record("effort_publish", torque_nm=np.zeros(7), reason="handback")
                        arm.spin_once(timeout_sec=0.0)
                    except BaseException as exc:
                        summary["cleanup_errors"].append(f"zero effort: {exc}")
                    self.sleep(self.config.command_period_sec)
                try:
                    arm.switch_to_default_control(timeout_sec=self.config.service_timeout_sec)
                    summary["restored"] = not mode_uncertain
                    if mode_uncertain:
                        summary["cleanup_errors"].append(
                            "restoration uncertain: interrupted activation RPC may still execute; "
                            "manual controller reconciliation required")
                except BaseException as exc:
                    summary["restored"] = False
                    summary["cleanup_errors"].append(f"controller handback: {exc}")
            for resource in (mpc, monitor, arm):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as exc:
                        summary["cleanup_errors"].append(f"close: {exc}")
            if summary["cleanup_errors"]:
                summary["status"] = "fault"
            log.close()
            (output / "summary.json").write_text(json.dumps(summary, default=_json, indent=2) + "\n")
