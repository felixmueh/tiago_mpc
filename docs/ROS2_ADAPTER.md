# TIAGo ROS 2 I/O adapter

`tiago_mpc.ros2.TiagoRightArmClient` ports the existing right-arm ROS transport
and PAL controller lifecycle into this repository. It has no dependency on the
old solver or old controller, and imports ROS only when constructing a client.
It does not implement an MPC loop, command watchdog, collision checking, effort
limits, reset/path verification or automatic restoration on `close()`.
The application must own those behaviors before using effort control.

The seven native joints are `arm_right_1_joint` through `arm_right_7_joint` in
that order. State positions and velocities, position trajectories and effort
commands all use native TIAGo coordinates. Do **not** reflect joint 6 here:
that conversion belongs only at the existing normalized GPU model boundary.
Full body joint dictionaries are retained for an external safety monitor.

## Source and dependencies

Port source: GATO branch `felix-feature/ros-tiago`, commit
`c4cf5cc466bc576fd7f5a86bc02d6690c248044a`, file
`tiago_src/gato_tiago/ros_tiago.py`. Its common transport/controller-switching
implementation is also on `felix-devel` (last adapter change `edae396`);
the ROS branch additionally supplies optional SEA absolute-position feedback
and velocity estimation. The focused controller mode, PAL interface and SEA
estimator tests are adapted from that same ROS branch. Historical measurements
and old solver/controller code are not copied.

Runtime requirements are NumPy, sourced ROS 2 `rclpy`, `builtin_interfaces`,
`controller_manager_msgs`, `rcl_interfaces`, `sensor_msgs`, `std_msgs` and
`trajectory_msgs`. The controller interface schema is explicitly PAL 4.32.2;
stock Humble controller-manager types are rejected before creating the ROS
node. Use a compatible installed PAL overlay, and select the intended DDS
domain/network before launching. This repository does not build that overlay
or select a physical robot network automatically.

Prefer sourcing ROS in the launch shell. `TIAGO_MPC_ROS_SETUP` (legacy
`GATO_ROS_SETUP` also accepted) can name a setup file for the inherited lazy
environment loader. It imports setup variables and can re-exec a script once
when native ROS libraries require startup environment changes. With neither
variable set, it tries `/opt/ros/humble/setup.bash`; this alone does not supply
the required PAL interface overlay.

## Feedback and the frequency fix

The old live process services just one `rclpy.spin_once` callback per command
tick. With both body `/joint_states` and high-rate SEA feedback ready, those
subscriptions compete for that single callback allowance; cached arm state can
then be reused between solves. The port's `spin_once()` services a bounded
burst: the first call can wait, and subsequent calls are nonblocking. It stops
when no state callback ran, after 64 state callbacks, or after a 2 ms drain
budget by default. `max_callbacks` and `drain_budget_sec` can be changed by the
I/O owner. A running callback may exceed the budget; it is not a hard deadline.
A non-state executor callback ends that burst, and the next poll resumes work.

The deterministic regression primes one encoder sample, then queues one body
and one SEA update before a command tick (consistent with depth-1 feedback).
The old implementation consumes only the body callback; the port also consumes
the arm update and establishes its velocity estimate.
This proves the callback-budget defect and its boundary fix. It does not prove
that this was the only cause of the historical roughly 41.5 Hz SEA throughput,
or increase a publisher configured for 50 Hz. No new physical rate was measured.

`latest_state()` is a cached nonblocking read. `read_state()` first polls
nonblocking, so a qualifying cached read adds no inherited 50 ms wait. It
retains cached-read behavior unless a condition is specified:

```python
state = arm.read_state(timeout_sec=2.0, max_age_sec=0.1)
next_state = arm.read_state(timeout_sec=0.5, after_seq=state.seq, max_age_sec=0.1)
```

`after_seq` requires an advancing arm observation, including while stationary;
`max_age_sec` limits elapsed monotonic receipt age. A zero timeout performs one
nonblocking poll. Missing controlled-arm velocity, malformed data, nonfinite
controlled positions/velocities and duplicate/reversed source stamps do not
refresh state. The ordinary legacy path used zero for missing velocity; the
port rejects that input explicitly. Restart the client after a ROS clock reset.
Uncontrolled wheel NaNs can remain in full body metadata. Source timestamps,
receipt timestamps and sequences remain distinct; solve rate is not fresh
measurement rate. No change-based history-frequency helper is ported.

Ordinary `/joint_states` retains the existing reliable depth-10 subscription.
Optional `sea_absolute=True` uses best-effort depth-1 for both body and SEA
subscriptions, preserving the prior adapter's QoS. SEA additionally requires
`sea_state_msgs/msg/SeaState`; its default topic is
`/arm_right_sea_state_broadcaster_controller/sea_joint_states`. It takes native
output encoder absolute position, derives velocity from advancing source
timestamps with the existing 10 ms time-constant filter, and requires two
recent samples after startup/gaps. Body freshness and arm freshness are both
needed for the merged state. No encoder or filter default is retuned here.

## Commands and controller lifecycle

Position trajectories and effort commands preserve ordering and reject wrong
sizes/nonfinite values. Multi-point trajectories start at `dt`, avoiding a
zero-time first point. These transport checks do not enforce physical limits.

The effort controller retains the existing robot-side name
`gato_arm_effort_forward_runtime`. Configuration loads and configures a public
forward effort controller. Switching verifies a unique known right-arm owner,
rejects unknown/mixed owners and incorrect claimed interfaces, uses one strict
activate/deactivate request, and confirms the resulting active mode. Handback
sets `set_last_command_interface_value_as_state_on_activation=False` before
reactivating `arm_right_controller`, so activation holds the measured pose
rather than an old trajectory. A topic subscriber alone does not establish
controller mode. Service timeouts do not expire actuator commands.

These operations are explicit methods. Constructing or closing the client
does not switch modes; closing destroys ROS resources. The application must
prime the effort stream, supervise stale states/commands and safety limits,
perform bounded zero-effort/handback on exit/error, and report unconfirmed
restoration. A publisher crash has no proven robot-side command timeout.
No physical execution is authorized or qualified by this port.

`active_arm_controller()` exposes the validated inventory read. An optional
`service_progress` callback lets the runner service state/safety/expiry checks
between bounded discovery, RPC and subscription wait slices. This uses the same
executor owner and does not cancel an already-dispatched RPC on interruption.

## Verification and safe inspection

From the repository root, with a sourced, compatible ROS environment:

```bash
python examples/ros2_read_state.py --samples 5
```

That example only subscribes and prints state; it never calls a controller
service or publishes a command. ROS-free unit tests use deterministic message,
executor and controller-manager doubles:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ros2_*.py
```

They cover callback contention/bounds, cache/freshness distinctions, native
coordinate order, command encoding, optional SEA estimation/gaps, PAL interface
rejection, controller exclusivity and confirmed handback. They do not establish
DDS delivery, installed PAL compatibility, actual callback rates, executor
latency, joint limit/collision safety or hardware restoration. Real ROS/PAL
simulation and supervised hardware validation remain pending. No GPU solver
change or benchmark comparison is part of this local adapter port. See
[the MPC integration proposal](ROS2_MPC_PROPOSAL.md) for how the unchanged
upstream `MPCController` is used by the implemented [supervised runner](ROS2_RUNNER.md).
