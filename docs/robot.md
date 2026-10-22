# The robot path

`RosPlant` (`tiago_mpc/robot.py`) is the plant for the TIAGo right arm on the
robot or in Gazebo; `TiagoRightArmClient` (`tiago_mpc/ros2.py`) does the ROS 2
I/O. The robot runs the same experiments and the same loop as simulation,
through `run_robot` on the wall clock. **Status:** tested on the CPU with fake
clients (`tests/test_robot.py`), never run against ROS, Gazebo or the robot.

## Running

Source ROS 2 and PAL's interface overlay in the launching shell (for example
`source /opt/ros/humble/setup.bash` plus the overlay, or PAL's setup on the
robot), and select the intended DDS domain and network. The package never
loads ROS itself; constructing the client raises a `RuntimeError` if the ROS
packages are missing, or if `controller_manager_msgs` does not match PAL 4.32.2.

- **Check-only (default):** `experiments/reach.py --plant robot` creates
  `RosPlant(execute=False)`, runs `start()` and `stop()`, and prints the checked
  settings. It reads the controller inventory but loads no controller, switches
  nothing, publishes nothing and builds no MPC. Nothing is saved.
- **Execution:** `--execute --target X_M Y_M Z_M` (target required). You type
  `EXECUTE` before the ROS client exists. Then `launch` (`experiments/common.py`)
  builds the `MpcProcess`, and `run_robot` calls `plant.start()`, so the startup
  checks see a state read after the build; meanwhile the arm stays under PAL's
  position controller. SIGTERM ends the run like Ctrl-C. `launch` closes the
  MPC and the plant at the end.
- **`--duration S`** ends a run after at most S seconds (otherwise the reach
  times out after 8 s and the figure eight runs its 24 s period); use it for
  first runs.
- **Before execution:** base and grippers stationary, the arm at rest at the
  comfortable pose `[-0.39, -1.73, -0.38, -2.35, 0, -1.21, 0.04]` rad. There is
  no automatic reset; check the path and the scene yourself.

## Processes

| Process | Runs |
|---|---|
| 1 (launching) | loop, task, law, `guard`, `Record`, `RosPlant` and the ROS client |
| 2 (spawned) | `MpcProcess`: builds the GATO controller and solves on the GPU |
| 3 (spawned) | the safety worker (`AsyncSafetyMonitor`), owned by `RosPlant`: no robot run without it |

## Where each check lives

| Where | Check |
|---|---|
| `RosPlant(...)` | refuses a disabled safety worker or one monitoring other joints than the right arm |
| `RosPlant.start()` | first state within 180 s; safety worker started with it; PAL's `arm_right_controller` active (existing effort control is not adopted; mixed or unknown arm owners are rejected); with `execute`, the effort controller loaded and configured; then a newer state passes `state()`'s checks, lies within 0.05 rad of the comfortable pose at joint-speed L1 ≤ 1 rad/s, and the safety worker has checked it |
| `RosPlant.state()` | receipt age ≤ 100 ms; finite stamp; seven finite q, qd within the URDF position and speed limits; sequence never reversed; a new sample has a later ROS stamp (a clock reset faults); full named body state; every other body joint within 0.02 rad (torso 5 mm) of its start value; the state goes to the safety worker, whose faults latch |
| loop (`run`) | `plant.start()` before the first tick; `task.done` before any command (a met goal never activates effort control); no command before the first plan; `plant.stop()` first on exit |
| `MpcProcess.result()` | every knot finite and within the raw effort limits; expected knot interval; the plan answers the pending request; an error reply or a dead process raises, polled every tick |
| `guard` (`loops.py`) | command finite and within the raw effort limits; never clipped; execution time before the lease end t₀ + min(100 ms, plan length), t₀ the plan's state time |

The checks raise; nothing retries. Limits are the reduced arm URDF's; the
tolerances are fields of `Checks` in `robot.py`.

## Activation and handback

- **The first `apply()`** checks the pose, speed and safety again, publishes the
  command to the inactive effort receiver, and switches PAL's
  `arm_right_controller` to `gato_arm_effort_forward_runtime` (a
  `ForwardCommandController` on the seven effort interfaces, joint order
  verified). The switch makes blocking service calls of up to 0.5 s each. Between
  ≤10 ms waits it reads and checks a fresh state and republishes the command.
  It must finish within 100 ms of the tick's state; otherwise activation faults.
  - The bound is 100 ms, not the plan's own lease, which `RosPlant` cannot see
    and which is shorter for knots under about 3 ms (62 ms for 31 knots of
    2 ms). Felix left it as is (2026-10-22), to review with the safety work.
- **Later `apply()`** calls publish the command.
- **`stop()`** after an activation attempt: five zero-torque publications 10 ms
  apart, then the switch back to `arm_right_controller`, which first gets
  `set_last_command_interface_value_as_state_on_activation=False` so it holds
  the measured pose. Errors are collected and raised; `restored` is recorded.
  An interrupted activation call leaves restoration uncertain (the server may
  still switch); `stop()` then raises and asks for manual controller
  reconciliation.
- **`close()`** stops the safety worker and the ROS client.

## Clock

`RosPlant` runs on `Clock.wall()`. `State.t` is the state's monotonic receipt
time, in the same domain; it starts the plan and the reference phase. The loop
reads the execution time `plant.now()` at every tick and samples the plan,
checks the lease and schedules with it, so a repeated or stale state cannot
extend a lease. The loop sets each tick's deadline one command period after the
previous one (a late tick skips the missed ones); `advance()` waits until the
clock reaches it. Freshness compares receipt times with `Clock.wall_time()`.
**Open for Gazebo:** `Clock.external()` fed from `/clock`, with the ROS stamps
as `State.t` (a comment in `RosPlant.__init__`); not implemented.

## ROS client

- **Port** of GATO `felix-feature/ros-tiago` `c4cf5cc`
  (`tiago_src/gato_tiago/ros_tiago.py`). Native joint coordinates; joint 6 is
  not reflected here.
- **Feedback:** `/joint_states`, reliable, depth 10. `--sea-absolute` adds the
  SEA output-encoder positions from
  `/arm_right_sea_state_broadcaster_controller/sea_joint_states` (needs
  `sea_state_msgs`), velocity from a 10 ms filter, and best effort depth 1 for
  both topics. Missing or nonfinite arm velocity never refreshes the state.
- **`spin_once()`** services a burst of up to 64 state callbacks or 2 ms, so a
  body update cannot starve the arm update. The arm and body each have their own
  freshness; SEA body-only updates get their own safety sequence.
- **Commands:** `std_msgs/Float64MultiArray` on
  `/gato_arm_effort_forward_runtime/commands`, seven torques in joint order.

## Not established

- No ROS installation on the host and no ROS container yet (deferred). The
  Gazebo check waits for Felix.
- Whether PAL's effort receiver expires commands when the publisher stops
  (TODO in `ros2.py`). SIGKILL or a host crash skips the handback; what the
  robot's onboard controllers then do is unknown.
- On the wall clock the plan in use expires 100 ms after its state; if the
  next plan arrives later, the run faults. The default reach with
  `SimPlant(clock=Clock.wall())` faulted this way after about 0.35 s
  (0.34–0.38 s in two runs), when one solve hit the ADMM iteration cap (46–61 ms);
  `--latency measured` reproduces it (2026-10-22).
- Startup needs a real full-body posture that passes the safety worker; the
  comfortable arm with every other joint at zero fails it ([safety.md](safety.md)).
