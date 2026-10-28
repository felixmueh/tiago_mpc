# The robot path

`RosPlant` (`tiago_mpc/robot.py`) is the plant for the TIAGo right arm on the
robot or in Gazebo; `TiagoRightArmClient` (`tiago_mpc/ros2.py`) does the ROS 2
I/O. The robot runs the same experiments as simulation, through
`run_real_time` on wall time (the rehearsal, `--plant rehearsal`, runs a
`SimPlant` on the same loop). **Status:** tested on the CPU with fake
clients (`tests/test_robot.py`), never run against ROS, Gazebo or the robot.

## Running

Source ROS 2 and PAL's interface overlay in the launching shell and select the
intended DDS domain and network. In the container (README, "In the container")
`source container/ros_simulation.sh` or `source container/ros_tiago.sh` does
both; elsewhere use PAL's setup on the robot. The package never
loads ROS itself; constructing the client raises a `RuntimeError` if the ROS
packages are missing, or if `controller_manager_msgs` does not match PAL 4.32.2.

- **Check-only (default):** `python -m experiments.reach --plant robot`: the main's
  `setup` (`experiments/common.py`) builds `RobotConfig(execute=False)`'s `RosPlant`,
  runs `start(WallClock())` and `stop()`, prints its `run_state` and returns exit
  status 0. It reads the controller inventory but loads no controller, switches
  nothing, publishes nothing and builds no MPC. Nothing is saved.
- **Execution:** `--execute --target X_M Y_M Z_M` (target required). You type
  `EXECUTE` in `setup`, before the ROS client exists and before anything compiles;
  anything else returns exit status 2. Only then does `setup` return
  `RobotConfig(execute=True)` and `RealTime`. The main has installed the stop
  signals first (`handle_stop_signals`). Then `run_experiment` creates the run's
  record, compiles GATO's module unless it is cached and builds the `MpcProcess`
  (`MpcProcess.compile`), builds the `RosPlant` from `RobotConfig(execute=True)`, and
  `run_real_time` calls `plant.start(WallClock())`, so the startup
  checks see a state read after the build, then `mpc.warmup` from a state read
  after them (GATO's warm-up solve); meanwhile the arm stays under PAL's
  position controller. Ctrl-C or SIGTERM sets the stop flag: the loop finishes its
  tick, the next one sends nothing, and `stop()` zeroes the torque and hands back;
  the run is saved as `interrupted`. A second signal raises `KeyboardInterrupt`
  (for the blocking first-state wait or a compile); `stop()` still runs in the
  loop's `finally`. The MPC process and the safety worker ignore SIGINT.
  `run_experiment` closes the plant (and with it the safety worker), then the MPC.
- **`--duration S`** ends a run after at most S seconds (otherwise the reach
  times out after 8 s and the figure eight runs its 24 s period); use it for
  first runs.
- **Before execution:** base and grippers stationary, the arm at rest within
  0.05 rad of `Q_START` (`tiago.py`), `[-0.39, -1.73, -0.38, -2.33, 0, -1.21,
  0.04]` rad. That is PAL's comfortable pose with joint 4 at −2.33 instead of
  −2.35 rad, inside the MPC's joint box. PAL's own pose passes the check but
  starts joint 4 0.0065 rad outside that box. There is no automatic reset;
  check the path and the scene yourself.

## Processes

| Process | Runs |
|---|---|
| 1 (launching) | compiling GATO's module before the run (`compile_binary`); loop, task, law, the fault checks, `RunRecord`, `RosPlant` and the ROS client |
| 2 (spawned) | `MpcProcess`: loads that module, builds the GATO controller and solves on the GPU |
| 3 (spawned) | the safety worker (`AsyncSafetyMonitor`), owned by `RosPlant`: no robot run without it |

## Where each check lives

| Where | Check |
|---|---|
| `RosPlant(config)` | refuses a disabled safety worker or one monitoring other joints than the right arm |
| `RosPlant.start()` | first state within 180 s; safety worker started with it; PAL's `arm_right_controller` active (existing effort control is not adopted; mixed or unknown arm owners are rejected); with `execute`, the effort controller loaded and configured; then a newer state passes `state()`'s checks, lies within 0.05 rad of the comfortable pose at joint-speed L1 ≤ 1 rad/s, and the safety worker has checked it |
| `RosPlant.state()` | receipt age ≤ 100 ms; finite stamp; seven finite q, qd within the URDF position and speed limits; sequence never reversed; a new sample has a later ROS stamp (a clock reset faults); full named body state; every other body joint within 0.02 rad (torso 5 mm) of its start value; the state goes to the safety worker, whose faults latch |
| loop (`run_real_time`) | `plant.start(WallClock())` (any other clock raises before a ROS call) and `mpc.warmup` before the first tick; `task.done` and the stop flag before any command (a met goal never activates effort control); no command before the first plan; every knot of a plan within the raw effort limits when it arrives, before any of its knots is commanded (`plan_fault`); `plant.stop()` first on exit |
| `MpcProcess.result()` | every knot finite; expected shapes and knot interval; the plan answers the pending request; an error reply or a dead process raises, polled every tick |
| `lease_fault`, `command_fault` (`loops.py`) | before the law, the tick's time before the lease end t₀ + min(100 ms, plan length), t₀ the plan's state time; then the command finite, seven torques, within the raw effort limits; never clipped |

The loop's checks (`plan_fault`, `lease_fault`, `command_fault`) return a reason
(`effort_limit`, `lease_expired`, `invalid_command`): the tick sends nothing,
reaches the tick callbacks, and the run ends as `fault` with that reason. The plant's
checks (`RosPlant`, the safety worker) and the MPC's raise: the run ends as
`error`. Nothing retries. Limits are the full URDF's; the tolerances are fields
of `Checks` in `robot.py`.

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
  the measured pose. Errors are collected and raised after the `handback` event
  (`restored`, `errors`); `run_state` keeps `restored` and `cleanup_errors`, which
  the record saves as `metadata.plant`. An interrupted activation call leaves
  restoration uncertain (the server may still switch); `stop()` then raises and
  asks for manual controller reconciliation.
- **Events:** `activated` when the switch is done (`switch_s`, how long it took),
  `handback` in `stop()`, after the loop; `run_experiment` takes the latter after the
  loop returns, and the console prints both.
- **`close()`** stops the safety worker and the ROS client.

## Latency

A plan starts at its state's time t₀, and `SampleKnot` commands the knot due
at the tick's time t. A plan that first takes effect at t − t₀ ≥ dt (one
knot interval) never runs its knot 0. Consecutive plans differ mainly in that
knot, so with every plan that late the loop loses the feedback and can
diverge. Each such plan is a `late_plan` event in both loops; the console prints
the first.

- On wall time (`run_real_time`) a plan takes effect at the first command tick
  after it arrives. With command ticks of one knot (10 ms), any solve delays it by a
  full knot; 2 ms ticks (the default) keep a solve of a few milliseconds within
  knot 0.
- A start outside the joint box makes every solve work at the ADMM iteration
  cap until a plan's knot 0 runs. With the former ADMM settings (cap 100,
  eps_rel 1e-3) and PAL's comfortable pose, the solves took 42–49 ms, and on
  wall time the lease ended the run. `Q_START` and the current ADMM settings (GATO's, except rho_u) keep
  the solves short.
- Evidence: `research/tiago-mpc-latency-analysis.md` in the thesis workspace
  (simulation, 2026-10-22).

## Time

`RosPlant` runs on wall time only: `start(clock)` refuses anything but the
loop's `WallClock`. `State.t` is the state's monotonic receipt time, on the same
clock; it starts the plan and the reference phase. `run_real_time` reads the
clock at every tick and samples the plan, checks the lease and schedules with
it, so a repeated or stale state cannot extend a lease. The loop sets each
tick's deadline one command period after the previous one; `advance()` waits
until the clock reaches it. A tick that starts a command period or more late
skips the missed ones and is a `late_tick` event (`late_s`, `skipped`), counted in
the results (`late_ticks`); not the tick after the first command, which switches
to effort control through blocking service calls, whose duration the `activated`
event records. Freshness compares receipt times with
`time.monotonic()`. **Open for Gazebo:** a third clock fed from `/clock`, with
the ROS stamps as `State.t` (a comment in `RosPlant.__init__`); not implemented.

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

- No ROS graph and no robot reached yet. The ROS container exists
  (`container/`, README) and its ROS imports work, but nothing has run against a
  controller manager, Gazebo or the robot. The Gazebo check waits for Felix.
  There is no ROS installation on the host.
- Whether PAL's effort receiver expires commands when the publisher stops
  (TODO in `ros2.py`). SIGKILL or a host crash skips the handback; what the
  robot's onboard controllers then do is unknown.
- On wall time the plan in use expires 100 ms after its state; if the
  next plan arrives later, the run faults. With a `SimPlant` on wall time,
  the robot's loop and `MpcProcess`, two runs each (2026-10-23, desktop GPU), the
  default reach and the latency analysis' 31.7 cm far reach arrived within the
  limits with 2 ms command ticks (0.11 s and 1.64–1.69 s; median solve 3 ms, at
  most 5 ms). With 10 ms command ticks every plan took effect a knot late: the
  default reach arrived (0.30–0.31 s), the far reach diverged and faulted on a
  plan above the raw effort limit after 2.7 s.
- Startup needs a real full-body posture that passes the safety worker; the
  comfortable arm with every other joint at zero fails it ([safety.md](safety.md)).
