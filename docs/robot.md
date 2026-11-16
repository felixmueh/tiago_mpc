# Robot operation

[RosPlant](../tiago_mpc/robot.py) runs the right arm through
[the ROS 2 client](../tiago_mpc/ros2.py) and `run_real_time`. It has CPU
fake-client coverage but has not been qualified against ROS, Gazebo or the
physical robot. The [collision worker](safety.md) is mandatory.

Source ROS 2 Humble and the PAL interface overlay in the launching shell.
The client requires PAL `controller_manager_msgs` 4.32.2; standard Humble
interfaces have incompatible layouts. See [interface provenance](../container/pal_interfaces/README.md).
The optional container is started with `container/docker.sh`; it uses host
networking and mounts the checkout's parent at `/workspace`. Its uv environment
is `.venv-container`. Source `container/ros_simulation.sh` for a same-host
simulator or `container/ros_tiago.sh` for the robot, after checking their DDS
and network settings for the intended system.

```bash
uv run python -m experiments.reach --plant robot
uv run python -m experiments.reach --plant robot --execute --target X_M Y_M Z_M --duration 1
```

The first command performs startup checks and reads controller inventory. It
sends no commands, switches no controllers, builds no MPC and saves no run.
Execution requires an explicit target and typing `EXECUTE` before compilation
or ROS client construction. Targets are metres in `torso_lift_link`.
`--plant rehearsal` uses a simulated plant on the same wall-time loop.

Before execution, keep the base and grippers stationary and place the arm at
rest within 0.05 rad of `Q_START` in [tiago.py](../tiago_mpc/tiago.py).
The full-body state must pass the collision worker; setting every other joint
to zero fails the clearance check. There is no automatic reset. Inspect the
scene, full motion and return path; world obstacles are not monitored.

The startup checks require PAL's `arm_right_controller` to own the arm.
The first command rechecks pose, speed and safety, then switches to
`gato_arm_effort_forward_runtime`. Subsequent commands publish seven native
joint torques. State freshness, finite solver results, plan leases and requested
and applied effort are checked during execution. Commands are never clipped.
Latency prediction preserves the original measurement age; it cannot renew a
lease. Exact checks and thresholds live in `robot.py`, `laws.py` and `loops.py`.

Ctrl-C or SIGTERM requests normal termination. After an activation attempt,
`stop()` publishes zero torque five times, 10 ms apart, and restores the
position controller to hold the measured pose. Inspect the `handback` event and
`metadata.plant.restored` / `cleanup_errors` before another run. An interrupted
switch can leave controller ownership uncertain and requires manual
reconciliation. SIGKILL and a host crash bypass cleanup; the effort receiver's
independent command-expiry behavior remains unestablished.

Activation is bounded by 100 ms from the tick's state, independently of the
plan's own lease; a shorter plan can expire during this operation. Gazebo
simulation-clock support is not implemented: this path uses monotonic wall
time. SEA feedback is opt-in (`--sea-absolute`), uses output-encoder positions
and a 10 ms velocity filter, and still needs live publisher interoperability
validation.
