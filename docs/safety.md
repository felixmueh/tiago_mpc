# The collision-safety worker

> **Awaiting Felix's review.** This describes `tiago_mpc/safety.py` as it is:
> ported from the old GATO `gato_tiago/safety_monitor.py`, last changed by the
> review fixes of 2026-10-21, and not reviewed in the refactor.

The safety worker checks every robot state for joint limits, clearance and
geometry speed, in its own spawned process, independently of the MPC. A
completed solve is not a collision check. The worker does not move the robot,
publish commands or manage controllers.

## Who runs it

Safety is the robot plant's job. `RosPlant` cannot be created without the
worker and starts it in `start()` with the first state, so no robot run happens
without it ([robot.md](robot.md)). `RosPlant.state()` submits every
state; `RosPlant` waits for a successful check of the start state, and again
right before activation. There is no loop-level safety hook; simulation runs no
safety check (a report-only one is deferred).

```python
monitor = AsyncSafetyMonitor(settings=CollisionSafetySettings(), initial_state=state)
monitor.wait_until_checked(state)   # blocks until this state passed
monitor.check(latest_state)         # every tick: submit, poll, raise on a fault
monitor.close()
```

- **Startup:** the worker builds the collision model and must report ready
  within 15 s; a failed build raises.
- **One check is outstanding at a time.** `check` submits a state with a new
  sequence number when none is pending and polls replies without waiting. A
  check pending longer than `check_timeout_sec` (50 ms) faults as stale, and the
  deadline is enforced before replies are read, so a late success cannot clear
  it. A dead worker faults at the next call.
- **Faults latch** until the monitor is closed: every later call raises.
  `safety_status`, `safety_fault` and `safety_message` describe the fault.
- **Sequence numbers:** `SafetyCheckState` carries q, qd, both timestamps, `seq`
  and the full named positions and velocities. With SEA feedback, body-only
  updates get their own safety sequence, so they are checked without making old
  arm data fresh. Freshness itself is `RosPlant`'s job.

## What each check tests

In order; the first failure is the fault:

1. finite timestamps, and finite positions and velocities for every joint of
   the runtime model (`state_invalid`);
2. position and speed limits of the controlled right-arm joints from the URDF
   (`joint_limit`);
3. clearance above 4 cm for every monitored collision pair
   (`collision_margin`);
4. a speed bound of at most 1 m/s for every monitored geometry,
   |linear velocity| + |angular velocity| × radius, with the radius from its mesh
   vertices or primitive (`collision_body_speed`).

Errors inside the worker are `worker_error`; timeouts are `worker_stale`.

| `CollisionSafetySettings` | Default |
|---|---|
| `min_distance_m` | 0.04 |
| `check_timeout_sec` | 0.05 |
| `max_monitored_geometry_speed_m_s` | 1.0 |
| `joint_position_margin_rad` | 0 |
| `joint_velocity_scale` | 1 |
| `controlled_joint_names` | the seven right-arm joints |
| `urdf_path`, `blacklist_path` | None (the overlay below), the blacklist below |

## Model

- **Source:** `models/tiago_pro/tiago_pro.urdf` with the collision elements of
  each link in the reviewed alpha-wrap model replaced by that model's, with its
  18 simplified meshes (`safety.collision_urdf_tree`, at run time). The reviewed
  URDF has outdated kinematics; it is kept byte for byte as
  `models/safety/collision_alpha_wrap_a3_o25_cap001/tiago_pro_collision_alpha_wrap_a3_o25_cap001_reviewed_source.urdf`
  (provenance: [models/README.md](../models/README.md)).
- **Blacklist:** `models/safety/comfortable_pose_collision_blacklist.json`,
  unchanged; the worker never adds entries.
- **Runtime model:** the overlay, its meshes as cached copies (`tiago.cache_meshes`), written to the cache
  (`<cache_root>/models/<hash>/safety_collision.urdf`); fixed base; wheel and finger joints locked
  at their start values (neutral when unobserved). It needs positions and
  velocities of the remaining 17 joints: torso lift, both arms, two head joints.
- **Coverage:** all 51 collision geometries are in the model; the 569 checked
  pairs are those not blacklisted that touch geometry below a controlled
  right-arm joint, including the right gripper. The rest of the body is an
  obstacle at its measured pose.

## Limits

- No full-body self-collision check, no world or scene obstacles, no moving
  base or grippers.
- Gripper mimic followers without feedback stay at neutral with a warning, so
  finger geometry can be wrong for an open gripper (accepted by Felix,
  2026-10-21).
- The comfortable arm pose with every other body joint at zero fails the 4 cm
  margin (base and torso against the right arm). The margin and blacklist were
  not changed for it; the robot's real full-body state must pass at startup.

`tests/test_safety_monitor.py` covers the model (sizes, provenance, meshes),
state completeness, limits, speed bounds against finite differences, the
zero-posture failure, and the spawned worker's faults, timeouts and death, on
the CPU without ROS.
