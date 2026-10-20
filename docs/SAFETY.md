# ROS-free collision safety worker

`tiago_mpc.safety` ports the old GATO `gato_tiago/safety_monitor.py` worker.
The runner must wait for a successful initial `wait_until_checked(state)` before
issuing effort and continue calling `check(latest_state)` independently of MPC
progress. A completed solver request is not a collision check. This module does
not move the robot, reset a posture, publish effort, or manage ROS controllers.

```python
from tiago_mpc.safety import AsyncSafetyMonitor, CollisionSafetySettings

monitor = AsyncSafetyMonitor(settings=CollisionSafetySettings(), initial_state=state)
try:
    monitor.wait_until_checked(state)  # Must succeed before effort starts.
    # In every control tick: monitor.check(latest_state)
finally:
    monitor.close()
```

`SafetyCheckState` retains `q`, `qd`, `stamp_sec`, `received_monotonic_sec`, `seq`,
`joint_positions`, and `joint_velocities`. The full named dictionaries determine
the collision configuration and velocity; the seven arm arrays alone are
insufficient. Each new observation needs a new sequence number. The runner is
responsible for input freshness and avoiding stale/replayed observations.

The API starts a separate Python **spawn** process, validates model creation,
then processes one outstanding check at a time. Startup has a 15 s initialization
timeout. `check` submits/polls without waiting for geometry computation;
`wait_until_checked` blocks for initial acceptance. Failures raise `RuntimeError`
and expose `safety_status`, `safety_fault`, and `safety_message`. Faults latch until
the monitor is closed/replaced. `close` is idempotent and joins or terminates the
child. Run entry points must use the normal `if __name__ == '__main__'` guard.

The preserved defaults are enabled safety, right-arm controlled joints, 0.04 m
minimum clearance, 0.05 s outstanding-check timeout, 1 m/s monitored geometry
speed bound, zero joint position margin, and URDF joint velocity limits at scale
1. Missing assets, incomplete/nonfinite named state, nonfinite timestamps and
invalid numeric settings fail closed. Startup/model/worker errors also fail
closed. Result classifications are `joint_limit`, `collision_margin`,
`collision_body_speed`, `state_invalid`, `worker_error`, and `worker_stale`.
Timeout status is `stale`; other check faults use status `fault`.

## Model and actual coverage

The reviewed `models/safety/collision_alpha_wrap_a3_o25_cap001/` URDF and its
simplified STL meshes, original alpha-wrap generation receipt, and
`comfortable_pose_collision_blacklist.json` are copied byte for byte. See
`models/safety/provenance.json` for the source commit and SHA-256 values. The
existing `.gitattributes` tracks all STL files with Git LFS. Materialize LFS files
before running safety. Historical absolute paths inside the copied generation
and review receipts remain provenance; runtime does not open those paths.

The original full URDF in `models/tiago_pro/tiago_pro.urdf` is unchanged. The
collision variant reuses original local meshes where it has no simplified mesh.
At model creation the port resolves mesh paths to this checkout and writes an
absolute-path runtime URDF under ignored `_expr_data/runtime/safety-model/`.
It never rewrites either source URDF. This works from other working directories;
there is no fallback to GATO or another checkout. Cache files are not portable
assets and are regenerated from the local source model.

The runtime is a fixed-base model with wheel joints and both grippers' finger
joints locked at initial named reference values when supplied, otherwise their
neutral references. It requires full position **and velocity** feedback for the
remaining 17 joints: torso lift, seven left arm, seven right arm, and two head
joints. It does not require wheel/finger feedback, and cannot qualify changing
finger positions or a moving mobile base.

All 47 collision geometries remain in the model. The default right-arm selection
checks the 513 nonblacklisted pairs touching geometry attached below a controlled
right-arm joint, including right-gripper geometry. Other body geometry provides
obstacles in these checks using the named torso/left-arm/head state. Pairs wholly
outside this monitored subtree are excluded. This is **not** a check of every
possible full-body self-collision, mobile navigation, or moving grippers; there
is no external scene/world obstacle model here. Position and velocity limit
checks apply to configured controlled joints. Geometry speed bounds use current
kinematics for each monitored geometry, `|linear_velocity| + |angular_velocity| *
radius`, with the radius computed from the loaded mesh vertices or primitive.
Unsupported/missing radius geometry fails model construction. The worker does
not add blacklist entries automatically.

## Verification and posture limitation

Run the bounded CPU checks with Pinocchio and NumPy installed:

```sh
python -m unittest discover -s tests -p 'test_safety*.py' -v
```

On 2026-10-20 all 14 checks passed using Python 3.10.12 and Pinocchio 3.9.0.
They cover real collision model construction outside the checkout working
directory, asset resolution, named-state completeness/finiteness, controlled
joint limits, geometry speed bounds against finite differences, and real spawned
worker fault/timeout/death handling. The healthy transport test uses an explicit
dummy checker in the child; actual geometry tests use the real checker and keep
the default 4 cm margin. No ROS, robot motion, or GPU was used.

The old `comfortable_high_clearance` start defines only right-arm values
`[-0.39, -1.73, -0.38, -2.35, 0, -1.21, 0.04]`. No complete recorded body posture
was available for qualification. With those arm values and every other updated
joint at zero, the real default check **fails**: base-link/right-arm pairs and
torso-base/right-arm-4 violate the 4 cm margin. This failure is retained as an
expected regression result; neither the margin nor blacklist was changed to
accept it. Seven arm angles cannot establish full-body clearance. The actual
robot's complete fresh named state must pass the initial gate before motion.
There is no qualified complete comfortable posture or physical deployment claim.

Raw verification logs and the explicit synthetic posture failure are retained in
`_expr_data/safety-port-20261020/`. The first test-development attempt is retained;
it incorrectly assumed that reducing a synthetic test's margin to zero supplied
a healthy geometry fixture. The final transport test isolates the checker
instead. No posture search, new blacklist decisions, solver changes or robot
experiments were performed.
