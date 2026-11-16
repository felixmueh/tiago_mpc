# Collision-safety coverage

[The safety worker](../tiago_mpc/safety.py) checks robot states independently
of the MPC in a spawned process. This module has not received the same review
as the core package. CPU tests cover its model and worker behavior; physical
execution remains unqualified. A completed MPC solve is not a collision check.

`RosPlant` requires the worker and waits for a successful check at startup and
before effort activation. Checks cover finite full-body state, right-arm joint
limits, at least 4 cm clearance for monitored pairs, and geometry speed at most
1 m/s, bounded by linear speed plus angular speed times geometry radius.
Faults latch; an outstanding check exceeding 50 ms or a dead worker faults the
run. Settings and lifecycle details are documented in `safety.py`.

The runtime model overlays 18 simplified alpha-wrap collision meshes onto the
canonical [full robot model](../models/README.md). The retained reviewed source
URDF supplies collision geometry only: its outdated kinematics are not used.
The blacklist is `models/safety/comfortable_pose_collision_blacklist.json`;
the worker never adds entries. Derived models and mesh copies go into the cache.

The model contains 51 collision geometries. Its 569 monitored pairs are the
non-blacklisted pairs touching geometry below a controlled right-arm joint,
including the right gripper. The rest of the body is an obstacle at its measured
pose. The runtime state requires positions and velocities for the torso, both
arms and head; wheel and finger joints are locked at their start values.

This is not a full-body self-collision check. It does not monitor world obstacles,
moving base or grippers. Unobserved gripper mimic followers remain neutral, so
finger geometry can be wrong for an open gripper. The comfortable arm pose with
all other body joints at zero fails the 4 cm margin; the actual full-body state
must pass without relaxing the margin or blacklist. Simulation does not run this
worker. See [robot operation](robot.md) for execution and handback limitations.
