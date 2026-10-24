# Models

- `tiago_pro/`: the full robot, `tiago_pro.urdf`, the only robot description
  here, with its meshes. Origin and the values written into it:
  [tiago_pro/README.md](tiago_pro/README.md).
- `safety/`: the safety check's collision inputs ([docs/safety.md](../docs/safety.md)).
  The reviewed alpha-wrap collision URDF (`…_reviewed_source.urdf`, renamed),
  its 18 meshes, `alpha_wrap_results.json` and the blacklist are byte for byte
  GATO's `tiago_configs/` at `fcc9621`, where `tiago_mpc/safety.py` was ported
  from `tiago_src/gato_tiago/safety_monitor.py`.
- `licenses/`: upstream licenses of the two meshes from PAL repositories.

Every other robot file is derived from `tiago_pro.urdf` at run time, into the
cache (`tiago_mpc/util.py:cache_root`), and never committed: the reduced right
arm (`arm_urdf`), its GPU copy (`runtime_urdf`) and the safety collision model.
`python -m tiago_mpc.tiago` prints the arm's and the GPU copy's paths.

The derived URDFs reference copies of their meshes in the cache, named by their
contents (`models/meshes/<sha256>.stl`), by relative paths
(`../meshes/<sha256>.stl` from `models/<hash>/<name>.urdf`; `util.cache_meshes`).
GATO spherizes the GPU copy's collision meshes into its module, but its code
generation key hashes only the URDF's bytes: with meshes named by their contents,
a changed mesh changes those bytes, and with relative paths the bytes and the
plant names do not depend on the checkout's path. The GPU copy keeps the
collision elements and drops the visual ones, which GATO never reads. Copying a
mesh that is still a Git LFS pointer fails (`git lfs pull`).

## End-effector point

Tasks track `arm_right_grasp_joint` (`tiago.GRASP_FRAME`), a massless frame
that `arm_urdf` adds 0.139288457525 m along the flange's +z
(`arm_right_tool_joint`, which stays the flange). It is the midpoint of the two
fingertip pads' inner faces at the neutral gripper posture, the posture at
which the gripper's inertia is summed into the tool link: a geometric
convention, not a calibrated TCP, and it does not follow the opening gripper.
The MPC always tracks it (there is no end-effector setting); GATO selects it through `ee_frame`, Pinocchio by name; the full URDF does not
have it.

Every task measures it through `tiago.ee_position`, including random100 (the
grasp-frame positions at its frozen goal postures) and the latency study's far
reach.
Replays record `end_effector_frame`; missing metadata means the flange. Old
flange-goal results, such as earlier random100 runs, must not be relabeled as
grasp-center results.
