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
cache (`tiago_mpc/tiago.py:cache_root`), and never committed: the reduced right
arm (`arm_urdf`), its GPU copy (`runtime_urdf`) and the safety collision model.
`python -m tiago_mpc.tiago` prints the arm's and the GPU copy's paths.

## End-effector point

Tasks track `arm_right_grasp_joint` (`tiago.GRASP_FRAME`), a massless frame
that `arm_urdf` adds 0.139288457525 m along the flange's +z
(`arm_right_tool_joint`, which stays the flange). It is the midpoint of the two
fingertip pads' inner faces at the neutral gripper posture, the posture at
which the gripper's inertia is summed into the tool link: a geometric
convention, not a calibrated TCP, and it does not follow the opening gripper.
GATO selects it through `ee_frame`, Pinocchio by name; the full URDF does not
have it.

Replays record `end_effector_frame`; missing metadata means the flange. Old
absolute goals must not be relabeled as grasp-center goals; random100 keeps the
flange for its frozen goals.
