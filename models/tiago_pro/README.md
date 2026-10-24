# TIAGo Pro model

`tiago_pro.urdf` is PAL Robotics' simulator URDF (xacro output of
`tiago_pro_description/robots/tiago_pro.urdf.xacro`, Apache-2.0, notice kept in
the file). Its original bytes are `models/tiago_pro/tiago_pro_sim_source.urdf`
in commit `d86e041`, copied from GATO's `sim_tiago_urdf.xml` at `3303eaf`
(SHA-256 `c8d51a1e…07d3e6`). Only the arm joints' `<dynamics>` differ: the
Coulomb friction c (Nm) and viscous damping b (Nm s/rad) below replace PAL's
1.0. The effort limits (Nm) are the supplied values too; Felix had already
written them into the source file, so they are not PAL's.

| Joint | Left c | Left b | Right c | Right b | Effort |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.47 | 7.42 | 1.66 | 4.7 | 43 |
| 2 | 3.52 | 8.71 | 1.15 | 9.69 | 43 |
| 3 | 1.04 | 4.75 | 1.16 | 5.3 | 26 |
| 4 | 1.35 | 3.65 | 0.912 | 4.2 | 26 |
| 5 | 0.779 | 3.65 | 1.25 | 3.66 | 26 |
| 6 | 0.773 | 2.82 | 0.473 | 2.95 | 26 |
| 7 | 0.945 | 1.79 | 0.56 | 2.95 | 26 |

The supplied armature (kg m², joints 1 to 7) is not in the URDF, because its
loaders (Pinocchio, GRiD 0.5.5) do not read it: left 0.511, 1.19, 0.472, 0.296,
0.424, 0.151, 0.14; right 0.746, 0.882, 0.211, 0.648, 0.326, 0.343, 0.27.

**The values are questionable:** supplied on 2026-10-21, not identified. The
physical fit of 2026-10-16 (right arm, joints 1 to 4) gives c = 2.01, 2.39,
2.03, 1.91 Nm and b = 4.44, 5.07, 2.87, 2.82 Nm s/rad; joints 5 to 7 and the
left arm have no fit. With link-only inertia, the supplied damping makes joint 7
stiff (time constant 0.26 ms; `tiago_mpc_visualizer/docs/friction.md`).

`meshes/`: 33 files from GATO's `TiagoProURDF/meshes/` and two from PAL
repositories ([`mesh_sources.json`](mesh_sources.json), licenses in
`../licenses/`). Mesh rights stay with their owners.
