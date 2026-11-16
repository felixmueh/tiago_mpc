# TIAGo Pro model

`tiago_pro.urdf` is PAL Robotics' simulator URDF (xacro output of
`tiago_pro_description/robots/tiago_pro.urdf.xacro`, Apache-2.0, notice kept in
the file). Its original bytes are `models/tiago_pro/tiago_pro_sim_source.urdf`
in commit `d86e041`, copied from GATO's `sim_tiago_urdf.xml` at `3303eaf`
(SHA-256 `c8d51a1e…07d3e6`). Only the arm joints' `<dynamics>` differ: the
Coulomb friction c (Nm) and viscous damping b (Nm s/rad) below replace PAL's
1.0. The right arm's values are fitted (2026-10-30), the left arm's are still
the supplied ones. The effort limits (Nm) were written into the
source file; they equal PAL's own TIAGo Pro arm limits (`pal_sea_arm`
`ab41a83`, `arm_limits/tiago-pro/arm.limits.xacro` and `spherical-wrist.limits.xacro`).

| Joint | Left c (supplied) | Left b (supplied) | Right c (fitted) | Right b (fitted) | Effort |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.47 | 7.42 | 2.06 | 4.96 | 43 |
| 2 | 3.52 | 8.71 | 2.25 | 6.67 | 43 |
| 3 | 1.04 | 4.75 | 1.93 | 3.21 | 26 |
| 4 | 1.35 | 3.65 | 1.62 | 2.66 | 26 |
| 5 | 0.779 | 3.65 | 1.78 | 3.57 | 26 |
| 6 | 0.773 | 2.82 | 1.03 | 1.86 | 26 |
| 7 | 0.945 | 1.79 | 0.94 | 2.16 | 26 |

The armature (kg m², joints 1 to 7) is supplied separately from the URDF. The right arm's fitted values, 0.43,
0.43, 0.19, 0.17, 0.21, 0.17, 0.18, are checked in as `ARMATURE_KG_M2` in
[`tiago_mpc/tiago.py`](../../tiago_mpc/tiago.py), beside the manually editable
`PD_KP` and `PD_KD` gains. `MpcParams.armature`, `SimConfig.armature` and
`LatencyCompensation.armature` default to it; `(0.,) * 7` turns it off.
Pinocchio receives its `model.armature` property; the predictor requires the
released GRiD/GATO armature support. Nothing is added to the XML or folded into
link inertias. The left arm's supplied armature is 0.511, 1.19, 0.472, 0.296,
0.424, 0.151, 0.14.

**Where the right arm's values come from.** They are fitted from one real
single-joint chirp per joint in Björn Plonka's realtoisaacsim (`de23016`; PAL's
position control, logged effort a fixed multiple of motor current): the logged
effort minus this URDF's rigid-body torque (Pinocchio), fitted as
I·q̈ + c·sign(q̇) + b·q̇ plus a dropped offset; the armature is the total
inertia I minus the link inertia. On a held-out trajectory with all joints
moving, they predict the effort better than the supplied values on every joint
(0.13–1.51 against 0.42–1.84 Nm RMS). Coulomb on joints 1 to 3 agrees with the
physical fit of 2026-10-16 through the torque interface (2.01, 2.39, 2.03 Nm).
The identification recordings and full analysis are not distributed here.

**Their limits:**
- One chirp per joint, logged at 40–77 Hz; the armature shifts by 0.03–0.10
  kg m² with a ±10 ms effort timing shift and is not separated from link
  inertia errors.
- Fitted between 0.02 and 0.87 rad/s. Breakaway is not observable in the
  chirps; separate standstill data show static friction 10–25 % above c on joints 2
  and 4, which `StickSlip` does not model.
- Unexplained: a pose-dependent offset of about ±1 Nm on joints 1 and 2 and a
  −0.35 Nm bias on joint 4.

**The supplied values** (right c = 1.66, 1.15, 1.16, 0.912, 1.25, 0.473, 0.56 Nm;
b = 4.7, 9.69, 5.3, 4.2, 3.66, 2.95, 2.95 Nm s/rad; armature 0.746, 0.882,
0.211, 0.648, 0.326, 0.343, 0.27 kg m²) were supplied for simulator matching; their physical interpretation has not
been established here. They remain useful as a plant that differs from the predictor.

**PD gains:** the stored PD default is the 20 rad/s, damping-ratio-1.4
simulation tune with the fitted armature; it is not
hardware-qualified. `util.inertia_gains` returns literal float tuples to copy
into these assignments, without writing them.

**Predictor stiffness:** The linearized explicit friction update requires
dt < 2/λ_max(M⁻¹ diag(b + c/w)). With the fitted values and armature this is
17 ms at the default width w = 0.1 rad/s (5.8 ms at w = 0.03, 30 ms at w = 0.2),
the minimum over 500 poses. With link-only inertia it is below 0.5 ms at any
width.

`meshes/`: 33 files from GATO's `TiagoProURDF/meshes/` and two from PAL
repositories ([`mesh_sources.json`](mesh_sources.json), licenses in
`../licenses/`). Mesh rights stay with their owners.
