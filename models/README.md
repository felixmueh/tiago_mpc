# Model provenance

The canonical current robot description is Felix's simulator URDF,
`/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`, selected on 2026-10-20:

- `tiago_pro/tiago_pro_sim_source.urdf` preserves its exact original bytes,
  including copyright, comments, package references and original dynamics tags.
- `tiago_pro/tiago_pro.urdf` is that canonical full model with both arms'
  supplied friction/damping, effort and custom armature metadata patched.
  The previous full model from `/workspace/GATO/TiagoProURDF/tiago_pro.urdf`
  was outdated and has been superseded; its original remains in Git history.
- `tiago_right/tiago_right_arm.urdf` was already extracted from this simulator
  model. Its torso-relative kinematics, limits, inertias and zero-posture
  gripper inertia lump remain unchanged. Its seven joint parameter fields and
  explanatory comments are patched, and a massless nominal grasp frame is added.
  The original reduced source came from
  `/workspace/GATO/gato/dynamics/tiago_right/tiago_right_arm.urdf` at `fcc9621`.
- `tiago_pro/meshes/*.stl` retains the 33 original mesh assets copied from
  `/workspace/GATO/TiagoProURDF/meshes/`. The two additional canonical full-model
  references are supplied from pinned official PAL repositories, for 35 total:
  `base_antena_link.stl` from `omni_base_robot` at `1a66854`, and `sick_tim551.stl`
  from `pal_urdf_utils` at `e8600a3`. Exact URLs, source commits and license
  locations are in `tiago_pro/mesh_sources.json`. These are official assets
  supplying the canonical references; equivalence to an unavailable installed
  PAL package is not claimed.

All canonical full and reduced mesh references resolve by basename to the
included assets. Both full-model visual and collision geometry load with finite
geometry after resolving source URIs against that directory. Package/file
references remain unchanged in the canonical source; the reaching example resolves its reduced
model references against the included directory in a generated runtime copy.
A separate generated GPU-coordinate URDF reflects joint 6's axis/limits;
the adapter maps position, velocity and torque signs at the solver boundary.

## End-effector point

New reaching/tracking use `arm_right_grasp_joint`, a fixed, massless frame
parented to `arm_right_tool_link` at local XYZ `(0, 0, 0.139288457525)` m,
with the flange's orientation. The old `arm_right_tool_joint` remains the flange.
GATO selects the new point via its existing `ee_frame` argument; Pinocchio uses
the same named frame. No GRiD or upstream GATO change is needed.

This nominal TCP is the midpoint between the two inward fingertip-pad centers
at zero gripper posture, matching the posture used for inertia aggregation.
In `fingertip.stl`, the flat contact face is at local x=-11.5 mm, y=7…47 mm,
z=±10 mm, giving pad center `(-11.5, 27, 0)` mm. Transforming both centers
through the canonical simulator URDF gives the offset above (idealizing the
tiny symmetry error from rounded URDF rotations). It is a geometric convention,
not a calibrated hardware TCP or a midpoint recomputed as the gripper opens.

The full simulator models retain their existing frames and exact bytes from
before this change. The viewer derives the additional TCP from the reduced URDF
when needed. Replay exports identify `end_effector_frame`; missing metadata
means the historical flange. Old absolute XYZ goals must not be relabeled as
grasp-center goals. The frozen-input reach check rejects a changed target frame;
the historical random100 runner explicitly passes `--ee-frame arm_right_tool_joint`
to preserve its frozen goals. New tasks use the grasp center by default.

## Actual arm joint parameters — Felix, 2026-10-21

These supplied actual values replace the previous range midpoints and guessed
wrist coefficients. Standard `friction` is Coulomb c in Nm, `damping` is viscous
b in Nm s/rad, and `limit effort` is actuator torque in Nm.

| Joint | Left c | Left b | Left armature | Right c | Right b | Right armature | Effort (both) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.47 | 7.42 | 0.511 | 1.66 | 4.7 | 0.746 | 43 |
| 2 | 3.52 | 8.71 | 1.19 | 1.15 | 9.69 | 0.882 | 43 |
| 3 | 1.04 | 4.75 | 0.472 | 1.16 | 5.3 | 0.211 | 26 |
| 4 | 1.35 | 3.65 | 0.296 | 0.912 | 4.2 | 0.648 | 26 |
| 5 | 0.779 | 3.65 | 0.424 | 1.25 | 3.66 | 0.326 | 26 |
| 6 | 0.773 | 2.82 | 0.151 | 0.473 | 2.95 | 0.343 | 26 |
| 7 | 0.945 | 1.79 | 0.14 | 0.56 | 2.95 | 0.27 | 26 |

Armature is in kg m^2 and stored in a custom joint child,
`<tiago_mpc armature="..."/>`. This is metadata only: the current Pinocchio and
GRiD URDF loaders do not activate it, and the application does not read it.
It is not folded into link inertias. Supplied kp/kd are deliberately omitted:
they belong to a different controller.

URDF friction/damping fields also require explicit runtime enablement. Ordinary
reaching still ignores passive joint torques. Existing friction-study protocols
explicitly supply their own historical coefficients and override generated copies;
updating these canonical files does not change old protocols, raw runs or figures.
Use a new identified protocol/output when testing the actual values. Existing
no-friction qualification is not a qualification of the newly supplied losses.

Prepare the active full model and reapply the retained arm's metadata with:

```bash
.venv/bin/python scripts/prepare_models.py
.venv/bin/python scripts/prepare_models.py --check
```

The script uses the preserved canonical source, patches the fourteen arm
joints, adds/checks the reduced arm's grasp frame, and does not repeat the
already verified gripper reduction. It is idempotent and
updates the ordinary `manifest.json` identities for the three URDFs and included
meshes. The example verifies those assets before codegen. Run CPU model checks
with `.venv/bin/python -m unittest discover -s tests -v`. CPU checks cover
coefficient values, metadata behavior, unchanged kinematics/inertias,
mesh completeness, finite STL vertices and full geometry loading.

The canonical simulator source retains its PAL Robotics Apache-2.0 copyright
and license notice. Mesh rights remain with their original owners; copying does
not relicense assets. The two added official PAL assets include their upstream
Apache-2.0 licenses in `licenses/`; no original asset license file was present
in the initially supplied `TiagoProURDF` folder.
