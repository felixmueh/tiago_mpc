# Model provenance

The canonical current robot description is Felix's simulator URDF,
`/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`, selected on 2026-10-20:

- `tiago_pro/tiago_pro_sim_source.urdf` preserves its exact original bytes,
  including copyright, comments, package references and original dynamics tags.
- `tiago_pro/tiago_pro.urdf` is that canonical full model with only the seven
  right-arm friction/damping tags and accompanying provenance comments patched.
  The previous full model from `/workspace/GATO/TiagoProURDF/tiago_pro.urdf`
  was outdated and has been superseded; its original remains in Git history.
- `tiago_right/tiago_right_arm.urdf` was already extracted from this simulator
  model. Its torso-relative kinematics, limits, inertias and zero-posture
  gripper inertia lump remain unchanged. Only its seven dynamics tags and
  explanatory comments are patched. The original reduced source came from
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

## Right-arm dynamics metadata

`friction` is the Coulomb coefficient c in Nm; `damping` is the viscous
coefficient b in Nm s/rad. Values came from Felix's identification notes and
the explicit illustrative wrist choices retained by the friction study:

| Right joint | c (Nm) | b (Nm s/rad) | Evidence / assumptions |
| --- | ---: | ---: | --- |
| 1 | 2.1 | 3.35 | User c estimate; b midpoint of 1.8–4.9, 2 runs |
| 2 | 2.6 | 4.15 | Midpoints c 2.3–2.9 / b 2.7–5.6, 8 runs |
| 3 | 2.0 | 2.45 | Midpoints c 1.8–2.2 / b 1.3–3.6, 8 runs |
| 4 | 1.95 | 2.25 | Midpoints c 1.8–2.1 / b 1.1–3.4, 8 runs |
| 5 | 0.8 | 0.5 | Assumed c above user's >0.7 lower bound; b assumed, unmeasured |
| 6 | 1.1 | 0.3 | User c approximately 1.1, one run; b assumed, unmeasured |
| 7 | 0.3 | 0.15 | Both assumed, unmeasured |

These are nominal metadata, not complete physical calibration or a stiction
model. Left-arm values are preserved exactly from the simulator source; no
symmetry assumption is applied. The current reaching application uses ordinary
Pinocchio ABA and the standard GATO/GRiD generator with `USE_JOINT_DYNAMICS=False`;
it does not apply these passive torques. This patch does not enable the earlier
experimental hard-Coulomb model or change the recorded 84/100 no-friction cohort.

Prepare the active full model and reapply the retained arm's metadata with:

```bash
.venv/bin/python scripts/prepare_models.py
.venv/bin/python scripts/prepare_models.py --check
```

The script uses the preserved canonical source, modifies no other joints and
does not repeat the already verified gripper reduction. It is idempotent and
updates the ordinary `manifest.json` identities for the three URDFs and included
meshes. The example verifies those assets before codegen. Run CPU model checks
with `.venv/bin/python -m unittest discover -s tests -v`. Fourteen checks pass,
including mesh completeness, finite STL vertices and full geometry loading.

The canonical simulator source retains its PAL Robotics Apache-2.0 copyright
and license notice. Mesh rights remain with their original owners; copying does
not relicense assets. The two added official PAL assets include their upstream
Apache-2.0 licenses in `licenses/`; no original asset license file was present
in the initially supplied `TiagoProURDF` folder.
