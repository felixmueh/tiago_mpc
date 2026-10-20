# Model provenance

Copied on 2026-10-20 from the existing local `/workspace/GATO` setup:

- `tiago_pro/tiago_pro.urdf`: exact bytes of
  `/workspace/GATO/TiagoProURDF/tiago_pro.urdf`.
- `tiago_pro/meshes/*.stl`: exact original mesh bytes from
  `/workspace/GATO/TiagoProURDF/meshes/`; all 33 distinct full-URDF mesh
  references are included, and these also cover the reduced arm references.
- `tiago_right/tiago_right_arm.urdf`: exact bytes of
  `/workspace/GATO/gato/dynamics/tiago_right/tiago_right_arm.urdf` at fork
  commit `fcc9621` (the copied file is tracked there). This model fixes the
  arm base at `torso_lift_link` and lumps the gripper inertia at the tool.

`manifest.json` gives SHA-256 and size for both source URDFs and every mesh.
The example verifies these hashes before codegen. No mesh simplification,
axis flipping, re-export or changes to either checked-in URDF were made.
Only referenced STL geometry is included; Unity import caches, `.meta` files,
prefabs and derived assets are omitted.

The full original has `package://meshes/...` references; use
`models/tiago_pro` as the package search root in a compatible viewer. The
reduced URDF preserves its original PAL package URIs; the example resolves
those filenames against the same mesh directory in a generated runtime copy.
A separate generated GPU-coordinate URDF reflects joint 6's axis/limits;
the adapter maps position, velocity and torque signs at the solver boundary.
Neither checked-in URDF is altered by this conversion.

Robot-description and mesh rights remain with their original owners. Copying
them here does not relicense them. No original asset license file was present
in the supplied `TiagoProURDF` folder.
