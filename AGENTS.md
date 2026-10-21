# Agent instructions

## Canonical TIAGo model and joint parameters — Felix, 2026-10-21

The current robot description is the simulator URDF originally captured at
`/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`. Preserve its exact source as
`models/tiago_pro/tiago_pro_sim_source.urdf`; the active full model is
`models/tiago_pro/tiago_pro.urdf`, with the explicit both-arm parameter patch.
`/workspace/GATO/TiagoProURDF/tiago_pro.urdf` is OUTDATED. Do not use that model
as the reference for extracting arms, inertias, limits, or new experiments.

The existing seven-joint arm reduction already comes from the correct simulator
URDF. Its fixed torso root and zero-position gripper inertia aggregation are
intentional. Apparent Euler-angle branches against the outdated model do not
justify extra frame changes. GRiD supports native signed axes; the pinned GATO
fixed-arm adapter still needs our generated joint-6 axis/limit conversion and
matching q/v/u sign mapping only at the GPU boundary. ROS and saved states stay
in native coordinates. Do not remove that workaround on GRiD-only evidence.

Felix supplied actual left/right arm values on 2026-10-21, superseding all
previous range midpoints and guessed wrist values. `scripts/prepare_models.py`
owns the exact table and reproducibly patches both arms in the active full URDF
and the right-arm reduction. See `models/README.md` for the complete table.
Use standard `dynamics` friction/damping and `limit` effort fields. Store armature
in a custom `<tiago_mpc armature="..."/>` joint child (kg m^2), explicitly metadata
only: neither current URDF loader enables it, and no runtime reads that tag.
Do not store kp/kd: Felix says those belong to a different controller.
Do not silently add armature to link inertia or enable it only in one model.

Existing experiment protocols and saved cohorts retain their historical
coefficients. Their explicit configuration can override canonical URDF losses;
new actual-value experiments require a separately identified protocol/output.
Do not relabel old figures or overwrite raw data with new coefficients.

URDF tags alone do not activate passive forces in Pinocchio ABA or the default
GRiD codegen path. Keep runtime loss enablement explicit, avoid double counting,
and retain the recorded hard-Coulomb startup failures. A metadata patch is not
a claim that matched friction MPC is qualified. Preserve historical models,
cohorts and outcomes rather than relabeling or silently regenerating them.

## Experiment storage — Felix, 2026-10-20

ALL new experiment data belongs in this checkout/worktree's `_expr_data/`,
including raw results, logs, intermediate analysis, archives/backups, and final
reports/figures. Keep `_expr_data/` ignored by Git. Do not write new data to
`/workspace/experiment_data/`, any shared sibling data directory, or GATO's
`thesis_results/` for this non-thesis work.

`/workspace/experiment_data_ma_frozen_archive/` is the frozen MASTER THESIS
archive (formerly `/workspace/experiment_data/`). Preserve historical raw data
and provenance; do not recreate the old directory or a compatibility alias.
Existing old path strings in raw records are historical, resolved using the
storage relocation manifest. This policy overrides older output instructions.

Use agents liberally for useful bounded tasks; root coordinates interpretation,
shared edits and sequential GPU use. Preserve scientific protocols and failures.
