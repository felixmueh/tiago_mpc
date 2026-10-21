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

## Experiment storage and presentation — Felix, 2026-10-21

ALL new experiment data belongs in this checkout/worktree's `_expr_data/`,
including raw results, logs, intermediate analysis, archives/backups, and final
reports/figures. Keep `_expr_data/` ignored by Git. The explicit exception is
portable replay handoffs: use **`impl/_expr_data/ingest/` on the host** and
**`/workspace/_expr_data/ingest/` in the container** (the same mounted directory).
This replaces the proposed shared `_expr_results` name. Use study/cohort
subfolders and descriptive, distinct `.npz` filenames; do not put full raw
archives or logs in the ingest directory. Preserve existing historical paths
and recordings instead of moving or rewriting them.

Do not write new data to `/workspace/experiment_data/` or GATO's
`thesis_results/` for this non-thesis work.

`/workspace/experiment_data_ma_frozen_archive/` is the frozen MASTER THESIS
archive (formerly `/workspace/experiment_data/`). Preserve historical raw data
and provenance; do not recreate the old directory or a compatibility alias.
Existing old path strings in raw records are historical, resolved using the
storage relocation manifest. This policy overrides older output instructions.

Unless Felix explicitly requests a particular plot, make the default handoff:

- Export recorded motion with `experiments/tools.py:save_replay` into the shared
  ingest directory. Use native joints, actual timestamps and the task reference;
  include saved predictions and recorded velocity/actuator torque/limits when
  available. These telemetry fields are optional. Do not invent motion or rerun retained
  experiments just to make a replay. The visualizer's export cookbook is
  `../tiago_mpc_visualizer/docs/replay-format.md` from the main checkout.
- Give the viewer URL (normally `http://192.168.178.82:8090`) and exact library
  path(s), such as `reaching/study/run_017`, so Felix can inspect them immediately.
- State the short finding, relevant recorded metrics/outcomes and material
  limitations. Preserve failures and original labels; selected replays do not
  replace aggregate cohort outcomes. Clearly identify simulation versus robot
  data, predicted versus executed motion, and the time basis.
- Link the raw source run and effective configuration/protocol; retain model
  and source identities. Explicitly label display-only subsampling/conversion
  and keep the complete raw measurements in their original owning checkout.

For timing/memory or other studies without recorded motion, use a compact
metrics table plus raw-output links. Produce the specific requested plot when
Felix asks for one; otherwise keep the replay/summary handoff concise.

Use agents liberally for useful bounded tasks; root coordinates interpretation,
shared edits and sequential GPU use. Preserve scientific protocols and failures.

## Worktree cleanup — Felix, 2026-10-21

Queue worktree cleanup after merging branches. At the end of the work session,
inspect loose changes/results and propose the exact paths and branches to
remove, stating where useful work is retained. Do not remove worktrees or delete
branches automatically; wait for approval. Do not interrupt ongoing tasks with
cleanup proposals.
