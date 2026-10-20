# Agent instructions

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
