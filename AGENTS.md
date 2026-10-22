# Agent instructions

Read `README.md` first. Felix's dated decisions, deferred work and open
questions are in `docs/decisions.md`; add an entry there when he decides
something lasting. The robot path is in `docs/robot.md`, the safety worker in
`docs/safety.md`.

## Keep it simple (Felix, 2026-10-22)

- Simpler and smaller while staying correct. Every piece and method earns its
  place; prefer deleting to adding.
- Plain functions and small dataclasses. No frameworks, registries, class
  hierarchies, compatibility shims or legacy protocols.
- Experiments stay easy to change, by composing pieces (plant, law, task,
  callback) or by editing code in a worktree.
- Parameters are Python dataclasses with defaults (`MpcParams` and `TIAGO` in
  `tiago_mpc/mpc.py`, the task dataclasses), overridden with `dataclasses.replace`.
  No JSON configurations.
- Types (Felix, 2026-10-22): mypy checks the core, `tiago_mpc/` without
  `safety.py` (`python -m mypy`, configured in `pyproject.toml`, also run by
  `tests/test_types.py`). Annotate public functions and methods; the interfaces
  are the structural protocols in `tiago_mpc/plan.py` (no inheritance, no
  run-time checks). Experiments stay untyped.
- Docs describe the current code only; git keeps the history.

## Every coefficient is a parameter (Felix, 2026-10-22)

The supplied friction, damping and armature values are questionable (see the
caution in the URDFs and `docs/real-robot-friction-plan.md` in the thesis
workspace). Develop with them anyway, but never bake a coefficient into an
algorithm, and test each controller against a plant whose coefficients differ
from the predictor's (for example the 2026-10-16 physical fits, or ±50 %). The
simulated plant may be more detailed than the predictor; enable such detail
explicitly and record both models. Never apply passive losses twice.

## Vocabulary

Use the thesis terms: execution controller, control update, control rate (of
control updates), command rate (of sent commands), "the robot's onboard
controllers". Not "playback" or "robot controller". Never assume that the knot
interval equals the control-update or command period.

## Model and coordinates

- The canonical full model is `models/tiago_pro/tiago_pro.urdf`.
  `/workspace/GATO/TiagoProURDF/tiago_pro.urdf` is OUTDATED: never use it for
  arms, inertias, limits or experiments.
- `scripts/prepare_models.py` owns the joint-parameter table, the grasp frame and
  `models/manifest.json`; regenerate rather than hand-edit. Armature stays
  metadata, never folded into link inertia; kp/kd are not stored.
- Native TIAGo coordinates everywhere (ROS, plants, records). Joint 6's sign map
  lives only at the GPU boundary; do not remove it on GRiD-only evidence.
- Do not modify generic GRiD code or generated headers; select frames through
  GATO's `ee_frame`.

## GRiD reference implementation (Felix, 2026-10-21)

When working on GRiD, consult [A2R-Lab/RBDReference](https://github.com/A2R-Lab/RBDReference):
the NumPy CPU reference for rigid-body dynamics, kinematics, analytical
derivatives and integration, with Pinocchio equivalence checks. Use it to
understand algorithms and validate generated GPU code. Check conventions and
version compatibility; reference features do not imply support in pinned GRiD.

## Robot safety

Keep the robot rules intact: check-only by default and typed confirmation for
execution; startup pose and speed checks; state freshness; every knot
validated; leases that end no later than the plan; never clip a torque or renew
a lease; zero torque and handback first on exit; a met goal never enables
effort control; a dead MPC process noticed within one tick. Keep the safety
model's 4 cm margin, blacklist and documented limits; never change them to make
a fixture pass. ROS imports stay lazy: the package, simulation and tests run
without ROS.

## Experiment data and handoff (Felix, 2026-10-21)

- All new data goes in this checkout's (or worktree's) `_expr_data/`: raw
  results, logs, analysis, figures. Keep it ignored by Git.
- Portable replays go to the shared ingest directory: `impl/_expr_data/ingest/`
  on the host, `/workspace/_expr_data/ingest/` in the container (the same
  mount), in study subfolders with distinct `.npz` names; no raw archives or logs.
- Never write to `/workspace/experiment_data/` or GATO's `thesis_results/`.
  `/workspace/experiment_data_ma_frozen_archive/` is the frozen thesis archive.
- Default handoff, unless Felix asks for a plot: export the motion with
  `experiments/replay.py:save_replay` into the ingest directory (native joints,
  actual timestamps, the task reference, saved plans and telemetry when
  available; see `../tiago_mpc_visualizer/docs/replay-format.md`). Give the
  viewer URL (normally `http://192.168.178.82:8090`) and the library paths, the
  short finding with its metrics and limitations, and links to the raw run.
  Say whether it is simulation or robot data, predicted or executed, and its
  time basis; label display-only subsampling. Do not invent motion or rerun
  retained experiments only for a replay. Without recorded motion, give a
  compact metrics table and raw-output links.
- Preserve failures, original labels and historical cohorts; do not relabel or
  regenerate them.

## Working here

- **Host and container share the checkout.** The container uses `.venv` and
  `.cache/gato`; on the host use `venv-host` with
  `GATO_CACHE_DIR=$PWD/.cache/host-gato`. Do not touch the other side's.
- **GPU:** one job at a time; check `nvidia-smi` first.
- **Agents:** use them for bounded tasks; the root agent coordinates
  interpretation, shared edits and GPU use.
- **Worktrees:** queue cleanup after merging. At the end of a session propose
  the exact worktrees and branches to remove and where useful work is kept;
  never remove them automatically.
- **Remotes:** this repository is `felixmueh/gato_tiago`; the GATO submodule is
  the fork `felixmueh/GATO`. Do not push or update the historical `felix-devel`
  branch.
