# Agent instructions

Read `README.md` first. Felix's dated decisions, deferred work and open
questions are in `docs/decisions.md`; add an entry there when he decides
something lasting. The robot path is in `docs/robot.md`, the safety worker in
`docs/safety.md`.

For control of the sibling visualizer, read
[the visualizer agent API guide](../tiago_mpc_visualizer/docs/agent-api.md).
Run its `scripts/control.py` from the visualizer checkout to inspect and patch
Live/presentation settings, send simulation commands, or select an MPC
checkout/worktree or solver profile. Open viewers reflect shared changes;
the guide documents partial patches, version checks and completion status.

## Felix owns this repository (Felix, 2026-10-24)

- Felix owns the structure and code of this repository. Implement his decisions
  as specified; discuss any deviation with him before implementing it: another
  mechanism, extra machinery or a different structure than he decided (for
  example, a lazily loading package root where he asked for plain re-exports).
- Do not commit, merge, rebase or push here. Leave the changes uncommitted in
  the working tree for Felix to review; he commits.

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
- The root uses plain re-exports (Felix, 2026-10-24). Importing `tiago_mpc` may load Pinocchio and
  GATO's Python package, but never ROS (only creating a `RosPlant` does) and never GATO's compiled
  module (only the MPC process); `test_importing_the_package_loads_no_ros_and_no_compiled_gato` checks this.
- Experiments use `import tiago_mpc as tm` (Felix, 2026-10-23); package modules import
  each other directly (`from .plan import State`), never through the root.
- Run experiments as modules, `python -m experiments.reach` (Felix, 2026-10-23), with the
  checkout's own environment: `uv run python -m …` or `.venv/bin/python -m …`. Each checkout
  and worktree has its own uv environment with `tiago_mpc` installed editable (Felix,
  2026-10-24), which also makes `experiments` importable, so this works from any directory.
  Never run one checkout's code with another checkout's environment; `-m` puts the current
  directory first, so do not run from another checkout's root either.
- Docs describe the current code only; git keeps the history.

## Every coefficient is a parameter (Felix, 2026-10-22)

The supplied friction, damping and armature values are questionable (see
`models/tiago_pro/README.md` and `docs/real-robot-friction-plan.md` in the thesis
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
- Every other robot file is derived from it at run time into the cache and never
  committed (Felix, 2026-10-23): the reduced arm and grasp frame (`tiago.arm_urdf`),
  the GPU copy (`runtime_urdf`), the safety collision model. Change the full URDF
  or the derivation, never a derived file. The armature is not in the URDF (no
  loader reads it), never folded into link inertia; kp/kd are not stored.
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

- All data goes in this checkout's (or worktree's) `_expr_data/` (Felix,
  2026-10-23): raw results, logs, analysis and figures, of experiments and of
  scratch runs, checks and internal investigations alike. No other results
  directory (such as `results/`). Keep it ignored by Git.
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

- **One uv environment per checkout** (Felix, 2026-10-24): `uv sync` creates it
  from the committed `uv.lock`, `.venv` on the host and `.venv-container` in the
  container (`UV_PROJECT_ENVIRONMENT`, README), with `tiago_mpc` installed
  editable. Host and container keep their own environments; do not touch the
  other side's. GATO is `gato-mpc` from PyPI; a local GATO checkout goes through
  `[tool.uv.sources]` on a branch or worktree (README), never `uv pip install -e`
  (the next `uv run` reverts it) and never on main.
- **One shared GATO cache** (Felix, 2026-10-23): the main checkout's
  `.cache/gato` (`/home/felix/_Uni/MasterThesis/impl/tiago_mpc/.cache/gato`)
  serves the main checkout and every worktree, on the host and in the container
  alike; `cache_root()` finds it through git, so leave `GATO_CACHE_DIR` unset
  (it remains an override). Derived files and GATO's modules are named by their
  contents, the module key including GATO's native sources, the toolchain and
  Python ABI, so host and container builds coexist, worktrees reuse main's
  compiled modules, and a worktree at another GATO gets its own. Plant names
  hash GATO's version, so the registrations of two GATO versions never
  overwrite each other (Felix, 2026-10-24). Where git cannot resolve the main
  checkout (a host-created worktree inside the container), the default falls
  back to the worktree's own `.cache/gato`.
- **GPU:** one job at a time; check `nvidia-smi` first.
- **Agents:** use them for bounded tasks; the root agent coordinates
  interpretation, shared edits and GPU use.
- **Worktrees:** a worktree and branch that you created and used for
  development may be removed without asking once merged (Felix, 2026-10-23),
  after moving any data worth keeping from its `_expr_data/` into the main
  checkout's. For all others, propose the exact worktrees and branches to
  remove at the end of a session and where useful work is kept; never remove
  them automatically.
- **Remotes:** this repository is `felixmueh/gato_tiago`. The GATO fork
  `felixmueh/GATO` stays on GitHub: historical records name its commits (for
  example `fa58597`). Do not push or update the historical `felix-devel` branch.
