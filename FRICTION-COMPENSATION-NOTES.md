# Friction compensation and armature comparison

The friction integration from `/workspace/tiago_mpc-friction-compensation`, branch
`experiment/friction-compensation`, is promoted to main on 28 October. It includes
the interface/model-data refactor and is based on `56db0fd`. The earlier format-2
reader `f66172f` is retained by tag `archive/friction-format2-f66172f`. Main's loose
ROS/container changes remain separate and uncommitted. Native dependency forks
remain local; no upstream publication or hardware qualification is implied.

Two local dependency worktrees carry the narrow extension:

- `/workspace/GRiD`, branch `experiment/tiago-armature-060`,
  upstream v0.6.0 `712df1c`. Generated ABA/Minv/CRBA/RNEA support explicit
  `JOINT_ARMATURE`. Pinned peers stay unchanged. Only fixed-base, scalar,
  cardinal-axis revolute joints are supported; unsupported models are rejected.
- `/workspace/GATO-tiago-armature-017`, branch `experiment/tiago-armature-017`,
  upstream v0.1.7 `c9da6a5`. Registration/build/codegen forward and record
  `joint_armature`; cache identity includes its values and actual emitter bytes.
  On top, `BSQP.admm_iters` (the ADMM inner budget, read from the native
  solver), cherry-picked from `feat/admm-iters-readback` (`17405d0`, worktree
  `GATO-admm-iters-readback`, for an upstream PR).

The TIAGo branch's uv sources select local GATO editable and a normal local
GRiD wheel. The wheel build stages GRiD's validated resources/peer provenance;
editable GRiD does not stage that bundle. No provenance gate was disabled.
The local dependency checkpoints are GRiD `4497ebe84237c8294e511b7477b556e5b9844ef0`
and GATO `69bd8d86d37eaf0439b8331cfe92b4449d7a5c5b`. They have not been pushed or
merged. The source overrides still use local paths, not portable remote revision
pins; package version labels remain the upstream base versions.

## Common model and six cases

All six cases use one coupled StickSlip plant, implicit viscous damping,
semi-implicit Euler with at most 1 ms substeps, the canonical URDF, and supplied
right-arm armature `[.746,.882,.211,.648,.326,.343,.27]` kg m². These armature
values are supplied, not identified. They enter both the plant and every
predictor, independently of friction. Pinocchio gets `model.armature` once;
its ABA/CRBA/RNEA already include it. Nothing is added to link inertias or XML.

Joints 1–4 use rounded physical thesis fits in the plant; wrists retain nominal
unidentified values. Predictor/compensation friction uses the nominal URDF
coefficients, preserving model mismatch. Static holding and sliding still share
one Coulomb threshold. SEA elasticity and actuator-delivery dynamics are absent.

Cases: MPC only, external friction feedforward, PD, both, friction inside MPC,
and friction inside MPC + PD. The last two have no external friction feedforward.
In this retained six-case cohort, PD used the original bare-link inertia
scaling and untuned 30 rad/s recipe. The current stored default is described below. External feedforward keeps
its original Tanh(.03) on planned velocity. Internal MPC uses Tanh(.2).

The .2 rad/s width is a numerical choice for 30 ms prediction knots, not an
identified friction transition. The near-zero damping screen uses the actual
augmented mass: with the supplied armature, the conservative dt*decay bound is
1.58 for nominal coefficients and 1.85 for fitted coefficients, below the scalar
Euler limit 2. At width .03, these bounds are 6.25/10.03; selected-pose coupled
limits are about 14.1/9.25 ms, shorter than the 30 ms step. Armature alone does
not justify a narrower width here. This is a damping/conditioning screen, not a
full nonlinear stability certificate. Widths were selected without task-RMS tuning.

This compares practical model/controller choices at their declared smoothing
widths; it does not isolate friction-placement architecture at one common width.

## Timing and costs

- Execution controllers, sent torque commands and fresh feedback: 500 Hz (2 ms),
  including PD. TO requests: 50 Hz (20 ms), independent of prediction knots.
- N32, 30 ms knots / .93 s horizon; EE running weight 4 and terminal weight 80,
  mu=1, qd_cost=.01, u_cost=1e-4, ADMM rho_u=5e-4; SQP5/PCG120.
- Virtual time, declared 5.5 ms solve latency, available at the next command tick
  (6 ms). Fixed latency has no wall-time solve budget. This is a representative
  delay, not calibrated end-to-end Jetson timing. No extra observation/transport
  delay or noise. Actual hardware feedback availability remains unverified.

`--feedback-period .02` retains the separate slow-feedback condition. Earlier
armature-off and slow-feedback recordings remain unchanged under `_expr_data/`.
No legacy result was relabelled or rerun merely to migrate its format.

## Current result and provenance

The current six-case run is
`_expr_data/friction-compensation/armature-20261026T164141Z/`.
Every case completed 24 s, with finite states and zero sampled raw q/v/effort
excess. True EE RMS is 26.37 / 9.79 / 21.47 / 8.59 / 17.75 / 15.07 mm in the
case order above. `RESULTS.txt`, `viewer-url.txt` and the independent audit are
beside the raw records. True plant motion and observed feedback remain separate;
replays depict executed truth, with every command and predicted horizon retained.

A wrapper inherited the application's `GIT_DIR`/`GIT_WORK_TREE`, making raw
foreign-dependency Git identity fields wrong or incomplete. Raw summaries and
original exports are preserved. `PROVENANCE-ADDENDUM.json` and
`dependency-provenance.json` identify the actual reviewed sources and binary
hashes. Derived replays have corrected source metadata plus `recorded_source`;
all motion arrays, labels and outcomes are unchanged. The future helper clears
those environment variables. No numerical rerun accompanied this correction.

Native headers, registration metadata and both compiled modules are retained in
`native/`; tracked patches and untracked source snapshots are under `source/`.
The local GRiD wheel's actual emitter digest is recorded, so version 0.5.5 alone
cannot be mistaken for official armature support.

## Run and verification

```bash
# Host: the checkout's .venv, independent of the container's environment.
uv sync
uv run python -m experiments.friction_compensation --dry-run
uv run python -m experiments.friction_compensation --output _expr_data/friction-compensation/<fresh-name>

# Container: relative Git pointers now work without global GIT_DIR overrides.
export UV_PROJECT_ENVIRONMENT=.venv-container
export UV_PYTHON_PREFERENCE=only-system
env -u PYTHONPATH -u GIT_DIR -u GIT_WORK_TREE uv sync --locked
env -u PYTHONPATH -u GIT_DIR -u GIT_WORK_TREE uv run python -m experiments.friction_compensation --dry-run
```

For a new GRiD source change, rebuild its local wheel with
`uv sync --reinstall-package grid-rbd`, then explicitly call `compile_binary`;
`find_binary` is intentionally a cheap lookup. Keep one GPU job at a time.

Checks: 337 application CPU checks (the full suite's 336 passed checks plus
its corrected local-source provenance check), 67 subtests, 87 final focused
checks; GRiD 25 CPU checks and GATO 19 host checks. None/zero armature preserves
old header bytes/cache behavior. The two consumed GPU model profiles passed
production checks plus eight independent sampled step/A/B checks each against
Pinocchio, with NumPy RBDReference mass/acceleration checks using its matching
prototype parser. This qualifies the consumed TIAGo path, not all GRiD algorithms
or hardware. No upstream performance benchmark claim is made.

Initial reference-harness failures are retained: mismatched parser API, then an
overly strict supplementary NumPy tolerance. A separate loose CPU reference check
passes; native GPU/Pinocchio tolerances were unchanged. No task ran before these
model checks passed. Six capped .24 s prefixes then passed before the full cases.
The dev viewer was paused with approval during each GPU window and restored in
finally blocks; receipts/logs and source backups are in
`_expr_data/armature-comparison-20261026/`.

## Additional 10 ms trial

`_expr_data/friction-compensation/armature10ms-20261026T192405Z` holds the six width-.2 cases;
its `-narrow` sibling holds only two additional internal-MPC cases at width .03.
TO remains 50 Hz and feedback/controllers 500 Hz. N32 yields a .31 s horizon.
All eight completed 24 s with zero sampled limit excess. The six direct cases
have higher RMS than their 30 ms counterparts. Narrower internal friction helps
RMS somewhat at 10 ms, with larger peak errors. Defaults remain the 30 ms profile.
The eight-track URL, full table, qualification and source receipts are beside
the direct comparison. `--cases` selects a subset without duplicate runs.

## Tuned execution-controller presets (27 October)

`experiments/controller_tuning.py` now contains explicit `TUNED_PD`,
`TUNED_FEEDFORWARD` and `TUNED_BOTH` simulation presets. PD uses effective
inertia M+A at Q_START, natural frequency 20 rad/s and damping ratio 1.4.
External FF uses the retained proximal coefficients, nominal wrists and width
.01 rad/s. The reusable laws are `tm.PlanFeedback` and `tm.FrictionFeedforward`
in `tiago_mpc/laws.py`; their tuned presets stay in the experiment. The later 28 October refactor makes these PD gains the stored
`tm.PlanFeedback()` default; named old tuning candidates retain explicit
historical gains. Internal MPC friction width is unchanged. `tm.inertia_gains(..., armature=...)` includes reflected inertia exactly once.

The study in `_expr_data/controller-tuning-20261027/` contains 70 complete
simulation episodes plus three preflight prefixes, the locked selection,
source snapshots, all failures and analysis. FF+PD improves training figure-eight
RMS from 8.59 to 4.38 mm and held-out 12 s tracking from 8.61 to 5.62 mm.
The half-loss stress is a counterexample: old FF+PD gives 7.49 mm versus the
tuned 8.86 mm. Full FF can overcompensate and oscillate when the actual losses
are smaller. See its `REPORT.md` for all tasks, 10 ms transfer and held-feedback
checks; `VIEWER-LINKS.md` loads the dated presentation replays together.

A fresh fit on retained real data did not transfer better than the thesis
prior; it is retained as negative evidence. Joint 1 lacks bidirectional
excitation, and wrists 5–7 remain unidentified. The practical manual procedure
is [controller-tuning-on-robot.md](docs/controller-tuning-on-robot.md): measure
timing/noise, start with conservative PD and partial proximal FF, and calibrate
the wrists separately. No robot contact or motion occurred. The presets are
not hardware-qualified; the package integration is promoted while the native
dependency forks remain local.

## Smoothness/sticking metrics and friction-free comparison (27 October)

`experiments/motion_metrics.py` adds active-plan-conditioned sustained near-rest
fractions and duration distributions, eligible joint time, exact-zero time,
velocity/torque ripple and command total variation per second. The tuning
experiment records full motion and values after the first second; its default
solve latency is now measured. `no_friction` removes plant friction only, keeping
armature, integration and frozen law/predictor parameters. The internal-MPC
friction cases use width .2 and the same tuned PD as the external comparison.

`_expr_data/controller-smoothness-20261027/REPORT.md` contains the completed
12-case matched comparison, plot, raw records and viewer links. Default EE RMS
is 26.37 mm with friction and 4.19 mm without; tuned FF+PD on the frictional
plant gives 4.41 mm. Keeping that FF+PD enabled on the friction-free plant gives
11.44 mm and more oscillation. All 12 complete 24 s with finite states and zero
sampled raw-limit excess. Commands/feedback are 500 Hz, TO 50 Hz, knots 30 ms,
with measured local solve latency. The earlier 76 full episodes were analyzed
read-only, preserving their original declared timing and raw bytes.

Stall thresholds (.005 actual/.02 planned rad/s, at least 50 ms) are explicit
analysis choices, not identified robot noise/friction parameters. Fractions use
eligible joint time and cannot replace tracking error. Internal-MPC and external
FF retain different coefficients/smoothing; this does not isolate architecture.

The separate static-friction design is in
`_expr_data/friction-realism-20261027/RECOMMENDATION.md`. Current StickSlip already
holds statically; the proposed refinement separates breakaway and sliding
thresholds while retaining coupled dynamics. No new physics was inserted into
this comparison, and static/Stribeck/wrist parameters remain uncalibrated.

## Friction integration, 28 October

Included in the promoted integration:

- Literal armature and tuned PD values live in `tiago.py`. `PlanFeedback()` reads
  the stored gains without computing them at import. The utility
  `inertia_gains` returns plain float tuples that print as matching assignments
  for manual editing; no model files are written automatically.
- `Friction` is coefficient-only. Simulation and MPC own `friction_law`;
  external FF owns `evaluation`. Numerical law types remain plain dataclasses.
  No separate static threshold or new stick/slip solver was introduced.
- Both package loops reject predictor friction plus built-in external FF, or
  nested external FF, before plant startup. Physical losses are allowed.
- Run records are format 3; older files are unchanged and format 2 remains
  readable using WIP `f66172f`. Portable replay version 1 is unchanged.
- Tuning candidates retain explicit historical gains for `old_pd`/`old_both`;
  the locked tuned presets use the stored model defaults. Their widths remain
  explicit on the FF laws.

Native dependency checkpoints remain GRiD `417a30d` and GATO `72c9d14`; current
uv sources still select sibling paths. Publishing immutable fork pins for main
is a separate step. No package/default-model physics, timing or native emitter
change was made by this follow-up, apart from the explicitly chosen PD default.

Predictive latency integration remains later work. The existing example only
received the new coefficient/law arguments; its CPU lookahead still does not
assign armature. It must receive an explicit propagation model before combination
with nonzero armature. That model may include physical friction even when
optimizer friction is disabled for external FF; this is not duplicate actuator
compensation. No new effort/speed clipping policy was implemented.

Current verification receipts are in `_expr_data/friction-integration-20261028/`.
The earlier WIP checks (352 package tests +67 subtests, 25 GRiD checks and 19 GATO
host checks) remain in `_expr_data/wip-checkpoint-20261028/`. No new GPU
measurements, robot motion or upstream performance qualification were performed.
