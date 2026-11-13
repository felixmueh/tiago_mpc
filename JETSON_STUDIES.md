# Jetson solve-time studies

Updated 2026-11-13. The PCG/BDSV comparison uses **batch size one only** at
N16/32/64/128/256/512. It uses relaxed limit barriers with ADMM disabled and
20 Hz control updates. The separate horizon × batch matrix keeps
N16/32/64/128/256/512 × B1/2/4/8/16 and its original timing protocol.

After fetching and checking out the temporary experiment branch, run this from
the repository root inside the existing Jetson container:

```bash
./run_jetson_studies.sh
```

It prepares the pinned source dependencies and this checkout's `.venv-container`,
then runs the histogram sweep and the horizon × batch matrix sequentially under
the shared `/workspace/.locks/heavy.lock`. CUDA builds use one job. The first run
needs network access for public source repositories and locked Python packages.
No existing sibling dependency checkouts are changed.

For a CPU-only protocol preview, or to select one study:

```bash
./run_jetson_studies.sh describe
./run_jetson_studies.sh histograms
./run_jetson_studies.sh matrix
./run_jetson_studies.sh histograms _expr_data/jetson-barrier-b1
./run_jetson_studies.sh all _expr_data/jetson-studies-repeat
```

`describe` also performs dependency/environment setup, but does not compile GATO
modules or run GPU solves. The default output root is `_expr_data/jetson-studies/`,
with `histograms/` and `nb-matrix/` subfolders. Existing study output directories
are never overwritten. An incomplete or unsupported histogram case does not
suppress the matrix. Individual exit statuses are printed and appended to
`launcher-status.log`; either incomplete study makes the launcher exit nonzero.
Ctrl-C/SIGTERM stops the launcher without starting the next study.

Both studies save statistics only: per-solve times/work counters, compact task
metrics, outcomes, source identities and plots. They do not save trajectories,
tick histories, plans or replays. Initialization is excluded from histogram
samples; matrix initialization statistics are retained separately. No robot or
ROS connection is opened. Clocks and power mode are never changed. Available
Jetson power/clock queries are recorded read-only; missing tools are recorded
without blocking the experiment.

## PCG/BDSV horizon sweep: batch one

N16/32/64/128/256/512, **B1 at every N**, plain PCG versus BDSV, one SQP
iteration, **PCG cap1,000**, three fresh repeats of one complete four-second figure-eight. The
native solver's batch size is checked before warm-up and recorded per update;
any batch size other than one is rejected. The simulator uses 500 Hz commands, 20 Hz control updates,
10 ms knots, zero declared solve latency and no latency compensation. The only
solver difference within a task pair is `linsys="pcg"` versus `linsys="bdsv"`.
ADMM is disabled; relaxed barriers use mu=0.003 and delta=0.05 with unchanged
planning bounds. Actual barrier rows, plain solver path and zero ADMM work are
verified. All completed episodes supply 80 timings.

Both plants use fitted friction with coupled StickSlip and Euler substeps;
both predictors omit friction. The execution controller samples plan torque
directly, without PD or friction feedforward. The winning trajectory is reused
without shifting, with automatic reseeding disabled. Prediction horizons are
0.15/0.31/0.63/1.27/2.55/5.11 s. Increasing N changes lookahead while per-knot
and terminal cost weights remain fixed. Subsets remain selectable on the
module CLI with `--knots`.

The default comparison has 36 episodes and 2,880 measured solves if all cases
complete. `REPORT.md`, histogram plots, scalar `stats.csv`/`stats.npz`, compact
case outcomes, warm-start error diagnostics and provenance remain in the output.
Every tail sample and failed or unsupported case is retained. Earlier desktop
cohorts stay separate.

Each update also records KKT, Schur, linear-solve, merit and line-search event
spans, initial-merit time, the accepted step and a completion wall timestamp.
These counters already exist in the native solver; no probe kernels are added.
No executed line search is represented by `line_search_ran=false` and a zero
placeholder `step_size`.

Each case runs in its own process so a native CUDA exit does not stop other
solvers or horizons. Scalar updates are streamed to `progress.jsonl` and
recovered after an abnormal worker exit; unknown task quality stays explicitly
unavailable. PCG can converge before its cap1,000; actual iterations and cap
hits are kept, including incomplete and failed cases. No horizon is silently
dropped to accommodate a GPU resource limit.

The GTX 1070 cap1,000 trial attempted all 36 cases. Both solvers completed
N16–128, and BDSV also completed N256: 27 episodes / 2,160 timings. PCG at
N256 and both paths at N512 failed during CUDA shared-memory setup. This GPU
allows 48 KiB per block; the N256 PCG scratch requests 72,368 bytes and the
N512 BDSV guard requests 57,632 bytes. These are resource failures, not PCG
iteration-cap exits. Keep all six horizons on Jetson and measure its actual
support separately. At N128, PCG still used at most 113 iterations; raising
the cap from120 to1,000 left iteration and prediction-error sequences unchanged.

## Reading timing distributions

The desktop BDSV bands moved between otherwise identical simulated updates;
the leading explanation is runtime scheduling/contention, not an established
intrinsic BDSV mode. On Jetson, compare full-update timing with work counts,
stage spans and clock/load logs before interpreting peaks. CUDA event spans can
include scheduling delays and are not exclusive arithmetic timings.

Run the timing section with the GPU otherwise idle, and keep power/clock
settings consistent across cells. The launcher serializes these studies but
does not stop unrelated applications. Clock/load monitoring uses `nvidia-smi`
when usable, otherwise `tegrastats --interval 200` writes `tegrastats.log` on
Jetson. Unavailable monitoring is explicit. `nvpmodel -q` and
`jetson_clocks --show` are optional read-only provenance queries, not launch gates.
Initialization and compilation stay outside the measured distributions.

## Horizon × batch matrix

N16/32/64/128/256/512 × B1/2/4/8/16, one SQP iteration, ordinary PCG without ADMM,
PCG cap 120, 30 ms knots, 10 ms updates, RK4 with at most 1 ms integration steps.
Each cell has two fresh-controller repeats of 300 measured updates, after
excluded initialization: 30 cells, 60 attempts and 18,000 measured solves if all
complete. Per-call scalar JSONL/CSV, cell tables, outcomes and heatmaps are kept.

This is the current stack using the thesis matrix's timing, seeding and scaled
weights. The canonical model, grasp frame and fitted armature are current. It
is a new cohort, separate from historical thesis measurements. The retained
GTX 1070 N32/SQP5 pilot remains a separate earlier protocol. A new N128/SQP1
desktop cohort uses the amended friction-mismatch workload. The full B1 horizon
sweep is prepared; no new desktop horizon sweep was run. Jetson measurements
and the SQP1 matrix remain pending.

## Reproducible dependency sources

Custom local commits were not published, so the branch carries small source
patches in `tools/jetson/`. The launcher fetches public base commits into this
checkout's ignored `_expr_data/jetson-deps/{GATO,GRiD}`, applies each patch to the
index and working tree, and verifies the exact resulting Git tree. It creates no
dependency commits and refuses unexpected existing source changes. The project
and lock file select only these isolated dependencies.

| Dependency | Public base commit | Validated local commit reproduced | Resulting tree |
| --- | --- | --- | --- |
| GATO 0.1.7 | `c9da6a58c7fec4c688f673b79de1c8267359098c` | `45a35094c746feaa2e2f2fcb943ed758d2dcab56` | `a9ccd156c9beefedc14a4941b90b601186f71c10` |
| GRiD 0.6.0 | `712df1c6d91e0c07ff5e482df3c5f703df364fb6` | `4497ebe84237c8294e511b7477b556e5b9844ef0` | `cf7a44d6ad528c54295b47772403aa40c8e16912` |

GRiD's pinned GLASS, URDFParser and RBDReference submodules are initialized for
its validated build resource bundle. Recorded dependency provenance therefore
shows the public base plus staged source changes; the tree hashes above identify
the complete validated source. Source snapshots and run provenance retain that
mapping. Python 3.10 and the existing CUDA 12.x/C++ toolchain are required.

To regenerate plots without GPU execution, use the appropriate module's
`--analyze PATH` option through this checkout's environment. Return the complete
output root, including all scalar attempt records and failure/provenance files.
