# Jetson solve-time studies

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
ROS connection is opened. MODE_40W is user-confirmed; clocks and power mode are
not changed and no host checks are required.

## Histogram sweep

N16/32/64/128/256/512, PCG versus factored BDSV, one SQP iteration, three fresh
repeats, one complete six-second figure-eight and one six-second 192.368 mm
reach plus hold. The simulator uses 500 Hz commands, 100 Hz control updates,
10 ms knots, zero declared solve latency and no latency compensation. The only
solver difference within a task pair is `admm_linsys="pcg"` versus
`admm_linsys="bdsv_factor"`. All completed episodes supply 600 timings.

The default sweep has 72 episodes and 43,200 measured solves if all cases
complete. `REPORT.md`, histogram plots, scalar `stats.csv`/`stats.npz`, compact
case outcomes and provenance remain in the output. Every tail sample and failed
case is retained. Arrival does not shorten the reaching distribution.

## Horizon × batch matrix

N16/32/64/128/256/512 × B1/2/4/8/16, one SQP iteration, ordinary PCG without ADMM,
PCG cap 120, 30 ms knots, 10 ms updates, RK4 with at most 1 ms integration steps.
Each cell has two fresh-controller repeats of 300 measured updates, after
excluded initialization: 30 cells, 60 attempts and 18,000 measured solves if all
complete. Per-call scalar JSONL/CSV, cell tables, outcomes and heatmaps are kept.

This is the current stack using the thesis matrix's timing, seeding and scaled
weights. The canonical model, grasp frame and fitted armature are current. It
is a new cohort, separate from historical thesis measurements. The retained
GTX 1070 N32/SQP5 pilot also remains a separate earlier protocol; these final
SQP1 sweeps have not been run on Jetson or GTX 1070 by this preparation session.

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
