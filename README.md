# TIAGo MPC

One right-arm position reach in internal simulation, using modern upstream
GATO for torque MPC and Pinocchio RK4 as the simulated plant. No ROS is used.

## Setup

Linux, an NVIDIA GPU, CUDA toolkit 12.x, a C++17 compiler, Python
3.10+ and Git LFS are required. CUDA must support the GPU's architecture.
Setup installs CMake 3.31.10 into the virtual environment.

```bash
git lfs install
git clone --recurse-submodules <this-repository> tiago_mpc
cd tiago_mpc
./scripts/setup.sh
.venv/bin/python examples/reach_sim.py
```

For this prepared checkout:

```bash
cd /workspace/tiago_mpc
.venv/bin/python examples/reach_sim.py
```

GATO is a Git submodule in `external/gato`, pinned to upstream commit
`aeacf16e7733df1a17dfd6627f3b5d15c2ba4387`. Its own dependencies pin
`grid-rbd==0.5.1`. The first run generates TIAGo dynamics and compiles just the
N32 CUDA module (several minutes); later runs reuse `.cache/gato`. Upstream
sources are unmodified. There is no dependency on the historical GATO fork.

## The example

`examples/reach_sim.py` uses the complete reaching defaults in
`configs/reaching.json`:

- Fixed-base, seven-joint right arm; comfortable high-clearance start.
- One reachable position goal, computed by FK from a nearby joint pose.
- Batch 1, N32, 10 ms prediction/control interval and RK4 plant steps at most 3 ms.
- SQP cap 5, PCG cap 120/tolerance .001; running position/velocity/effort costs
  4/.05/.0001, terminal position cost 800, mu 10 and regularization rho .01.
- Zero soft position/velocity/effort limit costs; upstream ADMM limit rows with
  100 iterations, relative tolerance .001 and q/qd/u penalties 1/.3/.0005.
- PCG for both SQP and ADMM; shifted warm starts and rho reset each feedback.
- Planning bounds: .1 rad position margin, 80% raw velocity, 85% raw effort.
- Arrival: position error at most 5 cm and joint-speed L1 at most 1 rad/s;
  timeout 8 simulated seconds, no dwell or final-approach stage.
- Position only: the tool orientation is unconstrained.

The source URDFs remain unchanged. A generated simulation copy resolves mesh
filenames to absolute local paths. A second, equivalent GPU model normalizes
joint 6's negative Z axis to positive Z and reflects its position limits.
The adapter converts that joint's position, velocity and torque signs at the
GPU boundary; the simulated plant and saved histories use native TIAGo
coordinates. Transforms, tool frame and inertias are retained. This is the
same coordinate convention used by the historical TIAGo wrapper: upstream's
fixed-arm adapter assumes zero topology-helper storage, whereas a directly
registered signed-axis model needs nonzero helper storage and traps on launch.
Startup checks compare end-effector FK and one GPU/Pinocchio dynamics step at
both task poses before starting the MPC loop. Both models use rigid-body
dynamics without passive joint damping/friction; the source URDF dynamics tags
are retained, and no physical-model validation is implied.

```bash
# Build/codegen and check model conventions, without a reaching episode:
.venv/bin/python examples/reach_sim.py --build-only
# Check holding the initial tool position for two simulated seconds:
.venv/bin/python examples/reach_sim.py --hold --timeout 2
# A fresh directory is required when naming an output explicitly:
.venv/bin/python examples/reach_sim.py --output _expr_data/reaching/my-first-reach
```

Runs save `summary.json`, `trajectory.npz` and `reach.png` in fresh timestamped
directories under `_expr_data/reaching/`. Histories contain all 3 ms plant states/applied
torques and per-solve timing/constraint violations. The selected JSON and configuration resolver are copied into each output.
The summary records source identities, effective settings, arrival and
independent simulated limit checks.
Exit status is nonzero on timeout, execution error or a simulated limit violation.
Failed outputs are retained; the script does not retry.

## Frozen random100 experiment

`experiments/random100/` includes the exact 100 random initial states and goal
inputs recovered from the historical experiment branch, with hashes and order.
These are random-posture starts, distinct from the comfortable-start cone
dataset. Run them with the current modern GATO/ADMM configuration:

```bash
.venv/bin/python experiments/random100/run.py --dry-run
.venv/bin/python experiments/random100/run.py
```

See `experiments/random100/README.md` for the explicit differences from the old
solver experiment. New outputs live under `_expr_data/`.
Arrival and physical-limit compliance are reported separately.

The current defaults are the tested ADMM-only configuration: **84/100 arrivals
and 100/100 sampled physical-limit passes**, compared with 69/100 arrivals and
100/100 limit passes for the retained soft+ADMM profile. The same frozen inputs
ran once in original order for each cohort. There were 16 gained arrivals and
one lost arrival; the selected diagnostic pilot overlaps the 100, so these
counts are not independent holdout evidence. No GPU rerun was needed to promote
the already tested configuration.

The winning profile is `zero_soft_state_rho10.json` at application commit
`c3f226b`, based on `e45c441`, in `/workspace/tiago_mpc-admm-only/`.
Its report, raw cohorts and per-case effective settings remain in that checkout's
`_expr_data/admm-only-reaching-20261020/`. The complete default pins its effective
N32/10 ms settings, including the .1 rad planning margin and native N32 ADMM PCG
policy. Preview is .31 s. See `VERIFICATION.md` for scope and verification.

Both commands above select `configs/reaching.json` when `--config` is omitted.
The runner's dry-run prints full effective solver, controller and bound settings;
runs copy the selected JSON and pass that snapshot to every case. Explicit
`--config` files retain their original partial-override semantics against the
frozen legacy base, so historical profiles reproduce their old settings:

```bash
.venv/bin/python experiments/random100/run.py --dry-run \
  --config experiments/configuration_study/profiles/n32_dt10ms_sqp5_margins_terminal10.json
```

To reproduce the original 30 ms/terminal80 example, pass an explicit JSON file
containing `{}`. To customize the new defaults, copy `configs/reaching.json`
and edit that complete file. The historical configuration-study profiles and
upstream GATO remain unchanged.

Simulation uses fixed pacing: computation time does not advance the plant.
Recorded solve times are diagnostic, not benchmark results. Collision checking
and hardware deployment are outside this example; planner constraint rows do
not certify physical safety.

## Models and storage

`models/tiago_pro/tiago_pro.urdf` is the original full supplied URDF, unchanged.
`models/tiago_pro/meshes/` includes all 33 referenced STL files. The reduced
`models/tiago_right/tiago_right_arm.urdf` is the existing seven-joint arm model,
also unchanged, with its gripper mass/inertia already lumped at the tool.
See `models/README.md` for provenance and hashes.

URDFs and code are ordinary Git files. Binary meshes use Git LFS, configured
by `.gitattributes`; clones must fetch the LFS objects. Generated headers,
binaries, environment and run results are ignored. No remote is configured
for this new repository.
