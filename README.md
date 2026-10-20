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

`examples/reach_sim.py` contains the complete example and its named settings:

- Fixed-base, seven-joint right arm; comfortable high-clearance start.
- One reachable position goal, computed by FK from a nearby joint pose.
- Batch 1, N32, 30 ms prediction/control interval and 3 ms RK4 plant steps.
- SQP cap 5, PCG cap 120; explicit position, velocity and effort costs.
- Upstream ADMM position/velocity/torque limit rows; linear solver policy PCG.
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
.venv/bin/python examples/reach_sim.py --output results/my-first-reach
```

Runs save `summary.json`, `trajectory.npz` and `reach.png` in fresh timestamped
directories under `results/`. Histories contain all 3 ms plant states/applied
torques and per-solve timing/constraint violations. The summary records source
identities, effective settings, arrival and independent simulated limit checks.
Exit status is nonzero on timeout, execution error or a simulated limit violation.
Failed outputs are retained; the script does not retry.

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
