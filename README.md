# gato_tiago

Right-arm reaching and figure-eight tracking in internal simulation, using
modern GATO for torque MPC and Pinocchio RK4 as the simulated plant. ROS 2 is
optional. This is the new application repository, separate from the historical
`felix-devel` implementation. The ROS runner's review fixes and accepted
limitations are listed in [its guide](docs/ROS2_RUNNER.md#review-fixes-and-accepted-limitations);
it is not qualified for physical execution.

## Setup

Linux, an NVIDIA GPU, CUDA toolkit 12.x, a C++17 compiler, Python
3.10+ and Git LFS are required. CUDA must support the GPU's architecture.
Setup installs CMake 3.31.10 into the virtual environment.

```bash
git lfs install
git clone --recurse-submodules https://github.com/felixmueh/gato_tiago.git tiago_mpc
cd tiago_mpc
./scripts/setup.sh
.venv/bin/python examples/reach_sim.py
```

For this prepared checkout:

```bash
cd /workspace/tiago_mpc
.venv/bin/python examples/reach_sim.py
```

GATO is a Git submodule in `external/gato`, pinned to commit
`f7be8b75dae562400f84816883e7bb848b9874fa`, upstream **GATO 0.1.2**
(`bd60b24`) plus our opt-in smooth-joint-friction and runtime-width contributions. Its dependencies
pin `grid-rbd==0.5.3`. The first run generates TIAGo dynamics and compiles just the
N32 CUDA module (several minutes); later runs reuse the shared GATO cache:
`GATO_CACHE_DIR` if set (host runs use `.cache/host-gato`), otherwise `.cache/gato`.
The simulations, the ROS worker and `examples/ros2_mpc.py` all use it. Plants are
named `tiago_right_<8 hex>` over the GPU URDF bytes, the end-effector frame and the
predictor friction configuration (`tiago_mpc/plant_cache.py`), so different variants
never overwrite each other and the same variant reuses its module; the hash is a
cache key, not provenance, and the name is saved as `plant_name`. Generated URDFs
live in `<cache>/models/<hash>/`. The
contribution adds opt-in passive prediction; default-off reaching retains
the existing rigid-body model. The submodule uses the published
`felixmueh/GATO` fork; its exact commit is pinned above. These contributions are
based on upstream 0.1.2, not historical `felix-devel`.

## Updating from the previous GATO pin

After updating this repository, run `git submodule sync --recursive`, then
`git submodule update --init --recursive`
and `.venv/bin/python -m pip install -r requirements.txt`. GATO 0.1.2 requires
GRiD 0.5.3; updating only the submodule leaves the old Python dependency installed.
Registered modules rebuild in this checkout's cache when their source/dependency
identity changes. Do not copy old `.so` files into the new installation.
The runtime-width pin changes the native friction ABI and requires one rebuild
of existing smooth-friction modules. Normal registration and module loading
handle cache invalidation; restart existing workers after updating the package.
Numeric smoothing-width changes then reuse the generated model and binary;
coefficients and friction enablement still require their own compiled identity.

This update preserves the reaching/tracking costs, ADMM settings, bounds,
warm-start policy and timing. The joint-6 GPU-boundary coordinate conversion
remains required: the separate native-axes PR is not included in this pin.
GRiD 0.5.3 handles rotated inertial origins; our current reduced arm already has
zero inertial RPY, so no inertia normalization or URDF change is needed.
Zero soft-limit weights remain explicit in both default configurations. GRiD
0.5.3 still evaluates their barrier expressions; it provides no public switch to
skip that work. This update leaves those calculations and ADMM unchanged.
The 0.1.2 validation repeated one full default tracking cycle (26.536 mm RMS)
and one default reach (.270 s to the existing 5 cm arrival gate); sampled raw
limits passed, including all returned tracking plans. See `VERIFICATION.md`.

## The example

`examples/reach_sim.py` uses the complete reaching defaults in
`configs/reaching.json`:

- Fixed-base, seven-joint right arm; comfortable high-clearance start.
- One reachable position goal, computed by FK from a nearby joint pose.
- Batch 1, N32, 10 ms prediction/control interval and RK4 plant steps at most 3 ms.
- SQP cap 5, PCG cap 120/tolerance .001; running position/velocity/effort costs
  4/.05/.0001, terminal position cost 8, mu 10 and regularization rho .01.
- Zero soft position/velocity/effort limit costs; upstream ADMM limit rows with
  100 iterations, relative tolerance .001 and q/qd/u penalties 1/.3/.0005.
- PCG for both SQP and ADMM; shifted warm starts and rho reset each feedback.
- Planning bounds: .1 rad position margin, 80% raw velocity, 85% raw effort.
- Arrival: position error at most 5 cm and joint-speed L1 at most 1 rad/s;
  timeout 8 simulated seconds, no dwell or final-approach stage.
- Position only: the tool orientation is unconstrained.

The canonical full model is the current simulator robot description; its
original bytes are retained separately. The full model carries both arms' supplied parameters; the reduction carries
the right-arm values. Armature is metadata only. A generated simulation copy
resolves mesh filenames to absolute local paths. A second, equivalent GPU model normalizes
joint 6's negative Z axis to positive Z and reflects its position limits.
The adapter converts that joint's position, velocity and torque signs at the
GPU boundary; the simulated plant and saved histories use native TIAGo
coordinates. Original transforms, flange and inertias are retained. Reaching
and tracking target the added nominal grasp-center frame, 139.3 mm along the
flange's +Z axis, between the finger pads at zero gripper posture. See
[the frame definition](models/README.md#end-effector-point). This is the
same coordinate convention used by the historical TIAGo wrapper: upstream's
fixed-arm adapter assumes zero topology-helper storage, whereas a directly
registered signed-axis model needs nonzero helper storage and traps on launch.
Startup checks compare end-effector FK and one GPU/Pinocchio dynamics step at
both task poses before starting the MPC loop. By default, reaching uses
rigid-body dynamics without passive joint damping/friction in both the plant and
the predictor. Losses are opt-in: add `joint_dynamics` and
`predictor_smoothing_rad_s` (as in `configs/tracking.json`) to a reaching
configuration for hard-sign losses in the plant and the smooth tanh law in the
predictor. Such a run became unstable in its first 10 ms control interval on
2026-10-22. The visualizer's
[friction diagnosis](../tiago_mpc_visualizer/docs/friction.md) attributes this
to viscous damping against joint 7's link-only inertia, which is stiff for
explicit 3 ms RK4 steps. Friction-enabled reaching is not qualified. No
physical-model validation is implied.

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

## random100: 100 random reaches

`experiments/random100/` runs 100 fixed random-posture starts with one position
goal each, in simulation only. See [its README](experiments/random100/README.md).

`examples/reach_sim.py` and the random100 runner select `configs/reaching.json`
when `--config` is omitted. An explicit `--config` file is resolved against an
old base configuration, not the current defaults; to vary the defaults, copy the
complete `configs/reaching.json` and edit it.

Both simulation examples share model preparation, configuration, controller
construction and limit setup with the ROS worker. Their simulation loops and
the ROS I/O/safety processes remain separate.

Both examples accept `--solve-time-mode fixed|measured`:

- `fixed` charges `--solve-time-s` seconds per solve; the default is zero,
  preserving instantaneous feedback in simulation.
- `measured` charges `MPCController.step`'s complete reported wall time,
  including host work, rather than native kernel timing.

Each solve returns a plan: one torque per knot. The execution controller
samples it: the plant applies the control of knot k = floor((t − t₀)/dt) at
time t, where t₀ is the time of the state the plan was computed from, a
zero-order hold per knot, as GATO's transcription assumes. With the control
update period equal to dt and zero latency this is the plan's first torque in
each interval. During charged time the previous plan continues (initially zero
torque); the new plan then runs from the offset of the charged time. A charged
solve longer than the whole plan would need a torque past its last knot; the
run then ends with an error. `--control-update-period-s` sets the nominal time
between control updates; it defaults to the knot interval `dt_s` and may differ from it.
Overruns extend the interval to the charged solve time. References follow the
resulting simulated timestamps; no wall-clock sleeping is used. Setup, plotting
and logging time are not charged. The upstream warm-start shift is unchanged:
GATO shifts by one knot per solve whatever `dt_elapsed` says (it only scores
hypotheses), so a control update period other than dt starts each solve from a
misaligned guess. Histories record solve-start times, charged latency and actual
update intervals alongside native/controller timings, and per plant substep the
applied torque, its knot and its plan's start time.

```bash
.venv/bin/python examples/reach_sim.py --solve-time-mode fixed --solve-time-s .003
.venv/bin/python examples/track_figure8.py --solve-time-mode measured \
  --output _expr_data/tracking/measured
```

Recorded solve times are diagnostic, not benchmark results. Collision checking
and hardware deployment are outside these examples; planner constraint rows do
not certify physical safety.

## Models and storage

`models/tiago_pro/tiago_pro_sim_source.urdf` preserves the exact canonical
simulator description from `/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`.
The active `models/tiago_pro/tiago_pro.urdf` replaces the outdated supplied full
model and applies explicit both-arm friction/damping and armature metadata.
The existing reduced arm already matched the simulator's kinematics/inertias and retained
gripper lump. The active full model now carries both arms' actual friction,
damping and effort values supplied on 2026-10-21; the reduction carries the
right-arm values. Armature is retained as custom metadata only, not enabled in
either dynamics model. kp/kd are omitted as they belong to a different controller.
Current reaching still ignores passive forces; historical experiments retain
their recorded coefficients and source identities.

All 35 canonical full-model mesh references are supplied, including two assets
from pinned official PAL repositories. The full visual/collision geometry loads
and all mesh vertices are finite. See `models/README.md` for the exact source,
asset licenses, actual coefficients and metadata semantics, and the reproducible
metadata preparation command.

URDFs and code are ordinary Git files. Binary meshes use Git LFS, configured
by `.gitattributes`; clones must fetch the LFS objects. Generated headers,
binaries, environment and run results are ignored. The application remote is
[`felixmueh/gato_tiago`](https://github.com/felixmueh/gato_tiago).


## Figure-eight tracking

`examples/track_figure8.py` uses `configs/tracking.json`. Felix selected running
end-effector XYZ weight **4** and terminal XYZ weight **8**. Joint-velocity
regularization remains **0.01** and actuator-torque regularization **1e-6**.
Orientation and joint-posture objectives are inactive. All three soft-limit
weights are zero; ADMM joint-position, velocity and actuator-effort constraints
remain enabled. The historical flange's 24 s check at 4/8 completed with
**26.536 mm full-cycle RMS**, 40.570 mm peak and 31.392 mm endpoint error; sampled
executed and returned-plan raw limits passed. This is the selected tuning, not
the preceding 8/1600 configuration that achieved 2.056 mm RMS.

The new nominal grasp-center default completed its own 24 s check with
**20.806 mm RMS** and 31.216 mm peak error; executed samples and all 12,000
saved plans passed raw joint/speed/torque limits. Changing the tracked point
also changes the Cartesian task, so these numbers do not isolate a controller
improvement. A short grasp-center reach passed the coarse 5 cm arrival gate
after 0.090 s (63.308 mm initial displacement); this is a frame/loop sanity check.
Raw validation is in `/workspace/tiago_mpc-grasp-frame/_expr_data/grasp-center/`.

The plant uses the supplied actual right-arm Coulomb/damping coefficients with
hard-sign friction; GATO uses tanh friction with **0.03 rad/s** smoothing plus
viscous damping. Armature is metadata only and remains inactive in both models.
Registration supplies the configured scalar or seven-joint smoothing widths to
each new solver automatically. The public
`solver.set_joint_friction_smoothing(widths)` method changes only that instance
between solves without rebuilding or resetting warm starts. The standard task
runners retain their configured width throughout a run. Controller setup details
(including ROS worker summaries) and tracking's `model_receipt.json` record the
effective seven float32 widths as `joint_friction_smoothing_rad_s`; it is `null`
when predictor friction is disabled. Record a new receipt after any deliberate
runtime change: the module hash alone no longer identifies the smoothing law.
N32, 10 ms knots and control period and 0.1 ms RK4 plant substeps track a 24 s
horizontal figure eight, 0.28 m wide and 0.18 m high. The default knot interval
changed from 2 ms to 10 ms on 2026-10-22; the tracking checks above ran at 2 ms
and do not qualify the 10 ms default. This uses exact state and fixed simulation
time: computation latency does not advance the plant. It is not a real-time or
physical-robot qualification. Collisions are not checked.

```bash
.venv/bin/python examples/track_figure8.py --output _expr_data/tracking/my-run --stop-on-raw-limit
# Only generate/build and check predictor values/Jacobians:
.venv/bin/python examples/track_figure8.py --output _expr_data/tracking/model-check --build-only
```

Use a fresh output directory and coordinate sequential GPU use. Saved native
joint states, task reference, actuator torques, predictions and effective settings
remain in that output. `--losses plant-only` retains plant losses and disables
predictor losses; `--losses neither` disables both for an explicit comparison.
Hard-sign prediction from the old compatibility adapter is not offered: its
reviewed historical implementation requires a different GATO pin. Explicit
`--config` files retain their own coefficients and settings.

Use `experiments/tools.py:save_replay` for the shared interactive viewer handoff;
see `../tiago_mpc_visualizer/docs/replay-format.md`. The completed experiment
branches and their raw results were removed at Felix's request after
the decisions were integrated or discarded. Reaching also uses running/terminal
XYZ weights 4/8; its other task-specific settings are unchanged.

The historical 4/8 tracking episode used application source `b582309` and final
study commit `199c899`. Its raw run, report and viewer replay were deleted at
Felix's request during cleanup on 2026-10-21.
The integration's effective tracking parameters match that episode; only the
configuration description and removal of the unused historical hard-predictor
adapter differ.

The one 4/8 reaching sanity episode stopped on the existing 5 cm arrival
gate (49.830 mm final error, speed L1 .624 rad/s), not after settling. Reaching
continues to use rigid-body plant and predictor dynamics with passive forces
inactive. It used integration source `4dde1f8`; its raw run and viewer replay
were also deleted during the authorized cleanup. Run a fresh episode and export
it with `experiments/tools.py:save_replay` to inspect the current defaults.

## Optional ROS 2 adapter

ROS 2 is optional; the base requirements do not install ROS packages.
The combined test suite uses optional development requirements:

```bash
python -m pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

`tiago_mpc/ros2.py` provides the ported PAL right-arm state, trajectory and
effort transport, optional SEA feedback, and explicit exclusive controller
handover. ROS imports are optional; the simulation example remains independent.
The adapter services a bounded burst of ready state callbacks and supports
explicit fresh-state reads. The process-based application is implemented in
`examples/ros2_mpc.py`, with separate MPC and ported collision-safety workers.
It uses the current grasp-center frame and configurations. The command below
checks fresh state and safety only; motion requires `--execute` and confirmation.

[Adapter usage and validation limits](docs/ROS2_ADAPTER.md) and
[the upstream MPCController integration proposal](docs/ROS2_MPC_PROPOSAL.md)
describe the native joint/sign convention and separate I/O process.
See [runner usage and verification](docs/ROS2_RUNNER.md) for the implementation.
The default runner invocation performs state/safety checks without commands:

```bash
python examples/ros2_mpc.py
```

The read-only inspection example publishes no commands or controller requests:

```bash
# Source the compatible PAL ROS overlay/network settings first.
python examples/ros2_read_state.py --samples 5
# ROS-free checks (requires pytest and installed simulation dependencies):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_ros2_*.py
```

No ROS/PAL simulation or hardware execution has been performed for this port.
