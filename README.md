# gato_tiago

Torque MPC for the TIAGo Pro right arm (seven joints, fixed torso root) with
GATO (submodule `external/gato`): reaching and figure-eight tracking of the
grasp-center frame ([models/README.md](models/README.md#end-effector-point)) in
simulation, and the same experiments on the robot or in Gazebo through ROS 2.

One loop runs at the command rate. At each command tick it reads the plant's
state, requests a control update when one is due, turns the latest plan into a
torque command with a control law (by default `sample_knot`: the knot due at
time t, counted from the time of the plan's state) and sends it to the plant,
which holds it until the next tick. The MPC runs in its own process; on the
robot, so does the safety check.

```
tiago_mpc/    tiago.py (TIAGo facts, URDF preparation), dynamics.py (friction laws),
              plant_cache.py, mpc.py (MpcParams, TIAGO, MpcInProcess, MpcWorker),
              plan.py (State, Plan, Tick), laws.py, loops.py (run, guard), sim.py (SimPlant),
              robot.py (RosPlant), ros2.py (ROS 2 client), safety.py, record.py (Record)
experiments/  reach.py, figure8.py, random100/, replay.py
scripts/      setup.sh, prepare_models.py, prepare_safety_model.py
```

## Setup

Linux, an NVIDIA GPU, CUDA toolkit 12.x with a host compiler it supports,
Python 3.10+ and Git LFS (for the meshes).

```bash
git lfs install
git clone --recurse-submodules https://github.com/felixmueh/gato_tiago.git tiago_mpc
cd tiago_mpc && ./scripts/setup.sh   # LFS files, submodule, .venv from requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q --ignore=external   # CPU tests, no ROS needed
```

- **GATO** is the fork `felixmueh/GATO`: upstream 0.1.2 plus opt-in smooth
  joint friction and trajectory capture, pinned by the gitlink; it needs
  `grid-rbd` 0.5.3. After a new pin: `git submodule sync --recursive`,
  `git submodule update --init --recursive`, reinstall `requirements.txt`.
- **The first run** generates the TIAGo dynamics and compiles one CUDA module
  (minutes), named `tiago_right_<hash>` over the GPU URDF, end-effector frame
  and predictor friction, and cached in `GATO_CACHE_DIR` (default `.cache/gato`).
- **Container and host share this checkout.** The container
  (`/workspace/tiago_mpc`) uses `.venv` and `.cache/gato`; the host uses
  `venv-host` with `GATO_CACHE_DIR=$PWD/.cache/host-gato` (nvcc 12.9, g++-14).

## Running the experiments

```bash
python experiments/reach.py                    # one reach from the comfortable pose
python experiments/figure8.py                  # one 24 s figure eight
python experiments/reach.py --latency measured --output _expr_data/reach/measured
python experiments/reach.py --command-period .002 --update-period .01
python experiments/random100/run.py --dry-run  # 100 reaches: experiments/random100/README.md
```

- **Simulation runs on a simulated clock.** `--latency` charges each solve a
  fixed time in seconds (default 0) or, with `measured`, its own wall time. A
  plan takes effect at the first command tick at or after its request time plus
  the charge; no new request is made while a plan is pending.
- **Rates:** `--command-period` sets the command rate, `--update-period` the
  control rate; both default to the knot interval `dt_s` (10 ms) and may differ
  from it. `--duration S` ends a run after at most S seconds.
- **reach:** `TIAGO` defaults, loss-free, 3 ms RK4 substeps; arrival within
  5 cm at a joint-speed L1 norm of at most 1 rad/s, timeout 8 s.
- **figure8:** `FIGURE8` (in `experiments/figure8.py`): URDF friction, hard sign
  in the plant (0.1 ms substeps) and tanh (width 0.03 rad/s) in the predictor.
- **Exit status** 1 on a timeout or a raw-limit excess.

## Writing an experiment

An experiment is a task (a frozen dataclass with `reference(t, n, dt)` and
`done(state, t)`), parameters, and a function that composes the pieces; `main`
only picks the plant. Swap a piece by passing another one:

```python
from dataclasses import replace
from functools import partial

from experiments.reach import Reach, reach
from tiago_mpc.laws import hold_first
from tiago_mpc.loops import Fixed, run_simulation
from tiago_mpc.mpc import TIAGO
from tiago_mpc.sim import SimPlant

params = replace(TIAGO, solver=TIAGO.solver.replace(mu=1.))
record = reach(SimPlant(substep=.003), run=partial(run_simulation, latency=Fixed(.005)),
               task=Reach(position_tol_m=.02), params=params, law=hold_first)
```

- A **control law** is a function `law(plan, t, state) -> command`;
  **callbacks** see every `Tick` (state, command, plan, whether it is new).
- `SimPlant(friction=...)` sets the **plant's** losses, `MpcParams.friction`
  the **predictor's**; the run records both.
- **Per-solve solver changes** come from the task's optional `updates(t)`,
  named in `UPDATES` (`mpc.py`). **Your own loop** can use `MpcInProcess`
  (the `MpcWorker` interface plus the raw GATO `controller`) directly.

## On the robot

Source ROS 2 with PAL's controller-manager interfaces (4.32.2) in the launching
shell; otherwise creating `RosPlant` fails with a clear error.

```bash
python experiments/reach.py --plant robot                                   # checks only
python experiments/reach.py --plant robot --execute --target X_M Y_M Z_M
python experiments/figure8.py --plant robot --execute --target X_M Y_M Z_M --duration 4
```

- **Check-only is the default:** the startup checks, then exit; no MPC, no
  controller switch, no command.
- **`--execute`** asks you to type `EXECUTE` and runs the experiment on the wall
  clock (`run_robot`): effort control from the first command, zero torque and
  PAL's position controller again on exit or a fault. `--target` (metres in
  `torso_lift_link`) is the reach target or the figure eight's `start_m`;
  `--duration` keeps first runs short; `--sea-absolute` uses SEA feedback.
- **Not run yet** against ROS, Gazebo or the robot: [docs/robot.md](docs/robot.md).

## Outputs

`Record` buffers one entry per command tick and writes at the end, into
`--output` or `_expr_data/<task>/<UTC time>/` (ignored by Git):

- `summary.json`: status, commits, source status, package versions, the
  complete `MpcParams`, the task, the plant model and settings (robot: checks,
  `ROS_DISTRO`, handback), and the results (arrival or tracking error,
  raw-limit excess, solve work).
- `trajectory.npz`: per tick the time, state (native q, qd) and command (NaN
  when none was sent); per plan its start time, knot states, knot torques and
  solve stats (`plan_*`); plus the experiment's arrays.
- `replay.npz`: the visualizer replay, exported after `record.save()` by
  `experiments/replay.py:save_replay` (`../tiago_mpc_visualizer/docs/replay-format.md`).

More: [docs/robot.md](docs/robot.md) (robot path and checks),
[docs/safety.md](docs/safety.md) (collision-safety worker),
[docs/decisions.md](docs/decisions.md) (Felix's decisions, deferred work, open
questions), [models/README.md](models/README.md) (model provenance).
