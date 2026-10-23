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

Each plant owns a `Clock` (`tiago_mpc/clock.py`: wall, simulation or external
time). The loop runs on the plant's execution time `plant.now()`: it samples
plans, checks leases and schedules ticks and control updates with it. `State.t`
is when the state was measured, in the same domain, and a plan starts at its
state's time; on the robot a state can be older than the tick and repeat.

```
tiago_mpc/    util.py (the cache: cache_root, cached, cache_meshes, digest),
              tiago.py (TIAGo facts, the derived URDFs), dynamics.py (friction laws),
              mpc.py (MpcParams, TIAGO, plant names, find_binary, compile_binary,
              Mpc, MpcProcess),
              plan.py (State, Plan, Tick; the protocols PlantInterface, MpcInterface,
              TaskInterface; Law, Callback),
              clock.py (Clock), laws.py, loops.py (run, guard),
              sim.py (SimPlant), robot.py (RosPlant), ros2.py (ROS 2 client), safety.py,
              record.py (Record)
experiments/  common.py (run_experiment, the mains' options), reach.py, figure8.py,
              random100/, replay.py
scripts/      setup.sh
```

## Setup

Linux, an NVIDIA GPU, CUDA toolkit 12.x with a host compiler it supports,
Python 3.10+ and Git LFS (for the meshes).

```bash
git lfs install
git clone --recurse-submodules https://github.com/felixmueh/gato_tiago.git tiago_mpc
cd tiago_mpc && ./scripts/setup.sh   # LFS files, submodule, .venv from requirements.txt
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q --ignore=external   # CPU tests, no ROS needed, and mypy
.venv/bin/python -m mypy                          # the type check alone (pyproject.toml)
```

- **GATO** is the fork `felixmueh/GATO`: upstream 0.1.2 plus opt-in smooth
  joint friction and trajectory capture, pinned by the gitlink; it needs
  `grid-rbd` 0.5.3. After a new pin: `git submodule sync --recursive`,
  `git submodule update --init --recursive`, reinstall `requirements.txt`.
- **GATO's module is compiled explicitly.** `compile_binary(params)` generates
  the TIAGo dynamics and compiles one CUDA module per variant (minutes, CPU
  only), `bsqpN<N>_tiago_right_<hash>` with the hash over the GPU URDF,
  end-effector frame and predictor friction, cached in `GATO_CACHE_DIR` (default
  `.cache/gato`); once cached it returns at once, `use_cache=False` rebuilds.
  `find_binary(params)` returns the cached module or None. `Mpc(params, binary)`
  and `MpcProcess(params, binary)` load exactly that file and never compile;
  `Mpc.compile(params)` and `MpcProcess.compile(params)` do both, as the mains do.
- **One robot file is committed,** `models/tiago_pro/tiago_pro.urdf`. The
  reduced arm, its GPU copy and the safety collision model are derived from it
  at run time (milliseconds) into the same cache, `models/<hash>/`, their meshes
  as copies named by their contents, `models/meshes/<sha256>.stl`, referenced by
  relative paths ([models/README.md](models/README.md)); never commit a derived file.
- **Container and host share this checkout.** The container
  (`/workspace/tiago_mpc`) uses `.venv` and `.cache/gato`; the host uses
  `venv-host` with `GATO_CACHE_DIR=$PWD/.cache/host-gato` (nvcc 12.9, g++-14).

## Running the experiments

```bash
python experiments/reach.py                    # one reach from the comfortable pose
python experiments/figure8.py                  # one 24 s figure eight
python experiments/reach.py --latency measured --output _expr_data/reach/measured
python experiments/reach.py --command-period .01   # command ticks at the knot interval
python experiments/random100/run.py --dry-run  # 100 reaches: experiments/random100/README.md
```

- **Simulation runs on `SimPlant`'s simulation clock**; `SimPlant(clock=Clock.wall())`
  runs in real time, with `run_robot`. `--latency` charges each solve a
  fixed time in seconds (default 0) or, with `measured`, its own wall time. A
  plan takes effect at the first command tick at or after its request time plus
  the charge; no new request is made while a plan is pending. Until the first
  command, `SimPlant` holds its start state, as PAL's position controller holds
  the arm until effort control starts.
- **Solve time and warm-up:** `MpcParams.max_sqp_time_s` (10 ms) stops a solve's
  SQP iterations by wall time, within `solver.max_sqp_iters` (5). It is for the
  wall clock: the simulation mains (`launch`) and random100 set it to None.
  Before the first tick, `run` calls `mpc.warmup` with the started plant's state,
  the arm still held: with `MpcParams.warmup` (default), GATO's warm-up solve
  (the solver's own iteration limit, no time limit) seeds the first solve, and
  its time counts against no plan's lease.
- **Limits:** the joint, speed and torque limits are ADMM rows with GATO's
  default settings except the control box (`AdmmLimits`: at most 10 inner
  iterations per SQP iteration, eps_rel 1e-2, rho 0.1, 0.03 and 5e-4 for the
  three boxes; GATO's rho_u is 1e-5, which the figure eight keeps). The
  joint box is the URDF's shrunk by `joint_margin_rad` (0.1 rad); `Q_START`
  (`tiago.py`), where the runs start, lies inside it.
- **Late plans:** a plan that first takes effect a knot interval or more after
  its state never runs its knot 0, and `sample_knot` may diverge; `run` warns
  once per run ([docs/robot.md](docs/robot.md#latency)).
- **Rates:** `--command-period` sets the command rate (default 2 ms, 500 Hz,
  in simulation and on the robot), `--update-period` the control rate (default
  the knot interval `dt_s`, 10 ms); either may differ from the knot interval.
  `--duration S` ends a run after at most S seconds.
- **reach:** `TIAGO` defaults, loss-free, 3 ms RK4 substeps, from `Q_START`
  (PAL's comfortable pose with joint 4 at −2.33 rad, not −2.35); arrival within
  5 cm at a joint-speed L1 norm of at most 1 rad/s, timeout 8 s.
- **figure8:** `FIGURE8` (in `experiments/figure8.py`): URDF friction, hard sign
  in the plant (0.1 ms substeps) and tanh (width 0.03 rad/s) in the predictor.
- **Exit status** 1 on a timeout or a raw-limit excess.

## Writing an experiment

An experiment is a task (a frozen dataclass with `at(t, n, dt)`, the reference
and solver updates at t, and `done(state, t)`) and a function that takes the
assembled plant and MPC and hands them to `run_experiment`
(`experiments/common.py`) with its own metrics.
`run_experiment` records the run, sets its status (done, timeout, interrupted or
error), saves it and exports the replay. `main` picks the plant and the
parameters through `launch`. Whoever creates the plant and the MPC closes them.
Swap a piece by passing another one:

```python
from dataclasses import replace
from functools import partial

from experiments.reach import Reach, reach
from tiago_mpc.laws import hold_first
from tiago_mpc.loops import Fixed, run_simulation
from tiago_mpc.mpc import TIAGO, Mpc
from tiago_mpc.sim import SimPlant

params = replace(TIAGO, solver=TIAGO.solver.replace(mu=1.), max_sqp_time_s=None)  # simulation clock
mpc = Mpc.compile(params)  # in this process; MpcProcess: its own
record = reach(SimPlant(substep=.003), mpc, run=partial(run_simulation, latency=Fixed(.005)),
               task=Reach(position_tol_m=.02), law=hold_first)
mpc.close()
```

- A **control law** is a function `law(plan, t, state) -> command`;
  **callbacks** see every `Tick` (execution time, state, command, plan,
  whether it is new). The pieces' interfaces are typed in `tiago_mpc/plan.py`:
  a plant, an MPC and a task need no base class, only what the protocols
  `PlantInterface`, `MpcInterface` and `TaskInterface` declare.
- `SimPlant(friction=...)` sets the **plant's** losses, `MpcParams.friction`
  the **predictor's**; the run records both.
- **Per-solve solver changes** come from the task's `at`, named in `UPDATES`
  (`mpc.py`). **Your own loop** can use `Mpc` (the `MpcProcess` interface plus
  the raw GATO `controller` and `step`, the latest solve's result), `run` and
  `Record` directly, without `run_experiment`.
- **Examples, two remedies for solve latency:** `experiments/plan_feedback.py`, a law
  (a PD around the plan), and `experiments/latency_compensation.py`, its own loop
  (solve from the predicted state); `experiments/latency_study.py` compares them.

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
- `trajectory.npz`: per tick the execution time (`time_s`), the state's time
  (`state_time_s`), state (native q, qd) and command (NaN when none was sent);
  per plan its start time, knot states, knot torques and solve stats
  (`plan_*`); plus the experiment's arrays. Times count from the first state's.
- `replay.npz`: the visualizer replay on execution time, exported after `record.save()` by
  `experiments/replay.py:save_replay` (`../tiago_mpc_visualizer/docs/replay-format.md`),
  with the `MpcParams` in `provenance.params`.

More: [docs/robot.md](docs/robot.md) (robot path and checks),
[docs/safety.md](docs/safety.md) (collision-safety worker),
[docs/decisions.md](docs/decisions.md) (Felix's decisions, deferred work, open
questions), [models/README.md](models/README.md) (model provenance).
