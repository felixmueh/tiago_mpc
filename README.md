# gato_tiago

Torque MPC for the TIAGo Pro right arm (seven joints, fixed torso root) with
GATO (`gato-mpc` 0.1.5 from PyPI): reaching and figure-eight tracking of the
grasp-center frame ([models/README.md](models/README.md#end-effector-point)) in
simulation, and the same experiments on the robot or in Gazebo through ROS 2.

A command loop runs at the command rate. At each command tick it reads the
plant's state, requests a control update when one is due, turns the latest plan
into a torque command with a control law (by default `sample_knot`: the knot due
at time t, counted from the time of the plan's state) and sends it to the plant,
which holds it until the next tick.

There are two loops (`tiago_mpc/loops.py`), named after their time.
`run_real_time` runs on wall time (`Wall`, `tiago_mpc/clock.py`): solves take as
long as they take, and a plan is used at the first tick after it arrives. It
drives the robot and the rehearsal, a `SimPlant` on the robot's architecture.
`run_virtual_time` runs controlled experiments in simulation on virtual time
(`Virtual`), which the loop keeps itself: the experiment declares how long a
solve lasts (`solve_latency`), and the loop waits for each solve and places its
plan at the request plus that latency. Either loop takes any MPC; `launch`
(`experiments/common.py`) gives `run_virtual_time` an in-process `Mpc` and
`run_real_time` an `MpcProcess`, which solves concurrently in its own process. Each loop starts the plant on its clock
(`plant.start(clock)`) and samples plans, checks leases and schedules ticks and
control updates on it. `State.t` is when the state was measured, on the same
time, and a plan starts at its state's time; on the robot a state can be older
than the tick and repeat. On the robot the safety check runs in its own process.

```
tiago_mpc/    __init__.py (the root: `import tiago_mpc as tm` re-exports what experiments use),
              util.py (the cache: cache_root, cached, cache_meshes, digest),
              tiago.py (TIAGo facts, the derived URDFs), dynamics.py (friction laws),
              mpc.py (MpcParams, TIAGO, plant names, find_binary, compile_binary,
              Mpc, MpcProcess),
              plan.py (State, Plan, Tick; the protocols PlantInterface, MpcInterface,
              TaskInterface; Law, Callback),
              clock.py (Virtual, Wall), laws.py, loops.py (run_real_time, run_virtual_time, guard),
              sim.py (SimPlant), robot.py (RosPlant), ros2.py (ROS 2 client), safety.py,
              record.py (Record)
experiments/  common.py (run_experiment, the mains' options), reach.py, figure8.py,
              random100/, replay.py
```

## Setup

Linux, an NVIDIA GPU, CUDA toolkit 12.x with a host compiler it supports,
[uv](https://docs.astral.sh/uv/) and Git LFS (for the meshes). uv provides
Python 3.10 (`.python-version`; `requires-python` is `==3.10.*`).

```bash
git lfs install
git clone https://github.com/felixmueh/gato_tiago.git tiago_mpc
cd tiago_mpc
git lfs install --local && git lfs pull   # the meshes
uv sync                                   # .venv from uv.lock, tiago_mpc editable, pytest and mypy
uv run python -m pytest -q                # CPU tests, no ROS needed, and mypy
uv run python -m mypy                     # the type check alone (pyproject.toml)
```

- **One environment per checkout:** `uv sync` creates the checkout's `.venv`
  from the committed `uv.lock`, with `tiago_mpc` installed editable, so each
  checkout and worktree runs its own code (`uv run …` or `.venv/bin/python …`),
  and `python -m experiments.<name>` works from any directory. Change a
  dependency in `pyproject.toml`, then `uv lock` and `uv sync`. `uv run` and
  `uv sync` restore the lock and undo a manual `uv pip install`.
- **GATO** is `gato-mpc` 0.1.5 from PyPI; it pins `grid-rbd` 0.5.5 and refuses
  any other installed GRiD. Predictor friction is registered with
  `joint_dynamics=True, friction_width=w`: GRiD generates the GPU URDF's
  damping and tanh Coulomb friction into the dynamics and derivatives, with the
  width compiled in. GATO's per-SQP-iteration trajectory capture (the old fork's)
  is not available. Each record's `summary.json` names the GATO it ran (`gato`:
  version, `direct_url.json`, `gato.__file__`, and an editable checkout's
  commit and source status).
- **GATO's module is compiled explicitly.** `compile_binary(params)` generates
  the TIAGo dynamics and compiles one CUDA module per variant (minutes, CPU
  only), `bsqpN<N>_tiago_right_<hash>` with the hash over GATO's version, the
  GPU URDF, grasp frame and predictor friction, cached in `GATO_CACHE_DIR` (default
  the main checkout's `.cache/gato`); once cached it returns at once,
  `use_cache=False` rebuilds.
  `find_binary(params)` returns the cached module or None. `Mpc(params, binary)`
  and `MpcProcess(params, binary)` load exactly that file and never compile;
  `Mpc.compile(params)` and `MpcProcess.compile(params)` do both, as the mains do.
- **One robot file is committed,** `models/tiago_pro/tiago_pro.urdf`. The
  reduced arm, its GPU copy and the safety collision model are derived from it
  at run time (milliseconds) into the same cache, `models/<hash>/`, their meshes
  as copies named by their contents, `models/meshes/<sha256>.stl`, referenced by
  relative paths ([models/README.md](models/README.md)); never commit a derived file.
- **Container and host share this checkout.** The host uses `.venv`, the
  container (`/workspace/tiago_mpc`) `.venv-container` (below); neither can use
  the other's interpreter. Both, and every worktree, share the main checkout's
  `.cache/gato` (`GATO_CACHE_DIR` overrides it): modules are keyed by content,
  toolchain and Python ABI, so builds of both sides coexist and worktrees reuse
  main's.

### A local GATO fork

To run tiago_mpc on a local GATO checkout, use a branch or worktree, never
main, and override the source in its `pyproject.toml` (the commented example
there):

```toml
[tool.uv.sources]
gato-mpc = { path = "../GATO", editable = true }   # the GATO checkout, relative to this one
```

Then `uv lock && uv sync`. GATO compiles the checkout's native sources, whose
content is part of its module key, so the fork's modules sit beside 0.1.5's in
the shared cache, and records name the checkout's commit. The changed
`pyproject.toml` and `uv.lock` stay on that branch. The fork must pin the same
`grid-rbd`, or change the pin in its own `pyproject.toml`. A manual
`uv pip install -e` does not work: the next `uv run` reinstalls the locked GATO.

### In the container

The container uses its own environment, `.venv-container`, with the system
Python 3.10 that ROS 2 Humble's `rclpy` is built for. The base image sets
`UV_PROJECT_ENVIRONMENT=/opt/gato-venv`; the overlay image
(`impl/GATO-notes/GATO/felixtools/Dockerfile`) overrides it:

```dockerfile
ENV UV_PROJECT_ENVIRONMENT=.venv-container   # relative: each checkout's root
ENV UV_PYTHON_PREFERENCE=only-system
```

```bash
export UV_PROJECT_ENVIRONMENT=.venv-container  # also set these in an already-running older container
export UV_PYTHON_PREFERENCE=only-system
env -u PYTHONPATH uv sync                    # the image's PYTHONPATH (GATO's tiago_src, python) precedes site-packages
env -u PYTHONPATH uv run python -m pytest -q
```

Use `uv run` or `.venv-container/bin/python`, not the activated
`/opt/gato-venv`, which is also first on the image's `PATH`. Robot runs source
ROS 2 in the launching shell as before, which puts `rclpy` on `PYTHONPATH`.
**Order when moving to this layout:** the container's old
environment was `.venv`, where the host's now goes, and the first host
`uv sync` replaces it. Create `.venv-container` in the container first, move
the container's work to it and delete the old `.venv` from inside the
container; only then run `uv sync` on the host.

## Running the experiments

```bash
python -m experiments.reach                    # one reach from the comfortable pose
python -m experiments.figure8                  # one 24 s figure eight
python -m experiments.reach --solve-latency measured --output _expr_data/reach/measured
python -m experiments.reach --plant rehearsal     # SimPlant on run_real_time, as on the robot
python -m experiments.reach --command-period .01   # command ticks at the knot interval
python -m experiments.random100.run --dry-run  # 100 reaches: experiments/random100/README.md
```

- **Timing.** `--plant sim` (the default) runs `run_virtual_time` with an in-process
  `Mpc`. `--solve-latency` says how long a solve lasts on the timeline: declared
  seconds (default 0, the explicit ideal) or `measured`, the solve's own wall time
  (`plan.stats["wall_s"]`). A plan takes effect at the first command tick at or
  after its request plus that latency; no new request is made while a solve is
  outstanding. A declared latency above 0 warns once when a solve really took
  longer and counts the overruns. `--plant rehearsal` runs the rehearsal:
  `SimPlant` on `run_real_time` with `MpcProcess`, as on the robot; a tick whose
  work ran past the next one is late, skips the missed ticks, warns once and is
  counted. `SimPlant(sync_real_time=True)` paces a virtual-time run to the wall clock
  for watching, without changing its results. The summary's `timing` records the
  loop, the command and update periods, `solve_latency` and the counts; the
  plant's `time` is `virtual` or `wall`. Until the first command, `SimPlant` holds
  its start state, as PAL's position controller holds the arm until effort
  control starts.
- **Solve time and warm-up:** `MpcParams.max_sqp_time_s` (10 ms) stops a solve's
  SQP iterations by wall time, within `solver.max_sqp_iters` (5). It follows the
  timing: `launch` sets it to None for a declared `--solve-latency` (the solve's
  wall time is irrelevant by assumption) and keeps it for `measured`, the
  rehearsal and the robot; random100 (a declared 0) sets None.
  Before the first tick, both loops call `mpc.warmup` with the started plant's
  state, the arm still held: with `MpcParams.warmup` (default), GATO's warm-up
  solve (the solver's own iteration limit, no time limit) seeds the first solve,
  and its time counts against no plan's lease.
- **Limits:** the joint, speed and torque limits are ADMM rows with GATO's
  default settings except the control box (`AdmmLimits`: at most 10 inner
  iterations per SQP iteration, eps_rel 1e-2, rho 0.1, 0.03 and 5e-4 for the
  three boxes; GATO's rho_u is 1e-5, which the figure eight keeps). The
  joint box is the URDF's shrunk by `joint_margin_rad` (0.1 rad); `Q_START`
  (`tiago.py`), where the runs start, lies inside it.
- **Late plans:** a plan that first takes effect a knot interval or more after
  its state never runs its knot 0, and `sample_knot` may diverge; both loops warn
  once per run ([docs/robot.md](docs/robot.md#latency)).
- **Rates:** `--command-period` sets the command rate (default 2 ms, 500 Hz,
  in simulation and on the robot), `--update-period` the control rate (default
  the knot interval `dt_s`, 10 ms; 0: back to back, a request at the tick the
  previous plan takes effect); either may differ from the knot interval.
  `--duration S` ends a run after at most S seconds.
- **reach:** `TIAGO` defaults, loss-free, 3 ms RK4 substeps, from `Q_START`
  (PAL's comfortable pose with joint 4 at −2.33 rad, not −2.35); arrival within
  5 cm with the grasp point slower than 2 cm/s, timeout 8 s.
- **figure8:** `FIGURE8` (in `experiments/figure8.py`): URDF friction, `StickSlip`
  Coulomb in the plant (1 ms substeps) and `Tanh(.03)` (width 0.03 rad/s) in the predictor.
- **Exit status** 1 on a timeout or a raw-limit excess.

## Writing an experiment

An experiment is a task (a frozen dataclass with `at(t, n, dt)`, the reference
and solver updates at t, and `done(state, t)`) and a function that takes the
assembled plant and MPC, a loop and its timing arguments and hands them to
`run_experiment` (`experiments/common.py`) with its own metrics.
`run_experiment` records the run, sets its status (done, timeout, interrupted or
error), saves it and exports the replay. `main` picks the plant and the
parameters through `launch`. Whoever creates the plant and the MPC closes them.
Swap a piece by passing another one:

```python
from dataclasses import replace

import tiago_mpc as tm
from experiments.reach import Reach, reach

params = replace(tm.TIAGO, solver=tm.TIAGO.solver.replace(mu=1.), max_sqp_time_s=None)  # a declared latency
mpc = tm.Mpc.compile(params)  # in this process; tm.MpcProcess: its own
record = reach(tm.SimPlant(substep=.003), mpc, run=tm.run_virtual_time, solve_latency=.005,
               task=Reach(position_tol_m=.02), law=tm.hold_first)
mpc.close()
```

- A **control law** is a function `law(plan, t, state) -> command`;
  **callbacks** see every `Tick` (the tick's time, state, command, plan,
  whether it is new). The pieces' interfaces are typed in `tiago_mpc/plan.py`:
  a plant, an MPC and a task need no base class, only what the protocols
  `PlantInterface`, `MpcInterface` and `TaskInterface` declare.
- `SimPlant(friction=...)` sets the **plant's** losses, `MpcParams.friction`
  the **predictor's**; the run records both.
- **Friction simulation:** a `Friction` holds the coefficients and its Coulomb law,
  plain data that each integrator interprets: `StickSlip()` (the default),
  `Sign()` or `Tanh(width_rad_s)`. `SimPlant(friction=urdf_friction())` uses
  semi-implicit Euler, implicit viscous damping and a coupled stick/slip solve.
  Joints hold at zero velocity while the needed friction lies within ±c, and slip
  at c. `urdf_friction(Sign())` instead evaluates Coulomb as `c*sign(v_old)` (it
  can chatter and creep near zero), `urdf_friction(Tanh(.03))` as
  `c*tanh(v_old/width)` (the predictor's law; it creeps below the threshold).
  Damping stays implicit in all three. They use the full Pinocchio mass matrix and
  bias at the old state and advance q with the new velocity. The default maximum
  substep is 1 ms (1 kHz), shortened to land on a command tick; this does not set
  the command or control rate. `integrator` is `"rk4"` or `"euler"`; the default
  is RK4 when loss-free, Euler with friction. `"rk4"` is GATO's RK4 when
  loss-free and, with `Tanh` or `Sign`, RK4 with the passive torque at every stage
  (the URDF's damping needs substeps far below 1 ms there); it refuses
  `StickSlip`, which exists only as the Euler step's solve. `passive_torque`
  likewise covers `Tanh` and `Sign` only. `MpcParams.friction` takes `Tanh` only.
  A saved `Friction` records its law's `name` (`"tanh"`, `"sign"`,
  `"stick_slip"`). Armature remains disabled. These plant choices do not qualify
  friction-enabled MPC or tracking.
- **Per-solve solver changes** come from the task's `at`, named in `UPDATES`
  (`mpc.py`). **Your own loop** can use `Mpc` (the `MpcProcess` interface plus
  the raw GATO `controller` and `step`, the latest solve's result), a loop and
  `Record` directly, without `run_experiment`; a loop of your own starts the plant
  with `Virtual()` or `Wall()` and returns a `tm.Outcome` for `run_experiment`.
- **Examples, two remedies for solve latency:** `experiments/plan_feedback.py`, a law
  (a PD around the plan), and `experiments/latency_compensation.py`, its own loop
  (solve from the predicted state); `experiments/latency_study.py` compares them.
- **Friction comparison:** `python -m experiments.friction_study --dry-run` describes
  48 simulation cells: those four execution controllers, the existing reach and
  figure eight, 0/20 ms declared latencies, and plant coefficients at 50/100/150 % of
  the URDF values. Predictors and gains stay fixed; reaching predicts no friction,
  tracking predicts nominal tanh friction. Run with
  `--output _expr_data/controller-friction/<fresh-name>`. Each cell runs once;
  this compares task behavior rather than wall-clock performance or robot readiness.

## On the robot

Source ROS 2 with PAL's controller-manager interfaces (4.32.2) in the launching
shell; otherwise creating `RosPlant` fails with a clear error.

```bash
python -m experiments.reach --plant robot                                   # checks only
python -m experiments.reach --plant robot --execute --target X_M Y_M Z_M
python -m experiments.figure8 --plant robot --execute --target X_M Y_M Z_M --duration 4
```

- **Check-only is the default:** the startup checks, then exit; no MPC, no
  controller switch, no command.
- **`--execute`** asks you to type `EXECUTE` and runs the experiment on wall
  time (`run_real_time`): effort control from the first command, zero torque and
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
- `trajectory.npz`: per tick its time (`time_s`), the state's time
  (`state_time_s`), state (native q, qd) and command (NaN when none was sent);
  per plan its start time, knot states, knot torques and solve stats
  (`plan_*`); plus the experiment's arrays. Times count from the first state's.
- `replay.npz`: the visualizer replay on the run's time, exported after `record.save()` by
  `experiments/replay.py:save_replay` (`../tiago_mpc_visualizer/docs/replay-format.md`),
  with the `MpcParams` in `provenance.params`.

More: [docs/robot.md](docs/robot.md) (robot path and checks),
[docs/safety.md](docs/safety.md) (collision-safety worker),
[docs/decisions.md](docs/decisions.md) (Felix's decisions, deferred work, open
questions), [models/README.md](models/README.md) (model provenance).

The [friction and breakaway handoff](docs/friction-breakaway-handoff.md) records
the physical-fit comparison and the proposed separate static/sliding plant model.
