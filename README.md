# gato_tiago

Torque MPC for the TIAGo Pro right arm (seven joints, fixed torso root) with
GATO (`gato-mpc` 0.1.5 from PyPI): reaching and figure-eight tracking of the
grasp-center frame ([models/README.md](models/README.md#end-effector-point)) in
simulation, and the same experiments on the robot or in Gazebo through ROS 2.

A command loop runs at the command rate. At each command tick it reads the
plant's state, requests a control update when one is due, turns the latest plan
into a torque command with a control law (by default `SampleKnot()`: the knot due
at time t, counted from the time of the plan's state) and sends it to the plant,
which holds it until the next tick.

There are two loops (`tiago_mpc/loops.py`), named after their time.
`run_real_time` runs on wall time (`WallClock`, `tiago_mpc/clock.py`): solves take as
long as they take, and a plan is used at the first tick after it arrives. It
drives the robot and the rehearsal, a `SimPlant` on the robot's architecture.
`run_virtual_time` runs controlled experiments in simulation on virtual time
(`VirtualClock`), which the loop keeps itself: the experiment declares how long a
solve lasts (`solve_latency_s`), and the loop waits for each solve and places its
plan at the request plus that latency. Their timings, `RealTime` and `VirtualTime`
(with the presets `ideal`, `constant_latency`, `as_fast_as_possible` and
`constant_rate`), hold each loop's settings and run it. Either loop takes any MPC;
given parameters, `run_experiment` (`experiments/common.py`) builds an `MpcProcess`,
which solves concurrently in its own process, for `RealTime` and an in-process `Mpc`
for every other timing. Each loop starts the plant on its clock
(`plant.start(clock)`) and samples plans, checks leases and schedules ticks and
control updates on it. `State.t` is when the state was measured, on the same
time, and a plan starts at its state's time; on the robot a state can be older
than the tick and repeat. On the robot the safety check runs in its own process.

Both loops support optional [latency compensation](docs/latency-compensation.md):
predict the state at the next plan's intended start under the ongoing commands,
then solve from that prediction. The measured state and its age remain separate.
Pass `compensation=LatencyCompensation(...)` in the timing; it defaults off.
Its physical friction and armature model is independent of the optimizer's model.

A run ends without an exception: the task done, the time up, a fault (a plan or
command beyond the raw effort limit, an expired lease, an invalid command) or a stop
request (Ctrl-C, SIGTERM) end the loop, which returns an `Outcome(status, reason, t)`.
Exceptions are real errors, among them the robot plant's own checks and a failed
solve.

```
tiago_mpc/    __init__.py (the root: `import tiago_mpc as tm` re-exports what experiments use),
              util.py (hz_to_s; the cache: cache_root, cached, cache_meshes, digest),
              tiago.py (TIAGo facts, the derived URDFs), dynamics.py (friction laws),
              mpc.py (MpcParams, TIAGO, plant names, find_binary, compile_binary,
              Mpc, MpcProcess),
              interfaces.py (State, Plan, Event, Tick; the protocols PlantInterface,
              MpcInterface, TaskInterface, Timing; ControlLaw, Callback),
              clock.py (VirtualClock, WallClock), laws.py (SampleKnot, HoldFirst,
              PlanFeedback, FrictionFeedforward),
              loops.py (run_real_time, run_virtual_time, RealTime, VirtualTime, Outcome,
              STOP, plan_fault, lease_fault, command_fault),
              prediction.py (LatencyCompensation and physical state propagation),
              sim.py (SimConfig, SimPlant), robot.py (RobotConfig, RosPlant),
              ros2.py (ROS 2 client), safety.py,
              record.py (the run format: to_dict, from_dict, CLASSES, provenance, Console,
              RunRecord, load_run)
experiments/  common.py (run_experiment, common_results, handle_stop_signals, and the mains'
              arguments and setup), reach.py, figure8.py, random100/, replay.py
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
uv run python -m pytest -q                # CPU tests, no ROS needed, mypy and the doctests (conftest.py)
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
  is not available. Each record's `summary.json` names the GATO it ran
  (`provenance.gato`: version, `direct_url.json`, `gato.__file__`, and an editable
  checkout's commit and source status).
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

`container/` holds the ROS 2 container, one image for the desktop and the
Jetson: system Python 3.10 (the one ROS 2 Humble's `rclpy` is built for), a
pinned uv, the CUDA toolchain, git, ROS 2 Humble and PAL's interface overlay
(`container/pal_interfaces/`, built natively; its README names the provenance).
It holds no GATO checkout and no Python environment.

```bash
container/docker.sh                        # desktop: build the image once, create or start the container, attach
container/docker.sh --target jetson        # on the Jetson: L4T r36.4.0, image and container tagged jetson-r36.4.0
```

The flags `--rebuild-image` (also recreates the container), `--recreate-container`,
`--restart` and `--no-attach` are in `--help`. The image has your user and group
IDs. The container uses the host network and IPC, and mounts the parent of this
checkout, `impl/`, at `/workspace`, so sibling worktrees and a local GATO under
`../GATO` (`[tool.uv.sources]`) resolve; the shell starts in the checkout
(`/workspace/tiago_mpc`).

Choose the ROS profile by sourcing it in the shell (they set the DDS variables
and load the overlay, which chains `/opt/ros/humble`):

```bash
source container/ros_simulation.sh         # domain 1, loopback only: the same-host PAL simulator
source container/ros_tiago.sh              # domain 2, Ethernet to the robot (10.68.0.1)
```

The image sets `UV_PROJECT_ENVIRONMENT=.venv-container` (relative: each
checkout's root) and `UV_PYTHON_PREFERENCE=only-system`:

```bash
uv sync                                    # creates .venv-container from uv.lock
uv run python -m pytest -q
```

Use `uv run` or `.venv-container/bin/python`. GATO's modules compile at run time
with the image's nvcc and host compiler into the shared `.cache/gato`.

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
  `Mpc`, on the main's `VirtualTime` (`VirtualTime.ideal()` for reach and
  figure8). `--solve-latency` says how long a solve lasts on the timeline: declared
  seconds (the preset's 0, the explicit ideal) or `measured`, the solve's own wall time
  (`plan.stats["wall_s"]`). A plan takes effect at the first command tick at or
  after its request plus that latency; no new request is made while a solve is
  outstanding. A solve that really took longer than a declared latency above 0 is
  an `overrun` event. `--plant rehearsal` runs the rehearsal:
  `SimPlant` on `run_real_time` (`RealTime`) with `MpcProcess`, as on the robot; a tick
  whose work ran past the next one is late, skips the missed ticks and is a
  `late_tick` event. `SimConfig(sync_real_time=True)` paces a virtual-time run to the
  wall clock for watching, without changing its results. The run records its timing
  as `config.timing`. Until the first command, `SimPlant` holds
  its start state, as PAL's position controller holds the arm until effort
  control starts.
- **Events:** what the loop or a plant detects at its own time (`late_tick`,
  `overrun`, `late_plan`, `pacing_behind`; the robot's `activated` and `handback`)
  rides on the tick (`Tick.events`; a plant hands its events over with
  `take_events()`) into the record's `events.json`. The console prints the first
  event of each kind (`Console`); the common results count `late_ticks`,
  `overruns` and `late_plans`.
- **Stopping:** Ctrl-C or SIGTERM sets `tm.STOP` (`handle_stop_signals`, which every
  main that runs the simulation or the arm calls first): the run ends at the next
  command tick as `interrupted`, the tick sending no command, and is saved; later
  runs of the same process end at their first tick, and the studies stop. A second
  signal raises `KeyboardInterrupt`, the hard abort for blocking calls (compiling,
  the robot's first state); the run is still saved. The MPC process and the safety
  worker ignore SIGINT; the cancelled `EXECUTE` prompt exits with status 2.
- **Solve time and warm-up:** `MpcParams.max_sqp_time_s` (10 ms) stops a solve's
  SQP iterations by wall time, within `solver.max_sqp_iters` (5). It follows the
  timing: when `run_experiment` builds the MPC, it keeps the budget only where the
  solve's wall time is the latency (`RealTime`: the rehearsal and the robot;
  `VirtualTime` with `measured`) and sets None for every other timing (a declared
  `--solve-latency`, `Compensated`: the solve's wall time is irrelevant by
  assumption); random100 (a declared 0) and the latency study build their own MPC
  with None.
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
  its state never runs its knot 0, and `SampleKnot` may diverge; each such plan is a
  `late_plan` event ([docs/robot.md](docs/robot.md#latency)).
- **Rates:** `--command-period` sets the command rate (default 2 ms, 500 Hz,
  in simulation and on the robot), `--update-period` the control rate (default
  10 ms, the knot interval of TIAGO and FIGURE8; 0: back to back, a request at the tick
  the previous plan takes effect); either may differ from the knot interval. Only
  the options given replace the timing's values. `--duration S` ends a run after at
  most S seconds. In code, a period written as a rate is `tm.hz_to_s(hz)`
  (`hz_to_s(500) == .002`).
- **reach:** `TIAGO` defaults, loss-free, 3 ms RK4 substeps, from `Q_START`
  (PAL's comfortable pose with joint 4 at −2.33 rad, not −2.35); arrival within
  5 cm with the grasp point slower than 2 cm/s, timeout 8 s.
- **figure8:** `FIGURE8` (in `experiments/figure8.py`): URDF friction, `StickSlip`
  Coulomb in the plant (Euler, 1 ms substeps) and `Tanh(.03)` (width 0.03 rad/s) in the
  predictor; status `done` at the end of its period.
- **Exit status** 1 on a timeout (the reach), a fault, an interruption or a raw-limit
  excess; 2 for a cancelled execution.

## Writing an experiment

An experiment is a task and an `evaluate`. The task is a frozen dataclass with
`at(t, n, dt)`, the reference and solver updates at t, `done(state, t)` and
`duration_s`, its own duration (reach's timeout, the figure eight's period).
`evaluate(record)` adds the experiment's columns and results to a run's
`RunRecord` and saves it; it reads the task from `record.config["task"]`, so the
same code runs on a saved run (`tm.load_run(path, classes=[Reach])`), and it skips
the metrics of a run without ticks. Reach's turns `done` into `arrived`.

`run_experiment(plant, mpc, task, law, timing, *, duration=None, out=None, **pieces)`
(`experiments/common.py`) runs one run for the task's duration, or at most
`duration` seconds. Given `MpcParams` as `mpc`, it compiles and builds the MPC
(minutes, unless GATO's module is cached): `MpcProcess` on `RealTime`, the in-process
`Mpc` on every other timing, with the wall-time budget only where the solve's wall
time is the latency (above), and closes it; a built MPC is the caller's, which
closes it. It records the pieces as the run's configuration (the duration it
resolved, the parameters the MPC loaded), builds the plant (`plant.build()`), runs
the timing's loop, sets the status (`done`, `timeout`, `fault`, `interrupted`;
`error` for a real error, raised again, a failed compile among them), adds the
common results, closes the plant, saves the run and exports the replay, also after
a fault or an error. Each main defines its defaults once and hands them to
`setup(args, plant, timing)`, which replaces only what the options name (`--plant`,
`--command-period`, `--update-period`, `--solve-latency`) and holds the robot's gate
(after the check-only run or a cancelled confirmation it returns the exit status
instead); the main then calls `run_experiment` and `evaluate` and prints the metadata
and the run directory. `arguments(description, argv, plants)` accepts only the plants an
experiment supports. Swap a piece by passing another one:

```python
from dataclasses import replace

import tiago_mpc as tm
from experiments.common import run_experiment
from experiments.reach import Reach, evaluate

params = replace(tm.TIAGO, solver=tm.TIAGO.solver.replace(mu=1.))
record = run_experiment(tm.SimConfig(substep=.003), params, Reach(position_tol_m=.02), tm.HoldFirst(),
                        tm.VirtualTime.constant_latency(.005))  # an in-process Mpc without the budget
evaluate(record)
print(record.metadata["status"], record.metadata["results"]["final_distance_m"])
```

- **Every piece is a frozen dataclass,** a value that compares and is recorded by
  `tm.to_dict` (with its type name) and rebuilt by `tm.from_dict`: the task,
  `MpcParams`, the plant's `SimConfig` or `RobotConfig` (`build()` makes the
  plant), the law and the timing. Extra keyword pieces of `run_experiment` (e.g.
  `case="pair_00026"`) are recorded too. Frozen configs hold tuples, not lists.
- A **control law** is a frozen dataclass called `law(plan, t, state) -> command`
  (`tm.SampleKnot()`, `tm.HoldFirst()`, `tm.PlanFeedback(kp, kd)`,
  `tm.FrictionFeedforward(friction, base_law)`); a **timing** one whose
  `run(plant, mpc, task, law, tick_callbacks, duration=...)` runs its loop
  (`VirtualTime`, `RealTime`, `Compensated`); **tick callbacks** see every `Tick` (the
  tick's time, state, command, plan, whether it is new, whether a control update
  started, its events). The pieces' interfaces are typed in `tiago_mpc/interfaces.py`:
  a plant, an MPC, a task and a timing need no base class, only what the protocols
  `PlantInterface`, `MpcInterface`, `TaskInterface` and `Timing` declare.
- `SimConfig(friction=..., friction_law=..., integrator=...)` sets the **plant's** losses,
  `MpcParams.friction` and `friction_law` the **predictor's**; the run records both.
- **Friction simulation:** a `Friction` holds only the Coulomb and viscous coefficients.
  Each consumer chooses its law: `StickSlip()` (the plant default),
  `Sign()` or `Tanh(width_rad_s)`. `SimConfig(friction=urdf_friction(), integrator="euler")`
  uses semi-implicit Euler, implicit viscous damping and a coupled stick/slip solve.
  Joints hold at zero velocity while the needed friction lies within ±c, and slip
  at c. Setting `friction_law=Sign()` instead evaluates Coulomb as `c*sign(v_old)` (it
  can chatter and creep near zero), `friction_law=Tanh(.03)` as
  `c*tanh(v_old/width)` (the predictor's law; it creeps below the threshold).
  Damping stays implicit in all three. They use the full Pinocchio mass matrix and
  bias at the old state and advance q with the new velocity. The default maximum
  substep is 1 ms (1 kHz), shortened to land on a command tick; this does not set
  the command or control rate. `integrator` is `"rk4"` (the default) or `"euler"`; a
  plant with friction names its integrator. `"rk4"` is GATO's RK4 when
  loss-free and, with `Tanh` or `Sign`, RK4 with the passive torque at every stage
  (the URDF's damping needs substeps far below 1 ms there); it refuses
  `StickSlip`, which exists only as the Euler step's solve. `passive_torque`
  likewise covers `Tanh` and `Sign` only, as a separate third argument.
  `MpcParams.friction_law` takes `Tanh` only. Numerical choices are recorded on
  their consumer by type name (`{"type": "StickSlip"}`), independently of coefficients.
  Armature defaults to zero; this experiment branch also accepts an explicit
  seven-joint reflected-inertia tuple in `SimConfig.armature` and `MpcParams.armature`.
  These plant choices do not qualify
  friction-enabled MPC or tracking.
- **Per-solve solver changes** come from the task's `at`, named in `UPDATES`
  (`mpc.py`). **Your own loop** can use `Mpc` (the `MpcProcess` interface plus
  the raw GATO `controller` and `step`, the latest solve's result) and starts the
  plant with `VirtualClock()` or `WallClock()`; given a timing dataclass whose `run`
  calls it and returns a `tm.Outcome`, it fits `run_experiment` (as `Compensated`).
  Per-plan data of an own loop travels in `plan.stats` (every key a `plans`
  column). Without `run_experiment`, `tm.RunRecord(out, **pieces)` is used on its
  own: pass it and `tm.Console()` as tick callbacks, and set the status, take the plant's
  events after the loop and save in a `finally` yourself.
- **Many runs:** an experiment's own frozen study classes, written once with
  `to_dict` before the first run, one `RunRecord` per case; `random100` is the
  example (`experiments/random100/run.py`).
- **Execution laws are package components:** `tm.PlanFeedback` adds PD around the
  interpolated plan; `tm.FrictionFeedforward` adds the specified `Tanh` or `Sign`
  losses at planned velocity, composing with `tm.SampleKnot()` by default or with
  another law. For example,
  `tm.FrictionFeedforward(tm.urdf_friction(), tm.PlanFeedback(), evaluation=tm.Tanh(.03))`.
  Both use native coordinates and are rebuilt by `tm.from_dict` and `tm.load_run`
  without extra law classes. `tm.inertia_gains(omega, damping_ratio=zeta, armature=values)`
  derives explicit PD gains from the effective inertia at `Q_START` (or a supplied
  `q`). The helper lives in `util.py` and keeps zero armature as its calculation
  default. `tm.PlanFeedback()` reads the checked-in `PD_KP` and `PD_KD` tuples in
  `tiago.py`: the chosen 20 rad/s, damping-ratio-1.4 simulation tune with armature.
  `ARMATURE_KG_M2` is stored there too; enabling it in plant/predictor remains explicit.
  Edit these literals manually. To calculate replacement values without writing files:

  ```python
  kp, kd = tm.inertia_gains(20., damping_ratio=1.4, armature=tm.ARMATURE_KG_M2)
  print(f"PD_KP = {kp!r}\nPD_KD = {kd!r}")
  ```

  Both package loops reject external friction feedforward with predictor friction
  enabled, or nested external feedforward, before starting the plant. Physical
  plant losses do not count as actuator compensation. They also warn
  (`RuntimeWarning`) when `PlanFeedback`, alone or under feedforward, runs without
  [latency compensation](docs/latency-compensation.md): the arm is held until the
  first plan takes effect, so the D term sees the plan's velocity after the first
  solve's latency as an error and can push the command beyond the effort limit at
  the start of a fast motion. The check covers the built-in composition; custom
  callable laws remain the caller's responsibility.
- **Examples, two remedies for solve latency:** `experiments/plan_feedback.py` uses
  the package PD law, and `experiments/latency_compensation.py` provides its own loop
  (solve from the predicted state) behind the timing `Compensated`;
  `experiments/latency_study.py` compares them.
- **Friction comparison:** `python -m experiments.friction_study --dry-run` describes
  48 simulation cells: those four execution controllers, the existing reach and
  figure eight, 0/20 ms declared latencies, and plant coefficients at 50/100/150 % of
  the URDF values. Predictors and gains stay fixed; reaching predicts no friction,
  tracking predicts nominal tanh friction. Run with
  `--output _expr_data/controller-friction/<fresh-name>`. Each cell runs once;
  this compares task behavior rather than wall-clock performance or robot readiness.
- **Execution-controller tuning:** [controller_tuning.py](experiments/controller_tuning.py)
  contains explicit simulation presets `TUNED_PD`, `TUNED_FEEDFORWARD` and `TUNED_BOTH`,
  selected on tracking and reaching with armature, then checked on different tasks,
  coefficients and timing. The historical `tm.PlanFeedback()` defaults stay unchanged.
  [Manual robot tuning](docs/controller-tuning-on-robot.md) explains conservative transfer;
  these presets are not hardware-qualified and the study never connects to ROS.
  The runner defaults to measured solve latency; `--solve-latency` also accepts
  declared seconds. `--scenario no_friction` removes only passive plant losses,
  retaining armature and the selected controller/predictor. `mpc_friction` and
  `mpc_friction_pd` select the internal-MPC friction comparison (no external FF).
  Evaluation records full/after-first-second plan-requested low-speed fractions,
  stall durations and eligible joint time, velocity/torque ripple, and torque
  total variation per second. Thresholds and units are recorded; these are
  descriptive metrics, not proof that a pause was caused by friction.

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
  time (`RealTime`, `RobotConfig(execute=True)`): effort control from the first
  command, zero torque and PAL's position controller again on exit, a fault or a
  stop request. `--target` (metres in
  `torso_lift_link`) is the reach target or the figure eight's `start_m`;
  `--duration` keeps first runs short; `--sea-absolute` uses SEA feedback.
- **Not run yet** against ROS, Gazebo or the robot: [docs/robot.md](docs/robot.md).

## Outputs

`RunRecord` (`tiago_mpc/record.py`, the one module of the format) writes the
provenance and configuration at the start and buffers one entry per command tick;
`save()` writes the run into `--output` or `_expr_data/<task>/<UTC time>/`
(ignored by Git). Plain JSON and npz, format 3; times count from the first state's:

- `summary.json`: `format`; `provenance` (commit, source status, package
  versions, GATO's source, `ROS_DISTRO`); `config`, the pieces (`task`, `params`,
  `plant`, `law`, `timing`, `duration`, extra ones), each `to_dict`-ed; `metadata`:
  `status`, `reason`, `end_t`, one flat `results` dict (the common results, then the
  experiment's, e.g. arrival or tracking error), `error` (a traceback) and `plant`
  (the robot's `activated`, `restored`, `cleanup_errors`; `{}` in simulation).
- `ticks.npz`, per command tick: `t`, `state_t`, `q`, `qd` (native), `command` (NaN
  when none was sent), `plan` (the plan in effect, −1 before the first), `update`
  (the control update started there, −1 for none), plus experiment columns.
- `plans.npz`, per plan: `t0`, `update_t` (the tick of its control update), `states`,
  `torques`, every solve stat, plus experiment columns. A plan that faulted is the
  last row; it never took effect, so no tick's `plan` points at it.
- `events.json`: every event, `{"t", "kind", "message", "data"}`.
- `replay.npz`: the visualizer replay on the run's time, exported from the record by
  `experiments/replay.py:export_replay` (`../tiago_mpc_visualizer/docs/replay-format.md`),
  with the `MpcParams` in `provenance.params`.

`tm.load_run(path, classes=[Reach])` reads a run back with the same interface
(`run.ticks["q"]`, `run.config["timing"]`, `run.metadata["status"]`, `run.events`):
a metric added later is the same code, and `run.save()` writes it back.
The loader rejects other format versions before reconstructing their configuration.
Format-2 runs remain readable with WIP `f66172f`; their files are unchanged.
Portable replay files keep their existing version.

More: [docs/robot.md](docs/robot.md) (robot path and checks),
[docs/safety.md](docs/safety.md) (collision-safety worker),
[docs/decisions.md](docs/decisions.md) (Felix's decisions, deferred work, open
questions), [models/README.md](models/README.md) (model provenance).

The [friction and breakaway handoff](docs/friction-breakaway-handoff.md) records
the physical-fit comparison and the proposed separate static/sliding plant model.
