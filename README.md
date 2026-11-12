# gato_tiago

Torque MPC for the TIAGo Pro right arm (seven joints, fixed torso root) with
GATO (`gato-mpc` 0.1.7): reaching and figure-eight tracking of the
grasp-center frame ([models/README.md](models/README.md#end-effector-point)) in
simulation, and the same experiments on the robot or in Gazebo through ROS 2.

A command loop runs at the command rate. At each command tick it reads the
plant's state, requests a control update when one is due, turns the latest plan
into a torque command with a control law and sends it to the plant, which holds it
until the next tick. The default law is `FrictionFeedforward()`: the planned torque,
plus PD about the plan at time t (`PlanFeedback()`, counted from the time of the
plan's state), plus the URDF's fitted friction at the planned velocity (tanh width
0.01 rad/s). `SampleKnot()` commands the knot due at t alone.

**Defaults live in the types and loops** (2026-11-12). `run_virtual_time` defaults to
`VirtualTimeSettings.constant_rate()` (500 Hz commands, 100 Hz updates, each solve
declared to take one update period: the worst case), `run_real_time` to
`RealTimeSettings()` (500 Hz commands, 100 Hz updates); both default to
`FrictionFeedforward()` and to latency compensation (`compensation="default"`, from
`default_compensation(settings)`). `run_experiment` passes the same defaults when it
gets no law or settings. Predictor friction is off by default
(`MpcParams.friction=None`, `TIAGO` and `FIGURE8`). Experiments that differ say so
explicitly.

There are two loops (`tiago_mpc/loops.py`), named after their time.
`run_real_time` runs on wall time (`WallClock`, `tiago_mpc/clock.py`): solves take as
long as they take, and a plan is used at the first tick after it arrives. It
drives the robot and the rehearsal, a `SimPlant` on the robot's architecture.
`run_virtual_time` runs controlled experiments in simulation on virtual time
(`VirtualClock`), which the loop keeps itself: the experiment declares how long a
solve lasts (`solve_latency_s`). With an asynchronous MPC it integrates the
independent plant interval before waiting at a possible activation boundary;
plans retain the request-plus-latency schedule. `run_virtual_time` requires a
`VirtualTimeSettings` object, with the presets `ideal`, `constant_latency`,
`as_fast_as_possible` and `constant_rate`. It holds only command period, update
period and solve latency. `run_real_time` requires `RealTimeSettings`, holding
the two periods. Both are plain dataclasses passed to their loop with `settings=`.
Either loop takes any MPC;
given parameters, `run_experiment` (`experiments/common.py`) builds an `MpcProcess`,
which solves concurrently in its own process, for `run_real_time` and an in-process
`Mpc` for other loops. Each loop starts the plant on its clock
(`plant.start(clock)`) and samples plans, checks leases and schedules ticks and
control updates on it. `State.t` is when the state was measured, on the same
time, and a plan starts at its state's time; on the robot a state can be older
than the tick and repeat. On the robot the safety check runs in its own process.

Both loops support optional [latency compensation](docs/latency-compensation.md):
predict the state at the next plan's intended start under the ongoing commands,
then solve from that prediction. The measured state and its age remain separate.
It is an explicit loop argument: `run_real_time` defaults to the previous observed
request-to-result delay, initially estimated from `settings.update_period` or
`settings.command_period` when the update period is zero. `run_virtual_time`
defaults to none. The standard reach, figure-eight and random100 experiments use
`default_compensation(settings)`: declared virtual latency selects the same fixed
forecast, while measured latency uses the previous observed delay with that seed.
`compensation=None` turns it off.
Its physical friction and armature model is independent of the optimizer's model.
Each compensated request makes one call to a frozen predictor. The loop keeps
the delay estimate; the predictor keeps no execution history. An older measurement
is propagated under the currently held command until the next command tick,
then under the active plan's command law on predicted feedback.

A run ends without an exception: the task done, the time up, a fault or a stop
request (Ctrl-C, SIGTERM or a run-owned cancellation flag) ends the loop, which
returns an `Outcome(status, reason, t)`. Robot execution validates the requested
and applied torque of each command. Effort handling follows evaluation of the
command law. Virtual-time simulation uses `effort_handling=EffortClip()` by
default, including when an experiment supplies its own law.
`RejectOnEffort()` stops on an excessive command;
`None` explicitly disables effort handling. Both policies accept configurable
per-joint effort limits. Neither rejects a prediction merely because an unapplied
future knot exceeds an effort limit. Command laws return `Command | Fault` and
must be pure when used for forecasting; the forecast uses the same safety and
effort policies.
Both loops use `decide_command` before sending a command: it checks the plan's
lease, evaluates the law, and applies effort handling. It returns `Command`
with requested and applied torques, or a `Fault` with the refusal reason.
Forecasting uses this same decision on hypothetical
states; a refusal makes the prediction unavailable and emits no execution event.
Exceptions are real errors, among them the robot plant's own checks and a failed
solve.

`run_virtual_time` is also the interactive simulation entrypoint. Both loops
require a finite, nonnegative duration and reject runs over ten minutes before
starting the plant. Virtual runs disable cyclic garbage collection and restore
the caller's previous GC policy on exit.
`safety=ExecutionSafety(max_plan_age_s=None)` disables the measurement-age expiry;
a finite plan still ends at its horizon. Its default law is `FrictionFeedforward()`;
`safety`, `effort_handling` and `compensation` are separate run arguments. A custom
law still receives the default safety and effort handling. Real-time execution
retains its mandatory strict checks and rejects clipping.

```python
settings = tm.VirtualTimeSettings.constant_latency(.005)
outcome = tm.run_virtual_time(
    plant, mpc, task, settings=settings, duration=1.,
    safety=tm.ExecutionSafety(max_plan_age_s=.1),
    effort_handling=tm.EffortClip(), compensation=None,
)
```

`ExecutionSafety` implements the structural `SafetyInterface` protocol. A user
can provide a different pure `validate(plan, t, state) -> Fault | None` check.
Validation runs before the law, and the loop applies only an accepted command.
Duration, callbacks, cancellation and initial conditions are also loop arguments.

Both loops expose three independent notification streams:

| Parameter | Payload | Reports |
| --- | --- | --- |
| `tick_callbacks` | `Tick` | Measured state and command decision |
| `solve_completed_callbacks` | `SolveCompleted` | Returned plan and its request |
| `diagnostic_callbacks` | `Diagnostic` | Overruns, lateness, breaches and plant diagnostics |

The loop receives plans directly from `mpc.result()` before notifying observers.
Each notification goes only to its own callback
group; diagnostics are not bundled into `Tick`. `before_tick_callbacks` is a
separate group of control hooks for pausing and capacity handling.
Real-time notifications follow the command decision and publication. Virtual-time
plant diagnostics reach observers before the pause checkpoint.

Every request keeps an owned `last_mpc_request_snapshot` of its state, reference
and solver updates, including when no observers are registered. A `SolveCompleted`
reports that snapshot and the returned native `Plan`, including a result which
never activates. `plan.stats['wall_s']` supplies the measured solve latency;
foreground timing belongs to optional profiling. `Tick.requested_command`
and `Tick.clipped` distinguish requested torque from `Tick.command` applied to
the plant. Observers should enqueue these owned numerical values and format them
outside the numerical path.

`RunRecord` records every returned plan through `record.solve_completed`, including
plans that never activate. Its tick callback records which plan was active;
`ticks['plan']` indexes the completed-plan rows. `run_experiment` registers both
callbacks, plus `record.diagnostic`, automatically.
`Tick.plan` is the active plan; `Tick.plan_activated` marks only the tick where it
takes effect. A returned plan may remain pending until its activation time while
the active plan continues supplying commands.

Each callback in `before_tick_callbacks` receives the boundary `State` and may
block on a condition/event to pause virtual time without losing pending plans or
controller warm state. Callbacks run once in registration order per command
boundary, before cancellation, result collection or another command. An outstanding
solve may finish while the loop is parked; its reply is consumed normally after resume. Set
`cancelled` to a run-owned flag's `is_set` method, then wake a paused callback to
terminate it. Cancellation is checked at this one boundary checkpoint. Once a
tick is admitted, it completes its command decision and physics interval before
another checkpoint, even if a solve or observer requests cancellation meanwhile.
Waiting for a solve can delay cancellation in wall time. A replacement run gets
a fresh flag/controller. `start_time_s` and
`initial_command` support anchored state and a previously held actual torque;
tasks still see time relative to the run's origin. Pacing is optional and belongs
to the caller or `SimPlant(sync_real_time=True)`, with only one pacing mechanism.
Pause requests raised by Tick observers take effect at the next ordinary
boundary. Pausing retains the invocation; reset cancels it and creates its replacement.
Results are collected once at the first possible activation boundary. For
declared latency this is the first eligible command tick; measured latency
conservatively waits at least until the next command tick to learn the result's
delay. This includes a measured duration of zero. Zero declared latency with no
future predicted origin waits immediately. A collected plan waits for its
scheduled availability and predicted origin before activation.

Each virtual-loop iteration handles one complete command tick, then advances the
plant to the next command time. The loop's periodic schedules are:

| Schedule | Period | Action |
| --- | --- | --- |
| Commands | `settings.command_period` | Read state, handle MPC work and publish a command |
| MPC updates | `settings.update_period` | Determine when a command tick may request an update |

Both use `Schedule`: `next_at` is the nominal next slot, and `advance`
consumes it while skipping missed slots without shifting the periodic phase.
The command schedule's pure `align` method rounds the forecast onto an eligible
command boundary without advancing the schedule. The command cursor advances
after publication.

A command tick reads the state, delivers diagnostics, runs pause callbacks and
checks cancellation. It receives and activates eligible earlier work, then may
submit one MPC request. If that new result is immediately eligible, the same tick
receives and activates it before deciding and publishing its command. Collection
checks use the scheduled time and predicted origin; a zero declared latency with
a future prediction still waits. Result reception and reporting share a small
helper, while plan selection and timing updates remain explicit in the loop.
Each tick follows this sequence once, so it submits at most one request.

`SimPlant.advance()` integrates the held command in `SimConfig.substep` steps,
shortening the final step to reach the next command time. This integration step
is independent of the command and MPC update periods. An asynchronous MPC can
solve during integration. A paced `SimPlant` waits once per command interval.
After the final tick, completion or a fault collects any outstanding solve without
activating it or advancing physics, leaving a caller-owned MPC ready for another run.

For full-physics observation, assign `SimPlant.step_callback(state)`. It returns
events to queue at each integrated endpoint. Events retain their detection time;
the loop delivers them at the next command boundary. Runtime callbacks are not
part of the serialized `SimConfig`.

```
tiago_mpc/    __init__.py (the root: `import tiago_mpc as tm` re-exports what experiments use),
              util.py (hz_to_s; the cache: cache_root, cached, cache_meshes, digest),
              tiago.py (TIAGo facts, the derived URDFs), dynamics.py (friction laws),
              mpc.py (MpcParams, TIAGO, plant names, find_binary, compile_binary, SolverSettings,
              Mpc, MpcProcess),
              interfaces.py (State, Plan, Command, Fault, Diagnostic, Tick, SolveCompleted; the protocols PlantInterface,
              MpcInterface, TaskInterface, SafetyInterface; ControlLaw, EffortHandling,
              TickCallback, SolveCompletedCallback, DiagnosticCallback),
              clock.py (VirtualClock, WallClock), laws.py (SampleKnot, HoldFirst,
              PlanFeedback, FrictionFeedforward, ExecutionSafety, RealTimeSafety, RealTimeEffort,
              EffortClip, RejectOnEffort, decide_command),
              loops.py (run_real_time, run_virtual_time, RealTimeSettings, VirtualTimeSettings, Outcome,
              STOP, default_compensation),
              prediction.py (LatencyCompensation and physical state propagation),
              sim.py (SimConfig, SimPlant), robot.py (RobotConfig, RosPlant),
              ros2.py (ROS 2 client), safety.py,
              record.py (the run format: to_dict, from_dict, CLASSES, provenance, Console,
              RunRecord, load_run),
              collision.py (Collision: GATO's collision rows; the stored covers arm_cover,
              body_cover, body_posture), data/collision/ (the stored covers, one JSON each)
experiments/  common.py (run_experiment, common_results, handle_stop_signals, and the mains'
              arguments and setup), reach.py, figure8.py, random100/, replay.py, obstacles.py,
              clearance.py (the exact clearances the covers approximate)
tiago_scripts/ offline generators of the checked-in data: arm_cover.py, body_cover.py,
              validate.py (tiago_scripts/README.md)
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
- **GATO** is `gato-mpc` 0.1.7 from `../GATO/.worktrees/tiago-combined`,
  selected in `pyproject.toml` and `uv.lock`. That local source supplies fixed-arm
  armature support, the native ADMM-budget accessor and per-SQP collision-row
  caching. It uses `grid-rbd` 0.6.0 from `../GRiD-tiago-armature-060` for armature
  dynamics. Both sibling sources are required for `uv sync --frozen` with this lock.
  Their native contents identify the compiled module in the shared cache.
  Predictor friction is registered with
  `joint_dynamics=True, friction_width=w`: GRiD generates the GPU URDF's
  damping and tanh Coulomb friction into the dynamics and derivatives, with the
  width compiled in. GATO's per-SQP-iteration trajectory capture (the old fork's)
  is not available. Each record's `summary.json` names the GATO it ran
  (`provenance.gato`: version, `direct_url.json`, `gato.__file__`, and an editable
  checkout's commit and source status).
- **The solver settings follow GATO's defaults** unless a preset sets a value
  (`MpcParams`' docstring lists them); records hold the values GATO's controller
  resolved and ran, read back from it.
  `tests/test_gato_defaults.py` lists the deviations of `TIAGO` and `FIGURE8` and
  fails when GATO's defaults differ from `tests/gato_defaults.json`; after checking
  the changes, `python tests/test_gato_defaults.py` rewrites that file.
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
content is part of its module key, so the fork's modules sit beside the release's in
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
  `Mpc`, on the main's `VirtualTimeSettings`. Reach, figure8, obstacles and random100
  use the defaults above: `VirtualTimeSettings.constant_rate()`, 500 Hz commands and
  100 Hz updates, each solve declared, not measured, to last one update period,
  10 ms, with latency compensation. `--solve-latency` says how long a solve lasts on the timeline: declared
  seconds (0: the ideal) or `measured`, the solve's own wall time
  (`plan.stats["wall_s"]`). A plan takes effect at the first command tick at or
  after its request plus that latency; no new request is made while a solve is
  outstanding. A solve that really took longer than a declared latency above 0 is
  an `overrun` event. `--plant rehearsal` runs the rehearsal:
  `SimPlant` on `run_real_time` (`RealTimeSettings`) with `MpcProcess`, as on the robot; a tick
  whose work ran past the next one is late, skips the missed ticks and is a
  `late_tick` event. `SimConfig(sync_real_time=True)` paces a virtual-time run to the
  wall clock for watching, without changing its results. The run records its timing
  as `config.settings`, with the loop name and execution components recorded
  separately. Until the first command, `SimPlant` holds
  its start state, as PAL's position controller holds the arm until effort
  control starts.
- **Diagnostics:** what the loop or a plant detects at its own time (`late_tick`,
  `overrun`, `late_plan`, `pacing_behind`; the robot's `activated` and `handback`)
  reaches `diagnostic_callbacks` independently of command ticks. A plant hands
  its diagnostics over with `take_diagnostics()`. `RunRecord.diagnostic` records
  them in `events.json`; `Console` prints the first diagnostic of each kind.
  The common results count `late_ticks`,
  `overruns` and `late_plans`.
- **Stopping:** Ctrl-C or SIGTERM sets `tm.STOP` (`handle_stop_signals`, which every
  main that runs the simulation or the arm calls first): the run ends as
  `interrupted` and is saved. Virtual time checks at its ordinary command-boundary
  checkpoint and can exit without a terminal Tick; real time records its next tick
  without a command. Later runs stop while the flag remains set. A second
  signal raises `KeyboardInterrupt`, the hard abort for blocking calls (compiling,
  the robot's first state); the run is still saved. The MPC process and the safety
  worker ignore SIGINT; the cancelled `EXECUTE` prompt exits with status 2.
- **Solve time and warm-up:** `MpcParams.max_sqp_time_s` (10 ms) stops a solve's
  SQP iterations by wall time, within `solver.max_sqp_iters` (5). It follows the
  timing: when `run_experiment` builds the MPC, it keeps the budget only where the
  solve's wall time is the latency (`RealTimeSettings`: the rehearsal and the robot;
  `VirtualTimeSettings` with `measured`) and sets None for every other timing (a declared
  `--solve-latency`, `Compensated`: the solve's wall time is irrelevant by
  assumption); random100 (a declared latency) and the latency study build their own MPC
  with None.
  Before the first tick, both loops call `mpc.warmup` with the started plant's
  state, the arm still held: with `MpcParams.warmup` (default), GATO's warm-up
  solve (the solver's own iteration limit, no time limit) seeds the first solve,
  and its time counts against no plan's lease.
- **Limits:** the joint, speed and torque limits are ADMM rows with GATO's
  default settings (`AdmmLimits`), except `TIAGO`'s rho_u of 5e-4 for the
  control box (the figure eight keeps GATO's). The
  joint box is the URDF's shrunk by `joint_margin_rad` (0.1 rad); `Q_START`
  (`tiago.py`), where the runs start, lies inside it.
- **Late plans:** a plan that first takes effect a knot interval or more after
  its state never runs its knot 0, and `SampleKnot` may diverge; each such plan is a
  `late_plan` event ([docs/robot.md](docs/robot.md#latency)).
- **Rates:** `--command-period` sets the command rate (default 2 ms, 500 Hz,
  in simulation and on the robot), `--update-period` the control rate (default
  10 ms, 100 Hz; a constant rate's declared latency follows it; TIAGO's knot interval is 30 ms, FIGURE8's 10 ms; 0: back to back, a request at the tick
  the previous plan takes effect); either may differ from the knot interval. Only
  the options given replace the timing's values. `--duration S` ends a run after at
  most S seconds. In code, a period written as a rate is `tm.hz_to_s(hz)`
  (`hz_to_s(500) == .002`).
- **reach:** `TIAGO` defaults (32 knots of 30 ms, planning speeds at 60 % of the raw
  limits), loss-free with the fitted armature, 3 ms RK4 substeps, the default
  execution, from `Q_START`
  (PAL's comfortable pose with joint 4 at −2.33 rad, not −2.35); arrival within
  5 cm with the grasp point slower than 2 cm/s, timeout 8 s.
- **figure8:** `FIGURE8` (in `experiments/figure8.py`): URDF friction, `StickSlip`
  Coulomb in the plant (Euler, 1 ms substeps), a loss-free predictor and the default
  friction feedforward; status `done` at the end of its period.
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

`run_experiment(plant, mpc, task, law, settings, *, loop=run_virtual_time, ...)`
(`experiments/common.py`) runs one run for the task's duration, or at most
`duration` seconds. Its `"default"` compensation and effort options select the
chosen loop's defaults; `None` explicitly disables the optional component.
The record stores the resolved objects. Given `MpcParams` as `mpc`, it first compiles and builds the MPC
(minutes, unless GATO's module is cached): `MpcProcess` for `run_real_time`, the in-process
`Mpc` for other loops, with the wall-time budget only where the solve's wall
time is the latency (above), and closes it; a built MPC is the caller's, which
closes it. A failed compile is a setup error and leaves no run directory. It then
records the pieces as the run's configuration (the duration it resolved, the MPC's
params, which GATO resolved), builds the plant (`plant.build()`), calls the supplied
loop with its settings, sets the status (`done`, `timeout`, `fault`, `interrupted`; `error` for a real
error, raised again, a failed build among them), adds the
common results, closes the plant, saves the run and exports the replay, also after
a fault or an error. Each main defines its defaults once and hands them to
`setup(args, plant, settings)`, which returns the plant, settings and loop function
and replaces only what the options name (`--plant`,
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

params = replace(tm.TIAGO, solver=replace(tm.TIAGO.solver, mu=1.))
record = run_experiment(tm.SimConfig(substep=.003), params, Reach(position_tol_m=.02), tm.HoldFirst(),
                        tm.VirtualTimeSettings.constant_latency(.005))  # an in-process Mpc without the budget
evaluate(record)
print(record.metadata["status"], record.metadata["results"]["final_distance_m"])
```

- **Configuration pieces are dataclasses,** recorded by `tm.to_dict` (with their type
  name) and rebuilt by `tm.from_dict`: the task,
  `MpcParams`, the plant's `SimConfig` or `RobotConfig` (`build()` makes the
  plant), the law, safety, effort handling, compensation and timing settings. Extra keyword pieces of `run_experiment` (e.g.
  `case="pair_00026"`) are recorded too. The package's configs are frozen and hold tuples;
  a supplied native cost menu retains its mutable terms and arrays.
- A **control law** is a frozen dataclass called `law(plan, t, state) -> Command | Fault`
  (`tm.SampleKnot()`, `tm.HoldFirst()`, `tm.PlanFeedback(kp, kd)`,
  `tm.FrictionFeedforward(friction, base_law)`); **timing settings** are plain data
  passed to an explicitly selected loop. **Tick callbacks** see every `Tick` (the
  tick's time, state, command, plan, whether the plan activated, whether a control update
  started, and requested/applied torque handling). The pieces' interfaces are
  typed in `tiago_mpc/interfaces.py`:
  plants, MPCs, tasks and safety checks need no base class, only what the protocols
  `PlantInterface`, `MpcInterface`, `TaskInterface` and `SafetyInterface` declare.
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
  Armature, a seven-joint reflected-inertia tuple in `MpcParams.armature`, `SimConfig.armature`
  and `LatencyCompensation.armature`, defaults to the fitted `ARMATURE_KG_M2` in all three
  (one chirp per joint; `models/tiago_pro/README.md`); `(0.,) * 7` turns it off.
  These plant choices do not qualify
  friction-enabled MPC or tracking.
- **Costs:** `MpcParams.cost` is always a native GATO `CostMenu`. `default_cost(n, ...)`
  constructs the default position, terminal position, joint-velocity and torque terms,
  plus optional orientation. Zero-weight terms are omitted; orientation is off by default.
  The constructor returns an ordinary menu, with no separate builder configuration.

  ```python
  task = Reach(orientation_rad=tuple(tm.ee_orientation(tm.Q_START)))
  params = replace(tm.TIAGO, cost=tm.default_cost(
      tm.TIAGO.N, orientation=.2, terminal_orientation=4.))
  ```

  Menu weights can be per knot, so changing the horizon also requires matching menu weights,
  for example `replace(tm.TIAGO, N=64, cost=tm.default_cost(64))`.

  Reference updates are explicit: `with_reference(menu, reference)` returns a menu with
  new frame-position and orientation targets. MPC `warmup` and `request` perform that
  operation when given a reference; pass None to use the menu's existing targets.
  Standard tasks supply `(n, 6)` references: grasp-frame position and rotation vector in
  `ROOT_FRAME`. Three NaN orientation values disable that knot's orientation target.
  Other native terms and targets are preserved. Constant orientations need only one
  rotation conversion; unchanged menus and references reuse the applied menu. Native
  menu mutations are detected. `MpcProcess` receives its own copy at startup.

  Custom objectives use the same type:

  ```python
  from gato.cost_menu import CostMenu, Term

  params = replace(tm.TIAGO, cost=CostMenu(
      frame_pos=Term(weight=4., target=(.4, -.2, .9)),
      input=Term(weight=1e-4)))
  # In your own loop, mpc.request(state, None) retains this menu's targets.
  ```

  `tm.to_dict`, `tm.from_dict` and run records rebuild native `CostMenu`, `Term`, and
  `LinearTerm`; NumPy arrays reload as tuples accepted by GATO. Arrival remains based
  on position and speed; tasks with an orientation target also record angular error.
- **Solver settings:** `MpcParams.solver` is `SolverSettings`, which follows GATO's
  numerical defaults and retains its joint-limit barrier weights. Tracking weights live
  exclusively in `cost`. The wrapper's parity test catches new fields and changed defaults
  in GATO. Both barrier and ADMM limits continue to apply with a menu. Legacy tracking
  setters (`cost_weights`, `knot_weights`, `posture`) are unavailable; the `external_force`
  task update remains available.
- **Per-solve solver changes** come from the task's `at`, named in `UPDATES`
  (`mpc.py`). **Your own loop** can use `Mpc` (the `MpcProcess` interface plus
  the raw GATO `controller` and `step`, the latest solve's result) and starts the
  plant with `VirtualClock()` or `WallClock()`. Pass its function with `loop=` and
  its settings dataclass to `run_experiment` (as for `run_compensated`). It returns
  a `tm.Outcome` and accepts the explicit execution components.
  Per-plan data of an own loop travels in `plan.stats` (every key a `plans`
  column). Without `run_experiment`, `tm.RunRecord(out, **pieces)` is used on its
  own: pass the record as a tick callback and its `diagnostic` method plus
  `tm.Console()` as diagnostic callbacks. Set the status and take the plant's
  diagnostics after the loop and save in a `finally` yourself.
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
  `ARMATURE_KG_M2` is stored there too, the default armature of predictor, plant and compensation.
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
  time (`RealTimeSettings`, `RobotConfig(execute=True)`): effort control from the first
  command, zero torque and PAL's position controller again on exit, a fault or a
  stop request. `--target` (metres in
  `torso_lift_link`) is the reach target or the figure eight's `start_m`;
  `--duration` keeps first runs short; `--sea-absolute` uses SEA feedback.
- **Not run yet** against ROS, Gazebo or the robot: [docs/robot.md](docs/robot.md).

## Outputs

`RunRecord` (`tiago_mpc/record.py`, the one module of the format) writes the
provenance and configuration at the start and buffers one entry per command tick;
`save()` writes the run into `--output` or `_expr_data/<task>/<UTC time>/`
(ignored by Git). Plain JSON and npz, format 4; times count from the first state's:

- `summary.json`: `format`; `provenance` (commit, source status, package
  versions, GATO's source, `ROS_DISTRO`, the 8-digit content hash of each stored
  collision cover); `config`, the pieces (`task`, `params`,
  `plant`, `law`, `settings`, `loop`, `compensation`, `safety`, `effort_handling`,
  `duration`, extra ones), each `to_dict`-ed; `metadata`:
  `status`, `reason`, `end_t`, one flat `results` dict (the common results, then the
  experiment's, e.g. arrival or tracking error), `error` (a traceback) and `plant`
  (the robot's `activated`, `restored`, `cleanup_errors`; `{}` in simulation).
- `ticks.npz`, per command tick: `t`, `state_t`, `q`, `qd` (native), `command` (NaN
  when none was sent), `plan` (the plan in effect, −1 before the first), `update`
  (the control update started there, −1 for none), plus experiment columns.
- `plans.npz`, per plan: `t0`, `update_t` (the tick of its control update), `states`,
  `torques`, every solve stat, plus experiment columns. A waiting plan that expired
  before activation is the last row; no tick's `plan` points at it.
- `events.json`: every event, `{"t", "kind", "message", "data"}`.
- `replay.npz`: the visualizer replay on the run's time, exported from the record by
  `experiments/replay.py:export_replay` (`../tiago_mpc_visualizer/docs/replay-format.md`),
  with the `MpcParams` in `provenance.params`; with an orientation target, the reference
  also has its rotation vectors (`reference_rotvec_rad`, NaN at a tick without one).

`tm.load_run(path, classes=[Reach])` reads a run back with the same interface
(`run.ticks["q"]`, `run.config["settings"]`, `run.metadata["status"]`, `run.diagnostics`):
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
