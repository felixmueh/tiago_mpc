# Decisions

Felix's lasting decisions for this repository, by topic, with dates and reasons
where known. Pending implementation is marked explicitly; git keeps superseded wording. The
2026-10-22 refactor implemented the approved design in the thesis workspace
(`agent-reading/tiago-mpc-design/`: requirements, design, implementation plan).

## Purpose and shape

- **2026-11-13: Jetson PCG/BDSV is a B1 horizon sweep** (Felix).
  Update the temporary Jetson experiment delivery to compare plain PCG and
  BDSV at N16/32/64/128/256/512 with **batch size one at every horizon**.
  Retain the four-second figure-eight, SQP1, limit barriers/ADMM off,
  20 Hz updates, direct torque, friction mismatch and unshifted warm starts.
  Keep the separate horizon × batch matrix unchanged, including B1/2/4/8/16.
  Check native batch identity and retain scalar stage times, accepted steps,
  completion timestamps and available clock/load logs to investigate timing
  bands. Three repeats per solver/N give 36 episodes / 2,880 planned updates.
  This updates the already authorized temporary `experiment/jetson-sqp1-stats`
  publication; the development worktree stays uncommitted.

- **2026-11-12: plain PCG/BDSV with barriers and slower updates** (Felix).
  Disable ADMM and use GATO's relaxed limit barriers, preserving the planning
  bounds. Compare `linsys="pcg"` and `"bdsv"`; the ADMM inner solver is no
  longer the comparison. Use a 20 Hz control-update rate (50 ms), chosen for
  Felix's request to slow updates, with 500 Hz commands. Keep the four-second
  figure-eight, N128/SQP1, PCG120, fitted plant friction, loss-free predictor,
  direct torque without PD/feedforward, unshifted retained warm starts and no
  reseeding. Use default barrier mu=0.003 and delta=0.05. Three repeats per
  solver produce 80 updates per run, 480 total if all cases complete. Preserve
  all earlier ADMM cohorts separately and retain timing/work/quality statistics.

- **2026-11-12: four-second figure-eight PCG/BDSV follow-up** (Felix).
  Try the same N128/SQP1 direct-torque/friction-mismatch comparison with a
  four-second figure-eight period. Record one full cycle with three fresh
  repeats per solver; the other settings stay unchanged. Do not rerun the
  reaching task. Keep this cohort separate from the six-second measurements.
  Felix also confirmed the absence of an external feedforward controller:
  execution samples planned torque only, without added friction feedforward or PD.

- **2026-11-12: harder PCG/BDSV presentation comparison** (Felix; supersedes
  the histogram workload below). Use N128, fitted friction in the simulated
  plant, a loss-free predictor and direct planned torque (`SampleKnot`), with
  no PD or friction feedforward. Reuse the winning trajectory without shifting
  (`warm_start="hold"`); automatic reseeding stays off. Keep the one-SQP budget,
  two six-second tasks, three repeats, zero declared solve latency and statistics-only
  recording. Run this new GTX 1070 cohort separately from the retained N32/SQP5
  pilot, and inspect the earlier GATO timing/CDF evidence. Timing multimodality
  is a hypothesis to measure. The thesis horizon × batch matrix is unchanged.

- **2026-11-12: PCG/BDSV presentation experiment** (Felix). Compare only
  `admm_linsys="pcg"` and `"bdsv_factor"` on a six-second figure-eight and a
  medium reach with identical simulator settings and zero declared solve latency,
  sweeping N16/32/64/128/256/512. Rerun the thesis timing matrix through N512
  with B1/2/4/8/16 as well. **Both studies use one SQP iteration and save only
  statistics, not trajectories.** Keep per-update solve-time distributions and
  variance as the primary histogram output. Prepare the studies for Felix to run
  on Jetson; no further GTX 1070 runs are needed. The earlier N32/SQP5 local
  pilot stays separately labelled. Automatic switching and other presentation
  experiment proposals are outside these studies.
  Delivery is a temporary `gato_tiago` branch with an executable script at the
  repository root; Felix pulls the branch and runs that script on Jetson. This
  explicitly authorizes publishing this temporary branch. Remove it after use,
  keeping the recorded statistics and reports.

- **2026-11-12: defaults live in the types and loops** (Felix; supersedes the timing and
  law of the 2026-10-30 entry below).
  - **Principle.** Parameters passed to GATO follow GATO's default (None) unless TIAGo
    needs another value, which `MpcParams` sets with a comment and
    `tests/test_gato_defaults.py` lists. Parameters only we use get their sensible
    default in our own types. Only experiments that differ set their own values.
  - **Timing.** `run_virtual_time` defaults to `VirtualTimeSettings.constant_rate()`:
    each solve declared to take one update period, the worst case, deterministic
    (measured latency makes simulated runs depend on GPU load). `VirtualTimeSettings`
    itself has no field defaults; its presets stay, including `constant_latency` for
    studies of latency at a fixed rate. `RealTimeSettings()` updates at 100 Hz, as the
    simulation (500 Hz commands in both).
  - **Compensation** defaults to `default_compensation(settings)` in both loops and
    in `run_experiment`; `None` turns it off.
  - **Law.** Both loops and `run_experiment` default to `FrictionFeedforward()`, the
    robot mains included: the URDF's fitted friction (`urdf_friction()`, refit B of
    2026-10-30), `PlanFeedback()`'s stored gains and a 0.01 rad/s tanh width.
  - **Predictor friction** is off by default: `FIGURE8` no longer predicts it, like
    `TIAGO` and `MpcParams`.
  - **Friction values.** The package and URDF hold only the refit B values; the
    16 October constants stay in the friction experiment, which pins its historical
    laws (`SampleKnot()` base, `Tanh(.03)`) explicitly. The latency study, the
    friction studies and `plan_feedback.py` pass `compensation=None` and the latency
    study its historical figure-eight predictor friction explicitly.
  - Random100 and obstacles follow the new defaults; their earlier runs were removed
    (Felix keeps backups) and are regenerated when needed.

- **2026-11-07: the package keeps only what the MPC needs; the body cover for the safety
  monitor is parked** (Felix, refactor of the entry below).
  - **`collision.py`** keeps `Collision` and the cover loading (`arm_cover`, `body_cover`,
    `body_posture`, `cover_hashes`, `cover_file`) with `cuboid`. `ExactCheck`, the reduced robot
    and the mesh helpers leave the package: the mesh helpers move to `tiago_scripts/validate.py`,
    the evaluation to `experiments/clearance.py`, which `obstacles.py` uses.
  - **Posture:** `BodyPosture` and `Collision.posture` are removed. The posture is the chosen
    body cover's (named joint positions); `body=None` uses the default cover's, because it is
    the same robot. The measured values are `tiago_scripts/body_cover.POSTURE`, the generator's
    input; a test compares each stored cover's posture with it.
  - **One robot model:** generation and evaluation use safety's `build_tiago_collision_model`
    with the full URDF's original meshes, every joint but the right arm's locked.
  - **No body cover for the safety monitor for now:** a cover of safety's alpha-wrap body grown
    by the monitor's 4 cm is parked on `experiment/body-cover-safety` (its NOTES.md: the fit
    exceeds the container's memory at 5 mm, and the default start pose has only 0.6 cm of
    monitor slack, so such a cover puts it inside; `todo.org`).
  - **Arm:** stays `arm/exact`. The monitor checks the arm's alpha-wrap meshes, about 1 cm outside
    the original ones, which the spheres do not certainly contain, so it may trip up to about
    1 cm early on the robot; noted at `collision.arm_cover` and in `todo.org`.
  - **Evaluation:** `experiments/clearance.py` measures the original meshes, as the removed
    `ExactCheck` did. `record.FORMAT` stays 4.

- **2026-11-07: stored collision covers, generated offline by `tiago_scripts`** (Felix).
  Nothing is fitted at run time any more; `Cover` and `fit_spheres` are gone.
  - **Scripts:** the package `tiago_scripts/`, run as `python -m tiago_scripts.<name>`
    (README there): `arm_cover` (the greedy's candidates, selected by an exact HiGHS MILP;
    a time limit or a non-optimal status fails, with no greedy fallback and no tie-break;
    it records the SciPy and HiGHS versions), `body_cover` (the weighted greedy plus
    redundant-shape removal, `--choice balanced|tighter`, one pass for both) and
    `validate` (the validation both use). The covers' viewer, the GPU distance bench and
    `tune_costs.py` were dropped. mypy does not check `tiago_scripts/`.
  - **Data:** one JSON per cover in `tiago_mpc/data/collision/`: `arm/exact.json` (28
    spheres per link; the default) and `body/balanced.json` (83 primitives; the default)
    and `body/tighter.json` (155, from the one-pass generator; the 2026-11-04 tighter had
    156 from a two-step run). Each holds the primitives by shape in the collision layouts,
    the frame, the posture (body), the inputs (URDF and per-part geometry sha256), the
    settings, a validation summary and the library versions; no absolute paths, no
    intermediates.
  - **Runtime:** `Collision.arm = "exact"` and `Collision.body = "balanced"` (or None) name
    the stored covers, which load when the MPC and the environment are built.
    `environment()` merges the body's spheres, capsules and cuboids with the obstacles'.
    No checks at load: `tests/test_collision.py` compares the stored inputs and posture with
    the current model. Records store the names and, in their provenance, an 8-digit content
    hash of each cover file; `record.FORMAT` became 4. Replays carry the body cover's
    capsules and boxes (`body_capsules`, `body_boxes`).

- **2026-11-04, revised 2026-11-07: body cover decimation is postponed** (Felix).
  On 2026-11-04 Felix decided to keep four static body covers, balanced and tighter, each
  original and decimated. On 2026-11-07 decimation was postponed: the decimated
  `tighter` leaves the torso base's mesh up to 1.2 mm outside its primitives (certified
  unreachable only for the current arm cover). Only complete covers land. The decimation,
  the four 2026-11-04 covers with their decimation records and their re-decimation
  against the exact arm cover are kept on branch `experiment/body-cover-decimation`; the
  follow-up (one enclosing box for the unreachable back) is in `todo.org`.

- **2026-11-03: retain local GATO armature and collision-cache work until upstream** (Felix).
  The shared-loop integration selects `../GATO/.worktrees/tiago-combined`
  and `../GRiD-tiago-armature-060` through `pyproject.toml` and `uv.lock`.
  Keep the fixed-arm armature support, native ADMM-budget accessor and per-SQP
  collision-row caching when merging; use these local sources until upstream
  supplies the required changes. Source identity remains part of the compiled
  module cache key and each run's provenance.

- **2026-11-03: use GATO cost menus for every MPC objective** (Felix).
  `MpcParams.cost` is always a native `gato.cost_menu.CostMenu`. `default_cost(n, ...)`
  constructs the default position, terminal position, joint-velocity, torque and optional
  orientation terms. It returns an ordinary menu, with no persistent builder object.
  `with_reference(menu, reference)` explicitly updates frame-position and orientation
  targets. Passing a reference to MPC warmup/request performs that update; passing None
  retains the menu's targets. Separate
  orientation fields on `MpcParams` and legacy tracking-weight updates are removed.
  `SolverSettings` wraps GATO's numerical settings and limit-barrier weights, excluding
  the four tracking weights. A parity test checks every remaining native field and
  default; its purpose is explained in a source comment. Settings and native menus
  use the existing dataclass serializer. Local construction skips inactive terms,
  shares constant rotation targets, and reuses unchanged menus/reference windows while
  checking for changes to native menu fields. GATO is unchanged.

- **2026-11-03: control readiness precedes lossless capture readiness** (Felix).
  Keep the numerical plan handoff independent of capture serialization. The
  visualizer reserves bounded recording storage and lets its independent
  consumer wait for publication; the solver retains diagnostic ownership until
  publication. Pause uses the next ordinary command-boundary checkpoint and
  retains outstanding solver replies until normal collection after resume.

- **2026-11-03: overlap virtual-time physics with asynchronous MPC** (Felix).
  Keep solver and plant in separate processes, submitting before the independent
  integration interval and collecting at a possible activation boundary. Retain
  declared/measured latency, request/origin timestamps, command laws and the
  existing virtual clock. The before-tick callback may park without collecting a
  pending reply. Cancellation ends the invocation at a command boundary, and
  caller teardown finishes outstanding solver work. Control and lossless recording use separate buffers in
  the visualizer. Forecast optimization remains deferred.

- **2026-11-03: one virtual-time engine for experiments and the visualizer** (Felix).
  Reuse `run_virtual_time` with small typed messages and a blocking callback at
  command boundaries. `SolveCompleted` records completed results independently
  of activation; existing `Tick` callbacks retain command decisions. Observer
  callbacks queue detached numerical data; formatting, geometry and delivery run
  independently. A run-owned cancellation flag is checked at the ordinary command
  boundary. Cancellation wakes a
  paused callback and ends that run; reset/reconfiguration reconstructs it. No
  optimizer checkpoint/restore or general event framework is added.

- **2026-10-27: compare sticking/smoothness and remove plant friction** (Felix).
  Extend the controller-tuning experiment with sustained low-speed episodes
  conditioned on motion requested by the active plan, duration distributions,
  velocity ripple and applied-command variation. Report tracking error and
  eligible request time with these metrics. The no-friction plant retains
  armature, timing, integrator and frozen controller estimates. Fresh comparisons
  use measured solve latency; retained runs keep their original declared timing
  and are analyzed read-only. Static-friction realism is a separate design
  investigation, not an unannounced change to this comparison's physics.

- **2026-10-27: plan feedback and friction feedforward are package laws** (Felix).
  `PlanFeedback`, `FrictionFeedforward` and the armature-aware `inertia_gains`
  helper live in `tiago_mpc/laws.py` with plain root re-exports. Experiments compose
  them through `tm`; record loading knows both law types, including their nested
  composition. The former experiment definitions are removed. Historical PD
  defaults and interpolation stay unchanged; tuned presets remain explicit in
  `experiments/controller_tuning.py`. This promotion adds no robot qualification.

- **2026-10-26: an experiment is a task and an `evaluate`; each main sets up its
  run** (Felix; the design is `agent-reading/tiago-mpc-design/experiment-structure-design.md`
  in the thesis workspace, its last section deciding the review's points). It replaced
  `launch` and the experiment functions `reach()` and `figure8()`, which took every
  system piece with a default only to hand it on, and the mains' copies of those defaults.
  - **An experiment** is its task dataclass and `evaluate(record)`: its metrics as a
    function of the record, reading the task from `record.config["task"]`, so the same
    code runs live and on `load_run(path, classes=[Reach])`. It skips the metrics of a
    run without ticks and ends with `record.save()`; reach's sets `"arrived"` for
    `"done"`. `TaskInterface.duration_s` is the task's own duration (reach's timeout,
    the figure eight's period); the loops never read it.
  - **Defaults once per main:** each main defines its plant config, parameters, task,
    law and timing once, calls `setup`, `run_experiment` and `evaluate` itself, then
    prints the metadata and the run directory (two lines, no `report` helper). Python
    callers (random100, the studies, the tests) pass every piece explicitly.
    `plan_feedback` and `latency_compensation` keep separate mains with their own setup.
  - **`run_experiment(plant, mpc, task, law, settings, *, loop=run_virtual_time, duration=None, out=None,
    **pieces)`:** `mpc` is `MpcParams` or a built MPC. Given parameters, it compiles
    (minutes unless cached; its docstring says so) after creating the record and before
    building the plant, `MpcProcess` on `RealTimeSettings`, else the in-process `Mpc`, and
    closes it in `finally`, also after a failed compile (a saved error). A built MPC
    (a fake, random100's `MpcProcess` per case) is the caller's, which closes it; its
    `params` are recorded. `duration=None` is the task's own, a number caps it; the
    record holds the resolved duration. This loosens the 2026-10-22 decision that a call
    site that may compile says so (above), and the 2026-10-24 one that the caller
    builds and closes the MPC (`launch`, Records and history).
  - **The robot gate is inside `setup(args, plant, settings) -> (plant, settings, loop) | int`.**
    `sim`: the main's timing with the given `--command-period`, `--update-period` and
    `--solve-latency` replaced (any timing dataclass with those fields, `Compensated`
    too; the loop is passed separately); `rehearsal`: `RealTimeSettings` with the
    given periods; `robot`: without `--execute` the check-only startup checks and exit
    status 0, with it the typed `EXECUTE` (2 when cancelled), and only then
    `RobotConfig(execute=True)` and `RealTimeSettings`. No ROS client and no compile before the
    confirmation. `arguments(description, argv, plants)` lets argparse reject a plant
    the experiment does not support (`plan_feedback` and `latency_compensation`:
    `sim` only). `run_compensated` refuses a `solve_latency_s` other than declared
    seconds (finite, >= 0; not `"measured"`) with a `ValueError` before the plant
    starts, as `run_virtual_time` does, so `latency_compensation`'s main has no check of
    its own; it honours `--target` as reach's does. `latency_study` keeps compiling once
    per task and passes its own `Mpc` per case.
  - **The budget rule, inverted:** when `run_experiment` builds the MPC, the wall-time
    budget (`max_sqp_time_s`) stays only where the solve's wall time is the latency
    (`RealTimeSettings`, `VirtualTimeSettings` with `"measured"`); every other timing, `Compensated`
    among them, gets None. It replaces the 2026-10-24 rule (Execution) that named the
    declared latency.
  - **`solve_latency_s`:** the unit stays in the name (Felix). `VirtualTimeSettings.solve_latency`
    and `Compensated.latency_s` became `solve_latency_s`, and so did the loops' keyword
    (`run_virtual_time`, `run_compensated`), `constant_latency`'s parameter, the
    record's settings and the `overrun` event's data key; the option stays
    `--solve-latency`. `record.FORMAT` became 2: `load_run` refuses a run saved before
    (format 1, the old keys) and names the code that wrote it (22 October: old
    recordings stay readable only by their own code).
  - **Callbacks:** `tick_callbacks`, `solve_completed_callbacks` and
    `diagnostic_callbacks` receive `Tick`, `SolveCompleted` and `Diagnostic`
    respectively. Their aliases are `TickCallback`, `SolveCompletedCallback` and
    `DiagnosticCallback`.
  - **`tiago_mpc/interfaces.py`**, renamed from `plan.py` (Felix's option a): one module
    for the data that crosses the interfaces (`State`, `Plan`, `Diagnostic`, `Tick`) and the
    protocols, the leaf of the import graph. `Law` became `ControlLaw` (`tm.ControlLaw`).
  - **`hz_to_s(hz)`** (`util.py`, `tm.hz_to_s`) writes a period as a rate:
    `hz_to_s(500)` and `hz_to_s(100)` are the timings' default command and
    control-update periods (exactly `.002` and `.01`).

- **2026-10-22: architecture review follow-up (implemented the same day).**
  Keep the component decomposition; share experiment assembly/lifecycle and
  launch plumbing while retaining task-specific definitions and metrics.
  The hard-coded MPC worker and duplicated experiment plumbing are one
  responsibility-boundary problem. Accepted names: `Mpc` and `MpcProcess`,
  replacing `MpcInProcess` and `MpcWorker`. Each plant owns a reusable
  clock (one less parameter to pass around): `plant.now()` returns
  `plant.clock.now()`, from `Clock.wall()`, `Clock.simulation(start=0.0)` or
  `Clock.external()`; `now()` reads execution time, `wall_time()` reads
  monotonic wall time, `advance(dt)` is local-simulation-only, and
  `update(t)` is external-only. `SimPlant` runs on the simulation clock or in
  real time on the wall clock. Pacing stays in `plant.advance(t)`; the loop
  runs on execution time and stays clock-agnostic. Keep measurement time
  (`State.t`, `Plan.t0`) separate from execution time in the same time domain.
  A type checker (mypy, run by `tests/test_types.py`) covers the core
  (`tiago_mpc/` without `safety.py`); experiment files stay untyped. The
  interfaces are the protocols `PlantInterface`, `MpcInterface` and
  `TaskInterface` in `plan.py`. A task's required `at(t, n, dt)` returns the
  reference and the named solver updates for a request at t (`None` when it
  changes nothing); it replaced `reference` and the optional `updates`.
  Additional experiment-setting metadata belongs to another project; safety
  internals and the unfinished ROS2 SEA path remain outside this review. The
  clock part (`Clock`, `plant.now()`) is superseded by the 2026-10-24 entry
  (two loops, `VirtualClock` and `WallClock`).
- **2026-10-23: the effort check belongs to the execution side.** `Mpc` and
  `MpcProcess` return what they planned (`result()` checks only that the plan is
  well formed and finite); execution validates each command before it is sent.
  Virtual time delegates effort handling to the selected command policy.
  Reason: a simulator such as the visualizer must
  be able to execute, clip or inspect an over-limit plan. Clipping stays a
  simulation setting: the robot path never clips a torque.
- **2026-10-24: the root re-exports with plain imports.** Felix chose proposal A: ordinary
  `from tiago_mpc.module import Name` lines (absolute since the same day) and an explicit `__all__`; a lazily loading root (PEP 562
  `__getattr__`) was rejected. Importing `tiago_mpc` may load Pinocchio and GATO's Python
  package (about 45 MB and a few hundredths of a second, and every process that matters uses
  them anyway), but never ROS (only creating a `RosPlant` does) and never GATO's compiled
  module (only the MPC process).
- **2026-10-22: compiling GATO's module is explicit.** An `Mpc` takes its
  binary and never compiles: `Mpc(params, binary)` and `MpcProcess(params,
  binary)` load exactly that file; `find_binary(params)` looks it up without
  compiling; `compile_binary(params, use_cache=True)` compiles it (minutes,
  CPU only) unless cached, and `use_cache=False` rebuilds. `Mpc.compile(params,
  use_cache=True)` does both, so a call site that may compile says so
  (loosened 2026-10-26, above: `run_experiment` compiles when given parameters).
  Reason: `Mpc(params)` silently compiled for minutes, and the visualizer needs
  settings changes that never compile. Also approved: `Mpc.step` keeps GATO's
  raw step result, and replays carry their `MpcParams` in `provenance.params`.
- **2026-10-22: four jobs.** TIAGo specifics for GATO (defaults, URDF
  preparation); experiments on the robot and in Gazebo (a simple ROS 2 adapter,
  separate processes, an independent safety check); data for the visualizer
  (lowest priority); experiments that are simple to write and change.
- **2026-10-22: simpler and smaller while staying correct.** Plain functions and
  small dataclasses; no shims or legacy protocols (rules in `AGENTS.md`).
- **2026-10-22: parameters are Python dataclasses** (`MpcParams`, `TIAGO`, task
  dataclasses), overridden with `replace`. JSON configurations, legacy profiles,
  override semantics and provenance hashing were removed. Each run records the
  commit, source status and complete effective parameters.
- **2026-10-23: experiments import `tiago_mpc as tm`.** Felix: experiments use
  `import tiago_mpc as tm` and `tm.<name>` instead of many `from tiago_mpc.<module>`
  lines. The root `tiago_mpc/__init__.py` re-exports the experiment-facing names with
  an explicit `__all__`; modules inside the package import each other directly, never
  through the root. `experiments/replay.py` keeps importing
  `tiago_mpc.tiago` directly: it needs only NumPy-level facts. Same day: experiments run as `python -m experiments.<name>`
  from the checkout's root and no longer insert `sys.path`; `tiago_mpc` is never installed
  (superseded 2026-10-24: each checkout's uv environment installs it editable; see Friction and GATO).

## Model and coefficients

- **2026-10-20: the canonical robot description is the simulator URDF**
  (`/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`; its original bytes are in
  `d86e041`). `/workspace/GATO/TiagoProURDF/tiago_pro.urdf` is outdated. The
  reduced right arm's fixed torso root and gripper inertia summed at zero
  posture are intentional.
- **2026-10-30: fitted right-arm coefficients, retuned PD gains, predictor width
  0.1** (Felix). The right arm's Coulomb, viscous and armature values are refit
  from realtoisaacsim's real single-joint chirps (`models/tiago_pro/README.md`;
  `research/realtoisaacsim-friction-identification.md` in the thesis workspace)
  and replace the supplied ones, which are almost certainly Isaac Sim matching
  values. The left arm keeps the supplied values. `PD_KP` and `PD_KD` are
  recomputed with the 2026-10-28 recipe for the fitted armature. The predictor's
  tanh width becomes 0.1 rad/s: with the fitted values its explicit step is
  stable only up to 5.8 ms at 0.03 and 17 ms at 0.1. Its low-speed accuracy
  matters little: by default the predictor is loss-free, and feedforward and PD
  handle friction.
- **2026-10-21: supplied joint parameters** for both arms (Coulomb, viscous,
  effort, armature) replace earlier midpoints and guesses
  (`models/tiago_pro/README.md`); the right arm's friction, damping and armature
  were replaced by fitted values on 2026-10-30. The armature is not in the URDF (no loader
  reads it) and never folded into link inertia. kp/kd are not stored: they
  belong to a different controller.
- **2026-10-23: the derived URDFs' meshes are cached copies named by their
  contents** (`<cache_root>/models/meshes/<sha256>.stl`), referenced by relative
  paths; the GPU copy keeps `<collision>` and drops `<visual>`. Reason: absolute
  mesh paths made the GPU copy's bytes, the plant names and every compiled GATO
  module specific to the checkout's path, and GATO's code generation key hashes
  the URDF's bytes but not the meshes it spherizes.
- **2026-10-23: one committed robot description; every other robot file is
  derived at run time and never committed.** `models/tiago_pro/tiago_pro.urdf`
  is PAL's simulator URDF with the arm values written in once. The reduced arm
  with the grasp frame (`tiago.arm_urdf`, ported from GATO's
  `generate_tiago_dynamics.py`), its GPU copy (`runtime_urdf`) and the safety
  collision model (`safety.collision_urdf_tree`) are written, content-addressed,
  into the cache that also holds GATO's modules (`GATO_CACHE_DIR`). Removed: the
  committed arm and active collision URDFs, the preserved sim source,
  `prepare_models.py`, `prepare_safety_model.py`, `models/manifest.json` and
  `models/safety/provenance.json`. Reason: a committed generated file can go
  stale when someone forgets to regenerate or commit it.
- **2026-10-22: the coefficients are questionable; every coefficient is a
  parameter.** Develop with them and test against plants with other
  coefficients. Identified values matter for tuning on the robot, not for the
  design.
- **2026-10-22: the simulated plant may be more detailed than the predictor**
  (losses or armature in the plant only), as on the real robot. Enable it
  explicitly and record both models. This replaced the rule against armature in
  only one model.
- **2026-10-30: the GPU model keeps joint 6's native axis** (Felix). GATO's
  fixed-base adapter supports axes other than +z since 0.1.4 (PR #21), so
  `runtime_urdf` no longer flips joint 6 and the solver works in native
  coordinates; the joint-6 sign map is gone. Flipped and native modules gave
  bitwise identical float32 results after the sign map (device step, KKT
  linearization, reach and figure-eight plans, runs at joint 6's limits) with
  GATO `72c9d14` and GRiD `417a30d` at N = 32
  (`research/joint6-sign-conversion-removal.md` in the thesis workspace).
- **2026-10-21: grasp-center frame.** Tasks track `arm_right_grasp_joint`, a
  massless frame 0.139 m along the flange's +z: the nominal midpoint of the
  finger pads at zero gripper posture, not a calibrated TCP. Summaries and
  replays record `end_effector_frame`; missing metadata means the flange.
- **2026-10-24: every task measures the grasp frame** (Felix), through
  `tiago.ee_position`, with no historical exceptions: random100 reaches the
  grasp-frame positions at its frozen goal postures (a new protocol; old
  flange results stay as they were and are not comparable), and the latency
  study's far reach targets the grasp frame.
  `FLANGE_FRAME` stays.

- **2026-10-24: the MPC tracks the grasp frame only** (Felix).
  `MpcParams.ee_frame` is removed; `mpc.py` uses `GRASP_FRAME`, and `reach()`
  has no frame check. Plant names are unchanged (they hash the grasp frame's
  bytes as before), so compiled modules stay valid. `FLANGE_FRAME` and `fk(q, frame)`
  stay for historical replays.

## Friction and GATO

- **2026-11-02: GATO 0.1.7, and the solver settings follow GATO's defaults** (Felix).
  - **GATO 0.1.7 and grid-rbd 0.6.0**, with the armature commits rebased onto them
    (`experiment/tiago-armature-017`, `experiment/tiago-armature-060`). On `pcg`, every
    run of the default reach, ten random100 pairs and the figure eight is bitwise equal
    to 0.1.5 (GTX 1070); 0.1.7 fixes `bdsv_factor`, which on 0.1.5 reached no target.
  - **Every setting with a GATO default follows it** unless a preset sets a value: the
    `SolverSettings` fields, and `None` in `AdmmLimits` and the controller settings
    `warm_start`, `reset_rho_each_step`, `admm_linsys` (GATO's choice by horizon: `pcg`
    at N32, not `auto`), `admm_threshold`, `linsys`, `bdsv_threshold` (both unused under
    ADMM) and `reseed_threshold`. Only set values reach GATO. Reason: during
    development, take GATO's improvements. The wrapper's parity test requires reviewing
    additions or changed defaults when GATO is updated.
  - **Records hold what GATO resolved and ran, not our inputs** (Felix, 2026-11-03):
    the MPC's `params` are read back from GATO once the controller is built
    (`MpcParams.resolved`: the exposed fields of the solver's `SolverParams`, the controller's attributes, and
    the ADMM settings the solver runs: its `admm_*` properties and the limit rows'
    penalties; `BSQP.admm_iters` is our GATO commit for an upstream PR), so a
    recorded run re-runs the same after a GATO update. GATO's shared ADMM `rho`, which
    every box overrides, is not recorded. Study files keep the study's definition, the
    inputs; the records hold what ran. `tests/test_record.py`
    checks that every field of every recorded configuration class (the package's and the
    experiments') survives the record.
  - **Build first, record once** (Felix, 2026-11-03): `run_experiment` builds the MPC
    before it creates the record, which takes the MPC's resolved params; nothing is
    overwritten later. A failed compile or model check is a setup error that leaves no run
    directory, which reverses the saved compile error of the `run_experiment` entry
    below; a failed plant build is still a saved error.
  - **Nothing changes silently:** `tests/gato_defaults.json` holds GATO's defaults and
    `tests/test_gato_defaults.py` fails when they change, naming each changed setting
    as followed or overridden; `python tests/test_gato_defaults.py` rewrites it. The
    same test lists the deviations of `TIAGO` and `FIGURE8` (the set values) and fails
    on an unlisted one.
  - **Pending:** validate the deviations (SQP iterations, PCG budget, cost weights,
    `rho_u`, the figure eight's `mu`) against all of GATO's defaults.
  - `admm_linsys="auto"` was compared on 0.1.7 (GTX 1070, 1 November): the same
    arrivals as `pcg`, a figure-eight RMSE 0.1 mm higher, and a longer solve-time
    tail (p99 10.1 against 8.2 ms with the 10 ms budget).

- **2026-10-30: the armature is on by default** (Felix). `ARMATURE_KG_M2` is
  the default of the predictor (`MpcParams.armature`), the simulated plant
  (`SimConfig.armature`) and the latency compensation's propagation model
  (`LatencyCompensation.armature`); `(0.,) * 7` turns it off. This supersedes keeping
  armature disabled and explicit (2026-10-23, below; the README since 2026-10-28). The
  values were supplied then and are fitted since the same day (above). Reason: the stored PD gains
  (`PlanFeedback()`, now the default law) are tuned with it; on the zero-armature models
  kd·T/I at 2 ms command ticks was 2 on joints 5 and 6 and 39 on joint 7, so the
  sampled PD and the compensation's forecast diverged at the wrist, and every default
  reach faulted. Results with armature come from another model than earlier
  armature-free runs; compare them only with that said.

- **2026-10-28: model-side armature and PD values; gain calculation is a utility**
  (Felix; implemented in the friction worktree after WIP `f66172f`). The literal
  `ARMATURE_KG_M2`, `PD_KP` and `PD_KD` tuples are manually editable in `tiago.py`.
  `util.inertia_gains` returns plain float tuples that can be printed as matching
  assignments; it never rewrites files or recalculates defaults at import.
  `PlanFeedback()` uses the tuned simulation gains (20 rad/s, damping ratio 1.4,
  including the armature; recomputed for the fitted armature on 2026-10-30). This changes the former 30 rad/s link-only
  default; named `old_pd`/`old_both` tuning candidates explicitly retain their
  previous gains. This is a default selection, not hardware qualification.
- **2026-10-28: separate friction coefficients from their numerical law**
  (Felix; interface refactor only, current physics retained). `Friction` holds
  `coulomb_nm` and `viscous_nm_s_per_rad`; `urdf_friction()` returns those data.
  `SimConfig.friction_law` selects `StickSlip`, `Sign` or `Tanh`;
  `MpcParams.friction_law` is `Tanh`; `FrictionFeedforward.evaluation` is `Tanh`
  or `Sign`. Defaults and arithmetic are unchanged except the chosen PD gains.
  Static holding and sliding still share one Coulomb threshold. Identical
  effective predictor settings retain their binary identity.
- **2026-10-28: reject duplicate friction compensation before startup** (Felix).
  Both package loops check the built-in FF composition once before starting
  the plant: FF plus predictor friction, or nested FF, is a configuration error.
  Physical plant and future propagation-model losses remain independent.
  Arbitrary custom callables remain the caller's responsibility. This guard adds
  no clipping policy; latency prediction is described under Execution.
- **2026-10-28: run format 3** records laws on their consumers, rather than in
  `Friction`. `load_run` refuses older formats before reconstruction. Existing
  files are unchanged; tag `archive/friction-format2-f66172f` reads format 2.
  Portable replay version 1 is unchanged. Friction was promoted onto main at
  `c0bcbc8`; the optional latency integration is described under Execution.
  Publishing the temporary armature fork pins remains pending.

- **2026-10-24: GATO 0.1.5 from PyPI, and a uv project** (Felix; plan in the
  thesis workspace, `research/tiago-mpc-gato-0.1.5-and-uv-plan.md`). This
  supersedes the fork entry of 2026-10-21 below.
  - **GATO:** `gato-mpc==0.1.5`, which pins `grid-rbd==0.5.5`, replaces the
    submodule `external/gato`. `requirements*.txt` and `scripts/setup.sh` are
    removed; README holds the setup commands. The GitHub fork `felixmueh/GATO`
    stays, because historical records name its commits.
  - **Environments:** one uv environment per checkout, `.venv` on the host and
    `.venv-container` in the container (`UV_PROJECT_ENVIRONMENT`, with
    `UV_PYTHON_PREFERENCE=only-system`), `requires-python ==3.10.*`, a
    `.python-version` of 3.10, and a committed `uv.lock`. `tiago_mpc` is
    installed editable into each; this supersedes "never installed" (2026-10-23)
    and the worktree `PYTHONPATH` rule. `experiments` stays an undeclared
    package and runs as `python -m experiments.<name>`.
  - **Costs:** all MPC objectives use GATO's cost menu (2026-11-03, Purpose and shape).
  - **Predictor friction:** upstream's `register_robot(..., joint_dynamics=True,
    friction_width=w)`: GRiD generates the GPU URDF's damping and tanh Coulomb
    friction into the dynamics and derivatives, the width compiled in. `BSQP`
    takes no friction argument. This implements the compile-time width of
    2026-10-22.
  - **A local GATO fork:** a `[tool.uv.sources]` editable path on a branch or
    worktree (README), never on main.
  - **Provenance:** records store `gato` (the installed version,
    `direct_url.json`, `gato.__file__`, and an editable checkout's commit and
    source status) instead of the submodule's commit.
  - **Plant names hash GATO's version,** so fork and 0.1.5 registrations never
    overwrite each other in the shared cache.
  - **Per-iteration trajectory capture** (the fork's) is absent in 0.1.5. The
    migration goes without it; the visualizer reports it as unsupported. How to
    restore it is open (below).
- **2026-10-21: GATO 0.1.2 plus opt-in smooth joint friction** and opt-in SQP
  trajectory capture, from the fork `felixmueh/GATO`, with `grid-rbd` 0.5.3
  (superseded 2026-10-24, above).
  GRiD still evaluates the barrier terms of zero-weight soft limits; there is no
  public switch.
- **2026-10-22: reaching is loss-free by default; friction is opt-in**
  (`f62c358`). The friction-enabled reaching default became unstable in its
  first 10 ms control interval; the visualizer's diagnosis
  (`../tiago_mpc_visualizer/docs/friction.md`) attributes this to viscous damping
  against joint 7's link-only inertia, stiff for explicit 3 ms RK4 substeps.
  Friction-enabled reaching is not qualified.
- **Tracking uses friction on both sides:** `StickSlip` Coulomb plus viscous in
  the plant, tanh Coulomb (width 0.03 rad/s) plus viscous in the predictor, both
  from the URDF. The plant change does not qualify tracking.
- **2026-10-23: compare both simulator friction proposals** (Felix). A fresh
  worktree from main, one commit for review before any merge. Both use Pinocchio
  semi-implicit Euler with implicit damping and a default 1 ms maximum step;
  Coulomb is either sign(v_old), or a coupled constraint solve allowing sticking
  at zero (the proposed default). No MPC tuning or predictor changes. Reaching
  stays loss-free by default; armature stays disabled (on by default since 2026-10-30).
- **2026-10-23: the plant's Coulomb law and integrator are choices** (Felix):
  `coulomb_law` is `"tanh"` (with the `Friction`'s width, explicit in the old
  velocity), `"sign"` or `"constraint"` (default); `integrator` is `"rk4"`
  (loss-free only) or `"euler"` (semi-implicit, loss-free without a `Friction`),
  by default implied by friction on/off. This supersedes "smoothing is
  predictor-only" for the plant; the predictor is unchanged. Its interface was
  replaced on 2026-10-24 (below).
- **2026-10-24: Coulomb laws as data** (Felix). A `Friction` holds the
  coefficients and `law`: `Tanh(width_rad_s)`, `Sign()` or `StickSlip()` (the
  default, the former `"constraint"`), frozen dataclasses without methods. Each
  integrator interprets the law with `match`, covering every case; each law
  stores its `name` so that `asdict` tells them apart, and `MpcParams.from_dict`
  rebuilds it through `dynamics.LAWS` (superseded by the run-data entry below: the
  type name tells them apart, `record.from_dict` rebuilds). `urdf_friction(law=StickSlip())` replaces
  the width argument; `check_coulomb_law` and `SimPlant`'s `coulomb_law` are gone.
  `MpcParams.friction` must be `Tanh`, the only law GATO predicts. RK4 with the
  passive torque at every stage (`plant_rk4`) is restored for `Tanh` and `Sign`;
  `StickSlip` stays Euler-only, and `passive_torque` refuses it. The plant names
  still hash the coefficients and the width alone, so they did not change.
  The 2026-10-28 coefficient/law separation above supersedes the container API.
- **2026-10-22: the smoothing width becomes compile-time only**, to diverge less
  from upstream GATO. Done 2026-10-24 with GATO 0.1.5 (`friction_width` at
  registration).

- **2026-10-24: breakaway follow-up handoff** (Felix). Document the next plant
  refinement with separate static/breakaway and sliding Coulomb values; keep
  the MPC predictor independently configurable and simpler. The
  [handoff](friction-breakaway-handoff.md) records recovered physical fits,
  saved-command evidence, implementation choices and bounded validation work.
  This request is documentation only; the current `StickSlip` still uses one
  Coulomb value for holding and sliding, and no calibration/default change is
  made by this handoff.

## Tasks

- **2026-10-21: end-effector position cost 4 running, 8 terminal**, for
  reaching and tracking. Earlier 8/1600 and 4/800 results do not qualify 4/8.
- **Tracking (`FIGURE8`):** joint-velocity cost 0.01, torque cost 1e-6, zero
  soft-limit costs (limits through ADMM only), a 24 s horizontal figure eight
  of 0.28 m by 0.18 m, 2 cm ahead of its start.
- **2026-10-22: tracking uses 10 ms knots** (2 ms before); checks at 2 ms do not
  qualify 10 ms.
- **2026-10-24: a reach arrives by the hand's speed.** Within 5 cm with the
  grasp point slower than 2 cm/s, replacing the joint-speed L1 norm of 1 rad/s,
  which let the hand move at up to about 0.4 m/s. Runs before this change used
  the old criterion, and the grasp-frame random100 protocol has no baseline yet.
- **Orientation is unconstrained** in both tasks by default. An optional orientation
  cost is available through `default_cost` or a native menu. Task references use rotation
  vectors in `ROOT_FRAME`, with NaN for no orientation target (README, "Costs").
  Weight tuning and arrival criteria remain open below.
- **Limits through ADMM are approximate:** returned plans can exceed the
  planning bounds, so the planning margins and independent command-policy
  checks stay. An ADMM return is no feasibility certificate.

## Execution

- **2026-11-07: name loop state by its role** (Felix).
  Use `active_plan` for the plan supplying commands and `pending_plan` for a
  completed plan waiting for its activation time. These can coexist. Remove
  the extra `new` plan reference; a per-tick `plan_activated` boolean records
  activation and supplies `Tick.plan_activated`. Name request decisions, unread
  solves, collection times and counters explicitly. Keep short `s` and `t` for
  measured state and command time. Naming changes do not alter saved columns or
  introduce additional execution state. Add short comments where timing semantics
  or estimator policy need context; do not repeat self-explanatory variable names.

- **2026-11-07: handle a whole command tick in one pass** (Felix).
  Each iteration handles a complete command tick, then advances the plant to the
  next command time. A command tick reads state, delivers
  diagnostics and runs the pause/cancellation checkpoint, receives and activates
  eligible earlier work, optionally submits one request, receives its result if
  immediately eligible, then decides and publishes its command. Use a small
  shared function for receiving, checking and reporting a result. Keep plan
  selection and timing updates explicit; do not re-enter a partially completed
  tick or track preparation/publication phases. The immediate path checks result
  eligibility, including a future predicted origin and the next-tick lower bound
  for measured latency. Preserve independent physics/solver overlap.
  Normal terminal drainage remains outside the loop. Cancellation
  returns from the command-boundary checkpoint without draining the reply.

- **2026-11-07: leave physics substeps to the plant** (Felix).
  The loop calls `plant.advance()` once per command interval. `SimPlant` already
  integrates with `SimConfig.substep`, shortening the last step at the command
  boundary and emitting substep observations through `step_callback`. Remove the
  separate loop physics schedule and period argument. Command period, MPC update
  period and integration step remain independently configurable. Optional wall
  pacing runs once per command interval; asynchronous solving can overlap the
  plant's integration.

- **2026-11-07: share periodic advancement and command alignment** (Felix).
  Use the same small `Schedule` class for commands and MPC updates.
  `next_at` exposes the nominal slot; `advance` consumes or skips slots without
  moving the periodic phase. Keep command-grid rounding in its pure `align`
  method, anchored to an actual eligible boundary so floating-point drift does
  not introduce actions between command ticks. Result and activation times stay
  plain deadlines checked inside the command tick. The update cursor controls
  request cadence there. Zero update period requests whenever the solver is free
  and no returned plan is waiting to activate, at most once per command tick.
  The command cursor advances after publication.

- **2026-11-06: record plans only through completed-solve callbacks** (Felix).
  `RunRecord.solve_completed` records every returned plan and its request time,
  including unused, expired and terminally drained results. Ticks record the
  active plan, mapped by identity to its saved row. Remove `Tick.faulted` and
  `Tick.plan_request_t` and the corresponding loop bookkeeping. A normal fault
  still ends execution before applying a refused command. `run_experiment`
  registers the recorder with all three notification groups. The saved columns
  and format stay the same; plan rows now cover every completed solve.

- **2026-11-06: remove foreground timing from the numerical loops** (Felix).
  Drop `controller_wall_s`, its `perf_counter` accumulation and the separate
  `SolveCompleted.wall_s` field. Completed solves retain their request snapshot
  and native plan, whose `stats['wall_s']` supplies the solver duration. Keep the
  observed real request-to-result delay used for compensation and virtual-time
  availability unchanged. Detailed foreground costs belong to optional profiling;
  the visualizer already measures its solver child's controller duration.

- **2026-11-06: use a sequence of before-tick callbacks** (Felix).
  `run_virtual_time` accepts `before_tick_callbacks=()`, matching the other
  callback groups. Each receives the boundary `State` in registration order,
  once per command boundary. Callbacks may block to pause the retained loop.
  Cancellation is checked after the group, before result collection and command
  evaluation. Keep this ordinary sequence without a callback dispatcher.

- **2026-11-06: make prediction one calculation before each MPC request** (Felix).
  Use a frozen callable predictor with a prepared physical model and explicit
  inputs. Remove its publication history, `sent` and `arrived` methods and hidden
  delay estimator. The loop owns the fixed or previous observed delay and chooses
  the prediction target. Propagate the latest measurement under the held command,
  then evaluate the active plan's command law and effort policy on predicted
  feedback at future command ticks. A virtual measurement is already at the
  command boundary. For older measurements, accept the approximation that the
  currently held command covered the interval since observation; do not add
  machinery to reconstruct earlier publications or plan changes. The physical
  workspace is reused, but predictions do not depend on previous calls.

- **2026-11-06: keep the university experiment framework straightforward** (Felix).
  Fix correctness issues when the solution is simple. Accept minor edge cases
  rather than complicating the numerical loop. Ordinary `try/finally` cleanup is
  sufficient; do not add special machinery to choose between competing observer
  and execution exceptions.

- **2026-11-06: separate typed notifications and keep every request snapshot** (Felix).
  The loops expose independent tick, solve-completed and diagnostic callbacks.
  `Diagnostic` replaces the generic `Event` payload; `Tick` contains no diagnostic
  bundle. Each notification has one delivery path, with no dispatcher or generic
  message callback. The plant supplies diagnostics through `take_diagnostics`.
  Keep an owned `last_mpc_request_snapshot` for every request, even without
  observers, and use it for the solve-completed notification. The clear name
  needs no explanatory comment. Snapshotting detaches mutable producer arrays
  once; it does not depend on which observers are registered.
  Real-time notifications run after command publication, preserving the execution
  order. Virtual-time plant diagnostics are delivered before the pause checkpoint.
  `RunRecord` receives ticks, completed solves and diagnostics separately. It
  normalizes diagnostic timestamps relative to the first recorded state, including diagnostics received
  before that tick. Console is a diagnostic callback. The saved `events.json`
  payload and replay schema stay unchanged.

- **2026-11-06: receive plans directly and emit notifications afterwards** (Felix).
  MPC owns request/result transport and result validation. The loop owns virtual
  availability, plan activation and the forecast's delay estimate. Keep those
  assignments visible in the loop instead of state-mutating nested functions.
  Notification helpers only report completed numerical work to observers; events
  do not supply plans or drive internal scheduling. A small amount of repeated
  result-handling code is preferable to hidden changes through `nonlocal`.

- **2026-11-06: leave effort and lease validation to execution policies** (Felix).
  Remove whole-plan effort rejection and `validate_plan` from the loops and
  public API. Unapplied excessive knot torques do not reject a plan. The command
  law and effort policy decide the command; real-time execution still checks
  requested and applied torque against the raw limits. Solver results remain
  structurally checked and finite. Remove the separate forecast-age preflight
  from the real-time loop; a requested plan may later be refused by the ordinary
  lease policy. Forecast commands use the same execution policies.

- **2026-11-06: keep finite durations and the runtime guard in native loops** (Felix).
  Both loops require finite, nonnegative seconds and retain the ten-minute
  `MAX_RUNTIME_MIN` guard. None does not select a streaming mode. Virtual runs
  disable cyclic GC for the invocation and restore the caller's policy on exit.
  There is no unsafe-disable flag in the native API. A long-running override and
  any pause-time GC handling belong to the visualizer's dedicated simulation
  process; its migration remains deferred until the native layout is final.

- **2026-11-06: pause the retained simulation loop at one ordinary checkpoint** (Felix).
  Before-tick callbacks run once per command boundary before result collection or
  command evaluation. They may block while an outstanding solve finishes in the
  worker. The ordinary result path consumes that reply after resume. Remove the separate
  quiescence flag, repeated boundary callback and deferred completion events.
  Pause retains the plant, solver, clock, schedule and forecast delay estimate. The
  cancellation flag is checked only after these callbacks at each boundary. Once
  admitted, a tick completes its command decision and physics interval before
  cancellation or pause is acknowledged; waiting for a solve is acceptable.
  Detected breaches may request a pause at an ordinary command boundary.
  Reset uses cancellation and constructs a fresh run. Tick-observer pause requests
  are acknowledged at the next boundary. With identical solver results and
  latencies, pausing preserves the numerical state and command sequence.

- **2026-11-06: collect virtual-time results only when they may be needed** (Felix).
  Keep one virtual-time implementation for synchronous and asynchronous MPCs.
  Submit before the independent integration interval and collect once at the
  first possible activation boundary. Declared zero latency collects immediately;
  measured latency collects no earlier than the next command tick, even for a
  reported zero duration. A result still waits for its measured availability and
  predicted origin. Update the loop's previous delay once per result.
  Activate an eligible plan before preparing the next request and reuse the
  current command decision for forecasting and application. An immediate ideal
  result invalidates that decision. Retain terminal drainage for controller reuse.

- **2026-11-06: derive compensation defaults from the configured timing** (Felix).
  Real-time compensation uses the previous observed request-to-result delay;
  its initial estimate is the update period, or the command period for back-to-back
  updates. `default_compensation(settings)` selects the same fixed delay for
  declared virtual latency and previous-delay mode for measured latency. Explicit
  compensation objects and None retain their chosen behavior. The standard
  experiments call this constructor after resolving their CLI timing settings.
  `RealTimeSafety` exposes only `validate`, matching `SafetyInterface`;
  `RealTimeEffort` supplies the separate callable effort boundary. Waiting plans
  use ordinary `ExecutionSafety.validate` without calling user safety early.

- **2026-11-06: consolidate execution checks outside the loops** (Felix).
  `RealTimeSafety` binds the mandatory lease and `RealTimeEffort` binds the raw
  effort checks around the selected user policies. Execution and forecasting
  share these policies. The
  string-returning plan, lease and command fault helpers are removed from the
  loops and public API. Waiting plans keep their mandatory expiry check before
  activation, without evaluating their law or user safety early.

- **2026-11-06: keep timing data separate from execution components** (Felix).
  `VirtualTimeSettings` holds only command period, update period and solve latency;
  `RealTimeSettings` holds the two periods. Neither has a `.run()` helper, and
  there is no `Timing` protocol. The runner receives the loop function and
  settings explicitly. Compensation, safety and effort handling are independent
  run arguments and are recorded independently. `ExecutionSafety` holds the
  maximum plan age and implements `SafetyInterface.validate(plan, t, state)`;
  users can supply their own pure check. Finite plan horizons remain mandatory.
  Control laws and effort handling return `Command | Fault`; safety returns
  `Fault | None`. `decide_command` validates before the law and applies effort
  handling to its accepted result. Custom laws retain default safety and effort
  handling. Real-time mandatory robot limits remain enforced. No generic
  validation framework is introduced.

- **2026-11-06: one execution decision for both loops and forecasting** (Felix).
  A plain `decide_command` function checks the active plan's age/horizon lease,
  evaluates its control law, then applies the effort policy. Its result is
  `Command(requested, applied)` or `Fault(reason, requested=None)`. The optional
  requested torque retains refused commands for observers. Forecast refusals
  make the prediction unavailable without an execution event. Waiting-plan expiry
  before activation remains separate. Real-time commands retain the strict 100 ms lease and raw effort
  rejection. No plan-modifier chain or general result framework is introduced.
  `RealTimeSettings` replaces the `RealTime` name; `run_real_time` requires
  `settings=`; the settings object is plain data.

- **2026-11-06: require one virtual-time settings object** (Felix).
  `VirtualTimeSettings` replaces the `VirtualTime` name and retains its preset
  constructors. `run_virtual_time` requires `settings=`; numerical configuration
  is held only in that object. Duration, compensation, safety, effort handling,
  callbacks, cancellation and initial conditions stay on the invocation.
  Configurations record the settings dataclass directly.

- **2026-11-04: simplify the shared loop and separate effort handling** (Felix).
  Virtual time has a separate
  `effort_handling=EffortClip()` parameter, also when a caller supplies its own law;
  `RejectOnEffort()` refuses excessive commands and `None` explicitly disables
  effort handling. The policies accept per-joint limits and act after feedback
  and feedforward. Invalid commands are refused before clipping. Forecasting uses
  the same law and effort policy; both must be pure and are not copied. A future
  refusal makes the forecast unavailable without an actuator event. Unapplied
  over-limit plan knots are not rejected. Real-time checks remain strict and never
  clip. Normal finite completion consumes outstanding solves so caller-owned
  controllers can be reused; cancellation keeps the simple early exit.

- **2026-11-03: configurable age expiry** (Felix).
  `ExecutionSafety.max_plan_age_s` is positive seconds or None; None disables the
  100 ms observation-age limit, including in forecasting. The finite horizon
  still bounds the plan; no implicit last-command extension is introduced.
  Propagation retains its own explicit model, independent of plant and optimizer.
  Recording capacity remains the caller's responsibility. Special pause-time
  memory management is deferred.

- **2026-10-30: warn about PD feedback without latency compensation** (Felix).
  The loops' setup check, renamed from `_check_compensation` to `_check_setup`,
  also warns (`RuntimeWarning`, no rejection) when `PlanFeedback`, alone or under
  `FrictionFeedforward`, runs without `LatencyCompensation`. Reason: simulated far
  reaches exceeded the raw effort limit at the first plan's takeover from the held
  arm, the D term reacting to the plan's velocity after the first solve's latency
  ([latency compensation](latency-compensation.md)).

- **2026-10-28: simplify prediction inputs and keep execution data off plans**
  (Felix). `request` and `_solve` take one `State | Prediction`; a prediction
  includes its original `measured_t`, while its `t` denotes the future initial
  condition. `Plan.measured_t` is a required float, equal to `t0` for measured
  inputs. No consumer fallback or separate optional prediction argument.
  Remove `_arrived`/`_activated` and lifecycle timing data from `Plan.stats`;
  it contains solver statistics. Keep only the recorder's request association
  on `SolveCompleted.request_time_s`, so superseded results retain the correct request.

- **2026-10-28: integrate optional latency prediction on main**
  (Felix; implementation delegated, KISS). Keep the two loops and their independent
  command/update rates. `LatencyCompensation` forecasts a separate `Prediction`
  for MPC; measured `State` stays unchanged for feedback and checks. Fixed forecast
  delay or previous completed request's delay with an initial estimate; no access
  to the current solve's future duration. CPU propagation reuses the simulator's
  Euler step with explicit independent friction/law/armature and a held-command
  approximation for old measurements. Early replies wait; late replies retain their
  predicted origin. `Plan.measured_t` preserves the measurement-age lease while
  `Plan.t0` defines trajectory time. Compensation defaults off (on since
  2026-10-30, the next entry); no solver,
  saturation-policy or visualizer change in this pass. See
  [latency compensation](latency-compensation.md).

- **2026-10-30: new defaults for the real MPC, which the simulation takes from it**
  (Felix; timing and law superseded on 2026-11-12, above). The real MPC's timing, `RealTimeSettings()`: 500 Hz commands, control updates at
  50 Hz (20 ms) and latency compensation on. The current real-time forecast uses
  previous observed latency, initialized from the update period. The simulation's timing is
  `VirtualTimeSettings.like(RealTimeSettings())`: the same periods, each solve
  declared, not measured, to last one control period; compensation is passed
  separately. `MpcParams`: 30 ms knots
  (`dt_s`, N stays 32: a 0.93 s horizon) and planning speeds at 60 % of the raw limits
  (`velocity_fraction`). The mains (reach, figure8, random100) use `PlanFeedback()`,
  the planned torque as feedforward plus PD about the plan, with its stored gains, which
  need the supplied armature, now on by default (Friction and GATO). Unchanged: the cost weights
  4/8, `FIGURE8`'s explicit 10 ms knots, the presets `ideal`, `constant_latency`,
  `constant_rate` and `as_fast_as_possible`, and the bare loops' parameter defaults
  (`law=SampleKnot()`, `compensation=None`). random100 runs from this date are a new
  protocol ([its README](../experiments/random100/README.md)).

- **2026-10-22: plans are sampled by knot.** The execution controller commands
  the torque of knot k = floor((t − t₀)/dt) from the time t₀ of the state the
  plan was computed from, a zero-order hold per knot (`sample_knot`). Command
  period, control-update period and knot interval are independent parameters.
- **GATO's warm start shifts one knot per solve**, whatever `dt_elapsed` says,
  so an update period other than dt starts from a misaligned guess. Kept: an
  elapsed-time resampling study (eight figure eights) found no practical benefit.
- **2026-10-22: one loop body on the plant's execution time.** `run_robot`
  (a plant on the wall clock) uses each plan on arrival; `run_simulation` (a
  plant on a simulation clock; each refuses the other kind) charges each solve
  a latency (`Fixed` or `Measured`), makes no new request while a plan is pending, and applies a
  plan from the first tick at or after its charged time. Results at nonzero
  latency changed with this. Past a plan's end, both loops fault. Superseded by
  the 2026-10-24 entry below (two loops, `VirtualClock` and `WallClock`).
- **2026-10-24: two loops, named after their time** (Felix). The timing model of
  `agent-reading/tiago-mpc-design/timing-model-fable.md` in the thesis workspace
  (sections 2–4), except one loop: Felix chose two. `run_real_time` runs on wall
  time: the robot and the rehearsal (a `SimPlant` on the robot's architecture,
  `--plant rehearsal`, Felix's name); plans are used from the first tick after
  they arrive, with no latency parameter. `run_virtual_time` runs controlled experiments on virtual time,
  which the loop keeps: `solve_latency`, required with no default, is declared
  seconds or `"measured"` (the solve's own `plan.stats["wall_s"]`, then used as
  the declared duration); the loop waits for each solve and places its plan at the
  request plus the latency. Both: a request at the first tick of each update slot
  while no solve is outstanding, and `update_period=0` back to back (a request at
  the tick the previous plan takes effect); a plan takes effect at the first tick
  at or after it is available; one outstanding solve. `Clock` became two classes,
  `VirtualClock(t)` (`reach(t)` finite, never backward) and `WallClock` (`reach(t)` sleeps,
  at once when late); a loop starts the plant with its clock, `plant.start(clock)`,
  and `now()` left the plant protocol. `RosPlant` refuses anything but `WallClock`.
  `SimPlant(sync_real_time=True)` paces a virtual-time run to the wall clock for
  watching (warns once when behind, never changes results; refused on wall time);
  it replaced `SimPlant`'s clock argument. `Fixed`, `Measured` and the old `run`
  went. Budget rule: `max_sqp_time_s=None` for a declared latency, the TIAGO
  budget for `"measured"`, the rehearsal and the robot (this replaces "simulations
  set None" below; inverted on 2026-10-26, Purpose and shape: the budget only on
  `RealTimeSettings` and `"measured"`; `solve_latency` became `solve_latency_s` the same
  day). Simulation (`--plant sim`) uses the in-process `Mpc`; the
  robot and the rehearsal keep `MpcProcess`, random100 its `MpcProcess` per case
  (Felix, same day). Warnings and counts: a late tick on wall time (once per run,
  not the tick after the first command, the robot's activation; all counted), a
  declared latency > 0 that a solve exceeded (once; overruns counted). The
  loops return an `Outcome` (done, late ticks, overruns), and the summary records
  the timing (`timing`: the loop, `command_period_s`, `update_period_s`,
  `solve_latency` for virtual-time runs, the counts); the plant's `settings` record
  `time` (`"virtual"` or `"wall"`; the replay's `time_basis` keeps
  `"simulation"` and `"wall"`). The warnings, counts, `Outcome`, `timing` and
  `settings` are superseded by the run-data entry below. Renamed the same day (Felix): "stepped time" became "virtual time",
  `run_stepped` → `run_virtual_time`, `Stepped` → `VirtualClock` (and `Wall` → `WallClock`), the plant's `time` `"stepped"` → `"virtual"`.
- **2026-10-22: one `Mpc` per run.** `run` calls `mpc.warmup(state, reference)`
  once before the first tick (arm still held); it resets the controller and, with
  `MpcParams.warmup` (default on), runs GATO's warm-up solve (its limits cannot be
  set through GATO's public API). Without a warm-up, the first request resets.
  It never resets again; a fresh controller is a new `Mpc`. Solver changes go
  only through named `updates`. GATO's SQP damping rho stays reset before every
  solve (2026-10-23: carrying it changed no outcome in random100, the far reach or the figure eight).
- **2026-10-23: fast solves and 500 Hz commands, so a plan takes effect within
  one knot.** Latency analysis (`research/tiago-mpc-latency-analysis.md`): the
  start pose lay outside joint 4's ADMM planning bound, every solve with latency
  hit the ADMM cap, and `sample_knot` diverges once a plan takes effect a knot
  late. Fixes: `Q_START` joint 4 at −2.33 (inside the box, guarded by a test);
  GATO's ADMM defaults except `rho_u` 5e-4 for reaching (ratio 5 to its control
  cost; GATO's 1e-5 made 46 of random100's first plans exceed the raw effort
  limit), the figure eight keeps 1e-5; default command period 2 ms, control
  updates at dt (at 50 Hz since 2026-10-30). `max_sqp_time_s` 0.01 s caps a solve on the wall clock;
  simulations set None (a wall-time budget is wrong on a simulated clock).
  `run` warns once when a plan is first commanded from its second knot; the
  remedies, feedback around the plan or latency compensation, are named there.
  GATO's u₀-at-arrival rule was dropped. random100 with these settings: 82
  arrivals, 18 timeouts, 0 errors (`_expr_data/random100-step16-20261023/`).
- **2026-10-22: no context managers.** `run` calls `plant.start()` before the
  first tick and `plant.stop()` first on exit; closing does not depend on order.
- **2026-10-22: records per command tick**, buffered and written at the end.
- **2026-10-22: garbage collection is paused during `run`** (full collections
  took 7–10 ms at 500 Hz). It must run in between; how is to be discussed.
- **2026-10-23: the simulator's friction step is semi-implicit Euler**, with
  `Tanh`, `Sign` and `StickSlip` Coulomb choices (see Friction and GATO). Loss-free plants
  keep Pinocchio RK4 by default (reach 3 ms); substep, integrator and friction
  remain `SimPlant` arguments.
- **2026-10-22: random100 runs one `MpcProcess` per case**, so a crash ends only
  that case.
- **2026-10-22: `save_replay` lives in `experiments/replay.py`**; experiments
  call it after `record.save()`.
- **2026-10-22: `--duration` is an optional cap**; a run ends at the earlier of
  the task's end and the cap.

## Robot

- **2026-10-21: three processes:** the loop with the ROS I/O, a spawned MPC
  process (`MpcProcess`) and a spawned safety worker. GATO's bindings keep the GIL during a
  solve, so only a separate process keeps the loop running. ROS imports stay
  optional and lazy.
- **2026-10-21: review fixes:** exact effort-controller joint order; the safety
  deadline enforced before late replies are read; SEA body-only updates get
  their own safety sequence without refreshing the arm.
- **2026-10-21: accepted limitation:** gripper mimic followers without feedback
  only warn and stay at neutral.
- **2026-10-28: the ROS container is owned by this repository** (`container/`).
  - **One image** for the desktop and the Jetson from one Dockerfile, with the CUDA
    devel base as a build argument: system Python 3.10, a pinned uv (the host's),
    ROS 2 Humble with the packages of the old GATO ROS layer, PAL's interface
    overlay, and the toolchain GATO's run-time compile needs. No GATO checkout, no
    Python environment, no ROS or DDS variables in the image; `.venv-container`
    comes from `uv sync`.
  - **The ROS profile is chosen by sourcing it** (`container/ros_tiago.sh`,
    `container/ros_simulation.sh`); the launcher only builds, creates or starts,
    and attaches.
  - **`--target desktop|jetson` stays explicit.** No session-capture script and no
    X11 forwarding.
- **2026-10-22: ROS is sourced in the launching shell**, otherwise a clear
  error; `ROS_DISTRO` is recorded.
- **2026-10-22: `State.t` is the measurement time;** on the robot the monotonic
  receipt time, in the domain of `RosPlant`'s wall clock. Gazebo would use
  `Clock.external()` fed from `/clock`, with ROS stamps as `State.t`; open
  (`docs/robot.md`). `Clock` is superseded (2026-10-24): Gazebo would need a
  third clock beside `VirtualClock` and `WallClock`.
- **2026-10-22: safety is the robot plant's job.** `RosPlant` owns the safety
  worker (it needs the ROS sequence numbers, and activation waits for it), so a
  robot run cannot happen without it. There is no loop-level safety hook.
- **2026-10-22: activation republishes until the state time + 100 ms**, not the
  plan's own lease end; left as is, to review with the safety work.

## Records and history

- **2026-10-24: run data and configuration** (Felix; the design is
  `agent-reading/tiago-mpc-design/run-data-design.md` v2 in the thesis workspace,
  its section 8 deciding the points left open).
  - **`record.py` is the one module of the format,** imported by everything that
    writes or reads a run: `to_dict`/`from_dict` (dataclasses with their type
    name; rebuilt from `CLASSES`, the package's config classes, imported at the
    module's top, plus the caller's; JSON arrays become tuples; two classes of one
    name raise), `provenance()` (commit, source status, versions, GATO's source,
    `ROS_DISTRO`; `gato_source` became its internal `_gato_source`), `Console`,
    `RunRecord(out, **pieces)` (renamed from `Record`) and `load_run(path,
    classes)`, which share one read interface (`config`, `ticks`, `plans`,
    `diagnostics`, `metadata`), and `FORMAT = 1` (2 since 2026-10-26). `Diagnostic` lives in `interfaces.py`, beside
    `Tick`, so that inside the package only the root imports `record.py`, and no
    import is lazy (Felix, same day).
  - **A run directory:** `summary.json` (`format`, `provenance`, `config`, the
    pieces written at the start, and `metadata`: `status`, `reason`, `end_t`, one
    flat `results`, `error`, `plant`), `ticks.npz` (with `plan` and `update`
    indices), `plans.npz` (with `update_t`), `events.json`, and `replay.npz`, the
    visualizer's contract, exported from the record. Every returned plan is saved
    from `SolveCompleted`, including plans that never activate; no tick's `plan`
    index points at an unused plan. Old recordings stay readable only by their own
    code (22 October).
  - **Diagnostics, not warnings:** the loops and plants emit `late_tick`, `overrun`,
    `late_plan`, `pacing_behind`, `activated` and `handback`; every occurrence is
    recorded, the console prints the first of each kind. Plants hand theirs to the
    loop with `take_diagnostics()`, once per tick and once after the loop (`stop()`'s).
    The late tick after the robot's activation is not emitted; its delay is the
    `activated` event's `switch_s`. `PlantInterface.settings` became `run_state`,
    the plant's final values.
  - **Configuration by construction:** every piece is a frozen dataclass and is
    recorded as given, without a wrapper or `Options`: the task, `MpcParams`,
    `SimConfig`/`RobotConfig` with `build()`, the laws `SampleKnot()` and
    `HoldFirst()` (replacing `sample_knot`, `hold_first`), the timings
    `VirtualTimeSettings` (presets `ideal`, `constant_latency`, `as_fast_as_possible`,
    `constant_rate`) and `RealTimeSettings` in `loops.py`, passed to the chosen loop.
    `SimConfig.integrator` is explicit, with no `None`: a plant with friction
    names its integrator. Fields that would not round-trip convert in
    `__post_init__` (`SimConfig.q`/`qd` to tuples, `CollisionSafetySettings`'s
    paths from strings); `MpcParams.from_dict`, the laws' `name` and
    `dynamics.LAWS` went.
  - **Ending without exceptions:** a loop returns `Outcome(status, reason, t)`:
    `done`, `timeout`, `fault` (`effort_limit`, `lease_expired`,
    `invalid_command`, from execution-safety checks that gate `plant.apply`) or
    `interrupted` (`stop_requested`, the module-level
    `STOP` flag checked every tick). `handle_stop_signals()` clears and sets the
    flag on the first SIGINT/SIGTERM; a second restores the defaults and raises
    `KeyboardInterrupt`. Every main that runs the simulation or the arm calls it
    first, so Ctrl-C behaves the same everywhere; the studies stop after an
    interrupted run. The MPC process and the safety worker ignore SIGINT (the
    parent stops the run and closes them); the cancelled `EXECUTE` prompt returns
    exit status 2. `PlanEnded` went: a law asked past the plan's last knot, which
    the lease check rules out, raises a plain `RuntimeError` (Felix, same day). The
    plant-side checks (`RosPlant._check`, the safety worker) stay exceptions, to be
    decided with the todo item "Rethink error handling in real-time runs".
  - **`run_experiment` returns the record,** building the plant inside its
    `try`, and saves and exports it in `finally`; the experiment adds its results
    and saves again. There is no `done_status`: the reach sets `"arrived"`
    itself, the figure eight ends as `done`. `RunRecord` also works on its own,
    with the same duties written out. `launch(args, experiment, plant, params,
    timing)` takes the main's `VirtualTimeSettings` explicitly (Felix, same day). Superseded
    on 2026-10-26 (Purpose and shape): `launch` went, each experiment's `evaluate`
    adds its results and saves, and `run_experiment` builds the MPC from parameters.
  - **Many runs:** no study class in the library; an experiment defines its own
    frozen study classes (random100: `Random100Config`, `Case`, `Random100Study`,
    `study.json` once before the first run, one `RunRecord` per case, stop on
    `interrupted`, `load(path)`).
  - **Runs are limited to `MAX_RUNTIME_MIN` = 10 minutes** (`loops.py`): both
    loops require a finite duration and raise before the plant starts for a
    longer run. Long buffered runs are not
    validated: the record keeps every tick in memory (about 2 KB per tick in
    simulation) and the loops pause garbage collection for the whole run, which
    stays (a check of 24 October found no cyclic garbage per tick on the CPU
    paths; rclpy unchecked). A loop of an experiment's own is not checked.

- **2026-10-21: completed experiment cleanup.** Felix removed ten completed
  experiment worktrees and branches with their raw results and replays, and the
  smooth-friction contribution checkout; their decisions are integrated or
  discarded. Do not describe those records as available.
- **Historical results keep their labels:** for example the frozen random100
  cohort (84/100 arrivals with 4/800 costs, loss-free, flange frame). New runs
  are new results; never relabel the historical ones.

## Deferred

- A `Stream` callback for live consumers such as the visualizer.
- A `Command` type for reference commands to the robot's onboard controllers.
- A single-process toggle.
- A report-only safety check in simulation.
- Laws beyond sampling the plan; a `configure` hook.
- The Gazebo check (the ROS container exists, 2026-10-28).
- Migrating the visualizer off `tiago_mpc_vis_baseline`.
- Publishing/pinning the armature extensions for main and upstream.

## Open with Felix

- Orientation tracking: the weights (running/terminal, untuned; 0.2/4 and 1/8 were tried once);
  whether arrival should include an orientation tolerance; the defaults for reach and
  figure8. All objectives now use a cost menu; orientation weights belong to the
  native menu, configured directly or through `default_cost`.

- How garbage collection runs between ticks or runs.
- Per-iteration trajectory capture on GATO 0.1.5 (under discussion, 2026-10-24):
  a port proposed upstream, a capture-only local fork, or reconstruction by
  truncated re-solves.
- Gazebo's clock: a third clock read from `/clock`, with ROS stamps as `State.t` (not implemented).
