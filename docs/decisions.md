# Decisions

Felix's lasting decisions for this repository, by topic, with dates and reasons
where known. Pending implementation is marked explicitly; git keeps superseded wording. The
2026-10-22 refactor implemented the approved design in the thesis workspace
(`agent-reading/tiago-mpc-design/`: requirements, design, implementation plan).

## Purpose and shape

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
  internals and the unfinished ROS2 SEA path remain outside this review.
- **2026-10-23: the effort check belongs to the execution side.** `Mpc` and
  `MpcProcess` return what they planned (`result()` checks only that the plan is
  well formed and finite); `run` checks every knot of a plan when it arrives and
  each command before it is sent. Reason: a simulator such as the visualizer must
  be able to execute, clip or inspect an over-limit plan. Clipping stays a
  simulation setting: the robot path never clips a torque.
- **2026-10-22: compiling GATO's module is explicit.** An `Mpc` takes its
  binary and never compiles: `Mpc(params, binary)` and `MpcProcess(params,
  binary)` load exactly that file; `find_binary(params)` looks it up without
  compiling; `compile_binary(params, use_cache=True)` compiles it (minutes,
  CPU only) unless cached, and `use_cache=False` rebuilds. `Mpc.compile(params,
  use_cache=True)` does both, so a call site that may compile says so.
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

## Model and coefficients

- **2026-10-20: the canonical robot description is the simulator URDF**
  (`/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`), kept byte for byte as
  `models/tiago_pro/tiago_pro_sim_source.urdf`.
  `/workspace/GATO/TiagoProURDF/tiago_pro.urdf` is outdated. The reduced right
  arm already came from the simulator model; its fixed torso root and gripper
  inertia lumped at zero posture are intentional.
- **2026-10-21: supplied joint parameters** for both arms (Coulomb, viscous,
  effort, armature) replace earlier midpoints and guesses (`models/README.md`).
  Armature is metadata that no loader reads, never folded into link inertia.
  kp/kd are not stored: they belong to a different controller.
- **2026-10-22: the coefficients are questionable; every coefficient is a
  parameter.** Develop with them and test against plants with other
  coefficients. Identified values matter for tuning on the robot, not for the
  design.
- **2026-10-22: the simulated plant may be more detailed than the predictor**
  (losses or armature in the plant only), as on the real robot. Enable it
  explicitly and record both models. This replaced the rule against armature in
  only one model.
- **2026-10-21: the joint-6 conversion stays.** The pinned GATO fixed-arm adapter
  needs joint 6's axis and limits normalized in a generated GPU URDF and the
  q, qd, u signs mapped at the GPU boundary; everything else is native. The
  upstream native-axes PR is not in the pin.
- **2026-10-21: grasp-center frame.** Tasks track `arm_right_grasp_joint`, a
  massless frame 0.139 m along the flange's +z: the nominal midpoint of the
  finger pads at zero gripper posture, not a calibrated TCP. Summaries and
  replays record `end_effector_frame`; missing metadata means the flange.
  random100's frozen goals stay on the flange (`arm_right_tool_joint`).

## Friction and GATO

- **2026-10-21: GATO 0.1.2 plus opt-in smooth joint friction** and opt-in SQP
  trajectory capture, from the fork `felixmueh/GATO`, with `grid-rbd` 0.5.3.
  GRiD still evaluates the barrier terms of zero-weight soft limits; there is no
  public switch.
- **2026-10-22: reaching is loss-free by default; friction is opt-in**
  (`c76b68d`). The friction-enabled reaching default became unstable in its
  first 10 ms control interval; the visualizer's diagnosis
  (`../tiago_mpc_visualizer/docs/friction.md`) attributes this to viscous damping
  against joint 7's link-only inertia, stiff for explicit 3 ms RK4 substeps.
  Friction-enabled reaching is not qualified.
- **Tracking uses friction on both sides:** hard-sign Coulomb plus viscous in
  the plant, tanh Coulomb (width 0.03 rad/s) plus viscous in the predictor, both
  from the reduced URDF.
- **2026-10-22: the smoothing width becomes compile-time only**, to diverge less
  from upstream GATO. Not done: the pinned fork still has the runtime-width
  commits; the friction work follows the refactor.

## Tasks

- **2026-10-21: end-effector position cost 4 running, 8 terminal**, for
  reaching and tracking. Earlier 8/1600 and 4/800 results do not qualify 4/8.
- **Tracking (`FIGURE8`):** joint-velocity cost 0.01, torque cost 1e-6, zero
  soft-limit costs (limits through ADMM only), a 24 s horizontal figure eight
  of 0.28 m by 0.18 m, 2 cm ahead of its start.
- **2026-10-22: tracking uses 10 ms knots** (2 ms before); checks at 2 ms do not
  qualify 10 ms.
- **Orientation is unconstrained** in both tasks.
- **Limits through ADMM are approximate:** returned plans can exceed the
  planning bounds, so the planning margins and the independent raw-limit checks
  on knots and commands stay. An ADMM return is no feasibility certificate.

## Execution

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
  latency changed with this. Past a plan's end, both loops fault.
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
  updates at dt. `max_sqp_time_s` 0.01 s caps a solve on the wall clock;
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
- **2026-10-22: the simulator stays as it was:** Pinocchio RK4 with optional
  hard-sign friction, the substep an argument (reach 3 ms, figure eight 0.1 ms),
  friction a `SimPlant` parameter. A real simulator comes later.
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
- **2026-10-22: ROS is sourced in the launching shell**, otherwise a clear
  error; `ROS_DISTRO` is recorded.
- **2026-10-22: `State.t` is the measurement time;** on the robot the monotonic
  receipt time, in the domain of `RosPlant`'s wall clock. Gazebo would use
  `Clock.external()` fed from `/clock`, with ROS stamps as `State.t`; open
  (`docs/robot.md`).
- **2026-10-22: safety is the robot plant's job.** `RosPlant` owns the safety
  worker (it needs the ROS sequence numbers, and activation waits for it), so a
  robot run cannot happen without it. There is no loop-level safety hook.
- **2026-10-22: activation republishes until the state time + 100 ms**, not the
  plan's own lease end; left as is, to review with the safety work.

## Records and history

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
- The ROS container, and with it the Gazebo check.
- Migrating the visualizer off `tiago_mpc_vis_baseline`.
- The real simulator (armature, constraint solving).
- The compile-time friction width.

## Open with Felix

- How garbage collection runs between ticks or runs.
- Gazebo's clock: `Clock.external()` from `/clock` with ROS stamps as `State.t` (not implemented).
