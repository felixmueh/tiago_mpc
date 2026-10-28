# Refactor run: working tracker

Untracked working file for the refactor of 2026-10-22. Only the coordinating
session edits it; sub-agents read it and report back. At the end, lasting
decisions move to `docs/decisions.md` and this file is deleted.

**Plan (approved by Felix):**
`/home/felix/_Uni/MasterThesis/agent-reading/tiago-mpc-design/implementation-plan.md`.
Design: `design.md`; requirements: `requirements.md` (same folder).

## Principle: KEEP IT SIMPLE

- Simpler and smaller while maintaining correctness. Every piece and method
  earns its place; prefer deleting to adding.
- Easy to experiment at every point: by composing pieces (another law, plant
  or task) and by changing code directly in a worktree.
- Plain functions and small dataclasses. No frameworks, registries or class
  hierarchies. No compatibility shims; no legacy protocols.
- Every coefficient is a parameter, never baked into an algorithm (the friction
  values are questionable). Experiments must be able to run a plant whose
  coefficients differ from the predictor's.
- Thesis glossary terms: execution controller, control update and control
  rate, command rate, "the robot's onboard controllers". No "playback".

## Rules for this run

- **Integration branch:** `cleanup/defects-and-defaults` in
  `../tiago_mpc-cleanup`. Each step gets its own branch and worktree from it;
  the coordinator reviews and merges. Nothing goes onto `main` without Felix.
- **Host without git-lfs:** create worktrees with `.git/mkwt.sh <path> <branch> <base>`
  in the main checkout (LFS files copied from `.git/lfs/objects`, marked assume-unchanged), and
  commit with `git -c core.hooksPath=/dev/null`. Never `git stash`.
- **Tests:** `GATO_CACHE_DIR=$PWD/.cache/host-gato
  /home/felix/_Uni/MasterThesis/impl/tiago_mpc/venv-host/bin/python -m pytest -q --ignore=external` (main's venv-host has grid-rbd 0.5.3 and mypy since 2026-10-23).
- **GPU:** one job at a time; check `nvidia-smi` first. Outputs go in the
  worktree's `_expr_data/`.
- **GPU checks (Felix, 2026-10-22):** no full random100 per step. Between
  steps, smoke-check a few cases against the baseline (a few arrivals plus
  one or two timeouts, e.g. ids 577 and 9637). The full cohort runs once at
  the end, or later, at the coordinator's discretion.
- **Equality (Felix, 2026-10-22):** bitwise-identical results help debugging
  but are not required; sensible reaches and tracking are enough.
- **Each step reports** its commits, test counts, and Python line counts
  before and after for `tiago_mpc/`, `experiments/`, `examples/` and `tests/`.

## Steps

| # | Step | Branch / worktree | Status |
|---|---|---|---|
| 0 | Merge the trims branch (`2eb9544`) | `refactor/plan-trims` | merged `2eb9544` (215 tests + 102 subtests) |
| – | Baseline: random100 cohort on the integration branch (reach and figure-eight references exist: `../tiago_mpc-plan-playback/_expr_data/plan-playback/`) | integration at `2eb9544`, `_expr_data/random100-baseline-20261022/` | done: 81 arrivals (all valid), 19 timeouts, 0 errors, 100/100 within limits |
| 1 | Parameters: `MpcParams` in `mpc.py`; delete `configs/`, `configuration.py` | `refactor/step1-params` (`../tiago_mpc-step1-params`) | merged `580f9bb` (215 + 102 tests; smoke 5/5 bitwise equal to baseline); lines tiago_mpc 3232→3210, examples 621→605, experiments 513→513, tests 2747→2685 |
| 2 | `Mpc` (`MpcInProcess`, `MpcWorker`), `Plan` with knot states, `updates`, `laws.py` | `refactor/step2-mpc` (`../tiago_mpc-step2-mpc`) | merged `f4a8d88` (219 + 102 tests; default reach bitwise equal to `580f9bb`; smoke 5/5 bitwise equal to baseline; worker and in-process plans bitwise equal); lines tiago_mpc 3210→3129, examples 605→583, experiments 513, tests 2685→2672 |
| 3 | `loops.py` (fake-plant CPU tests), `SimPlant`, `Record`, experiments, delete the old simulation scripts; GPU gate | `refactor/step3-loop` (`../tiago_mpc-step3-loop`) | merged `9482086` (197 + 102 tests; reach and smoke 5/5 bitwise; figure eight all 2400 commands bitwise, last state differs by the summed clock); lines tiago_mpc 3129→3482 (save_replay moved in), experiments 513→565, examples 583→121, tests 2672→2679 |
| 4 | `RosPlant`, robot tests, 500 Hz timing test; delete `ros2_mpc.py` and `runner.py` leftovers | `refactor/step4-robot` (`../tiago_mpc-step4-robot`) | merged `51864a3` (204 + 102 tests; 500 Hz loop p99 0.08 ms with GC paused; smoke 5/5 equal); lines tiago_mpc 3482→3323, experiments 565→611, examples 121→0, tests 2679→2659 |
| 5 | `tiago.py` (TIAGo facts), final layout, replay export to `experiments/replay.py` | `refactor/step5-tiago` (`../tiago_mpc-step5-tiago`) | merged `fb41698` (203 + 102 tests; reach and smoke bitwise equal to step 4 except timings; plant names unchanged); lines tiago_mpc 3323→3073, experiments 611→836 (replay.py moved in), tests 2659→2625 |
| 6 | Docs, README, AGENTS.md split into rules plus `docs/decisions.md` | `refactor/step6-docs` (`../tiago_mpc-step6-docs`) | merged `7cabfbd` (rebased on step 5; docs ~1616→728 lines; names checked against the code) |
| 7 | Whole-system review in a sub-agent (Felix, 2026-10-22): simplicity first, then correctness; known problems (`todo.org`, this tracker) are fine | on `7cabfbd` | done: `agent-reading/tiago-mpc-design/system-review.md` (14 simplifications, 4 low correctness findings, safety notes); awaiting Felix |
| 8 | Review fixes: correctness 1–4, simplifications 1, 3, 5, 6, 8, 9, 11, 13, 14, PDF report dropped. Felix: SEA path stays (future), `from_dict` stays, `save_replay` stays general (item 10 skipped), safety notes for his review; random100 GPU locks kept | `refactor/step8-review-fixes` (`../tiago_mpc-step8-review`) | merged `ac5ec81` (203 + 67 tests; reach and smoke bitwise equal to step 5; plant names unchanged; 0.5 s figure eight ok); lines tiago_mpc 3073→2851, experiments 836→813, tests 2625→2543 |
| 9 | Time contract (codex_review.md item 1, Felix's amendment): `Clock` class (`wall`, `simulation`, `external`), owned by the plant; `plant.now()` uses `plant.clock.now()`; the loop runs on execution time, `State.t` stays measurement time; pacing stays in `plant.advance` | `refactor/step9-clock` (`../tiago_mpc-step9-clock`) | merged `8cfec7e` (209 + 67 tests; sim-clock reach and smoke bitwise unchanged; wall-clock reach: known lease fault at 0.34 s); tiago_mpc 2851→2928, tests 2543→2628 |
| 10 | `Mpc`/`MpcProcess` naming; one shared experiment lifecycle function (experiments pass `mpc`); `Mpc.request` rejects an unread plan; zero-solve summary; clock/loop match check | `refactor/step10-lifecycle` (`../tiago_mpc-step10-lifecycle`) | merged `942038e` (213 + 67 tests; reach, smoke, 0.5 s figure eight bitwise unchanged except timings); `experiments/common.py` (`run_experiment`, `launch`); tiago_mpc 2928→2935, experiments 813→834, tests 2628→2671 |
| 11 | Types for the core: Protocols for plant, Mpc, task; aliases for law and callback; a type checker over `tiago_mpc/` (not experiments) | `refactor/step11-types` (`../tiago_mpc-step11-types`) | merged `1749889` (214 + 67 tests incl. mypy 2.4.0; Protocols `Plant`, `MpcLike`, `Task`, aliases `Law`, `Callback` in `plan.py`); tiago_mpc 2935→3013, tests 2671→2712. Container `.venv` needs `pip install -r requirements-dev.txt` |
| 12 | Felix: protocols named `PlantInterface`, `MpcInterface`, `TaskInterface`; required `task.at(t, n, dt) -> (reference, updates)` replaces `reference` and the optional `updates`. Then merge 9–12 to main (Felix: "after these fixes merge") | `refactor/step12-interfaces` (`../tiago_mpc-step12-interfaces`) | merged `6963dc2`; reach and smoke unchanged; **main fast-forwarded to `6963dc2`** (214 + 67 tests); Felix's working copy of `docs/decisions.md` updated (plant clock, `MpcProcess`, interfaces, `task.at`, Gazebo) |
| 13 | Felix (visualizer migration): explicit compile, `find_binary`, `compile_binary(params, use_cache=True)`, `Mpc(params, binary)` / `MpcProcess(params, binary)` load only, `Mpc.compile`; `Mpc.step` raw GATO result; replays carry `provenance.params` | `refactor/step13-compile` (`../tiago_mpc-step13-compile`) | merged into the integration branch `0f83789` (218 + 67 tests incl. mypy; reach and 5 smoke cases bitwise equal to `6963dc2` except timings); on main since `b3597b0`/`7385f06` (another session rebased onto it); main's `venv-host` fixed (grid-rbd 0.5.3, GATO metadata 0.1.2, Felix's OK); tiago_mpc 3011→3093, experiments 834→838, tests 2712→2788; decision added to Felix's working copy of `docs/decisions.md` |
| 14 | Felix: `SimPlant` holds the start pose until the first command (as PAL's position controller does until activation). **main fast-forwarded to `b3597b0`** before this step (221 + 67 tests) | merged `7385f06`, **main fast-forwarded** (222 + 67 tests); zero-latency reach and smoke bitwise unchanged; hold on both clocks, so `Fixed`/`Measured` latency results change (by design); wall-clock reach now faults at 0.14–0.19 s (every solve at the ADMM cap); worktree and branch removed |
| 15 | Felix: fixes A and C, configurable, on by default: `max_sqp_time_s=0.01` (replaces `budget_s`), `warmup: Warmup | None = Warmup(max_sqp_iters=10, max_sqp_time_s=0.2)` via GATO's `controller.warmup()` (falls back to `warmup: bool` if GATO can't apply the limits); `mpc.warmup(state, reference)` called by `run` before the first tick; exactness check with both off; Felix: no wall-time budget on the simulation clock (sim mains and random100 set `max_sqp_time_s=None`; no code guard, misuse is user error) | `refactor/step15-budget` (`../tiago_mpc-step15-budget`) | merged into integration `c6bfc8c` only (main stays `7385f06` until the latency analysis); 227 + 67 tests; both off = bitwise baseline. GATO's public API cannot limit the warm-up, so `warmup: bool = True` (one `controller.warmup()`, 5 SQP, ~42 ms at the ADMM cap). Extra commit `6779969`: wall-clock `SimPlant.state()` integrates up to now. Wall clock with both: no lease fault, but the arm diverges and a plan exceeds the raw effort limit at 0.32/0.39 s. Worktree kept for the analysis |
| – | Experiment B (Felix's interest, not merged): warm start shifted by elapsed knots vs GATO's one-knot shift | `experiment/elapsed-shift` (`../tiago_mpc-elapsed-shift`) | done: hypothesis refuted; saturation follows latency (Fixed 0.02/0.045 diverge or error at zero budget); elapsed shift changes nothing; report kept in main's `_expr_data/elapsed-shift/` (REPORT.md, table.md, metrics.json); worktree and branch removed (Felix) |
| – | Latency analysis (Felix: "before we decide anything"): GATO mode B vs tiago_mpc; draft `research/tiago-mpc-latency-analysis.md`. Root cause shown: start pose 0.007 rad outside joint 4's ADMM planning bound → every solve at the ADMM cap with latency; 0.05 rad margin or GATO's ADMM budget (cap 10, eps_rel 1e-2) → 3–4 ms solves. Felix: "the actual problem is the high ADMM count; we can't have slow solves". Adding: how other MPC frameworks apply plans on real robots | `experiment/latency-analysis` (`../tiago_mpc-latency-analysis`) | done: `research/tiago-mpc-latency-analysis.md` (286 sim runs; literature section on how other systems apply plans); worktree kept as cleanup candidate |
| 16 | Felix: start pose inside the ADMM planning box (+ test), GATO's default ADMM settings, warning when a plan is first commanded from its second knot (index 1), with a comment naming the remedies (feedback around the plan; latency compensation), default command period 2 ms (500 Hz); full random100 + figure eight | `refactor/step16-admm` (`../tiago_mpc-step16-admm`) | done, NOT merged (`e42ae04`, 230 + 67 tests): Q_START joint 4 −2.35 → −2.33; full GATO ADMM defaults break random100 (50 arrived, 46 first-plan raw-effort errors) because of rho_u 1e-5; rho_u 5e-4 → 82/18/0. Felix: rho_u 5e-4 for TIAGO, figure eight keeps 1e-5 (`31c405a`). **main fast-forwarded to `31c405a`** with steps 15 and 16 (230 + 67 tests); worktrees and branches removed; final cohort kept as `_expr_data/random100-step16-20261023/` (82/18/0). Wall clock at 2 ms ticks: reach 0.11 s, far reach 1.64–1.68 s, solves 2.8–3.4 ms |
| – | Research: impact of resetting rho every solve vs adapting (reset per goal) | `experiment/rho-reset` (`../tiago_mpc-rho-reset`) | done: `research/tiago-mpc-rho-reset.md`; the reset concerns GATO's SQP damping `SolverParams.rho`; carrying it collapses it to 1e-8 within ~7 solves; outcomes unchanged (random100 82/100 both ways, goal 3, figure eight); Felix kept the reset; worktree, branch and data removed 2026-10-23 (note marks it) |
| 13 | Second review of `6963dc2`: [P1] refresh execution time after a potentially blocking solve, recheck duration and lease before commanding; [P2] wall-clock `SimPlant.apply` integrates elapsed time under the previous command first | `refactor/step13-timing` (`../tiago_mpc-step13-timing`) | merged `b3597b0`, rebased onto another session's `08ba60e` (replay provenance) and `0f83789` (explicit GATO compile; worktree `../tiago_mpc-step13-compile`, not ours); 221 + 67 tests combined; sim-clock reach and smoke unchanged; wall-clock reach: known lease fault at ~0.36 s. Open: hold the start pose in a wall-clock `SimPlant` until the first command? |
| 17 | Felix (visualizer limit handling): effort check on the execution side; `Mpc.result` returns over-limit plans, `run` checks each plan on arrival (`checked`), `guard` each command | `refactor/execution-effort-check` | **main fast-forwarded to `436b419`** (233 + 67 tests incl. mypy); random100 577/9637/3546/3331/1033, default reach and a far reach with an effort error bitwise equal to `31c405a` except timings; worktree and branch removed; decision added to Felix's working copy of `docs/decisions.md` |

## Decisions that steer the steps (2026-10-22)

- **Simulator:** take the least work now. Move today's Pinocchio RK4 and
  optional hard-sign friction into `SimPlant` unchanged. The substep stays an
  argument (reach 3 ms, figure eight 0.1 ms). Friction coefficients are
  `SimPlant` parameters, with the URDF as the default. The real simulator
  (armature, constraint solving) comes later; reach results may change.
- **Records:** per command tick. Recording resolution changes are accepted.
- **random100:** one worker per case.
- **Robot clock:** `State.t` is the monotonic receipt time. Gazebo may need the
  ROS stamp (real-time factor); keep that open, with a comment in `RosPlant`.
- **Nonzero-latency results** may change (plans take effect at the next tick).
- **Replay export (Felix):** `save_replay` leaves the package for
  `experiments/replay.py`; experiments call it after `record.save()` (step 5).
- **Controller reset:** one `Mpc` per run; it resets on its first request
  and never again. Task generations go. A fresh controller is a new `Mpc`.
- **`updates` only** (no `configure` hook); warm-up is an `MpcParams` field.
- **No context managers;** the loop runs `plant.start()` before the first tick
  and `plant.stop()` first on exit.
- **Friction agent:** starts after this refactor; there are no friction
  branches to merge.
- **Visualizer:** stays on `../tiago_mpc_vis_baseline` until it migrates to
  the pieces, after step 4.

## Done before the plan (on the integration branch)

- `c76b68d`: loss-free reaching default, friction opt-in.
- `74e0bc3`: review defects and the torque TODO.
- `c63e8e6`: legacy profiles, shims and hashing removed.
- `952b730`: ROS must be sourced.
- `5649cdb`: plant identity and shared cache.
- `f813162`: plan sampling and the 10 ms tracking default.
- `c7eb5c4`: `ROS_DISTRO` in the run summary.

## Notes for later steps (from step 1)

- **Step 3:** `FIGURE8` sits in `mpc.py` until it moves to `experiments/figure8.py`;
  until then importing `tiago_mpc.mpc` loads Pinocchio and reads the URDF.
  The figure-eight plant name changed (hashes `asdict(Friction)`), so its
  first GPU run recompiles (minutes). random100 runs `reach_sim` in a
  subprocess that only gets `--ee-frame`; one worker per case removes that.
- **Step 6:** `AGENTS.md` still says reaching defaults to friction (around
  lines 13–14 and 151–153).
- **Visualizer:** imports `reach_sim` helpers; adapt when it migrates.

From step 2:
- **Warm-up runs on the first request** (GATO's warm-up needs a state and a
  reference). Step 3: `Measured()` latency must charge only the last solve,
  not the warm-up.
- **`plan.stats["position_error_m"]`** (FK inside the Mpc) serves the
  runner's arrival check; it moves to the task (step 3 or 4).
- **`Mpc.details`** (build details, incl. limits) is not in the design; it
  goes once `tiago.py` and `RosPlant` own the limits (steps 4 and 5).
- **Import isolation was relaxed:** the runner process now imports gato's
  Python package and (via `FIGURE8`) Pinocchio; the test only checks
  `rclpy`. Step 3/4: assert no compiled GATO module loads in the loop
  process, and no Pinocchio once `FIGURE8` leaves `mpc.py`.
- **Behaviour changes (for Felix):** a simulated plan beyond the raw effort
  limits now raises instead of finishing with `limits_pass=False`; the
  runner lost its 180 s worker start timeout and its state and safety
  polling while the worker builds (arm stays in PAL position control).
- **Saved outputs changed:** native planned states and torques instead of
  raw `xu`; `solve_work.json` keys `sqp`/`pcg`/`admm`/`admm_linsys`.
- **Step 6:** `docs/ROS2_RUNNER.md` still describes the worker protocol,
  `WorkerConfig` and `--budget-ms`.

From step 3:
- **Loop order (deviation, for Felix):** a plan requested in a tick is
  collected in the same tick, so zero latency uses it at its own tick (as
  before). The last tick sends no command but reaches the callbacks.
- **`guard(command, plan, t, effort)`**: effort from `mpc.details` until
  `tiago.py` owns the limits.
- **Removed main options** (now composition): `--hold` (`law=hold_first`),
  figure-eight `--losses` (pass `SimPlant` without friction), `--build-only`,
  `--input`, `--stop-on-raw-limit`. Added `--command-period`,
  `--update-period`, `--latency`.
- **Task defaults** are computed by FK when `experiments/` is imported.
- **Step 6:** README, `VERIFICATION.md`, AGENTS.md still name `reach_sim.py`,
  `track_figure8.py`, `experiments/tools.py:save_replay` (now in
  `record.py`).

From step 4 (Felix's answers, 2026-10-22; findings are in `todo.org`):
- **`--duration`:** back as an optional cap (step 5 adds it).
- **Activation republish bound:** comment only, left as is (step 5).
- **GC paused in `run`:** comment only; must be enabled in between, to be
  discussed (step 5 adds the comment).
- **`safety=` in `run`:** deleted (Felix; step 5). Safety is `RosPlant`'s job.

From step 6:
- **After main merge:** `research/tiago-pal-controller-interfaces.md` (thesis
  workspace) links `impl/tiago_mpc/docs/ROS2_ADAPTER.md`; point it at
  `docs/robot.md`.
- **Unverified doc claims to show Felix:** host toolchain (nvcc 12.9,
  g++-14) from memory notes; friction-reaching attribution from the
  visualizer's `friction.md`; historical 84/100 cohort and elapsed-shift study
  (raw data removed earlier).

From step 8 (behaviour changes to show Felix):
- `check_model` is one check for both predictors; loss-free is now compared
  with GATO's trapezoidal step (not RK4), the friction check also runs at
  `Q_REACH`.
- `ros2.py`: a failed `load_controller`/`configure_controller` reply now
  raises (before, a node of that name counted as success).
- Interrupted runs are saved as `status: "interrupted"`; random100 builds
  one `MpcWorker` before the first case to catch setup failures.
- `PATH` prepend in `mpc.py` stays (venv cmake version is part of GATO's
  module key), now with a comment.

**Totals vs `main`:** Python outside tests 4298 → 3664 (tiago_mpc 3078 →
2851, experiments 598 → 813, examples 622 → 0); tests 2472 → 2543; Markdown
1927 → 754.

## Decisions after codex_review.md (Felix, 2026-10-22)

- Clock on the plant (one less parameter), as a reusable `Clock` class:
  `plant.now()` uses `plant.clock.now()`. `SimPlant` can run on the
  simulation clock or in real time.
- Type checker for the core; experiment files stay untyped.
- Integration branch stays `cleanup/defects-and-defaults` (= main at
  `ac5ec81`); main only with Felix's OK.

From step 9 (for the main merge): update `docs/decisions.md` lines on "one
loop body, two clocks", "`State.t` on the robot is the monotonic receipt
time" and the Gazebo question to the plant clock. Open: `guard` has no
tolerance (the tick at nominal t0 + 100 ms still commands, by rounding); a
wall-clock `SimPlant` integrates a command from the state's time (≤ 0.17 ms
early).

From step 10 (for the main merge): `docs/decisions.md` says "MPC worker" (random100, three processes); rename to `MpcProcess`. Behaviour: a figure eight cut short by `--duration` reports `timeout`; the record is created after the MPC build; summary keys reordered.

Felix (2026-10-22, after the latency analysis): the way to go is fast solves
plus 500 Hz command ticks (delay under one knot, `sample_knot` then uses knot
0); GATO's u₀-at-arrival rule is dropped; feedback around the plan (an
impedance law on the plan's knot states) and latency compensation are the
remedies once delays reach a knot (warning comment points to them).

## Merged to main (2026-10-22, Felix: "merge to main now then I can review")

- `main` fast-forwarded to `ac5ec81`; 203 + 67 CPU tests pass there.
- Felix's uncommitted AGENTS.md section was already in the new AGENTS.md
  verbatim; his working copy was backed up to the session scratchpad
  (`AGENTS.main-working-copy.md`, `.diff`) and restored before the merge.
- `research/tiago-pal-controller-interfaces.md` links repointed to
  `docs/robot.md` (quotes attributed to `ROS2_ADAPTER.md` at `20162aa`).
- Full random100 on main: `_expr_data/random100-refactor-20261022/`: 81 arrivals, 19 timeouts, 0 errors, 100/100 within limits; all 100 cases equal to the baseline (status, time, final distance, final speed).
- Cleanup done (Felix approved): step and feature worktrees removed after moving their `_expr_data` to `_expr_data/refactor-20261022/<worktree>/`; baseline moved to `_expr_data/random100-baseline-20261022/`; merged branches deleted, plus `experiment/admm-configuration` (merged) and `fix/default-reaching-friction` (`d18a1b6`, superseded by the loss-free default).
- Kept: `../tiago_mpc-cleanup` (its `venv-host` is the only host venv with grid-rbd 0.5.3), `../tiago_mpc_vis_baseline`, and the visualizer-trial container worktree; the other six container worktrees were removed later (see the cleanup section).
- Not pushed.

## Open for Felix

- **AGENTS.md on main has uncommitted edits.** Step 6 folds that working copy
  into the split unless Felix commits it first.
- **Robot checklist for the PAL controllers:**
  `../../research/tiago-pal-controller-interfaces.md`.
- **Friction diagnosis versus the completed 10 ms tracking run:** for the
  friction agent.

## Known issues

- `venv-host` on main has `grid-rbd` 0.5.1; GATO 0.1.2 requires 0.5.3.
- Visualizer: `test_solver_boundary` errors, independently of the baseline
  switch.
- GATO fork: `test/expected_tests.txt` lacks the five trajectory-capture tests.
- Plant names include absolute mesh paths, so checkouts compile separately.

## Deferred

- The `Stream` callback (queue or ring buffer for the visualizer).
- The `Command` type for reference commands.
- A single-process toggle.
- Laws beyond sampling the plan; the `configure` hook.
- The ROS container.
- Migrating the visualizer.
- The real simulator.
- The compile-time friction width (friction agent).

## End-of-run cleanup (done 2026-10-22, approved by Felix)

- **Removed worktrees (refactor):** the step worktrees 1–8,
  `../tiago_mpc-plan-trims`, `-plan-playback`, `-plant-cache`, `-ros-env`.
  Their `_expr_data` moved to `_expr_data/refactor-20261022/<worktree>/`; the
  random100 baseline to `_expr_data/random100-baseline-20261022/`.
- **Removed container worktrees** (`/workspace/...`; data not preserved, per
  Felix): `tiago_mpc-default-friction`, `-gato012`, `-grasp-frame`,
  `-ros2-review-fixes`, `-runtime-friction`, `-shared-setup` (directories and
  `.git/worktrees/` metadata).
- **Deleted branches:** the step branches, `refactor/plan-trims`,
  `feature/plan-playback`, `feature/plant-identity-cache`,
  `fix/ros-env-error`, `experiment/admm-configuration`,
  `fix/default-reaching-friction` (`d18a1b6`), `integration/gato-0.1.2`,
  `feature/grasp-center-frame`, `fix/ros2-review`,
  `refactor/shared-mpc-setup`, `integration/runtime-friction-width`
  (`e29ffd0`).
- **Kept:**
  - `../tiago_mpc-cleanup` + `cleanup/defects-and-defaults`: its `venv-host`
    is the only host venv with grid-rbd 0.5.3 (main's has 0.5.1); remove
    after upgrading main's venv.
  - `../tiago_mpc_vis_baseline` + `vis-baseline`: until the visualizer
    migrates.
  - `visualizer-trial/stick-slip-plant` and its container worktree (Felix).
  - `docs/friction-default-validation` branch (Felix; its worktree removed).
  - `experiment/collision-avoidance`, `experiment/self-collision` (unmerged
    experiments, not part of this cleanup).
  - `../GATO-compile-time-width` (friction agent's call).
  - Caches (`.cache/gato` is the container's).
- **Steps 9–12 (Felix):** worktrees removed with their check outputs; branches
  `refactor/step9-clock`, `step10-lifecycle`, `step11-types`,
  `step12-interfaces` deleted.
- **2026-10-23 (Felix):** main's venv-host has grid-rbd 0.5.3; mypy installed from `requirements-dev.txt` (230 + 67 tests pass with it). `../tiago_mpc-cleanup` and `cleanup/defects-and-defaults` to be removed after the final review agent finishes. `research/tiago-mpc-rho-reset.md` deleted; references to it removed. Push of main: no credentials on this host, Felix pushes (`git -c core.hooksPath=/dev/null push origin main`; no LFS files in the push; submodule commit `fa58597` is on the fork).
- **Left:** delete this tracker after Felix's review of main.
