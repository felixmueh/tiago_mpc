# Reaching: goal posture, torque reference and orientation schedule (handoff)

30 October 2026. A research and experiment task for the next agent. Find out
why random reaches get stuck or fault, and which inputs and cost changes fix it,
backed by the literature and by runs. **Nothing here is implemented.** Felix asked
for this handoff instead of doing the work in the session that wrote it.

## Felix's request

Dictated (speech-to-text: "course description" means a cost description, i.e. a
cost term):

> I have the feeling that there's a lot of research about this and maybe actually
> only use MPC for the reaching or something like this. What we need to do is
> reduce torque regularization by including the final posture as a baseline for
> the joints, like the torque of the final posture. I think this is possible with
> a new cost description. So yeah, this should be a well-studied problem. Do some
> research on these reaches and then how to improve this. And I think in general
> what I kind of want is maybe some tuning, especially if this is like done in
> research where you slowly ramp up the orientation cost or like change the
> weights where in the beginning, depending on trajectory length, I think the
> orientation cost should only be high at the point when we reach it. So maybe
> from the existing previous trajectory we can calculate when we are kind of
> close at what time, and then we ramp up the orientation cost at this time. And
> as long as we're not there, either we keep it low in total, or we just have a
> higher cost for the last node, and we don't define the orientation in between.
> Because defining it in between seems silly. Until we, of course, we're super
> close, then we have basically reached on all nodes, so all nodes should be
> high. Does this make any sense? This will fix a lot of the problems with
> orientation, I think. But I also do a lot of research around this, how this is
> typically done.

> I think I did some research already which you might be able to start out with.
> Do some research on what this needs. Can we give the final pose as an input?
> Sensibly, reliably? I think you found that it's worse, but maybe we also
> fucked some things up. Just try different things to see what works, what
> doesn't work and what is supported by research.

## Start from the current code

Read [AGENTS.md](../AGENTS.md), [README.md](../README.md) and
[decisions.md](decisions.md). Felix owns the structure and commits: leave every
change uncommitted. Paths below are container paths. On the host, `/workspace/` is
`impl/` in the thesis workspace (`/var/home/felix/_Uni/MasterThesis/impl/`).

- **Main (`64dd698`) has the new defaults since 30 October:**
  - 50 Hz control updates, with a declared 20 ms latency and latency compensation (`RealTimeSettings()`, `VirtualTimeSettings.like`)
  - dt 0.03 s, N 32, velocity fraction 0.6
  - planned torque plus PD (`PlanFeedback()`)
  - the supplied armature in predictor, plant and forecast
  - costs 4/8
  - [decisions.md](decisions.md) records them.
- **Orientation tracking is not on main yet.** It is uncommitted work in two places:
  - `/workspace/tiago_mpc-orientation`, branch `experiment/orientation-tracking`. Felix is reviewing it. Do not edit it.
  - the study worktree below.
- **Interface:**
  - `MpcParams.orientation_cost` / `orientation_N_cost` (running / last knot).
  - The reference is (N, 6) per knot: position, then a world rotation vector. NaN orientation slots mean no target at that knot (`cost_menu()` in `tiago_mpc/mpc.py`).
- **Work in the study worktree `/workspace/tiago_mpc-random100-orientation`.**
  - Branch `experiment/random100-orientation` at `64dd698`, plus the orientation diff, all uncommitted.
  - It has the random100 reach-time study: `experiments/random100/orientations.py`, `reach_time.py` (options `--orientation W W_N`, `--law sample_knot`).
  - Add your code there.
- **GATO is the local checkout `/workspace/GATO-tiago-armature`** (branch `experiment/tiago-armature-015`; see `[tool.uv.sources]` in `pyproject.toml`). If you need GATO changes, create your own GATO worktree on a new branch from it and point only your worktree's sources at it. Do not edit existing GATO checkouts.

## The cost menu already has the pieces

`/workspace/GATO-tiago-armature/python/gato/cost_menu.py` (module docstring and
`CostMenu`):

- **Terms:**
  - `state` (x minus target, i.e. a joint-posture target)
  - `input` (u minus target, i.e. a torque reference)
  - `frame_pos`, `orientation`, `velocity`, linear features
- **Activations:** quadratic, pseudo-Huber and barrier activations.
- **Per knot:** every target and weight can be given per knot (`(N, size)`), or per batch entry and knot.
- **Gravity torque reference:** `input.target = "gravity"` with `CostMenu.q_ref` (N, nq) regularizes the torque toward the gravity torque of a reference configuration, per knot (fixed-base plants). This is Felix's "torque of the final posture" with `q_ref` = the goal posture at every knot, or g(q) along a reference path.
- **Torque toward zero is the current default.** `tiago_mpc`'s `cost_menu()` rebuilds GATO's tracking objective (position, qd, u toward zero with `u_cost` 1e-4) plus orientation.
- **The input term is off at the last knot** (no control there).

- **The menu is applied only while the orientation cost is on** (`MpcParams.orientation`). Without it, GATO's native tracking objective runs.
  - That objective has its own running posture anchor: the `"posture"` entry of `UPDATES` in `tiago_mpc/mpc.py`, i.e. `set_q_pos_cost` / `set_q_nom`, which also applies the joint sign map. See `BSQP` in GATO's `python/gato/interface.py`.
  - So for position-only variants, either use that anchor, or apply the menu with orientation off. Say which you did.

So posture targets, gravity torque references and per-knot orientation weights
or NaN targets need no new GATO code. Check the joint-6 sign map at the GPU
boundary for any joint-space target (`tiago.py`, the GPU boundary in `mpc.py`).
The native `"posture"` update already applies it; a menu `state` target needs it
too.

## Evidence so far (simulation, 30 October)

The data is in `/workspace/tiago_mpc-random100-orientation/_expr_data/`:
- `random100-reach-time-N32/` and `-N64/`: raw runs with replays.
- `random100-reach-time-plots/`, holding:
  - `difficult_cases_N32.json` (the named case lists)
  - `weight_sweep.json`, `law_compare.json`, `first_plan_iterations.json`, `oriented100.json`
  - the plots
- Inputs: `experiments/random100/inputs/random_<id>.npz`, with `initial_state`, `source_goal_q` (a known goal posture) and `target_xyz`. The oriented inputs, with IK goal postures, are in `_expr_data/random100-oriented-inputs/`.

Settings for all runs:
- the new defaults with costs 20/40
- orientation 1/8 when on
- each case runs until arrival (5 cm, 2 cm/s), at most 30 s

### Results

| Condition (N 32) | Not arrived at 30 s (faults) | Median / p90 arrival |
|---|---|---|
| Position goals, original starts | 14 (5) | 1.22 / 1.68 s |
| Pose goals, position arrival | 28 | 1.78 / 2.49 s |
| Pose goals, full pose ≤ 0.1 rad | 47 (6) | 1.90 / 2.68 s |
| Oriented starts and goals, orientation cost 0 | 10 (8) | 1.22 / 1.92 s |

- **No long tail:** nothing arrives after 8.1 s. 64 knots (1.9 s horizon) changes nothing.
- **The position timeouts are local minima at joint limits,** not unreachable goals. In all 9, the known goal posture lies inside the planning box (URDF range minus 0.1 rad). The arm instead ends in another posture family, pressed against the box:

| Case | Start distance | End distance | Largest joint gap to `source_goal_q` | Joints at the box |
|---|---|---|---|---|
| pair_08783 | 0.64 m | 0.126 m | 2.50 rad | 1, 2 |
| pair_07065 | 0.50 m | 0.068 m | 3.29 rad | 4, 5, 6 |
| pair_08327 | 1.11 m | 0.201 m | 3.05 rad | 1, 2, 4 |
| pair_03196 | 1.08 m | 0.057 m | 4.79 rad | 1 |
| pair_06906 | 1.22 m | 0.240 m | 4.90 rad | 1, 4 |
| pair_07607 | 1.02 m | 0.178 m | 3.04 rad | 2 |
| pair_04105 | 0.68 m | 0.214 m | 4.42 rad | 1, 3, 4, 5, 6 |
| pair_09624 | 0.99 m | 0.054 m | 3.28 rad | 2 |
| pair_00577 | 1.07 m | 0.424 m | 4.46 rad | 1, 2, 3, 4 |

- **Orientation:**
  - 19 pose cases stall within 5 cm with the orientation error stuck at 0.10–0.43 rad.
  - Stronger constant orientation weights trade position arrival for faults. On the 47 difficult pose cases (full pose ≤ 0.1 rad / position arrival / faults):
    - 1/8: 0 / 19 / 6
    - 2/16: 10 / 15 / 8
    - 5/40: 8 / 8 / 19
    - 10/80: 2 / 2 / 31
  - The faults all happen in the first 0.3 s, with the initial wrist rotation planned at the effort bound.
- **Faults, two causes:**
  - **`effort_limit` at t = 0 (11 of 19):** the first plan's own torque is 100–118 % of the raw limit, against its 85 % bound.
    - The ADMM torque rows have not converged. GATO's ADMM converges across warm-started solves (`gato/bsqp/bsqp.cuh`, comment at `enable_limit_admm`). The first plan has only one warm-up solve behind it: `Mpc.warmup` → `MPCController.warmup`, called before the first tick in both loops (`tiago_mpc/loops.py`).
    - 50 ADMM iterations per SQP iteration fix it; 20 SQP iterations do not.
    - The warm-up runs before the clock starts, so a longer or converged warm-up costs no run time.
  - **`prediction_unavailable` at 0.04–0.06 s (7):** PD on top of a near-bound plan exceeds the raw limit in the compensation's forecast. They disappear with `SampleKnot`.
- **Hypothesis to check:** regularizing the torque toward zero could hold the arm near the goal. Gravity torques are tens of Nm. At u_cost 1e-4, the torque term per knot is of order 0.1. The position term is 20·d², i.e. 0.05 at 5 cm, and orientation 1·θ² is 0.02 at 0.15 rad. Compare the gradients, not only the values.

## Felix's earlier work: start here

- **His notes,** in `/workspace/GATO-notes/GATO/felixnotes/`:
  - `HANDOFF_MPC_INPUT_REFERENCE.md` (u_ref = g(q) input reference)
  - `MPC_COST_FUNCTION_REPORT.md`
  - `REACHING_BATCH_RESEARCH_2026-10-17.md`, `REACHING_BATCH_CODE_MAP_2026-10-17.md`
  - `TRACK_REACHING_BATCH_HYPOTHESES_2026-10-17.md` (CPU IK interior goal postures, goal-witness routes)
  - `HANDOFF_REACHING_LANDSCAPE.md` (directed full-pose IK, branch postures)
  - `BATCHED_MULTISTART_RESEARCH_2026-08-10.md`, `TIAGO_TRULY_MULTIMODAL_DESIGN_2026-08-11.md`
  - `HANDOFF_REACHING_MISSES_LITERATURE_2026-10-16.md`, `PLAN_RANDOM_REACHING_BASELINE_2026-10-15.md`
  - `experiment_tracker.md`, `experiment_result.md`
- **Thesis-workspace notes,** on the host only (the dev container mounts only `impl/`):
  - in `/var/home/felix/_Uni/MasterThesis/research/`:
    - `joint-limits-constraints-vs-penalties.md` (trapped postures as constrained local minima)
    - `cone-reaching-mpc-nonarrival-literature.md`
    - `trajectory-optimization-initialization-and-planning.md`
    - `mpc-near-goal-weight-scheduling-textbooks.md`, `mpc-prediction-horizon-extension-textbooks.md`
    - `random-reaching-hard-misses/`
  - the paper library `/var/home/felix/_Uni/MasterThesis/MasterThesisPapers/`

Felix remembers that giving the final pose or posture as an input was tried and
came out worse. Find where:
- what exactly was tried, on which revision, with which plant and defaults
- whether it could be an artefact. Examples:
  - other defaults then: 10 ms knots, no armature, other costs, no latency compensation
  - the posture used only as a warm start that the solver left
  - a joint target in the wrong coordinates (joint 6's sign map)
  - `u_ref` at the wrong knot
  - an IK posture outside the planning box
  - a posture in the wrong family, reachable only across a joint limit

## Questions

1. **Local minima and joint-limit traps in goal reaching with redundant arms.** How does the literature handle them?
   - IK or a planner picks the goal posture or path, and MPC only tracks or refines ("only use MPC for the reaching").
   - Posture or joint-space regularization toward an IK or nominal posture.
   - Null-space posture tasks.
   - Warm starts from a planner.
   - Multiple seeds or batched hypotheses: GATO is batched, so can one batch carry several posture hypotheses?
   - Which posture to choose: reachable without crossing a joint limit from the start, closest to the start, or seeded from it?
2. **Torque regularization toward a reference:** g(q_goal) at all knots, g(q) per knot (Crocoddyl's `ResidualModelControlGrav`), or toward zero.
   - Is a stall from regularizing toward zero a known effect?
   - What do Crocoddyl's examples and torque-MPC papers on manipulators use (e.g. Kleff et al., Dantec et al.)?
3. **Scheduled weights for reaching:**
   - terminal-only goal costs versus running costs, and ramps over the horizon
   - weights by progress or by predicted arrival
   - minimum-time reaching, shrinking versus receding horizons
   - Pitfalls: plan jumps and warm-start mismatch when weights switch, a chattering predicted arrival (hysteresis, smooth ramps), effects on stability.
4. **Felix's three proposals, judged explicitly:**
   - the goal-posture torque reference
   - the orientation weight ramped at the predicted arrival time, estimated from the previous plan
   - orientation only at the last knot (NaN in between) until arrival is near, then on all knots

## Experiments

Baseline: the current defaults, costs 20/40, orientation 1/8. Run the difficult
cases first (`difficult_cases_N32.json`):
- the 9 position timeouts and 5 position faults
- the 47 difficult pose cases

Variants, each alone, then sensible combinations:
- **a. Warm-start seed:** a joint-space path toward the goal posture, given to the warm-up (`MPCController.reset(x0, xu_warm=...)`).
- **b. Posture cost:** a `state` term toward the goal posture, last knot only and/or small running.
- **c. Torque reference:** `input.target = "gravity"` with `q_ref` = the goal posture, and g(q) along a reference path if that is cheap.
- **d. Orientation schedule:** orientation at the last knot only (NaN elsewhere) until the previous plan's predicted arrival enters the horizon, then from the predicted arrival knot on, ramped, with hysteresis.
- **e. t = 0 effort faults:** repeated warm-up solves until the torque bound holds, or a warm-up with more ADMM iterations.
- **Goal postures:** compare the known `source_goal_q`, an IK posture closest to the start, and, if cheap, a posture checked to be reachable without crossing a limit.

Report per variant:
- arrivals, both position and full pose at 0.1 and 0.2 rad
- median arrival time
- faults and limit violations
- solve time against the 20 ms budget on the GTX 1070

Rerun promising variants on all 100 cases, both conditions, to check that the
easy cases still work.

**Constraints:**
- **GPU:** in the dev container (`tiago-mpc-dev`), one job at a time with the shared locks. Other sessions' series may hold them: wait, never break a lock, and never kill processes you did not start.
- **Data:** in the study worktree's `_expr_data/`, under new labels; do not relabel existing series.
- **Do not edit:**
  - main
  - `tiago_mpc-orientation`
  - the visualizer checkouts
  - the 8090/8091 servers

## Deliverables

- **Report:** `/workspace/tiago_mpc-random100-orientation/_expr_data/reaching-variants-20261030/report.md`, in Felix's report style (answer first, descriptive headings, sources at the end of each section). It has:
  - the literature findings per question, with primary sources and section or equation locators, labelled established / analogy / hypothesis
  - what came of Felix's earlier "worse" finding
  - an experiment table per variant
  - a ranking with what to make default
  - the open decisions
  - If you run on the host, also save the literature part as `/var/home/felix/_Uni/MasterThesis/research/reaching-mpc-local-minima-and-weight-scheduling.md`.
- **Replays:** two or three, with `experiments/replay.py:save_replay` (the orientation version) in `/workspace/_expr_data/ingest/reaching-variants-20261030/`, e.g. one stuck case at baseline and with the best variant.
  - Give links of the form `http://192.168.178.82:8091/?replay=reaching-variants-20261030/<name>` (repeat `replay=` to compare). 8091 runs the orientation visualizer.
  - Say what the data is: simulation, executed motion, simulated time.
