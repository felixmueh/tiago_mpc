# Runtime configuration diagnosis

This study starts from the unchanged random100 ADMM cohort at application
`36dae3c` / upstream GATO `aeacf16`. Only public runtime configuration is
changed: iteration caps, tolerances, per-row penalties, bounds and, separately,
knot/control resolution. The driver exposes JSON overrides and retains a few
diagnostic plans. Model assets, GPU solver and GRiD source remain unchanged.

The declared pilot is `[8990,1124,309,6536,8306,3546,8048]`: two torque failures,
the position failure, both speed failures, one clean arrival and one clean
timeout. This is a deliberately selected diagnostic set, not a random success
rate. Each named configuration gets one episode per assigned ID, with its own
fresh directory. Initial states, position goals, eight-second observation,
arrival criteria and raw physical-limit definitions stay frozen. No retry or
filtering changes the old cohort. Results and all intermediate attempts live
under `../experiment_data/tiago_mpc/configuration-study-20261020/`.

The profiles document the adaptive sequence: ADMM cap100; then torque-row rho
scaled50x to match the effort cost's50x change versus upstream's tuning;
then relative exit1e-3; then SQP20; then an explicit margin comparison.
The initial failed solves use all ten ADMM iterations per SQP iteration.
All100 initial postures are inside the original0.1rad planning margins.

Saved first-stage plans distinguish mismatch with the solver's discrete
trapezoidal dynamics from mismatch with actual30ms Pinocchio RK4 execution.
This is a bounded physical prediction check, not KKT certification or a new
solver repair investigation. More SQP reduces the former in the speed cases;
the larger execution mismatch remains. The shorter-step profile uses
N96/10ms (0.95s preview), versus N32/30ms (0.93s). It also follows upstream's
native horizon-dependent ADMM inner backend: PCG at N32, reused factorization
at N96. Stage-count/cost accumulation differs; the comparison is a practical
configuration assessment, not an isolated integrator or backend ablation.

```bash
.venv/bin/python experiments/random100/run.py --dry-run \
  --config experiments/configuration_study/profiles/admm100.json \
  --pairs 8990 1124 309 6536 8306 3546 8048
.venv/bin/python experiments/configuration_study/summarize.py \
  ../experiment_data/tiago_mpc/configuration-study-20261020
```

Selected configurations must be assessed on the complete frozen100 before any
whole-set improvement claim. Keep physical compliance, arrivals, computation
cost and execution errors separate. Timings are diagnostic; there is no
real-time or robot qualification and no KKT residual acceptance gate.

## Selected configuration and complete comparison

`profiles/n32_dt10ms_sqp5_margins_terminal10.json` is the selected explicit
configuration. The initial parameters were a port of the existing TIAGo
reaching configuration (`/workspace/GATO/tiago_src/gato_tiago/config.py`,
including the approved reaching costs), combined with the pinned upstream
controller's ADMM defaults (`external/gato/python/gato/controller.py`,
`LIMIT_DEFAULTS`). They were not untouched upstream solver defaults, nor a
previously qualified combination for these random starts. The N32/30 ms,
SQP5/PCG120 experiment setup was explicit. The original100 also differs from
the historical experiment's SQP1000 and effort cost1e-5.

| Parameter | Initial cohort | Selected cohort |
| --- | ---: | ---: |
| ADMM iteration cap | 10 | 100 |
| ADMM relative exit tolerance | 0.01 | 0.001 |
| Torque-row ADMM rho | 1e-5 | 5e-4 |
| Planning velocity bound | 2.4 rad/s | 2.0 rad/s (80% physical) |
| Planning effort bounds | 42.9 / 25.9 Nm | 36.55 / 22.1 Nm (85% physical) |
| Knot/control interval | 30 ms | 10 ms |
| Terminal position weight | 80 | 800 |

The 50x torque-row penalty scaling was an informed candidate based on our
effort cost1e-4 versus upstream default2e-6; it was tested, not treated as a
mathematical scaling guarantee. Stronger ADMM removed the pilot torque/position
failures; margins at30 ms still left both speed failures. SQP20 reduced the
discrete-model defects but left large execution mismatch. Shortening the
interval helped the latter; terminal800 improved short-preview goal progress.
The selected profile retains SQP5, PCG120, running costs, original0.1 rad
planning position margins, input states, physical grading limits and arrival
criteria. N remains32, shortening preview from0.93 to0.31 s. The separate
basic reach passes at5.77 s/49.997 mm with all sampled physical limits passing.
The preceding terminal80 and terminal240 basic checks timed out and are retained.

The selected pilot had5/7 arrivals and7/7 limit passes. Its subsequent frozen
100-case run had69 arrivals,100 whole-run physical-limit passes,69 satisfying
both,31 timeouts and zero execution errors. Initial100 had57/81/47 respectively.
All100 ran once per configuration in the original order. The pilot influenced
selection, and its seven cases also occur in the100; this is not a holdout.
Returned plans can still cross conservative internal bounds: executed physical
compliance is assessed separately from finite-budget ADMM plan feasibility.

The N96/10 ms SQP20 attempt had a severe finite first-case regression (108.4 rad
position excess,140.5 rad/s speed excess). It was stopped after one completed
case; the second was interrupted and five held. `STOP.json` preserves that
status. N96 selects a different native ADMM backend; no isolated timestep cause
or solver defect is established. No solver repair was attempted. Continue from
the selected N32 profile rather than resuming the held N96 configuration.

Full comparison, paired changes, independently audited raw trajectories,
all intermediate cohorts and PDF/plot exports are retained at:
`/workspace/experiment_data/tiago_mpc/configuration-study-20261020/`.
`REPORT.md` / `REPORT.pdf` describe results and limitations;
`build_comparison.py` regenerates the saved-data comparison. The original
cohort remains separately at
`/workspace/experiment_data/tiago_mpc/random100-admm-sqp5-20261020/`.
Separate verified archives are in the adjacent `backups/` directory.

## 10 ms horizon extension pilot

The follow-up at N64 and N128 is retained at
`/workspace/experiment_data/tiago_mpc/horizon-study-20261020/`
(`REPORT.md`, self-contained `REPORT.pdf`, independent raw audit and module
receipts). On the same seven selected cases, explicit ADMM inner PCG gave
N64: 4 arrivals/7 whole-run physical passes; N128: 3 arrivals/7 physical
passes, versus selected N32: 5 arrivals/7 physical passes. All completed once,
with zero execution errors. These are diagnostic pilot counts, not random-task
success estimates; no full100 N64/N128 run was made in this session.

Native-default N64/N128 each completed only case8990 with severe finite
position/speed violations; each remaining six-case pilot is held in HOLD.json.
These horizons select bdsv_factor by default. The separately named
`n64_dt10ms_sqp5_margins_terminal10_pcg.json` and
`n128_dt10ms_sqp5_margins_terminal10_pcg.json` preserve PCG explicitly.
This policy comparison does not establish a causal backend defect. Source,
model, costs, frozen inputs and grading remain unchanged; timings use the
fixed simulation clock and are diagnostic. User priorities then moved to
friction/damping tracking and rendering. All negative cohorts remain retained.
