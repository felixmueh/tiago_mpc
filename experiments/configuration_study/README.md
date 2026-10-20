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
