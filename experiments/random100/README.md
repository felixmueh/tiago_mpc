# Frozen random100 on modern GATO

These are the exact ordered 100 random start/goal pairs from GATO branch
`felix-expr/random-reaching`, not the comfortable-start cone distribution.
`manifest.json` pins that branch's commit and every original NPZ hash.
Initial velocities are zero; each task starts independently from its own
recorded random posture. Goal configurations are FK witnesses, never controller
seeds or posture targets. No task is filtered or resampled.
These frozen XYZ goals refer to the historical flange (`arm_right_tool_joint`);
the runner explicitly selects that frame. They are not the new grasp-center
tasks. Its per-case summaries and cohort configuration record the frame.

```bash
.venv/bin/python experiments/random100/run.py --dry-run
.venv/bin/python experiments/random100/run.py
```

The second command runs all 100 sequentially in separate processes, once each.
Default outputs are a fresh timestamped directory under
`_expr_data/`. The runner acquires the shared GPU locks;
individual timeouts, limit violations and numerical failures retain their
outputs and independent cases continue. Shared setup failures hold the rest.
A 180 s process watchdog is distinct from the 8 s simulated observation.

The application defaults are `configs/reaching.json`: B1/N32,
10 ms control/knots, at most 3 ms Pinocchio RK4, fixed simulation pacing, SQP5/PCG120,
5 cm position plus speed-L1<=1 rad/s, no dwell, orientation or final approach.
Joint 6's native/GPU coordinate conversion is unchanged. The historical
reference's orientation slots are preserved as inputs but upstream ignores
them. The first actual cold solve supplies the first applied torque.
The default has terminal position weight800, zero soft limit costs, ADMM100 /
eps_rel .001, q/qd/u penalties1/.3/.0005, and planning bounds .1 rad /80%/85%.
It is the already tested 84-arrival/100-limit-pass configuration, promoted from
`c3f226b`; the retained soft+ADMM cohort had69 arrivals and100 limit passes.
Evidence stays at `/workspace/tiago_mpc-admm-only/_expr_data/admm-only-reaching-20261020/`.
Selection used an overlapping pilot; this is not holdout or robot evidence.

`--dry-run` reports complete effective settings and the selected JSON hash.
The selected JSON is copied to the cohort and passed explicitly to each case.
Explicit `--config` profiles continue to override the frozen legacy base,
including its30 ms interval and soft costs when those fields are absent.
Copy and edit the complete `configs/reaching.json` to vary the new defaults.

This is a separately identified successor using modern GATO and ADMM limit
rows. Historical settings in `historical_protocol.json` are context, not the
new runtime configuration. Other differences are SQP5 versus historical1000,
ordinary effort cost 1e-4 versus1e-5, resetting adaptive rho each feedback
versus retaining it, and upstream's soft-cost implementation/bounds versus
the fork's compiled C2/interior costs. These changes do not isolate an ADMM
effect. ADMM uses upstream's finite budget/residual exit and automatic inner
linear-solver selection; actual inner modes and work are recorded.

Arrival and whole-episode simulated physical-limit compliance are separate.
No torque clipping, candidate veto, early limit-stop, rescue controller,
retuning, KKT acceptance gate or robot execution is included. Per-case output
retains all plant states/controls, per-solve native/controller timings, SQP/PCG/
ADMM work and predicted row violations, input/source identities and failures.
The root summary retains every assigned ID in its denominator. Numerical
errors prevent a full-limit-pass claim even when the retained prefix was valid.

Preparation checks: all 100 input bytes match the source manifest; all native
FK targets match exactly, initial states lie inside raw position limits, and
the original seeds match state-hold/zero-control initialization. Initial
Cartesian distances range 0.139–1.382 m. Existing same-module convention and
small-reaching evidence is reused; the first assigned random task is counted,
not discarded as a warmup. GATO/GRiD sources and build are unchanged.

Analyze the completed saved cohort without a GPU:

```bash
.venv/bin/python experiments/random100/analyze.py <output-directory>
```

This checks the recorded initial states and independently recomputes joint,
speed and torque excesses from all retained plant samples, then produces a
case table, arrival/error plot and compact report beside the raw recordings.

The completed 20 October 2026 cohort is at
`_expr_data/random100-admm-sqp5-20261020/` relative to the
repository root. Its `REPORT.md` / self-contained `REPORT.pdf`, `analysis.json`
and full per-task histories contain the observed results; generated results
remain outside Git.
