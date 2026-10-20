# Canonical model correction — 20 October 2026

The current robot description is `/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`.
Its exact bytes are retained as `models/tiago_pro/tiago_pro_sim_source.urdf`
(SHA-256 `c8d51a1e8cf363740a394ea272b9961a122a7bcfc418a6186088552cd907d3e6`).
The active full URDF now uses that description with seven explicit right-arm
friction/damping tags and identification/assumption comments. The previously
included full model was outdated. The reduced arm already matched the simulator
source; only dynamics metadata changed, without altering arm transforms,
limits, inertias or its gripper lump. Left-arm dynamics remain canonical.

CPU structural checks establish that the active full model differs from the
preserved source only in the seven dynamics tags/comments, and all retained
reduced links/joints match the canonical source. A three-state Pinocchio check
compares full-model reduction against the standalone arm's torso-relative tool
FK, mass matrix and gravity at atol/rtol 1e-8. Coefficients, known-versus-assumed
comments, exact raw bytes and repeatable preparation are checked separately.
Current runtime Pinocchio ABA / standard GRiD codegen ignore these passive tags;
no friction force or codegen option was enabled, and
the historical 84/100 cohort retains its original source/configuration identity.

A fresh default reach at `d86e041` subsequently passed under all three shared
GPU locks. It arrived after 0.25 simulated seconds with 33.60 mm position error,
speed L1 0.9753 rad/s and zero saved raw position/velocity/effort excess. The
arm's kinematics/inertias are unchanged; these results match the preceding
default smoke. Raw source/config/history and log are in
`_expr_data/canonical-model-correction-20261020/default-smoke/` and
`default-smoke.log`. This is one operation check, not a new 100-case study or
performance claim.

Commands and retained CPU output:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/prepare_models.py --check
```

Fourteen CPU tests pass, including all canonical mesh references, finite vertices
in all 35 STL assets, and full visual/collision geometry loading (51 objects
each; 70 links, 69 joints and 10 mimic followers). The initial
mesh-completeness failure is retained; the two missing original asset names were
subsequently supplied from pinned official PAL repositories (`omni_base_robot`
`1a66854` and `pal_urdf_utils` `e8600a3`) with their upstream licenses. Their
source URLs are recorded in `models/tiago_pro/mesh_sources.json`; no equivalence
to an unavailable installed PAL package is claimed. CPU evidence and download
receipts stay in `_expr_data/canonical-model-correction-20261020/`.
No placeholder geometry, source-URI rewrite, archive migration or GPU rerun
was performed.

The verification below is historical and retains its original model identities.

## Reaching default promotion — 20 October 2026

`configs/reaching.json` now supplies the default for `examples/reach_sim.py`
and `experiments/random100/run.py`. It pins the effective configuration already
tested as `zero_soft_state_rho10.json` at `c3f226b`, based on application
`e45c441`; upstream GATO stays at `aeacf16e7733df1a17dfd6627f3b5d15c2ba4387`.
No solver, model, tracking or experiment-input changes were made.

The frozen 100 comparison produced 84 arrivals, 100 sampled physical-limit passes,
16 timeouts and no execution errors, versus 69/100 arrivals and 100/100 limit
passes with retained soft+ADMM. There were 16 gained and one lost arrival.
The candidate was selected using seven cases also in the 100; this is not
independent holdout evidence. Finite-budget returned plans can violate the
conservative planning bounds; sampled executed-limit passes do not establish
hard feasibility, collision safety or physical robot qualification. Fixed
simulation pacing does not charge solve time to the plant.

Original report, configurations, raw data, timings, failures and source receipts
remain at `/workspace/tiago_mpc-admm-only/_expr_data/admm-only-reaching-20261020/`.
This promotion reuses that qualified configuration without another 100-case
cohort or performance claim. No retained results were moved or relabeled.

Focused CPU verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/random100/run.py --dry-run
```

Five tests pass: the complete selected defaults, every explicit historical
study profile, an explicit empty config reproducing the original example,
full default dry-run output, and copied default config propagation to a mocked
case process. No CUDA solver is constructed. CPU logs and both default/retained-profile
dry-run receipts are in `_expr_data/reaching-default-promotion-20261020/`.
Explicit historical profiles
retain their old base; unspecified speed/effort fractions are recorded as null
with the compiled absolute .1 rad/s / .1 Nm offsets. No-config runs select and
record the new complete JSON.

A direct smoke run at committed source `5a38b9b` also passed, using all three
shared GPU locks after the elapsed-shift experiment released the device:

```bash
.venv/bin/python examples/reach_sim.py \
  --output _expr_data/reaching-default-promotion-20261020/default-smoke
```

The normal comfortable-start example arrived after 0.25 simulated seconds with
33.60 mm position error and speed L1 0.9753 rad/s. Every saved state and applied
torque passed raw URDF limits; the command returned zero. This is one basic
operation check, separate from the frozen random-start study. Its source,
configuration, trajectory and log remain beside the CPU receipts.

The evidence below describes the initial 30 ms configuration, before promotion.

## Initial simulation verification — 20 October 2026

Application source: `638022f1757533257974237c45c206d2b4bd344b`.
GATO: unchanged upstream `aeacf16e7733df1a17dfd6627f3b5d15c2ba4387`.
Hardware/build: GTX 1070 (sm61), CUDA 12.9.86, GCC 11.4.0, CMake 3.31.10,
Python 3.10.12, Pinocchio 3.9.0, GRiD 0.5.1; release float N32 module.

Executed commands, from this checkout:

```bash
.venv/bin/python examples/reach_sim.py --build-only
.venv/bin/python examples/reach_sim.py --hold --timeout 2 \
  --output results/hold-20261020T194021.906203Z
.venv/bin/python examples/reach_sim.py \
  --output results/reach-20261020T194023.881420Z
```

| Check | Result |
| --- | --- |
| Original full and reduced URDF bytes | Identical to supplied sources |
| Assets | All 33 mesh SHA-256 hashes match; LFS object check passes |
| Full URDF geometry | Pinocchio loads 47 visual and 47 collision objects |
| Adapted GPU versus native Pinocchio step, start / target | Maximum absolute state differences 0.0000311 / 0.0000271; tolerance 0.002 |
| Two-second hold | Final/max position error 15.0 mm; speed/position/torque limits pass |
| Reach | Initial error 71.9 mm; arrival at 49.98 mm after 5.82 simulated seconds |
| Reach terminal speed L1 | 0.132 rad/s, below the 1 rad/s arrival threshold |
| Reach simulated physical limits | No excess over URDF position, speed or effort limits at retained plant samples |
| Reach peak speed / torque | 0.244 rad/s / 10.91 Nm |
| Numeric histories | All retained arrays finite; 194 controller solves, 1,941 plant states |

Both episodes returned zero. Each was run once, sequentially with the shared
GPU locks held. Times inside the summaries are diagnostics, not benchmark
comparisons. The reach uses the explicit loose 5 cm arrival criterion and
stops on arrival; it demonstrates basic operation, not centimetre accuracy or
settled tracking. The hold has a 15 mm drift. No orientation, collision,
real-time scheduling or physical robot qualification is claimed.

Raw outputs, source copies, histories, plots and logs remain under `results/`
(ignored). `results/environment.txt` records installed packages. `reach.png`
was inspected for legible axes and complete rendering.

## Retained setup failures

- Initial Pinocchio import failed with unconstrained `cmeel-urdfdom` 6.0.0;
  Pinocchio 3.9 requires the urdfdom 4 / tinyxml2 10 wheel ABI. Both compatible
  versions are now explicit requirements.
- First configure failed because the system CMake did not support `block`;
  `results/build-cmake-failed.log` preserves that configure output. The
  application uses the pinned virtual-environment CMake.
- Registering the native negative-axis URDF directly compiled, then trapped
  in `sim_forward`. `results/build-model-cmake-fixed.log` and
  `results/model-memcheck.log` preserve this failure. GRiD emitted seven
  topology helpers; upstream's fixed-arm adapter supplies a null helper
  pointer. The generated equivalent positive-axis representation emits zero
  helpers and passes the bounded model comparison. The application maps the
  sixth joint's q/v/torque signs; GATO and GRiD sources remain unchanged.
  This is an integration workaround, not a general solver repair.
