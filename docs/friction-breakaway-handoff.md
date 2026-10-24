# TIAGo friction and breakaway handoff

24 October 2026. Implement separate breakaway and sliding friction in the
simulated plant as the next model refinement. This handoff records the agreed
direction, evidence and remaining work; the split is **not implemented**.
The request in this turn is documentation only.

The physical fits are a better supported candidate for right-arm joints 1–4
than the current supplied coefficients. They improve local predictions on two
retained recordings, but leave a wrist mismatch and do not establish full-motion
or independent validation. Keep the predictor independently configurable and
simpler than the plant.

## Start from the current code

Read [AGENTS.md](../AGENTS.md), [README.md](../README.md) and
[decisions.md](decisions.md). Felix owns the structure and commits: leave
implementation changes uncommitted for his review. The checkout currently has
ongoing edits based on `ff7ee86`; inspect them before editing shared files.

The 24 October working tree uses the following API, which replaces the older
string-valued Coulomb-law interface:

- [dynamics.py](../tiago_mpc/dynamics.py): `Friction(coulomb_nm,
  viscous_nm_s_per_rad, law=StickSlip())`; the law is plain frozen data:
  `StickSlip()`, `Sign()` or `Tanh(width_rad_s)`.
- [sim.py](../tiago_mpc/sim.py): `SimConfig(friction=..., integrator=...)`, built into a `SimPlant`.
  `StickSlip` uses semi-implicit Euler with implicit viscous damping and the
  coupled `_stick_slip` solve. The default maximum substep is 1 ms. RK4 supports
  loss-free dynamics or explicit `Sign`/`Tanh` friction, but rejects `StickSlip`.
- [tiago.py](../tiago_mpc/tiago.py): `urdf_friction(law=StickSlip())` reads
  coefficients from the canonical full URDF. Derived arm/GPU URDFs are cached,
  not committed. Native joint coordinates stay unchanged.
- [mpc.py](../tiago_mpc/mpc.py): `MpcParams.friction` accepts `Tanh` only.
  Reaching predicts no friction by default; tracking predicts the URDF values
  with `Tanh(.03)`. `record.from_dict` reconstructs recorded law data by type name.

The current stick/slip model uses the **same** per-joint `coulomb_nm` value for
holding and sliding. `passive_torque` handles `Sign`/`Tanh`; it deliberately
rejects `StickSlip`, whose friction is solved within the integration step.
Armature and motor/SEA elastic dynamics remain absent.

## Physical estimates and their limits

The [model provenance](../models/tiago_pro/README.md) and the thesis
[evidence ledger](../../GATO-notes/GATO/felixnotes/EXPERIMENT_USAGE_MATRIX_2026-10-18.md)
(joint-loss fits, line 327 at discovery) retain these rounded October 16 values:

| Right joint | Current Coulomb, Nm | Fitted Coulomb, Nm | Current viscous, Nm s/rad | Fitted viscous, Nm s/rad |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.66 | 2.01 | 4.70 | 4.44 |
| 2 | 1.15 | 2.39 | 9.69 | 5.07 |
| 3 | 1.16 | 2.03 | 5.30 | 2.87 |
| 4 | 0.912 | 1.91 | 4.20 | 2.82 |

The ledger reports breakaway/stiction approximately **10–25% above sliding
friction**, with broadly consistent fits between the two feedback sources.
This is a reason to investigate separate thresholds, not a calibrated static
coefficient for every joint. Do not silently assign `static = 1.2 * sliding`.
If ratios are used for sensitivity checks, expose and label them as assumptions.

Joints 5–7 have no reliable Coulomb-plus-viscous fit in the retained summaries.
The wrist hardly moved in the principal recordings. One joint-6 movement near
1.1 Nm and other wrist threshold observations are qualitative evidence, not a
complete calibration. Do not infer left-arm symmetry.

The original fitting script and `fits.json` were referenced at thesis-side
`research/physical-tiago-20261016/model-mismatch/`, with discussion in
`agent-reading/physical-tiago-impact/notes/model-mismatch.md`. They are not in
the current mount. Recovered numbers therefore lack verified uncertainty
intervals and the full original fitting/filtering procedure.

## Completed comparison with recorded commands

The [saved analysis](../_expr_data/physical-friction-comparison-20261024/REPORT.txt)
contains the method and limitations. Its directory also retains
`configuration.json`, the dynamics/model snapshot, `source.patch`, input hashes,
`check_predictions.py`, prediction arrays, `prediction_metrics.json`,
`coefficients.csv` and `local_prediction_comparison.csv`.

Two physical 24 s figure eights were checked: motor feedback
`20261016T151324.040280Z-figure8_24s` and SEA output feedback
`20261016T144344.880937Z-figure8_24s`. Only the first-four Coulomb and viscous
coefficients changed; wrist coefficients and the coupled rigid-body model stayed
fixed. Each prediction reset from a recorded state and integrated to the next,
roughly 20–24 ms later, with published commands held between publication times.

| Feedback and timing convention | Joints 1–4 position RMS: current → fitted, mrad | Reduction |
| --- | ---: | ---: |
| Motor, callback receipt | 0.732 → 0.426 | 41.8% |
| Motor, source stamps with constant clock offset | 0.570 → 0.308 | 46.0% |
| SEA, arm callback receipt | 0.831 → 0.417 | 49.8% |
| SEA, source stamps with constant clock offset | 0.830 → 0.416 | 49.9% |

There were 1,186 motor and 993 SEA observation intervals. Joint-6 position error
increased **44–47%** through the coupled simulated response, despite unchanged
wrist coefficients. This supports the fitted values as a proximal-joint
candidate, not a complete seven-joint calibration.

These recordings contributed to the original fits. The comparison is not
held-out validation, a 24 s free rollout, or a claim of sub-milliradian accuracy
over a complete motion. Recorded torques are **published commands**, not measured
actuator torque. Instantaneous delivery is an assumption. SEA velocity is a
filtered derivative; observation and actuation timing remain uncertain.

The original inputs are under the existing read-only legacy mount:

```text
/workspace/experiment_data/physical-tiago/raw/physical-tiago-20261016/
  jetson-import/_expr_results/physical-arm/physical_robot/<run>/tiago_right/
```

Use `full_joint_state_history.jsonl`, `command_history.jsonl`, command metadata
and lifecycle events. For SEA, the true arm receipt is
`feedback_metadata.arm_received_monotonic_sec`; the CSV's merged freshness time
can refer to an earlier body observation. Bound prediction intervals after the
first task publication and before `handback_request`.

**Reproduction note:** the retained script/snapshot predates the current law
dataclasses. It used the older `plant_euler(..., friction, "constraint")` API.
Do not run it unchanged against today's package. Port only the API calls, or
reproduce with the preserved numerical snapshot, retaining the original outputs
and identifying any new analysis separately. No rerun is needed just to migrate
the format.

## Proposed breakaway behavior

Use a per-joint static threshold `tau_s` and sliding Coulomb torque `tau_c`, with
finite values `tau_s >= tau_c >= 0` and nonnegative viscous coefficient `b`:

- While stuck, friction supplies the holding torque required by the **coupled**
  dynamics, up to `±tau_s`.
- Once slipping, the resisting bias is `tau_c * sign(v) + b * v`.
- When motion stops, the joint can stick again. Treat stopping, reversal and
  restart consistently, without numerical chatter or spontaneous motion.

`tau_s == tau_c` must reproduce the existing model to practical tolerances.
Keep current behavior available and make the richer plant choice explicit.
The exact field placement/naming is still open; follow the existing frozen-data
and integrator design, without adding a framework or separate configuration
system. Record the effective static/sliding coefficients and law with each run.

The present solver minimizes a convex step objective:

```text
A = M(q) + h * diag(b)
r = M(q) * v + h * (u - bias(q, v))
minimize 0.5 * v_next.T * A * v_next - r.T * v_next
         + h * sum(c * abs(v_next))
```

Its sticking bounds and sliding force both come from the same `c`. A larger
static bound and smaller sliding force do not follow from simply replacing
that coefficient in this objective. Choose and review a consistent regime or
active-set treatment; any persistent sticking/slipping state must reset with
the plant on initialization, Reset and other state replacements. Independent
per-joint torque clipping or a bare `v == 0` switch is not a substitute for the
coupled solve. Retain implicit damping and apply passive losses once.

A Stribeck speed transition is a possible later refinement if recordings support
it. A full hysteresis/LuGre model is not needed for the first static/sliding split.
Keep the MPC predictor's smooth law independently configurable; changing plant
physics does not require changing generic GATO/GRiD code or controller tuning.

## Bounded implementation and validation work

1. Settle the data representation and coupled breakaway algorithm against the
   current working tree. Keep coefficient provenance explicit; separate the
   algorithm change from switching to fitted coefficients.
2. Extend the relevant CPU checks in [test_friction.py](../tests/test_friction.py)
   and [test_sim.py](../tests/test_sim.py). Cover holding below `tau_s`, breakaway
   above it, continued sliding under a load between `tau_c` and `tau_s`, stopping,
   reversal, zero losses, `tau_s == tau_c`, coupled motion and friction opposing
   sliding. Check practical substep sensitivity and resets if modes are retained.
3. Check parameter serialization through `SimConfig` and `MpcParams` (`record.to_dict`) and the
   existing MPC/experiment tests. Predictor friction remains `Tanh`; plant-only
   static coefficients must not silently alter the compiled predictor.
4. Reuse the saved-command comparison for a small, identified successor. Change
   static/sliding separation and coefficient calibration independently so their
   effects can be distinguished. Report proximal and wrist errors separately.
   Use a different retained motion for stronger validation when appropriate.
5. Only after basic physics checks, run a bounded controller comparison with the
   fitted vector and explicitly labelled static estimates. The prior common
   ±50% scale does not represent the fitted vector: joint 2 needs about twice the
   Coulomb term and half the damping. Preserve original failures and cohorts.

The earlier [48-cell controller comparison](../../tiago_mpc-controller-friction/_expr_data/controller-friction-20261023/REPORT.txt)
used source `57cfe79`, single-threshold constraint friction and 1 ms steps. It
does not validate a split-threshold model. At half nominal friction, all four
zero-delay tracking controllers violated recorded speed limits; the PD cases
also violated position limits. Keep those failures visible.

Use relevant CPU tests first; this documentation task runs no new tests or
experiments. Future logs, measurements and figures belong in the owning
checkout's ignored `_expr_data/`; portable replay exports use the shared ingest
directory. Treat the archive as frozen even where it is mounted under its old
`/workspace/experiment_data` name. Do not recreate or write that path, export
new work to `thesis_results/`, or change historical results. Robot qualification
and supervised physical execution remain separate from simulation evidence.

## Model references

- [MuJoCo joint friction loss](https://mujoco.readthedocs.io/en/latest/computation/#friction-loss):
  a joint dry-friction constraint illustrates the single-limit simplification.
- [Drake stiction and sliding](https://drake.mit.edu/doxygen_cxx/group__mbp__continuous.html):
  idealized separate static and kinetic regimes.
- [Simscape rotational friction](https://www.mathworks.com/help/simscape/ref/rotationalfriction.html):
  separate breakaway/Coulomb parameters with a Stribeck transition.
