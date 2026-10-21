# Agent instructions

## Optional ROS 2 runner on main — Felix, 2026-10-21

`examples/ros2_mpc.py` integrates the PAL adapter with three processes: parent
ROS I/O/watchdogs, a spawned GATO MPC worker, and a spawned collision-safety
worker. ROS packages remain lazy, optional imports; base simulation and package
imports must work with no ROS installation. Keep CUDA construction/solves in
the MPC child and geometry checks independent of it. Keep command leases,
freshness, fault handling, default checks-only invocation and explicit execution
confirmation intact. No ROS/PAL or physical execution is qualified by this port.

The worker uses the current nominal grasp TCP and task configurations. Reaching
keeps its loss-free predictor; `configs/tracking.json` explicitly enables the
existing smooth friction/damping predictor at width .03 rad/s and supplies the
figure-eight dimensions/edge clearance. Do not apply passive losses twice or
infer a 2 ms ROS update rate from tracking's knot interval; I/O remains 10 ms by
default. Existing ADMM parameters and zero soft costs are unchanged.

The safety model is regenerated from the canonical full simulator URDF with
the retained reviewed alpha-wrap collision overlay. Preserve its 4 cm margin,
blacklist and documented coverage/limitations; do not alter them to accept a
synthetic comfortable-arm/body-zero fixture. Full fresh named body state must
pass the initial gate. See `docs/ROS2_RUNNER.md` and `docs/SAFETY.md`.

Integration checks: 147 CPU tests plus 118 subtests pass without ROS installed;
real spawned N32 GPU workers pass three requests each for reaching/tracking,
with native Pinocchio simulation, finite/raw-bounded torques, preserved 100 ms
leases and clean exit. Raw checks remain in
`/workspace/tiago_mpc/_expr_data/ros2-worktree/ros2-main-integration/`. These are bounded
integration checks, not a physical or complete ROS task qualification.

Felix authorized removal of the merged `integration/ros2-adapter` branch and
`/workspace/tiago_mpc-ros2` worktree after integration. Its ignored validation
files were retained under main's `_expr_data/ros2-worktree/`; embedded original
paths remain historical. Current review notes are in
`_expr_data/ros2-code-review-20261021/`; the confirmed findings are summarized in
`docs/ROS2_RUNNER.md`.

Felix requested fixes for exact effort-controller joint ordering, collision
deadline enforcement before polling late replies, and SEA body-only safety
updates. These pass 161 CPU tests plus 118 subtests. Keep arm freshness separate
from the safety sequence. Missing gripper mimic followers are warning-only by
explicit request; neutral fallback geometry remains a limitation. He accepted
the existing separate configuration-file reads; no snapshot change is requested.
The later consolidation shares model/configuration/controller setup between
simulation and the ROS worker through normal package imports. Keep setup in
`configuration.py`, `model_runtime.py`, `joint_dynamics.py` and `controller.py`;
do not add dynamic example imports or duplicate setup in callers.
Simulation supports `fixed` (configurable seconds, default zero) and `measured`
(complete controller-step wall time) solve latency. Hold the previous torque
during the charge, starting with zero torque; apply the new torque for the
remaining nominal period, and extend overruns to the charged duration. Record
simulated solve-start/update/latency times. Reference phase follows simulated
time. This is simulation-only and does not change ROS leases or safety gates.
Logs remain in `/workspace/tiago_mpc-ros2-review-fixes/_expr_data/ros2-review-fixes/`.

The application remote is `https://github.com/felixmueh/gato_tiago.git`. Its GATO
submodule points to `felixmueh/GATO` so the pinned smooth-friction commit is
fetchable on `integration/tiago-friction-0.1.2`. This is separate from historical
`felix-devel`; do not push or update that old branch as part of application work.

## Nominal grasp-center frame — Felix, 2026-10-21

Use `arm_right_grasp_joint` for new reaching/tracking: the reduced URDF contains
a massless fixed frame at tool-local `(0, 0, .139288457525)` m, zero RPY.
`scripts/prepare_models.py` owns its addition/check. This is the nominal midpoint
of the contact pads at zero gripper posture, derived from the simulator URDF and
mesh; it is not hardware calibration or a moving-gripper TCP. See models/README.md.
Select it through GATO's existing `ee_frame`; do not modify generic GRiD code.
Keep full simulator URDF bytes and the flange frame unchanged. Viewers must
derive the extra frame from the reduced model. Save `end_effector_frame` in
summaries/replay exports. Missing historical metadata means flange; do not
reinterpret old XYZ references or qualification results at the new point.
The grasp-center default passed 16 application CPU checks, frame/sign mapping,
native GPU/Pinocchio dynamics,
one coarse reach and one full 24 s tracking check (20.806 mm RMS; executed
samples and all 12,000 saved plans inside raw limits). Raw data remains in
`/workspace/tiago_mpc-grasp-frame/_expr_data/grasp-center/`. Existing 26.536 mm
and .270 s checks below describe the historical flange, not the new TCP.

## Canonical TIAGo model and joint parameters — Felix, 2026-10-21

The current robot description is the simulator URDF originally captured at
`/workspace/GATO/tiago_tools/sim_tiago_urdf.xml`. Preserve its exact source as
`models/tiago_pro/tiago_pro_sim_source.urdf`; the active full model is
`models/tiago_pro/tiago_pro.urdf`, with the explicit both-arm parameter patch.
`/workspace/GATO/TiagoProURDF/tiago_pro.urdf` is OUTDATED. Do not use that model
as the reference for extracting arms, inertias, limits, or new experiments.

The existing seven-joint arm reduction already comes from the correct simulator
URDF. Its fixed torso root and zero-position gripper inertia aggregation are
intentional. Apparent Euler-angle branches against the outdated model do not
justify extra frame changes. GRiD supports native signed axes; the pinned GATO
fixed-arm adapter still needs our generated joint-6 axis/limit conversion and
matching q/v/u sign mapping only at the GPU boundary. ROS and saved states stay
in native coordinates. Do not remove that workaround on GRiD-only evidence.

Felix supplied actual left/right arm values on 2026-10-21, superseding all
previous range midpoints and guessed wrist values. `scripts/prepare_models.py`
owns the exact table and reproducibly patches both arms in the active full URDF
and the right-arm reduction. See `models/README.md` for the complete table.
Use standard `dynamics` friction/damping and `limit` effort fields. Store armature
in a custom `<tiago_mpc armature="..."/>` joint child (kg m^2), explicitly metadata
only: neither current URDF loader enables it, and no runtime reads that tag.
Do not store kp/kd: Felix says those belong to a different controller.
Do not silently add armature to link inertia or enable it only in one model.

Existing experiment protocols and saved cohorts retain their historical
coefficients. Their explicit configuration can override canonical URDF losses;
new actual-value experiments require a separately identified protocol/output.
Do not relabel old figures or overwrite raw data with new coefficients.

URDF tags alone do not activate passive forces in Pinocchio ABA or the default
GRiD codegen path. Keep runtime loss enablement explicit, avoid double counting,
and retain the recorded hard-Coulomb startup failures. A metadata patch is not
a claim that matched friction MPC is qualified. Preserve historical models,
cohorts and outcomes rather than relabeling or silently regenerating them.

## Reaching and tracking defaults — Felix, 2026-10-21

Felix selected running end-effector XYZ cost **4** and terminal XYZ cost **8**.
`configs/tracking.json` owns tracking settings: qd_cost=.01, u_cost=1e-6,
all three soft-limit costs zero (ADMM-only limits), actual right-arm losses,
tanh predictor width .03 rad/s, hard-sign plant, N32/2 ms and .1 ms RK4 plant.
The separate 24 s 4/8 tracking check completed (26.536 mm full-cycle RMS;
sampled executed and planned raw limits passed); old 8/1600 results do not
qualify 4/8. Keep armature inactive
in both models. Preserve historical protocols and raw cohorts unchanged.
Reaching also uses running/terminal XYZ weights 4/8 in `configs/reaching.json`;
only its terminal weight changes from the preceding 4/800 profile. Other reaching
settings remain task-specific and unchanged. The historical 84/100 cohort used
4/800 and does not qualify this new terminal weight. One bounded 4/8 sanity reach
passed the existing 5 cm arrival gate at .270 s with sampled raw limits passing;
this is not broad reaching or settling qualification. Do not rerun or relabel
the frozen historical cohort.

## Completed experiment cleanup — Felix, 2026-10-21

Felix authorized removal of the ten completed app experiment/integration
worktrees and branches (ADMM-only, friction/damping, loss diagnosis, native-axis
spike, smooth friction, smooth reaching, zero control cost, paired reach render,
larger figure eight and tracking defaults), including their raw results, session
archive copies and associated viewer replays. Their decisions are integrated or
discarded. Do not describe those deleted records as still available or recreate
the archives. Main's current code/defaults, the separate ROS2 work and the upstream
axes PR remain. After creating the friction issue, Felix also authorized removal
of `/workspace/GATO-smooth-friction` and `feature/smooth-joint-friction`, including
its drafts and benchmark data. Smooth friction remains available through main's
pinned GATO commit `37a6768d575c4a09e2ec90e51a7c4029568e3fec`
(upstream 0.1.2 plus the friction patch, using grid-rbd 0.5.3; supersedes the
preceding 7918a83 pin).
The 0.1.2 update preserves all task configs, costs, ADMM settings and model
files. Fifteen CPU checks and one fresh default tracking/reaching pair passed;
new raw validation is retained in
`/workspace/tiago_mpc-gato012/_expr_data/gato012/`. Zero-weight soft-limit
expressions are still evaluated by GRiD; there is no public skip switch.
The native-axes PR is not yet adopted by main: its
joint-6 GPU boundary conversion remains required at the current pin.

## Experiment storage and presentation — Felix, 2026-10-21

ALL new experiment data belongs in this checkout/worktree's `_expr_data/`,
including raw results, logs, intermediate analysis, archives/backups, and final
reports/figures. Keep `_expr_data/` ignored by Git. The explicit exception is
portable replay handoffs: use **`impl/_expr_data/ingest/` on the host** and
**`/workspace/_expr_data/ingest/` in the container** (the same mounted directory).
This replaces the proposed shared `_expr_results` name. Use study/cohort
subfolders and descriptive, distinct `.npz` filenames; do not put full raw
archives or logs in the ingest directory. Preserve existing historical paths
and recordings instead of moving or rewriting them.

Do not write new data to `/workspace/experiment_data/` or GATO's
`thesis_results/` for this non-thesis work.

`/workspace/experiment_data_ma_frozen_archive/` is the frozen MASTER THESIS
archive (formerly `/workspace/experiment_data/`). Preserve historical raw data
and provenance; do not recreate the old directory or a compatibility alias.
Existing old path strings in raw records are historical, resolved using the
storage relocation manifest. This policy overrides older output instructions.

Unless Felix explicitly requests a particular plot, make the default handoff:

- Export recorded motion with `experiments/tools.py:save_replay` into the shared
  ingest directory. Use native joints, actual timestamps and the task reference;
  include saved predictions and recorded velocity/actuator torque/limits when
  available. These telemetry fields are optional. Do not invent motion or rerun retained
  experiments just to make a replay. The visualizer's export cookbook is
  `../tiago_mpc_visualizer/docs/replay-format.md` from the main checkout.
- Give the viewer URL (normally `http://192.168.178.82:8090`) and exact library
  path(s), such as `reaching/study/run_017`, so Felix can inspect them immediately.
- State the short finding, relevant recorded metrics/outcomes and material
  limitations. Preserve failures and original labels; selected replays do not
  replace aggregate cohort outcomes. Clearly identify simulation versus robot
  data, predicted versus executed motion, and the time basis.
- Link the raw source run and effective configuration/protocol; retain model
  and source identities. Explicitly label display-only subsampling/conversion
  and keep the complete raw measurements in their original owning checkout.

For timing/memory or other studies without recorded motion, use a compact
metrics table plus raw-output links. Produce the specific requested plot when
Felix asks for one; otherwise keep the replay/summary handoff concise.

Use agents liberally for useful bounded tasks; root coordinates interpretation,
shared edits and sequential GPU use. Preserve scientific protocols and failures.

## Worktree cleanup — Felix, 2026-10-21

Queue worktree cleanup after merging branches. At the end of the work session,
inspect loose changes/results and propose the exact paths and branches to
remove, stating where useful work is retained. Do not remove worktrees or delete
branches automatically; wait for approval. Do not interrupt ongoing tasks with
cleanup proposals.
