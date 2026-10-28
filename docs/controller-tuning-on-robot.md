# Manual tuning of PD feedback and external friction feedforward

27 October 2026. Use the simulation study to choose candidates, then transfer
conservatively with an operator present. This is a proposed manual procedure;
it does not add a robot tuning runner or authorize motion.

The current [robot path](robot.md) is covered by CPU fake-client tests and has
not been qualified against ROS, Gazebo or the physical robot.
The package laws `tm.PlanFeedback` and `tm.FrictionFeedforward` live in
[laws.py](../tiago_mpc/laws.py) and compose with either plant through the same
control-law interface; use `import tiago_mpc as tm`.
[PlanFeedback's main](../experiments/plan_feedback.py) accepts simulation only;
[friction_compensation.py](../experiments/friction_compensation.py) is also a
simulation experiment. Neither is a runnable physical tuning preset. A short
standalone script still needs to compose the selected law with `RobotConfig`,
`RealTime`, `setup`, `handle_stop_signals` and `run_experiment`, preserving the
existing confirmation, checks, leases and handback. Reuse these paths rather
than publishing torque directly. The holding/stroke references and component
logging below also need that script; they are not existing joint-sweep presets.

**The knobs and the comparison to preserve.** The implemented external law is

```text
u_sent = u_plan + kp * (q_plan - q) + kd * (v_plan - v)
         + c_ff * tanh(v_plan / w_ff) + b_ff * v_plan
```

The planned state is sampled at command time; feedback keeps its original
measurement time. [PlanFeedback](../tiago_mpc/laws.py) interpolates
the plan with its constant-acceleration transcription. It compares against
the latest held feedback without extrapolating that sample to command time;
sensor delay therefore remains part of the tuning problem.
`tm.FrictionFeedforward(friction, base_law=...)` adds losses at the interpolated
planned velocity, not the measured velocity.

| Quantity | Units | Current way to set it |
| --- | --- | --- |
| `kp`, `kd` | Nm/rad; Nm s/rad | `tm.PlanFeedback(kp=seven_values, kd=seven_values)`; scalars also work |
| Sliding `c_ff`, viscous `b_ff` | Nm; Nm s/rad | `Friction(coulomb_nm=..., viscous_nm_s_per_rad=...)` |
| `w_ff` | rad/s | `FrictionFeedforward(..., evaluation=Tanh(w_ff))`; smooths reversals |
| Reflected armature | kg m² | Explicit `MpcParams.armature`; the simulation also has `SimConfig.armature` |
| Command / control-update periods | s | `RealTime(command_period=.002, update_period=.02)` for 500/50 Hz requests |

For a PD-plus-feedforward candidate use
`tm.FrictionFeedforward(friction, tm.PlanFeedback(kp=..., kd=...))`. Coefficient
scales are applied to the coefficient tuples before construction; there is no
separate scale argument. Keep native joint signs, including joint 6.

For this external-feedforward comparison keep predictor friction off. The
alternative friction-in-MPC cases already plan that loss; adding the same
nominal loss again would duplicate compensation. Do not retune MPC weights,
armature, filter, timing and execution law together. The current simulation
reference is 30 ms knots, N32, position/terminal weights 4/80, mu=1,
qd_cost=.01, u_cost=1e-4, rho_u=5e-4, control updates 50 Hz and commands 500 Hz.
Its 500 Hz feedback and declared 5.5 ms solve delay are simulation assumptions,
not measurements of robot sensor or end-to-end latency.
The locked simulation presets in [controller_tuning.py](../experiments/controller_tuning.py)
use 20 rad/s, zeta=1.4 and external width .01 rad/s. These values were selected
on two simulation tasks; the lower initial physical gains below are deliberate.

1. **Prepare without effort control.** Read [robot.md](robot.md) and
   [safety.md](safety.md). In the existing container, after sourcing the ROS/PAL
   overlays, the operator can run this existing check-only command from this
   checkout with its own environment:

   ```bash
   .venv-container/bin/python -m experiments.reach --plant robot
   ```

   Without `--execute`, this reads state and controller inventory and performs
   startup/safety checks; it sends no torque, switches no controller and builds
   no MPC. It does not save a timing recording. Use `--sea-absolute` only when
   that is the intended feedback path and its interface is available.
   Preserve the 4 cm collision margin and blacklist. Check the actual full-body
   posture, nearby world obstacles, the complete trial path and return path;
   the worker does not monitor world obstacles. Keep base/grippers stationary.
   Start every trial from the reviewed comfortable pose under PAL's position
   controller; the code provides no automatic reset.

2. **Audit timing and noise passively before picking gains.** Start with the
   retained recordings. If new data is needed, use a reviewed subscription-only
   recorder while PAL retains position control; the check-only command above is
   not such a recorder. Do not activate effort control just to measure noise.
   Record distinct arm-source stamps, arm callback monotonic receipts, sequence
   numbers, full-body receipts, native q/qd and the selected feedback source.
   Report update intervals and their tails, repeated-sample share, gaps,
   stationary q/qd noise and quantization. A 500 Hz command loop does not create
   500 Hz sensor information.

   [SEA velocity](../tiago_mpc/ros2.py) is differentiated output position with a
   10 ms filter, not an independent instantaneous velocity measurement. Motor
   feedback and SEA output feedback measure different sides of the transmission;
   retain both when available and choose one for the controller. Do not switch
   sources during a gain comparison. Do not infer low noise from a wrist velocity
   stream that remains zero despite observed position changes. Later effort runs
   must additionally retain plan-state age, plan activation time, `late_tick`,
   `late_plan`, lease faults and publication times on the same monotonic clock.

3. **Establish gentle holding, with external feedforward off.** Use an already
   reviewed gravity/plan torque baseline, not zero-torque open-loop holding.
   Choose one joint's small, smooth position perturbation, keep the others
   holding, and inspect all seven responses. A concrete first proposal for the
   operator's path review is ±0.005 rad, requested speed no more than .01 rad/s,
   with 2 s holds; these numbers are not a certified safe envelope for every
   posture or joint. The trial script must declare its displacement, speed,
   duration and additional-effort bounds and abort beyond them. Current generic
   robot checks enforce the raw URDF limits, not this smaller trial envelope.

   Initialize gains from the diagonal of the effective inertia at the intended
   posture, `m_i = diag(M_link(q) + diag(a))_i`:

   ```text
   kp_i = m_i * omega_i²
   kd_i = 2 * zeta_i * m_i * omega_i
   ```

   [inertia_gains](../tiago_mpc/util.py) accepts an explicit
   `armature` tuple: `tm.inertia_gains(omega, damping_ratio=zeta, armature=values)`.
   Its zero-armature default preserves the old link-only gains. Pinocchio's
   `model.armature` already enters CRBA; do not add it a second time. The
   [supplied armature](../models/tiago_pro/README.md) is still an assumption,
   not an identified transmission model.

   As a conservative **starting heuristic**, consider omega=3 rad/s (about
   0.48 Hz), zeta=1, then 5 rad/s only if timing, noise, torque and observed
   response support it. These are natural-frequency parameters of a local
   single-joint approximation, not certified bandwidths of the coupled SEA
   robot. The simulation candidates 10/20/30 rad/s are not physical starting
   gains. Diagonal PD does not cancel coupling or transmission elasticity.
   The scalar inertia/gain relation and the hazards of excessive gains are
   described in [Modern Robotics §11.4](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-4-motion-control-with-torque-or-force-inputs-part-1-of-3/).

   Increase the natural-frequency parameter in small steps, changing one joint
   at a time and deriving both gains together. If it rings, first reduce the
   frequency and inspect timing; adjust damping around the same frequency only
   when velocity quality supports it. Increasing kd can amplify velocity noise.
   Check `kd_i * stationary_velocity_noise_i` against the available torque
   margin. Record holding bias separately from settling and oscillation; do not
   remove gravity/model bias by simply increasing kp. The distinction between
   torque PD and model compensation is explained in
   [MIT's manipulator-control notes](https://manipulation.mit.edu/force.html).

4. **Identify sliding loss with paired steady-speed strokes.** Keep the modest
   holding gains fixed and external feedforward off. For one joint at a time,
   use smooth acceleration/deceleration and traverse the same short, reviewed
   position interval in both directions. Begin at ±.01 rad/s; consider ±.03
   and ±.05 rad/s only if actual motion, clearance and effort margin permit.
   Use multiple speed magnitudes to distinguish Coulomb intercept from viscous
   slope. Allow enough travel for a steady plateau; if the available safe
   interval is too short, retain an inconclusive result rather than calling an
   acceleration segment steady. These speeds are proposed test inputs, not
   evidence of hardware qualification.

   Fit plateau interiors after acceleration and filter settling, using measured
   speed rather than its requested value. Match comparable positions and
   thermal/payload conditions, repeat both directions, and inspect directional
   asymmetry instead of forcing it into one coefficient. Do not pool reversals,
   stationary samples or handback commands into the sliding fit. This approach
   follows the paired constant-velocity identification experiment in
   [Johansson, 2017, §6.1.1](https://publications.lib.chalmers.se/records/fulltext/252903/252903.pdf#page=70);
   its robot's numerical settings are not TIAGo settings.

   With a validated joint-torque measurement, define
   `r = torque - rnea(q, v, acceleration)` and fit
   `r_i ≈ c_i * sign(v_i) + b_i * v_i` on those plateaux. If acceleration is
   neglected, first demonstrate that its contribution is small. Keep the
   coupled rigid-body gravity/Coriolis terms; differencing opposite directions
   cancels gravity only when posture and other conditions match. Use the total
   effort in the residual, not only the PD contribution. Without calibrated
   delivered torque, call this an **effective command-residual fit**: sent
   torque also includes drive tracking, transmission elasticity and model
   error. Neither `/commands` nor this repository's `ticks.command` proves
   torque delivery. Even an `effort` telemetry field needs its meaning, scale
   and timestamp checked before using it as an actuator-torque measurement.

   The existing physical fits provide proximal priors, but the fresh analysis
   found joint 1 almost entirely excited in one direction, so it cannot validate
   a symmetric joint-1 fit. Their retained physical
   figure eights did not adequately excite joints 5–7; zero motion does not identify
   zero friction. Do not fit their Coulomb/viscous terms, armature or SEA
   dynamics from these records. A conservative transfer candidate can leave
   wrist external feedforward off until separate small wrist strokes are
   reviewed, explicitly distinguishing it from the nominal-wrist simulation.
   Steady-speed data also cannot identify rotor inertia reliably.

5. **Add external feedforward gradually, then recheck PD.** Freeze the fitted
   coefficient shape and compare a partial feedforward candidate with the
   FF-off baseline; for example begin with 25% of supported c/b estimates,
   retaining the same PD, then increase only if both directions and reversals
   improve without creep, noise or torque peaks. The fraction is a proposed
   cautious transfer step, not a fitted result. An initial external `Tanh(.03)`
   is a smoother transfer proposal than the simulation-selected `.01`; record
   the choice separately from the MPC predictor's width.
   Change c, b and width independently. Check b against the residual-versus-speed
   slope and c against its signed intercept across multiple plateaux; use
   near-zero reversal behavior to assess width. Avoid choosing all three from
   one tracking RMS.

   `tanh(0)=0`: this feedforward cannot supply an independent breakaway kick.
   Record rest-to-motion thresholds with dwell time separately from sliding
   loss, and do not substitute the peak breakaway torque for c or silently set
   static friction to 1.2 times Coulomb friction. Static and sliding regimes
   differ ([Drake's friction model](https://drake.mit.edu/doxygen_cxx/group__mbp__continuous.html)).
   The current simulated StickSlip model uses one threshold, and no separate
   breakaway law is implemented; see the dated
   [breakaway handoff](friction-breakaway-handoff.md). A torque-ramp/breakaway
   controller would be separate reviewed work, not part of this procedure.

6. **Keep a candidate only if it transfers and hands back cleanly.** Repeat the
   small holds, paired strokes and then a different reviewed slow path or reach
   with frozen gains. Keep fitting trials and validation trials distinct;
   adjacent samples from one figure eight are not independent validation.
   Compare joint errors, reversal peaks, hold drift, settling, torque RMS/peaks
   and command variation, state age, lateness and faults—not just end-effector
   RMS. Retain every failed trial and the last acceptable setting. Keep the
   robot's initial thermal condition and payload comparable and report changes.

   Stop for growing oscillation, unexpected drift/contact, poor velocity data,
   unexpected controller ownership, loss of effort margin or a trial-envelope
   breach. Existing stale-state, physical-limit, lease and safety-worker faults
   already terminate the run; do not relax them to finish a trial. Ctrl-C/SIGTERM
   requests the normal stop path. `RosPlant.stop()` attempts five zero-torque
   publications 10 ms apart and position-controller handback. Inspect the
   `handback` event and `metadata.plant.restored/cleanup_errors` before resetting
   or beginning another trial. Uncertain handback needs manual controller
   reconciliation. SIGKILL, a host crash and an emergency stop are different
   paths; the effort receiver's independent command-expiry behavior remains
   unestablished. Keep the operator's hardware emergency-stop procedure ready.

Each future trial should print its exact seven kp/kd/c/b values, widths,
armature/model identities, feedback/filter choice, actual timing assumptions,
motion envelope and duration before the existing typed `EXECUTE` gate. Save
raw data, source/configuration and outcomes in a fresh directory under this
checkout's `_expr_data/`. A tuning recorder should additionally retain planned,
PD and feedforward torque components; current `ticks.command` records their
total. Use a single native coordinate convention and preserve original raw
recordings and provenance. This note has performed no ROS contact or motion.
