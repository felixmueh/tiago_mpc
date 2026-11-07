# Latency compensation

Both package loops accept optional `LatencyCompensation`. It predicts the arm state
at the intended start of the next plan while the previous plan continues running.
MPC solves from that prediction, with the reference window starting at the same
future time. `run_real_time` uses the previous observed request-to-result delay by
default. Its initial estimate is the update period, with the command period as a
fallback for back-to-back updates. Standard simulations pass
`default_compensation(settings)`: declared virtual latency selects that fixed
delay; measured latency selects the previous observed delay. It does not change the
command rate, control-update schedule or MPC knot interval.

```python
from dataclasses import replace
import tiago_mpc as tm

compensation = tm.LatencyCompensation(
    latency_s="previous",
    initial_latency_s=.01,
    friction=tm.urdf_friction(),
    friction_law=tm.StickSlip(),
    armature=tm.ARMATURE_KG_M2,
    substep=.001,
)
real_settings = tm.RealTimeSettings(
    command_period=.002,       # 500 Hz torque commands, including PD/FF
    update_period=.02,         # 50 Hz control updates; 0 requests back to back
)

# The same compensation on a simulation clock, charging actual solve wall time:
simulation_settings = tm.VirtualTimeSettings(
    command_period=.002,
    update_period=.02,
    solve_latency_s="measured",
)
```

Pass the settings, selected loop and `compensation=compensation` to
`experiments.common.run_experiment`. It builds a concurrent `MpcProcess` for
`run_real_time` and an in-process `Mpc` for simulation. Interactive callers can
pass `MpcProcess`, `settings=simulation_settings` and `compensation=compensation`
to `run_virtual_time` to overlap solving with independent plant integration.

## Delay estimate and plan activation

`latency_s=.01` uses a fixed 10 ms forecast. `latency_s="previous"` uses a previous
completed request's observed delay; the first request uses `initial_latency_s`.
It never uses the current solve's duration before that solve completes. This
estimate is separate from `VirtualTimeSettings.solve_latency_s`, which determines when
the simulated result actually becomes available.

`default_compensation(settings)` centralizes the loop-dependent choice and seeds
previous-delay mode from the configured periods. A fixed forecast remains an
explicit `LatencyCompensation(latency_s=...)` choice; changing the update rate
does not silently change that explicit model.

On wall time, computing the forecast also costs time. The target is chosen before
that CPU rollout, and the request is sent afterwards; its remaining lookahead can
therefore be shorter than the configured estimate. Virtual time keeps that CPU
work out of the simulated timeline, as it
does for the rest of the simulation's computation.

An early result waits until its predicted origin. A late result retains that
origin and executes the part of the plan due at the actual command tick. For
example, a plan predicted for 30 ms and received at 34 ms starts 4 ms into its
trajectory. It is never relabelled as a plan starting at 34 ms. The old plan
continues during solving and early-result waiting, within its existing lease.
There is one pending plan; back-to-back updates do not queue future plans while
one is waiting to activate.

A collected result can activate when
`tick_time >= max(request_time + solve_latency, predicted_origin)`. Zero declared
latency can therefore activate at the request tick through the same time
inequality as any other delay. Each command tick admits at most one new request,
including ideal back-to-back updates.

## Physical propagation and feedback

Propagation has its own friction coefficients, Coulomb law, armature and maximum
Euler substep. It reuses the coupled dynamics step from the simulator, assigning
armature to the Pinocchio model once. The defaults are loss-free with the fitted
armature (`ARMATURE_KG_M2`), as are the predictor's and the simulated plant's; enable
friction explicitly, as above. The optimizer's model
and the real/simulated plant remain separately configured.

Each loop calls the predictor once before submitting a compensated MPC request.
The predictor is a frozen callable with a prepared physical model; it keeps no
command history or delay estimate. The loop chooses the target time from its
fixed forecast or previous observed solve delay.

Propagation starts from the latest measured state under the currently held
command. At each future command tick it evaluates the active plan's command law
on the predicted state, including the selected effort handling. Before effort
control starts, the arm is treated as held by its onboard position controller.
In virtual time the measurement is already at the current command boundary.
For an older measurement, propagation assumes the current command was held since
that measurement; it does not reconstruct earlier publications or plan changes.
Future feedback is also a model assumption: a forecast cannot know later sensor
noise, disturbances or how an unexpected delay will change the actual feedback.

Actual `PlanFeedback` continues to use the latest measured state at every command
tick. Prediction does not replace the measured feedback used by PD or the robot's
checks. Friction feedforward can be combined with physical propagation friction:
one adds actuator torque and the other models opposing resistance. The existing
guard still rejects external FF combined with optimizer friction, or nested FF.
Simulation selects its effort policy separately from the command law, as described below.

Both loops warn when `PlanFeedback` runs without compensation. Without it, the
first plan starts at its state's time while the arm is still held, so when it
takes effect after the first solve's latency L, its velocity is already about
a₀·L ahead of the arm, where a₀ is the planned initial acceleration. The D term
adds about kd·a₀·L to the first knot's torque, which a far reach plans at the
effort bound. In simulated far reaches (2026-10-30) this exceeded the raw effort
limit, while compensation whose `initial_latency_s` covered the first solve
removed the excess. A forecast shorter than the actual first solve leaves the
part due to the shortfall.

## Timestamps, leases and records

The measured `State` remains unchanged. `Mpc.request` and `MpcProcess.request`
take one initial condition, `State | Prediction`, with the same three positional
arguments: `request(initial, reference, updates=None)`. A measured input uses
`State.t`; a predicted input uses `Prediction(t, q, qd, measured_t=source_time)`.
Its `t` is the future origin and `measured_t` is the original observation time.
There is no separate optional prediction argument.
`Plan.t0` is the time represented by its initial coordinates; `Plan.measured_t`
is a required float retaining the original measurement time. Without prediction,
callers set it equal to `t0`. Real-time execution expires a plan at the earlier of:

- `measured_t + 0.1` seconds, the existing measurement-age allowance;
- `t0 + len(torques) * dt`, the end of its trajectory.

Virtual-time execution uses `ExecutionSafety.max_plan_age_s` for the first limit
(default 0.1 s); None disables that age limit in execution and prediction.
The plan's finite horizon remains. Forecasting still preserves the original
measurement time and does not renew an observation's age.

Prediction therefore cannot renew an old measurement's lease. Existing effort,
state-freshness and robot safety checks remain in force. A non-compensated plan
uses its original state time for both timestamps.

An unsupported forecast ends the loop with `prediction_unavailable`: for example,
the observation is later than the current time, the target is beyond a valid
lease, or a forecast command would fail validity or effort checks. Simulation
uses the same law and separate effort policy for actual and predicted commands:
`effort_handling=EffortClip()` means propagation uses the clipped torque, and
`RejectOnEffort()` makes an excessive future command return prediction_unavailable
without an actual actuator event. `None` explicitly disables effort handling.
Forecast-compatible laws and effort policies must be pure; forecasting evaluates
them directly without copying their state. Safety implements
`validate(plan, t, state) -> Fault | None`; forecasting uses that same object.
`decide_command` performs the same
lease, law and effort checks in both execution and forecasting, returning
`Command` or `Fault`; forecasting consumes a refusal without an actuator event. On
wall time this preparation happens after the current valid command and callbacks;
the loop stops and hands back immediately if preparation fails.

`Plan.stats` contains solver statistics. `SolveCompleted` carries the plan and
its request time; `RunRecord.solve_completed` uses that time for `plans.update_t`.
Every completed plan is saved, including an early result superseded before use
or a returned plan whose lease expires before activation. Tick records identify
the active plan by its row in this completed-plan table.
Execution does not copy plans to add arrival/activation timestamps to their
statistics. Run records retain the compensation configuration and measured and
predicted origin times. Robot `State.t` is still monotonic receipt time; delay before receipt and
unmeasured actuator/transport delay are not identified by solve timing. A raw
measurement and a prediction at different times must not be interpreted as a
same-time prediction-error measurement. CPU and simulation checks do not qualify
this controller for physical execution.
