"""Latency compensation in simulation: solve from the state predicted for the moment the
plan takes effect, and date the plan from then (Findeisen and Allgöwer; Mastalli et al.).

With compensation disabled and VirtualTimeSettings.solve_latency_s=L,
run_virtual_time solves from the state measured at tick t, dates the plan t and
lets it take effect at t + L, so SampleKnot enters it at knot floor(L / dt).
Once L reaches one knot interval, knot 0, where each new solve's correction sits, never
runs, and the reach diverges (docs/robot.md, Latency). run_compensated predicts the state
at t + L under the commands the loop sends until then, solves from it and dates the plan
t + L, so every plan starts at its knot 0. This experiment has its own loop to compare
implicit and GPU propagation models with the package's latency predictor. Its timing
is Compensated, which any experiment takes
(run_experiment builds the in-process Mpc for it, without the wall-time budget):

    record = run_experiment(SimConfig(substep=.003), TIAGO, Reach(), SampleKnot(),
                            Compensated(solve_latency_s=.02), loop=run_compensated)
    evaluate(record)  # reach's

The other remedy needs only a control law (plan_feedback.py); latency_study.py compares
both, alone and together, with stock SampleKnot. The main runs one compensated reach.

Run from the checkout's root: python -m experiments.latency_compensation --solve-latency .02
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import sys
from typing import Literal

import numpy as np
import pinocchio as pin

from experiments.common import arguments, handle_stop_signals, run_experiment, setup
from experiments.reach import Reach, evaluate
import tiago_mpc as tm

_TOLERANCE = 1e-9  # as loops.py: absorbs the rounding of summed periods


def predict(plan, law, state, at, command_period, step, substeps=1, *,
            safety=tm.ExecutionSafety(), effort_handling=tm.EffortClip()):
    """The state at tick time at, from state, under the commands the loop sends at the
    ticks in between: law(plan, tick, predicted state), each held for one command period,
    so a law that depends on the state is predicted too. Without a plan the arm is held.
    step(x, u, h) is the predictor's model (gpu_step or implicit_step); each tick may be
    split into substeps equal steps under the same command."""
    t, x = state.t, np.r_[state.q, state.qd]
    while plan is not None and t < at - _TOLERANCE:
        decision = tm.decide_command(plan, t, tm.State(t, x[:7], x[7:]), law,
                                     safety=safety, effort_handling=effort_handling)
        if isinstance(decision, tm.Fault):
            return None
        u = decision.applied
        for _ in range(substeps):
            x = step(x, u, command_period / substeps)
        t += command_period
    return tm.Prediction(at, x[:7], x[7:], measured_t=state.t)


def gpu_step(mpc):
    """GATO's own model: BSQP.sim_forward (GATO python/bindings.cu, gato/bsqp/kernels/sim.cuh)
    runs one step of the solver's device integrator, the constant-acceleration
    (trapezoidal) step its transcription uses (check_model's step), in float32, with the
    predictor's friction and knot 0's external wrench. It reads only the model and writes
    only its own output buffer: the solver's warm start, duals and rho are untouched.

    The step is explicit: with the URDF's friction in the predictor, one 2 ms step makes
    joint 7 diverge without armature (0.0008 kg m^2 against b plus the tanh slope c/w: a time
    constant of tens of microseconds), unless split into substeps below twice that."""
    return lambda x, u, h: np.asarray(mpc.controller.solver.sim_forward(x[None], u[None], h), dtype=float)[0]


# TODO: implicit_step ignores the external_force solver update (mpc.UPDATES), which
# gpu_step includes (knot 0's wrench); a task that sets it needs gpu_step or the wrench here.
def implicit_step(friction, friction_law=tm.Tanh(.03)):
    """The same arm model (Pinocchio on the arm URDF, which check_model ties to GATO's),
    one semi-implicit Euler step that is implicit in the passive torque p(qd) only,
    linearised:
        qd+ = qd + (M + h D)^-1 h (u - bias(q, qd) - p(qd)),   q+ = q + h qd+
    with D = diag(dp/dqd) = diag(b + (c/w) sech^2(qd/w)) (dynamics.passive_torque). The
    passive torque is the stiff term (joint 7's time constant is tens of microseconds), so this
    keeps a 2 ms step stable; inertia and bias stay explicit. Linearised at qd, it misses the
    tanh's steep slope at zero: near rest joint 7's speed chatters by about h c / (m + h b)
    (0.15 rad/s at 2 ms), bounded, with a negligible position error. friction: the
    predictor's (MpcParams.friction), None loss-free. Native coordinates."""
    model = pin.buildModelFromUrdf(str(tm.arm_urdf()))
    data = model.createData()

    def step(x, u, h):
        q, qd = x[:7], x[7:]
        slope = (0. if friction is None else np.asarray(friction.viscous_nm_s_per_rad)
                 + np.asarray(friction.coulomb_nm) / friction_law.width_rad_s / np.cosh(qd / friction_law.width_rad_s)**2)
        force = u - pin.rnea(model, data, q, qd, np.zeros(7)) - tm.passive_torque(qd, friction, friction_law)
        qd_next = qd + np.linalg.solve(pin.crba(model, data, q) + h * np.diag(slope * np.ones(7)), h * force)
        return np.r_[q + h * qd_next, qd_next]
    return step


def run_compensated(
    plant,
    mpc,
    task,
    law=tm.SampleKnot(),
    tick_callbacks: Sequence[tm.TickCallback] = (),
    *,
    settings,
    duration,
    compensation=None,
    safety=tm.ExecutionSafety(),
    effort_handling=tm.EffortClip(),
    solve_completed_callbacks: Sequence[tm.SolveCompletedCallback] = (),
    diagnostic_callbacks: Sequence[tm.DiagnosticCallback] = (),
):
    """Run a declared-latency simulation with this experiment's propagation model.

    Each request is made from the state predicted (predict) at the tick its plan
    takes effect, and dated then.
    compensate=False requests from the measured state, dated now: run_virtual_time's timing.
    prediction_substep_s: the longest prediction step (None: one step per command tick);
    predictor: "implicit" (implicit_step, the default) or "gpu" (gpu_step).
    Virtual time only (it starts the plant with VirtualClock(), which RosPlant refuses);
    compensating needs the in-process Mpc (its solver). solve_latency_s is
    declared seconds only (finite, >= 0), else a ValueError before the plant starts.
    Returns an Outcome.

    As in run_virtual_time: the warm-up before the first tick, one outstanding solve, the next request
    at the first control-update slot at or after its plan took effect, each command checked
    after evaluating the law, a fault or STOP ending the run, late_plan
    diagnostics, the last tick without a command, plant.stop() first on exit. Left out: the
    garbage-collection pause and the overrun diagnostics. Completed solves retain detached
    request inputs; prediction diagnostics also retain the requested state for evaluation.
    """
    if compensation is not None:
        raise ValueError("run_compensated uses its settings propagation model; compensation must be None")
    command_period, update_period, solve_latency_s = (settings.command_period, settings.update_period,
                                                       settings.solve_latency_s)
    compensate = settings.compensate
    prediction_substep_s, predictor = settings.prediction_substep_s, settings.predictor
    if not (isinstance(solve_latency_s, (int, float)) and math.isfinite(solve_latency_s) and solve_latency_s >= 0):
        raise ValueError(f"solve_latency_s must be declared seconds (finite, >= 0), not {solve_latency_s!r}")
    ahead = math.ceil(solve_latency_s / command_period - _TOLERANCE)  # a plan takes effect at the first tick at or after t + L
    substeps = 1 if prediction_substep_s is None else math.ceil(command_period / prediction_substep_s - _TOLERANCE)
    step = None
    if compensate:
        step = gpu_step(mpc) if predictor == "gpu" else implicit_step(mpc.params.friction, mpc.params.friction_law)
    clock, update_due = tm.VirtualClock(), tm.Schedule(update_period)
    active_plan, pending_plan, start, plans_activated = None, None, None, 0
    plant.start(clock)
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        deadline = None
        while True:
            s, t = plant.state(), clock.now()
            start = s.t if start is None else start
            done, stopped = task.done(s, t - start), tm.STOP.is_set()
            end = done or stopped or t - start >= duration - _TOLERANCE
            plan_activated = pending_plan is not None and t >= pending_plan[0] - _TOLERANCE
            if plan_activated:  # (time it takes effect, plan)
                active_plan, pending_plan = pending_plan[1], None
            fault, request_update = None, False
            if pending_plan is None and not end and update_due(t):
                request_update, at = True, t
                for _ in range(ahead):  # summed as the loop sums its deadlines: the same tick times
                    at += command_period
                x = (predict(active_plan, law, s, at, command_period, step, substeps,
                             safety=safety, effort_handling=effort_handling) if compensate else s)
                if x is None:
                    return tm.Outcome("fault", "prediction_unavailable", t - start)
                reference, updates = task.at(x.t - start, mpc.N, mpc.dt)
                if isinstance(x, tm.Prediction):
                    initial = tm.Prediction(x.t, x.q, x.qd, measured_t=x.measured_t)
                else:
                    initial = tm.State(x.t, x.q.copy(), x.qd.copy(), deepcopy(x.joint_positions),
                                       deepcopy(x.joint_velocities))
                    initial.q.flags.writeable = initial.qd.flags.writeable = False
                reference = np.array(reference, copy=True)
                reference.flags.writeable = False
                last_mpc_request_snapshot = initial, reference, deepcopy(updates)
                mpc.request(*last_mpc_request_snapshot)
                solved = mpc.result(block=True)  # blocks: the solve takes no virtual time
                assert solved is not None
                pending_plan = (at, solved)
                if ahead == 0:  # zero latency: this tick
                    active_plan, pending_plan = pending_plan[1], None
                    plan_activated = True
                if solve_completed_callbacks:
                    completed = tm.SolveCompleted(t, *last_mpc_request_snapshot, solved)
                    for callback in solve_completed_callbacks:
                        callback(completed)
                if compensate:
                    diagnostic = tm.Diagnostic(t, "prediction_request", "latency example requested a predicted state",
                                               dict(t0=initial.t - start,
                                                    requested_state=np.r_[initial.q, initial.qd].tolist()))
                    for callback in diagnostic_callbacks:
                        callback(diagnostic)
            if plan_activated:
                plans_activated += 1
            command, requested_command, clipped = None, None, False
            if active_plan is not None and not end:
                if (plan_activated and t >= active_plan.t0
                        and active_plan.index(t - active_plan.t0) >= 1):  # uncompensated only
                    diagnostic = tm.Diagnostic(
                        t, "late_plan", f"a plan took effect {1e3 * (t - active_plan.t0):.1f} ms after "
                        f"its state, {1e3 * active_plan.dt:g} ms knots: its knot 0 never "
                        "runs (docs/robot.md, Latency)",
                        dict(plan=plans_activated - 1, delay_s=t - active_plan.t0))
                    for callback in diagnostic_callbacks:
                        callback(diagnostic)
                decision = tm.decide_command(active_plan, t, s, law, safety=safety,
                                             effort_handling=effort_handling)
                requested_command = decision.requested
                if isinstance(decision, tm.Fault):
                    fault = decision.reason
                else:
                    command = decision.applied
                    clipped = decision.clipped
                    plant.apply(command)
            for diagnostic in plant.take_diagnostics():
                for callback in diagnostic_callbacks:
                    callback(diagnostic)
            tick = tm.Tick(t, s, command, active_plan, plan_activated, request_update, requested_command, clipped)
            for callback in tick_callbacks:
                callback(tick)
            if fault is not None:
                return tm.Outcome("fault", fault, t - start)
            if end:
                return tm.Outcome("done" if done else "interrupted" if stopped else "timeout",
                                  "stop_requested" if stopped and not done else None, t - start)
            deadline = t if deadline is None else deadline
            deadline += command_period
            plant.advance(deadline)
    finally:
        plant.stop()  # zero torque first


@dataclass(frozen=True)
class Compensated:
    """run_compensated's timing, its settings with the defaults of a 20 ms latency."""
    solve_latency_s: float = .02
    command_period: float = tm.hz_to_s(500)
    update_period: float = tm.hz_to_s(100)
    compensate: bool = True
    prediction_substep_s: float | None = None
    predictor: Literal["implicit", "gpu"] = "implicit"


def first_tick_of_each_plan(record):
    """The index of the tick at which each plan of record took effect."""
    plans, first = np.unique(record.ticks["plan"], return_index=True)
    return first[plans >= 0]


def requested_states(record):
    """Exact experiment inputs, from the record's prediction_request diagnostics."""
    requests = {e.t: e.data for e in record.diagnostics if e.kind == "prediction_request"}
    values = []
    for update_t, t0 in zip(record.plans["update_t"], record.plans["t0"]):
        diagnostic = requests[update_t]
        if not math.isclose(diagnostic["t0"], t0, abs_tol=_TOLERANCE):
            raise ValueError("prediction diagnostic does not match its plan origin")
        values.append(diagnostic["requested_state"])
    return np.asarray(values, dtype=float).reshape(-1, 14)


def main(argv=None):
    """One compensated reach on SimPlant, --solve-latency seconds (declared; default the timing's)."""
    handle_stop_signals()
    args = arguments(__doc__, argv, plants=("sim",))
    task = Reach() if args.target is None else Reach(tuple(args.target))
    plant, settings, loop = setup(args, tm.SimConfig(substep=.003), Compensated(), loop=run_compensated)  # sim: never a status
    record = run_experiment(plant, tm.TIAGO, task, tm.SampleKnot(), settings, loop=loop,
                            duration=args.duration, out=args.output)
    evaluate(record)
    effect = first_tick_of_each_plan(record)
    actual = np.c_[record.ticks["q"][effect], record.ticks["qd"][effect]]
    error = (np.abs(actual - requested_states(record)[record.ticks["plan"][effect]])
             if len(effect) else np.zeros((0, 14)))
    record.metadata["results"]["prediction_error_max"] = dict(q_rad=float(error[:, :7].max(initial=0.)),
                                                              qd_rad_s=float(error[:, 7:].max(initial=0.)))
    record.save()
    print(json.dumps(record.metadata, indent=2, default=str))
    print(f"Saved {record.out}")
    return 0 if record.metadata["status"] == "arrived" and record.metadata["results"]["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
