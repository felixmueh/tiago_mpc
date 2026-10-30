"""Latency compensation in simulation: solve from the state predicted for the moment the
plan takes effect, and date the plan from then (Findeisen and Allgöwer; Mastalli et al.).

run_virtual_time(..., solve_latency_s=L) solves from the state measured at tick t, dates the
plan t and lets it take effect at t + L, so SampleKnot enters it at knot floor(L / dt).
Once L reaches one knot interval, knot 0, where each new solve's correction sits, never
runs, and the reach diverges (docs/robot.md, Latency). run_compensated predicts the state
at t + L under the commands the loop sends until then, solves from it and dates the plan
t + L, so every plan starts at its knot 0. It is its own loop, not run_virtual_time: run_virtual_time
requests from the measured state, and its late-plan check, plan.index(t - plan.t0), raises
for a plan dated in the future. Its timing is Compensated, which any experiment takes
(run_experiment builds the in-process Mpc for it, without the wall-time budget):

    record = run_experiment(SimConfig(substep=.003), TIAGO, Reach(), SampleKnot(),
                            Compensated(solve_latency_s=.02))
    evaluate(record)  # reach's

The other remedy needs only a control law (plan_feedback.py); latency_study.py compares
both, alone and together, with stock SampleKnot. The main runs one compensated reach.

Run from the checkout's root: python -m experiments.latency_compensation --solve-latency .02
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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


def predict(plan, law, state, at, command_period, step, substeps=1):
    """The state at tick time at, from state, under the commands the loop sends at the
    ticks in between: law(plan, tick, predicted state), each held for one command period,
    so a law that depends on the state is predicted too. Without a plan the arm is held.
    step(x, u, h) is the predictor's model (gpu_step or implicit_step); each tick may be
    split into substeps equal steps under the same command."""
    t, x = state.t, np.r_[state.q, state.qd]
    while plan is not None and t < at - _TOLERANCE:
        u = law(plan, t, tm.State(t, x[:7], x[7:]))
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
    tick_callbacks=(),
    *,
    command_period,
    update_period,
    duration,
    solve_latency_s,
    compensate=True,
    prediction_substep_s=None,
    predictor="implicit",
):
    """run_virtual_time(..., solve_latency_s=solve_latency_s), but each request is made from the
    state predicted (predict) at the tick its plan takes effect, and dated then.
    compensate=False requests from the measured state, dated now: run_virtual_time's timing.
    prediction_substep_s: the longest prediction step (None: one step per command tick);
    predictor: "implicit" (implicit_step, the default) or "gpu" (gpu_step).
    Virtual time only (it starts the plant with VirtualClock(), which RosPlant refuses);
    compensating needs the in-process Mpc (its solver). solve_latency_s is
    declared seconds only (finite, >= 0), else a ValueError before the plant starts.
    Returns an Outcome.

    As in run_virtual_time: the warm-up before the first tick, one outstanding solve, the next request
    at the first control-update slot at or after its plan took effect, each plan checked
    on arrival and each command before it is sent, a fault or STOP ending the run, late_plan
    events, the last tick without a command, plant.stop() first on exit. Left out: the
    garbage-collection pause and the overrun events. Prediction inputs are retained in
    experiment events, separate from solver statistics.
    """
    if not (isinstance(solve_latency_s, (int, float)) and math.isfinite(solve_latency_s) and solve_latency_s >= 0):
        raise ValueError(f"solve_latency_s must be declared seconds (finite, >= 0), not {solve_latency_s!r}")
    ahead = math.ceil(solve_latency_s / command_period - _TOLERANCE)  # a plan takes effect at the first tick at or after t + L
    substeps = 1 if prediction_substep_s is None else math.ceil(command_period / prediction_substep_s - _TOLERANCE)
    step = None
    if compensate:
        step = gpu_step(mpc) if predictor == "gpu" else implicit_step(mpc.params.friction, mpc.params.friction_law)
    clock, due, plan, pending, start, plans = tm.VirtualClock(), tm.Schedule(update_period), None, None, None, 0
    pending_request_t = 0.
    plant.start(clock)
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        deadline = None
        while True:
            s, t = plant.state(), clock.now()
            events = []
            start = s.t if start is None else start
            done, stopped = task.done(s, t - start), tm.STOP.is_set()
            end = done or stopped or t - start >= duration - _TOLERANCE
            previous_plan = plan
            new = pending if pending is not None and t >= pending[0] - _TOLERANCE else None
            plan_request_t = pending_request_t if new is not None else None
            if new is not None:  # (time it takes effect, plan)
                plan, pending = new[1], None
            fault, update, faulted = None, False, None
            if pending is None and not end and due(t):
                update, at = True, t
                for _ in range(ahead):  # summed as the loop sums its deadlines: the same tick times
                    at += command_period
                x = predict(plan, law, s, at, command_period, step, substeps) if compensate else s
                mpc.request(x, *task.at(x.t - start, mpc.N, mpc.dt))
                if compensate:
                    events.append(tm.Event(t, "prediction_request", "latency example requested a predicted state",
                                           dict(t0=x.t - start, requested_state=np.r_[x.q, x.qd].tolist())))
                solved = mpc.result(block=True)  # blocks: the solve takes no virtual time
                assert solved is not None
                fault = tm.plan_fault(solved)
                if fault is None:
                    pending = (at, solved)
                    pending_request_t = t
                    if ahead == 0:  # zero latency: this tick
                        new, plan, pending = pending, pending[1], None
                        plan_request_t = t
                else:
                    faulted = solved  # it never takes effect
                    new, plan, plan_request_t = None, previous_plan, t
            if new is not None:
                plans += 1
            command = None
            if plan is not None and not end and fault is None:
                if new is not None and t >= plan.t0 and plan.index(t - plan.t0) >= 1:  # uncompensated only
                    events.append(tm.Event(t, "late_plan", f"a plan took effect {1e3 * (t - plan.t0):.1f} ms after "
                                                           f"its state, {1e3 * plan.dt:g} ms knots: its knot 0 never "
                                                           "runs (docs/robot.md, Latency)",
                                           dict(plan=plans - 1, delay_s=t - plan.t0)))
                fault = tm.lease_fault(plan, t)  # before the law: never past the last knot
                if fault is None:
                    command = np.asarray(law(plan, t, s), dtype=float)
                    fault = tm.command_fault(command)
                    if fault is None:
                        plant.apply(command)
                    else:
                        command = None
            tick = tm.Tick(t, s, command, plan, new is not None, update, (*events, *plant.take_events()), faulted,
                           plan_request_t if new is not None or faulted is not None else None)
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

    def run(self, plant, mpc, task, law, tick_callbacks, *, duration):
        return run_compensated(plant, mpc, task, law, tick_callbacks, duration=duration, **asdict(self))


def first_tick_of_each_plan(record):
    """The index of the tick at which each plan of record took effect."""
    plans, first = np.unique(record.ticks["plan"], return_index=True)
    return first[plans >= 0]


def requested_states(record):
    """Exact experiment inputs, from current events or the retained older plan column."""
    if "requested_state" in record.plans:
        return record.plans["requested_state"]
    requests = {e.t: e.data for e in record.events if e.kind == "prediction_request"}
    values = []
    for update_t, t0 in zip(record.plans["update_t"], record.plans["t0"]):
        event = requests[update_t]
        if not math.isclose(event["t0"], t0, abs_tol=_TOLERANCE):
            raise ValueError("prediction event does not match its plan origin")
        values.append(event["requested_state"])
    return np.asarray(values, dtype=float).reshape(-1, 14)


def main(argv=None):
    """One compensated reach on SimPlant, --solve-latency seconds (declared; default the timing's)."""
    handle_stop_signals()
    args = arguments(__doc__, argv, plants=("sim",))
    task = Reach() if args.target is None else Reach(tuple(args.target))
    plant, timing = setup(args, tm.SimConfig(substep=.003), Compensated())  # sim: never a status
    record = run_experiment(plant, tm.TIAGO, task, tm.SampleKnot(), timing,
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
