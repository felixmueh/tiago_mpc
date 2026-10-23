#!/usr/bin/env python3
"""Latency compensation in simulation: solve from the state predicted for the moment the
plan takes effect, and date the plan from then (Findeisen and Allgöwer; Mastalli et al.).

run_simulation(..., latency=Fixed(L)) solves from the state measured at tick t, dates the
plan t and lets it take effect at t + L, so sample_knot enters it at knot floor(L / dt).
Once L reaches one knot interval, knot 0, where each new solve's correction sits, never
runs, and the reach diverges (docs/robot.md, Latency). run_compensated predicts the state
at t + L under the commands the loop sends until then, solves from it and dates the plan
t + L, so every plan starts at its knot 0. It is its own loop, not run: run requests from
the measured state, and its late-plan check, plan.index(t - plan.t0), raises for a plan
dated in the future. It has run's calling convention; bind its own arguments:

    reach(SimPlant(substep=.003), Mpc.compile(replace(TIAGO, max_sqp_time_s=None)),
          run=partial(run_compensated, latency_s=.02))

The other remedy needs only a control law (plan_feedback.py); latency_study.py compares
both, alone and together, with stock sample_knot. The main runs one compensated reach.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import arguments
from experiments.reach import reach
from tiago_mpc.laws import sample_knot
from tiago_mpc.loops import Fixed, Schedule, checked, guard
from tiago_mpc.mpc import TIAGO, Mpc
from tiago_mpc.plan import State, Tick
from tiago_mpc.record import Record
from tiago_mpc.sim import SimPlant

_TOLERANCE = 1e-9  # as loops.py: absorbs the rounding of summed periods


def predict(mpc, plan, law, state, at, command_period, substeps=1):
    """The state at tick time at, from state, under the commands the loop sends at the
    ticks in between: law(plan, tick, predicted state), each held for one command period,
    so a law that depends on the state is predicted too. Without a plan the arm is held.

    The predictor's own model steps it: BSQP.sim_forward (GATO python/bindings.cu,
    gato/bsqp/kernels/sim.cuh) runs one step of the solver's device integrator, the
    constant-acceleration (trapezoidal) step its transcription uses (check_model's step),
    in float32, with the predictor's friction and knot 0's external wrench. It reads only
    the model and writes only its own output buffer: the solver's warm start, duals and
    rho are untouched. It works in GPU coordinates, so the sign map mpc.signs applies.

    The step is explicit: with the URDF's friction in the predictor, one 2 ms step makes
    joint 7 (0.0008 kg m^2 against 2.95 N m s/rad, plus the tanh slope) diverge, so each
    tick can be split into substeps equal steps under the same command."""
    t, x, ss = state.t, np.r_[state.q, state.qd], np.tile(mpc.signs, 2)
    while plan is not None and t < at - _TOLERANCE:
        u = law(plan, t, State(t, x[:7], x[7:])) * mpc.signs
        for _ in range(substeps):
            x = ss * np.asarray(mpc.controller.solver.sim_forward(
                (x * ss)[None], u[None], command_period / substeps), dtype=float)[0]
        t += command_period
    return State(at, x[:7], x[7:])


def run_compensated(plant, mpc, task, law=sample_knot, callbacks=(), *, command_period, update_period,
                    duration, latency_s, compensate=True, prediction_substep_s=None):
    """run_simulation(..., latency=Fixed(latency_s)), but each request is made from the
    state predicted (predict) at the tick its plan takes effect, and dated then.
    compensate=False requests from the measured state, dated now: run_simulation's timing.
    prediction_substep_s: the longest prediction step (None: one step per command tick).
    Simulation clock only; compensating needs the in-process Mpc (its solver and signs).

    As in run: the warm-up before the first tick, one outstanding solve, the next request
    at the first control-update slot at or after its plan took effect, each plan checked
    on arrival and each command guarded, the last tick without a command, plant.stop()
    first on exit. Left out: the garbage-collection pause and the execution time of a
    blocking solve (both matter on the wall clock only), the per-tick result() that
    notices a dead MpcProcess, and the late-plan warning. A Record among the callbacks
    gets the loop's settings and, per plan, the requested state (predicted or measured)
    and the plant's state at the tick the plan took effect.
    """
    if plant.clock.kind != "simulation":
        raise ValueError("run_compensated models the latency on a simulation clock; the robot keeps run_robot")
    ahead = math.ceil(latency_s / command_period - _TOLERANCE)  # a plan takes effect at the first tick at or after t + L
    substeps = 1 if prediction_substep_s is None else math.ceil(command_period / prediction_substep_s - _TOLERANCE)
    due, plan, pending, start, taken = Schedule(update_period), None, None, None, []
    plant.start()
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        deadline = None
        while True:
            s, t = plant.state(), plant.now()
            start = s.t if start is None else start
            done = task.done(s, t - start)
            end = done or t - start >= duration - _TOLERANCE
            new = pending if pending is not None and t >= pending[0] - _TOLERANCE else None
            if new is not None:  # (time it takes effect, requested state, plan)
                plan, pending = new[2], None
            if pending is None and not end and due(t):
                at = t
                for _ in range(ahead):  # summed as the loop sums its deadlines: the same tick times
                    at += command_period
                x = predict(mpc, plan, law, s, at, command_period, substeps) if compensate else s
                mpc.request(x, *task.at(x.t - start, mpc.N, mpc.dt))
                pending = (at, x, checked(mpc.result(block=True)))  # blocks: the solve takes no simulated time
                if ahead == 0:  # zero latency: this tick
                    new, plan, pending = pending, pending[2], None
            if new is not None:
                taken.append((new[1], s))
            command = None
            if plan is not None and not end:
                command = guard(law(plan, t, s), plan, t)
                plant.apply(command)
            for callback in callbacks:
                callback(Tick(t, s, command, plan, new is not None))
            if end:
                return done
            deadline = t if deadline is None else deadline
            deadline += command_period
            plant.advance(deadline)
    finally:
        try:
            plant.stop()  # zero torque first
        finally:
            for record in callbacks:
                if isinstance(record, Record):
                    note(record, taken, start or 0., dict(
                        loop="run_compensated", latency_s=latency_s, effective_latency_s=ahead * command_period,
                        compensate=compensate, prediction_substeps=substeps, law=getattr(law, "__name__", repr(law))))


def note(record, taken, start, settings):
    """The loop's settings and the per-plan prediction error into record."""
    requested = np.array([np.r_[x.q, x.qd] for x, _ in taken]).reshape(-1, 14)
    actual = np.array([np.r_[s.q, s.qd] for _, s in taken]).reshape(-1, 14)
    error = np.abs(actual - requested)
    record.arrays.update(effect_time_s=np.array([s.t - start for _, s in taken]),
                         requested_state=requested, effect_state=actual)
    record.summary.update(loop=settings, prediction_error_max=dict(
        q_rad=float(error[:, :7].max(initial=0.)), qd_rad_s=float(error[:, 7:].max(initial=0.))))


def main(argv=None):
    """One compensated reach on SimPlant, --latency seconds (fixed)."""
    args = arguments(__doc__, argv)
    if args.plant != "sim" or not isinstance(args.latency, Fixed):
        raise SystemExit("latency_compensation.py: simulation with a fixed --latency only")
    mpc = Mpc.compile(replace(TIAGO, max_sqp_time_s=None))  # in this process: predict uses its solver
    record = reach(SimPlant(substep=.003), mpc, run=partial(run_compensated, latency_s=args.latency.s),
                   out=args.output, command_period=args.command_period, update_period=args.update_period,
                   duration=args.duration)
    print(json.dumps({k: v for k, v in record.summary.items() if k not in ("params", "versions")},
                     indent=2, default=str))
    mpc.close()
    return 0 if record.summary["reached"] and record.summary["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
