"""Two command loops: one for the real robot running in sync with wall time,
 and one for experiments, running in virtual time.

The timing of each loop, with diagrams, is explained in its docstring: run_real_time
(the robot and the rehearsal) and run_virtual_time (experiments). Their timings,
RealTime and VirtualTime, carry each loop's settings and run it. A run ends without an
exception when the task is done, the time is up, a fault is found or a stop is
requested (STOP); the loop returns how in its Outcome.

A plan's lease ends at min(measured_t + LEASE_S, t0 + the plan's length), so neither
prediction nor a future t0 renews its source observation's age allowance, and it never
runs past its last knot (31 knots of 2 ms span only 62 ms). Without compensation,
measured_t equals t0. Both loops check it at every tick before the law
runs (lease_fault); a plan still in effect after its lease ends the run with the fault
"lease_expired".

The two loops are kept separate to improve readability of the real_robot loop.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import gc
import math
import threading
from typing import Literal
import warnings

import numpy as np

from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.laws import FrictionFeedforward, PlanFeedback, SampleKnot
from tiago_mpc.interfaces import (Callback, ControlLaw, Event, MpcInterface, Plan, PlantInterface, TaskInterface,
                                  Tick)
from tiago_mpc.prediction import LatencyCompensation, _Predictor, _tick_at_or_after
from tiago_mpc.tiago import limits
from tiago_mpc.util import hz_to_s

LEASE_S = .1  # a plan may be commanded at most this long after its measured source state
_TOLERANCE = 1e-9  # absorbs the rounding of summed periods
# Not validated for long runs. Records stored in memory and GC disabled.
# Only raise after verification or handling these issues yourself.
MAX_RUNTIME_MIN = 10
# The late-plan event of both loops: knot 0 holds the correction each new plan makes,
# and a plan that first takes effect a knot interval or more after its state never runs
# it (docs/robot.md, Latency). Remedies (thesis workspace, research/tiago-mpc-latency-
# analysis.md, "How other MPC systems apply their plans"): (a) feedback around the plan
# at the command rate, e.g. u = u_plan(t) + Kp (q_plan(t) - q) + Kd (qd_plan(t) - qd) on
# the plan's knot states (OCS2: Riccati gains; BiConMP: joint impedance); (b) latency
# compensation: solve from the state predicted at the plan's application time under the
# committed torques and date the plan from then (Findeisen and Allgöwer; Mastalli et al.).

# A stop request: both loops check it every tick and end the run as interrupted. Set by
# experiments.common.handle_stop_signals (Ctrl-C, SIGTERM), which also clears it.
STOP = threading.Event()


@dataclass(frozen=True)
class Outcome:
    """How a loop ended: status "done" (the task is done), "timeout" (the duration
    ended), "fault" or "interrupted" (STOP); reason, for a fault "effort_limit",
    "lease_expired", "invalid_command" or "prediction_unavailable", for an interruption "stop_requested", else
    None; t: the last tick's time, counted from the first state's."""
    status: str
    reason: str | None
    t: float


def _ended(done: bool, stopped: bool, t: float) -> Outcome:
    """The Outcome of a last tick without a fault."""
    if done:
        return Outcome("done", None, t)
    return Outcome("interrupted", "stop_requested", t) if stopped else Outcome("timeout", None, t)


class Schedule:
    """Control updates every period from the first call: due at the first call at or
    after each slot, at most once per call; missed slots are skipped. Schedule(0) is
    due at every call."""

    def __init__(self, period):
        self.period, self.next = period, None

    def __call__(self, t):
        if self.period == 0:
            return True
        if self.next is None:
            self.next = t
        slots = math.floor((t - self.next) / self.period + _TOLERANCE)
        if slots < 0:
            return False
        self.next += (slots + 1) * self.period
        return True


def plan_fault(plan: Plan) -> str | None:
    """"effort_limit" if a knot torque exceeds the raw effort limit, else None: any knot
    may be commanded before its lease ends, so each plan is checked when it arrives."""
    return "effort_limit" if np.any(np.abs(plan.torques) > limits().effort) else None


def lease_fault(plan: Plan, t: float) -> str | None:
    """"lease_expired" once the plan's lease (the module docstring) has ended at t, else
    None, so a law is never asked past the last knot."""
    return "lease_expired" if t >= min(plan.measured_t + LEASE_S, plan.t0 + len(plan.torques) * plan.dt) else None


def command_fault(command: np.ndarray) -> str | None:
    """"invalid_command" unless the command is seven finite torques, "effort_limit" beyond
    the raw effort limits, else None."""
    command, effort = np.asarray(command, dtype=float), limits().effort
    if command.shape != effort.shape or not np.isfinite(command).all():
        return "invalid_command"
    return "effort_limit" if np.any(np.abs(command) > effort) else None


def _check_setup(mpc: MpcInterface, law: ControlLaw, compensation: LatencyCompensation | None) -> None:
    """Check the run's configuration before startup: raise on invalid configurations, warn
    about known problems. Covers the built-in law composition; arbitrary user callables
    remain the caller's responsibility."""
    base = law
    if isinstance(law, FrictionFeedforward):
        if isinstance(law.base_law, FrictionFeedforward):
            raise ValueError("Friction feedforward cannot wrap another friction feedforward law")
        if mpc.params.friction is not None:
            raise ValueError("Friction feedforward requires predictor friction=None to avoid double compensation")
        base = law.base_law
    if isinstance(base, PlanFeedback) and compensation is None:
        warnings.warn("PlanFeedback without latency compensation: when the first plan takes effect after its "
                      "solve latency, the arm is still held, and the D term can push the command beyond the "
                      "effort limit at the start of a fast motion. Pass compensation=LatencyCompensation(...) "
                      "with an initial_latency_s that covers the first solve (docs/latency-compensation.md).",
                      RuntimeWarning, stacklevel=3)


def _late_plan(plan: Plan, t: float, index: int) -> Event:
    """The late-plan event of plan number index, which takes effect at t."""
    return Event(t, "late_plan", f"a plan took effect {1e3 * (t - plan.t0):.1f} ms after its state, "
                                 f"{1e3 * plan.dt:g} ms knots: its knot 0 never runs (docs/robot.md, Latency)",
                 dict(plan=index, delay_s=t - plan.t0))


def run_real_time(
    plant: PlantInterface,
    mpc: MpcInterface,
    task: TaskInterface,
    law: ControlLaw = SampleKnot(),
    tick_callbacks: Sequence[Callback] = (),
    *,
    command_period: float,
    update_period: float,
    duration: float,
    compensation: LatencyCompensation | None = None,
) -> Outcome:
    """Run task on plant on wall time until task.done(state, t) or for duration seconds:
    the robot (RosPlant), or a rehearsal of it with a simulated arm (SimPlant).

    Time is the computer's monotonic clock (WallClock), and everything takes its real time:
    the solve, the plant, the tick callbacks. MpcProcess solves concurrently while the loop
    keeps ticking, and a plan is used at the first tick after it arrives; on the robot
    a state's own age adds to the delay. RealTime(update_period=.01, compensation=None);
    the solves took 3.5 and 4.5 ms:

      wall time (ms)  0   2   4   6   8   10  12  14  16  18
      command ticks   |   |   |   |   |   |   |   |   |   |
      control updates R0                  R1
      solve           [------]            [--------]
      in effect       old=====P0======================P1===
                              ^ the first tick after the plan arrived

    Each tick sends its command before it requests a plan, so an in-process Mpc, which
    solves inside the request, delays only the next tick: the ticks it misses
    meanwhile are late.

    Ticks are command_period apart. A late tick, one that starts a command period or
    more after its deadline because the work before it ran past the next tick, skips
    the missed ticks and is reported as a late_tick event, unless it follows the first
    command (on the robot, that command switches to effort control through blocking
    service calls, whose duration is the activated event's switch_s).

    The task's time starts at 0 with the first state: done and the duration see the
    tick's time, and at, which gives a request its reference, sees the time of the state
    the request starts from. Before the first tick, mpc.warmup runs
    from the started plant's state, the arm still held, so its time counts against no
    plan's lease. Each plan is checked when it arrives (plan_fault), each command before
    it is sent (lease_fault, command_fault): within the plan's lease. A fault ends the
    run; its tick sends nothing, and a faulting plan never takes effect (the tick carries it as Tick.faulted,
    so the record keeps it). Polling result() at every tick also
    notices a dead MPC process within one tick (it raises). Each plan that takes effect a
    knot interval or more after its state, so that its knot 0 never runs, is a late_plan
    event (docs/robot.md, Latency). The last tick, which sees the task done, the time up
    or STOP set, sends no command but reaches the tick callbacks. Garbage collection pauses
    while the loop runs: a full collection over the buffered ticks takes several
    milliseconds. A duration over MAX_RUNTIME_MIN raises before the plant starts.

    Optional compensation predicts the solve's initial condition without changing the
    measured State. Early replies wait for their predicted origin, late replies retain
    their age. The physical propagation model and delay forecast are explicit settings.
    A failed forecast ends as prediction_unavailable; the current valid command and
    callbacks may already have run before that request-time fault is detected.
    """

    # This is written as inline code for clarity. The cost is duplication with the virtual-time loop and potentially more maintenance.
    # Time will tell if this decision should be revisited.

    if duration > 60 * MAX_RUNTIME_MIN:
        raise ValueError("Not validated for long runs. Records stored in memory and GC disabled.")
    _check_setup(mpc, law, compensation)
    clock, due = WallClock(), Schedule(update_period)
    plan, solving, start, deadline = None, False, None, None
    predictor = None if compensation is None else _Predictor(compensation, LEASE_S, command_fault)
    waiting, requested_at, request_tick = None, 0., 0.
    sent, plans = 0, 0
    plant.start(clock)  # startup checks on a fresh state; the robot accepts wall time only

    # NOTE: IMPORTANT: Fine for exprs, but for real world use: DON'T use a GC language!
    gc.disable()  # for the whole run: full collections took 7-10 ms at 500 Hz
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        while True:
            s, t = plant.state(), clock.now()  # the newest state, and the tick's time
            events = []
            if deadline is None:
                deadline = t  # the first tick
            elif t >= deadline + command_period:  # late: the work before it ran past the next tick
                if sent != 1:  # the tick after the first command: the activated event has its delay
                    events.append(Event(t, "late_tick", f"a command tick started {1e3 * (t - deadline):.1f} ms after "
                                                        f"its deadline, {1e3 * command_period:g} ms ticks: the "
                                                        "missed ticks were skipped",
                                        dict(late_s=t - deadline,
                                             skipped=math.floor((t - deadline) / command_period + _TOLERANCE))))
                deadline = t  # skip the missed ticks
            start = s.t if start is None else start
            done, stopped = task.done(s, t - start), STOP.is_set()
            end = done or stopped or t - start >= duration - _TOLERANCE
            new, fault, faulted = mpc.result(), None, None  # a plan that arrived since the last tick
            if new is not None:
                solving, fault = False, plan_fault(new)
                if predictor is not None:
                    predictor.arrived(clock.now() - requested_at)
                    fault = fault or lease_fault(new, t)
                    if fault is None:
                        waiting, new = new, None
                if fault is None:
                    if new is not None:
                        plan, plans = new, plans + 1
                else:
                    new, faulted = None, new  # it never takes effect
            if waiting is not None and fault is None:
                fault = lease_fault(waiting, t)
                if fault is not None:
                    faulted, waiting = waiting, None
                elif t >= waiting.t0 - _TOLERANCE:
                    new, waiting = waiting, None
                    plan, plans = new, plans + 1
            command = None
            if plan is not None and not end and fault is None:
                if new is not None and plan.index(t - plan.t0) >= 1:
                    events.append(_late_plan(plan, t, plans - 1))
                fault = lease_fault(plan, t)  # before the law: never past the last knot
                if fault is None:
                    command = np.asarray(law(plan, t, s), dtype=float)
                    fault = command_fault(command)
                    if fault is None:
                        sent_at = clock.now() if predictor is not None else 0.
                        plant.apply(command)
                        if predictor is not None:
                            predictor.sent(sent_at, command)
                        sent += 1
                    else:
                        command = None
            update = not end and fault is None and not solving and waiting is None and due(t)
            tick = Tick(t, s, command, plan, new is not None, update, (*events, *plant.take_events()), faulted,
                        request_tick if new is not None or faulted is not None else None)
            for callback in tick_callbacks:
                callback(tick)
            if fault is not None:
                return Outcome("fault", fault, t - start)
            if end:
                return _ended(done, stopped, t - start)
            if update:  # the command is out: now ask for a new plan
                request_tick = t
                if predictor is None:
                    mpc.request(s, *task.at(s.t - start, mpc.N, mpc.dt))
                else:
                    now = clock.now()  # after publication and callbacks; prediction work follows
                    next_tick = _tick_at_or_after(now + 1e-9, deadline + command_period, command_period)
                    at = _tick_at_or_after(now + predictor.latency_s, next_tick, command_period)
                    predicted = predictor.predict(s, now, at, plan, law, command, next_tick, command_period)
                    if predicted is None:
                        return Outcome("fault", "prediction_unavailable", clock.now() - start)
                    requested_at = clock.now()
                    mpc.request(predicted, *task.at(predicted.t - start, mpc.N, mpc.dt))
                solving = True
            deadline += command_period
            plant.advance(deadline)  # the plant holds the command until then
    finally:
        try:
            plant.stop()  # zero torque and handback, first
        finally:
            gc.enable()


def run_virtual_time(
    plant: PlantInterface,
    mpc: MpcInterface,
    task: TaskInterface,
    law: ControlLaw = SampleKnot(),
    tick_callbacks: Sequence[Callback] = (),
    *,
    command_period: float,
    update_period: float,
    duration: float,
    solve_latency_s: float | Literal["measured"],
    compensation: LatencyCompensation | None = None,
) -> Outcome:
    """Run task on plant on virtual time until task.done(state, t) or for duration
    seconds: a controlled experiment in simulation (SimPlant).

    Three effects set a simulated run's timing: how long a plan takes to arrive (its
    solve latency), when plans are requested (the update schedule, above), and how
    fast the simulator computes. This loop sets the first two and keeps the third out
    of the results.

    Virtual time. The loop keeps the time (VirtualClock) and moves it itself, tick by
    tick. Nothing takes time unless the loop writes it in, and it writes in two
    things: the ticks, command_period apart, and the solves, at solve_latency_s. The
    simulator's own computing and the loop's work take none; they only make the run
    slower or faster than real time. (On wall time, in run_real_time, everything
    takes its real time instead.) The loop waits for each solve, places its plan at
    the request plus the latency, and runs the ticks up to then with the previous
    plan. One in-process solve at the tick at 4 ms; the boxes show what the computer
    does, along its wall clock:

      computer        [tk][tk][tk][  solve 3.5 ms  ][tk][tk][tk]
      wall clock      0    2    4                   7.5  9.5  11.5
      virtual time    0    2    4                   6    8    10
                      the computing took no virtual time; on the timeline the
                      solve lasts solve_latency_s (its plan is placed at
                      4 + solve_latency_s), and the ticks at 6 and 8 run after it.

    solve_latency_s, required, says how long a solve lasts on the timeline:
      a number    declared, in seconds: the plan is available that long after the
                  request, whatever the solve really took. A solve that took longer
                  is an overrun event (0, the explicit ideal, never is).
      "measured"  this computer's own solve time (plan.stats["wall_s"]): the loop
                  waits for the solve, then uses its time as the declared one.

    The timing VirtualTime holds these settings; its constructors are the common
    ones. VirtualTime.ideal(): each plan in effect at the tick of its request
    (solve_latency_s 0). Three more, 2 ms command ticks:

      VirtualTime.constant_latency(solve_latency_s=.005), a declared latency,
      update_period=.01 (the preset's own latency is 50 ms):

      time (ms)       0   2   4   6   8   10  12  14  16  18  20
      command ticks   |   |   |   |   |   |   |   |   |   |   |
      control updates R0                  R1                  R2
      solve           [---------]         [---------]
      in effect       old=========P0==================P1=======
                                  ^ the first tick at or after the plan is available

      VirtualTime.as_fast_as_possible(), solve_latency_s="measured", update_period=0;
      the solves took 3.5 and 4.1 ms of wall time:

      time (ms)       0   2   4   6   8   10  12  14  16
      command ticks   |   |   |   |   |   |   |   |   |
      control updates R0      R1          R2
      solve           [------][-------]
      in effect       old=====P0==========P1===========
                              ^ R1 at the tick P0 takes effect

      VirtualTime.constant_rate(), a constant control rate, solve_latency_s =
      update_period = .01: a request at every slot, its plan in effect at the next
      (an overrun event if a solve took longer):

      time (ms)       0   2   4   6   8   10  12  14  16  18  20
      command ticks   |   |   |   |   |   |   |   |   |   |   |
      control updates R0                  R1                  R2
      solve           [-------------------[-------------------]
      in effect       old=================P0==================P1

    Watching. SimPlant(sync_real_time=True) holds each tick until the wall clock has
    caught up, so the run can be watched at real speed; falling behind is a
    pacing_behind event and never changes the results. It is not a rehearsal: a
    rehearsal runs a SimPlant on run_real_time.

    As in run_real_time: the task's t counts from the first state's time, the
    warm-up runs before the first tick, each plan is checked (here when its solve
    returns) and each command, a fault ends the run, each plan that never runs its
    knot 0 is a late_plan event, the last tick (also on STOP) sends no command,
    garbage collection pauses while the loop runs, and a duration over MAX_RUNTIME_MIN
    raises before the plant starts.

    Optional compensation uses a separate physical model and a fixed or previous-delay
    forecast. The current solve's duration controls availability only, never its own
    forecast. Measured State timestamps and the observation-relative lease are retained.
    """
    if not (solve_latency_s == "measured" or isinstance(solve_latency_s, (int, float))
            and math.isfinite(solve_latency_s) and solve_latency_s >= 0):
        raise ValueError("solve_latency_s must be seconds (finite, >= 0) or 'measured', "
                         f"not {solve_latency_s!r}")
    if duration > 60 * MAX_RUNTIME_MIN:
        raise ValueError("Not validated for long runs. Records stored in memory and GC disabled.")
    _check_setup(mpc, law, compensation)
    clock, due = VirtualClock(), Schedule(update_period)
    plan, pending, available, start = None, None, 0., None  # pending: the solved plan
    predictor = None if compensation is None else _Predictor(compensation, LEASE_S, command_fault)
    requested_at, pending_received = 0., False
    updates, plans = 0, 0
    plant.start(clock)
    gc.disable()  # full collections took 7-10 ms at 500 Hz
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        while True:
            s, t = plant.state(), clock.now()
            events = []
            start = s.t if start is None else start
            done, stopped = task.done(s, t - start), STOP.is_set()
            end = done or stopped or t - start >= duration - _TOLERANCE
            new, fault, update, faulted = None, None, False, None
            plan_request_t = None
            if pending is not None and t >= available - _TOLERANCE:
                if predictor is not None and not pending_received:
                    predictor.arrived(available - requested_at)
                    pending_received = True
                if predictor is None or t >= pending.t0 - _TOLERANCE:
                    plan_request_t = requested_at
                    new, pending = pending, None
            prepared_plan, prepared_command = None, None
            # Between requests nothing is polled: a dead MpcProcess is noticed at the next request.
            if pending is None and not end and due(t):
                update, updates = True, updates + 1
                request_at = t
                if predictor is None:
                    mpc.request(s, *task.at(s.t - start, mpc.N, mpc.dt))
                else:
                    prepared_plan = new if new is not None else plan
                    if prepared_plan is not None:
                        fault = lease_fault(prepared_plan, t)
                        if fault is None:
                            prepared_command = np.asarray(law(prepared_plan, t, s), dtype=float)
                            fault = command_fault(prepared_command)
                    at = _tick_at_or_after(t + predictor.latency_s, t, command_period)
                    predicted = (None if fault is not None else predictor.predict(
                        s, t, at, prepared_plan, law, prepared_command, t + command_period, command_period))
                    if predicted is None:
                        fault = fault or "prediction_unavailable"
                    else:
                        mpc.request(predicted, *task.at(predicted.t - start, mpc.N, mpc.dt))
                if fault is not None:
                    update = False
                else:
                    solved = mpc.result(block=True)  # wait for it: no virtual time passes
                    assert solved is not None
                    fault = plan_fault(solved)
                    if fault is None:
                        pending, took = solved, solved.stats["wall_s"]
                        requested_at, pending_received = request_at, False
                        latency = took if solve_latency_s == "measured" else solve_latency_s
                        if 0. < latency < took:
                            events.append(Event(t, "overrun", f"a solve took {1e3 * took:.1f} ms, longer than "
                                                              f"solve_latency_s={1e3 * latency:g} ms",
                                                dict(plan=updates - 1, wall_s=took, solve_latency_s=latency)))
                        available = t + latency
                        if t >= available - _TOLERANCE:
                            if predictor is not None:
                                predictor.arrived(latency)
                                pending_received = True
                            if predictor is None or t >= pending.t0 - _TOLERANCE:
                                plan_request_t = requested_at
                                new, pending = pending, None
                    else:
                        faulted, plan_request_t = solved, request_at
                        new = None  # an earlier ready plan sent no command before this fault
            if new is not None:
                plan, plans = new, plans + 1
            command = None
            if plan is not None and not end and fault is None:
                if new is not None and plan.index(t - plan.t0) >= 1:
                    events.append(_late_plan(plan, t, plans - 1))
                fault = lease_fault(plan, t)  # before the law: never past the last knot
                if fault is None:
                    command = (prepared_command if predictor is not None and plan is prepared_plan
                               else np.asarray(law(plan, t, s), dtype=float))
                    assert command is not None
                    fault = command_fault(command)
                    if fault is None:
                        plant.apply(command)
                        if predictor is not None:
                            predictor.sent(t, command)
                    else:
                        command = None
            tick = Tick(t, s, command, plan, new is not None, update, (*events, *plant.take_events()), faulted,
                        plan_request_t if new is not None or faulted is not None else None)
            for callback in tick_callbacks:
                callback(tick)
            if fault is not None:
                return Outcome("fault", fault, t - start)
            if end:
                return _ended(done, stopped, t - start)
            plant.advance(t + command_period)  # the plant integrates to it, and the clock moves there
    finally:
        try:
            plant.stop()  # zero torque first
        finally:
            gc.enable()


@dataclass(frozen=True)
class VirtualTime:
    """run_virtual_time's timing: its three settings, without defaults; the common
    timings are the constructors below, which its docstring describes."""
    command_period: float
    update_period: float
    solve_latency_s: float | Literal["measured"]
    compensation: LatencyCompensation | None = None

    @classmethod
    def ideal(cls, command_period: float = hz_to_s(500), update_period: float = hz_to_s(100)) -> VirtualTime:
        """Each plan in effect at the tick of its request."""
        return cls(command_period, update_period, 0.)

    @classmethod
    def constant_latency(
        cls,
        solve_latency_s: float = .05,
        command_period: float = hz_to_s(500),
        update_period: float = hz_to_s(100),
    ) -> VirtualTime:
        """Each plan available solve_latency_s seconds after its request."""
        return cls(command_period, update_period, solve_latency_s)

    @classmethod
    def as_fast_as_possible(cls, command_period: float = hz_to_s(500)) -> VirtualTime:
        """Each solve takes its measured time, and the next is requested when its plan takes effect."""
        return cls(command_period, 0., "measured")

    @classmethod
    def constant_rate(cls, period: float = hz_to_s(100), command_period: float = hz_to_s(500)) -> VirtualTime:
        """A request every period, its plan in effect a period later."""
        return cls(command_period, period, period)

    @classmethod
    def like(cls, real: RealTime) -> VirtualTime:
        """The simulation of real: its periods and compensation, each solve declared to last one
        control period. VirtualTime.like(RealTime()) simulates the real MPC's defaults."""
        return cls(real.command_period, real.update_period, real.update_period, real.compensation)

    def run(
        self,
        plant: PlantInterface,
        mpc: MpcInterface,
        task: TaskInterface,
        law: ControlLaw,
        tick_callbacks: Sequence[Callback],
        *,
        duration: float,
    ) -> Outcome:
        return run_virtual_time(plant, mpc, task, law, tick_callbacks, duration=duration,
                                command_period=self.command_period, update_period=self.update_period,
                                solve_latency_s=self.solve_latency_s, compensation=self.compensation)


@dataclass(frozen=True)
class RealTime:
    """run_real_time's timing. Its defaults are the real MPC's, which the simulation takes with
    VirtualTime.like (Felix, 2026-10-30): 500 Hz commands, 50 Hz control updates and latency
    compensation with a fixed forecast of one control period."""
    command_period: float = hz_to_s(500)
    update_period: float = hz_to_s(50)
    compensation: LatencyCompensation | None = LatencyCompensation(latency_s=hz_to_s(50))

    def run(
        self,
        plant: PlantInterface,
        mpc: MpcInterface,
        task: TaskInterface,
        law: ControlLaw,
        tick_callbacks: Sequence[Callback],
        *,
        duration: float,
    ) -> Outcome:
        return run_real_time(plant, mpc, task, law, tick_callbacks, duration=duration,
                             command_period=self.command_period, update_period=self.update_period,
                             compensation=self.compensation)
