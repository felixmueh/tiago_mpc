"""Two command loops: one for the real robot running in sync with wall time,
 and one for experiments, running in virtual time.

The timing of each loop, with diagrams, is explained in its docstring: run_real_time
(the robot and the rehearsal) and run_virtual_time (experiments). Their timings,
RealTime and VirtualTime, carry each loop's settings and run it. A run ends without an
exception when the task is done, the time is up, a fault is found or a stop is
requested (STOP); the loop returns how in its Outcome.

Robot execution keeps the strict 100 ms measurement-age lease and complete-plan
effort checks. Virtual-time execution makes the age limit configurable, retains
finite plan horizons, and applies an explicit effort policy after the control law.
Both loops preserve measured_t independently of a compensated future origin.

The two loops are kept separate to improve readability of the real_robot loop.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import gc
import math
import threading
from time import perf_counter
from typing import Literal
import warnings

import numpy as np

from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.laws import EffortClip, FrictionFeedforward, PlanFeedback, SampleKnot, _apply_effort
from tiago_mpc.interfaces import (Callback, ControlLaw, EffortHandling, Event, MessageCallback, MpcInterface, Plan,
                                  PlantInterface, Prediction, SolveCompleted, State, TaskInterface, Tick)
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


def _simulation_lease_fault(plan: Plan, t: float, max_plan_age_s: float | None) -> str | None:
    expires = plan.t0+len(plan.torques)*plan.dt
    if max_plan_age_s is not None:
        expires = min(expires, plan.measured_t+max_plan_age_s)
    return "lease_expired" if t >= expires else None


def _check_setup(mpc: MpcInterface, law: ControlLaw, compensation: LatencyCompensation | None) -> None:
    """Check the run's configuration before startup: raise on invalid configurations, warn
    about known problems. Covers the built-in law composition; arbitrary user callables
    remain the caller's responsibility."""
    base = law
    if isinstance(base, FrictionFeedforward):
        base = base.base_law
        if isinstance(base, FrictionFeedforward):
            raise ValueError("Friction feedforward cannot wrap another friction feedforward law")
        if mpc.params.friction is not None:
            raise ValueError("Friction feedforward requires predictor friction=None to avoid double compensation")
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
    predictor = None if compensation is None else _Predictor(compensation, LEASE_S)
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
    duration: float | None,
    solve_latency_s: float | Literal["measured"],
    compensation: LatencyCompensation | None = None,
    effort_handling: EffortHandling | None = EffortClip(),
    max_plan_age_s: float | None = LEASE_S,
    start_time_s: float = 0.,
    message_callbacks: Sequence[MessageCallback] = (),
    before_tick: Callable[[State], None] | None = None,
    quiesce_at: Callable[[State], bool] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    initial_command: np.ndarray | None = None,
) -> Outcome:
    """Run one controlled simulation with the same engine for experiments and Live.

    Commands and requests follow virtual time. A solve takes no virtual time;
    its plan waits for declared/native measured latency, while the old command
    continues. Requests have at most one pending plan; missed slots are skipped.

    effort_handling applies to the complete law's torque (EffortClip by default);
    None leaves effort unchanged. Finite plans still end at their horizon. max_plan_age_s
    additionally limits age from the measured observation; None disables that age
    limit. The compensation model is configured independently of the plant.

    Tick callbacks retain command records. Message observers receive queued plant
    Events before the next command, SolveCompleted before activation, and Tick
    after the command decision. before_tick sees every command-boundary state and
    may block to pause virtual time. Cancellation is rechecked after blocked work,
    before any command or further physics. A cancelled run is never restarted;
    construct its replacement with a fresh cancellation flag.

    Asynchronous requests overlap the independent plant interval. Results are
    polled, then awaited at the first possible activation tick (the next command
    tick for unknown measured latency). quiesce_at can require collection before
    an observer parks at an earlier reporting boundary. A pause requested by a
    Tick observer is collected/gated before physics; before_tick can therefore
    see the same timestamp twice. No in-process MPC is
    made asynchronous implicitly; MpcProcess implements the nonblocking protocol.

    duration=None has no preset time endpoint. Streaming callers own their history
    capacity; the finite-run limit remains for ordinary buffered experiments.
    Finite runs temporarily disable cyclic GC; streaming keeps its existing policy.
    The original policy is restored on exit. start_time_s supports anchored runs;
    task.at/done still receive time relative to this run's initial clock.
    """
    if not (solve_latency_s == "measured" or isinstance(solve_latency_s, (int, float))
            and math.isfinite(solve_latency_s) and solve_latency_s >= 0):
        raise ValueError("solve_latency_s must be seconds (finite, >= 0) or 'measured', "
                         f"not {solve_latency_s!r}")
    if not math.isfinite(command_period) or command_period <= 0:
        raise ValueError("command_period must be finite positive seconds")
    if not math.isfinite(update_period) or update_period < 0:
        raise ValueError("update_period must be finite nonnegative seconds")
    if duration is not None:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite nonnegative seconds or None")
        if duration > 60*MAX_RUNTIME_MIN:
            raise ValueError("Not validated for long runs. Records stored in memory and GC disabled.")
    if (max_plan_age_s is not None and
            (not math.isfinite(max_plan_age_s) or max_plan_age_s <= 0)):
        raise ValueError("max_plan_age_s must be finite positive seconds or None")
    if not math.isfinite(start_time_s):
        raise ValueError("start_time_s must be finite")
    if initial_command is not None:
        initial_command = np.array(initial_command, dtype=float)
        if _apply_effort(initial_command, None)[1] is not None:
            raise ValueError("initial_command requires seven finite native torques")
    _check_setup(mpc, law, compensation)
    clock, due = VirtualClock(start_time_s), Schedule(update_period)
    plan: Plan | None = None
    pending: Plan | None = None
    available, requested_at = start_time_s, start_time_s
    predictor = None if compensation is None else _Predictor(compensation, max_plan_age_s, effort_handling)
    updates, plans = 0, 0
    held_command = initial_command

    def stopping() -> bool:
        return cancelled() or STOP.is_set()

    def emit(message: Tick | SolveCompleted | Event) -> None:
        for callback in message_callbacks:
            callback(message)

    solving = False
    request_inputs: tuple[State | Prediction, np.ndarray, dict | None] | None = None
    controller_wall_s = 0.
    earliest_result = start_time_s
    deferred_events: list[Event] = []

    def collect_result(block: bool, events: list[Event]) -> Plan | None:
        nonlocal solving, available, controller_wall_s
        began = perf_counter()
        solved = mpc.result(block=block)
        controller_wall_s += perf_counter()-began
        if solved is None:
            if block:
                raise RuntimeError("MPC returned no plan for a blocking result")
            return None
        solving = False
        if stopping():
            return None
        took = solved.stats["wall_s"]
        if not isinstance(took, (float, int)) or not math.isfinite(took) or took < 0:
            raise ValueError("MPC wall_s must be finite nonnegative seconds")
        if message_callbacks:
            assert request_inputs is not None
            captured, captured_reference, changed = request_inputs
            emit(SolveCompleted(requested_at, captured, captured_reference, changed, solved, controller_wall_s))
        latency = took if solve_latency_s == "measured" else solve_latency_s
        available = requested_at+latency
        if 0. < latency < took:
            event = Event(requested_at, "overrun", f"a solve took {1e3*took:.1f} ms, longer than "
                          f"solve_latency_s={1e3*latency:g} ms",
                          dict(plan=updates-1, wall_s=took, solve_latency_s=latency))
            events.append(event)
            emit(event)
        return solved

    def take_ready_plan(t: float) -> Plan | None:
        nonlocal pending
        if pending is None or t < available-_TOLERANCE:
            return None
        if predictor is not None:
            predictor.arrived(available-requested_at)
            if t < pending.t0-_TOLERANCE:
                return None
        ready, pending = pending, None
        return ready

    plant.start(clock)
    gc_enabled = gc.isenabled()
    if duration is not None:
        gc.disable()
    try:
        if stopping():
            return Outcome("interrupted", "stop_requested", 0.)
        if initial_command is not None:
            plant.apply(initial_command)
            if predictor is not None:
                predictor.sent(start_time_s, initial_command)
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])
        while not stopping():
            s, t = plant.state(), clock.now()
            events = list(plant.take_events())
            # Events retain their physics-endpoint timestamps, but are delivered
            # at this command boundary before another actuator decision.
            for event in events:
                emit(event)
            events[:0] = deferred_events
            deferred_events = []
            if stopping():
                break
            if solving:
                pending = collect_result(t >= earliest_result-_TOLERANCE or
                               quiesce_at is not None and quiesce_at(s), events)
            if stopping():
                break
            if before_tick is not None:
                before_tick(s)
            if stopping():
                break
            done = task.done(s, t-start_time_s)
            end = done or duration is not None and t-start_time_s >= duration-_TOLERANCE
            new = take_ready_plan(t)
            fault: str | None = None
            update = False
            plan_request_t = requested_at if new is not None else None
            prepared_plan: Plan | None = None
            prepared_requested: np.ndarray | None = None
            prepared_command: np.ndarray | None = None
            if pending is None and not solving and not end and due(t):
                update, updates = True, updates+1
                solve_started = perf_counter()
                initial: State | Prediction = s
                if predictor is not None:
                    prepared_plan = new if new is not None else plan
                    if prepared_plan is not None:
                        fault = _simulation_lease_fault(prepared_plan, t, max_plan_age_s)
                        if fault is None:
                            prepared_requested = np.asarray(law(prepared_plan, t, s), dtype=float)
                            prepared_command, fault = _apply_effort(prepared_requested, effort_handling)
                    at = _tick_at_or_after(t+predictor.latency_s, t, command_period)
                    predicted = (None if fault is not None else predictor.predict(
                        s, t, at, prepared_plan, law,
                        held_command if prepared_requested is None else prepared_command,
                        t+command_period, command_period))
                    if predicted is None:
                        fault = fault or "prediction_unavailable"
                    else:
                        initial = predicted
                if fault is not None:
                    update = False
                else:
                    reference, changed = task.at(initial.t-start_time_s, mpc.N, mpc.dt)
                    if message_callbacks:
                        if isinstance(initial, Prediction):
                            captured: State | Prediction = Prediction(initial.t, initial.q, initial.qd,
                                                                      measured_t=initial.measured_t)
                        else:
                            captured = State(initial.t, initial.q.copy(), initial.qd.copy(),
                                             deepcopy(initial.joint_positions), deepcopy(initial.joint_velocities))
                            captured.q.flags.writeable = captured.qd.flags.writeable = False
                        captured_reference = np.array(reference, copy=True)
                        captured_reference.flags.writeable = False
                        request_inputs = captured, captured_reference, deepcopy(changed)
                    requested_at = t
                    latency_bound = command_period if solve_latency_s == "measured" else solve_latency_s
                    earliest_result = max(t+latency_bound, initial.t)
                    mpc.request(initial, reference, changed)
                    solving = True
                    controller_wall_s = perf_counter()-solve_started
                    pending = collect_result(solve_latency_s == 0., events)
                    if stopping():
                        break
                    ready = take_ready_plan(t)
                    if ready is not None:
                        new, plan_request_t = ready, requested_at
            if new is not None:
                plan, plans = new, plans+1
            command: np.ndarray | None = None
            requested = prepared_requested
            clipped = False
            if stopping():
                break
            if plan is not None and not end and fault is None:
                if new is not None and plan.index(t-plan.t0) >= 1:
                    event = _late_plan(plan, t, plans-1)
                    events.append(event)
                    emit(event)
                fault = _simulation_lease_fault(plan, t, max_plan_age_s)
                if fault is None:
                    if plan is prepared_plan and prepared_requested is not None:
                        requested, command = prepared_requested, prepared_command
                    else:
                        requested = np.asarray(law(plan, t, s), dtype=float)
                        command, fault = _apply_effort(requested, effort_handling)
                    clipped = command is not None and bool(np.any(command != requested))
                    if stopping():
                        break
                    if fault is None and command is not None:
                        plant.apply(command)
                        held_command = command
                        if predictor is not None:
                            predictor.sent(t, command)
                    else:
                        command = None
            if solving and (end or fault is not None):
                collect_result(True, events)
                if stopping():
                    break
            if fault in ("effort_limit", "invalid_command") and requested is not None:
                rejected = Event(t, "command_rejected", f"command refused: {fault}",
                                 dict(reason=fault, requested_torque=(requested.tolist()
                                     if np.isfinite(requested).all() else None)))
                events.append(rejected)
                emit(rejected)
            tick = Tick(t, s, command, plan, new is not None, update, tuple(events), None,
                        plan_request_t if new is not None else None, requested, clipped)
            for callback in tick_callbacks:
                callback(tick)
            emit(tick)
            if stopping():
                break
            if fault is not None:
                return Outcome("fault", fault, t-start_time_s)
            if end:
                return _ended(done, False, t-start_time_s)
            # A Tick observer may request a pause after submission (for example
            # an applied-effort breach). Collect before parking, without another
            # actuator decision or physics step at this same command boundary.
            if quiesce_at is not None and quiesce_at(s):
                if solving:
                    # This Tick has already reached its callbacks. Carry only
                    # its later events to the next Tick; emit() already sent them.
                    pending = collect_result(True, deferred_events)
                if before_tick is not None and not stopping():
                    before_tick(s)
                if stopping():
                    break
            plant.advance(t+command_period)
    finally:
        try:
            plant.stop()
        finally:
            if gc_enabled:
                gc.enable()
            else:
                gc.disable()
    return Outcome("interrupted", "stop_requested", clock.now()-start_time_s)


@dataclass(frozen=True)
class VirtualTime:
    """run_virtual_time's timing: its three settings, without defaults; the common
    timings are the constructors below, which its docstring describes."""
    command_period: float
    update_period: float
    solve_latency_s: float | Literal["measured"]
    compensation: LatencyCompensation | None = None
    max_plan_age_s: float | None = LEASE_S
    effort_handling: EffortHandling | None = EffortClip()

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
        duration: float | None,
        message_callbacks: Sequence[MessageCallback] = (),
        before_tick: Callable[[State], None] | None = None,
        quiesce_at: Callable[[State], bool] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        start_time_s: float = 0.,
        initial_command: np.ndarray | None = None,
    ) -> Outcome:
        return run_virtual_time(plant, mpc, task, law, tick_callbacks, duration=duration,
                                command_period=self.command_period, update_period=self.update_period,
                                solve_latency_s=self.solve_latency_s, compensation=self.compensation,
                                max_plan_age_s=self.max_plan_age_s, effort_handling=self.effort_handling,
                                message_callbacks=message_callbacks,
                                before_tick=before_tick, quiesce_at=quiesce_at, cancelled=cancelled, start_time_s=start_time_s,
                                initial_command=initial_command)


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
