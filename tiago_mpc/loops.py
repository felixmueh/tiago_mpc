"""Two command loops for wall-time robot execution and virtual-time experiments.

The timing of each loop is explained in its docstring: run_real_time
(the robot and the rehearsal) and run_virtual_time (experiments). RealTimeSettings and
VirtualTimeSettings carry each loop's timing settings. A run ends without an
exception when the task is done, the time is up, a fault is found or a stop is
requested (STOP); the loop returns how in its Outcome.

A plan's lease sets how long it may supply commands. In real time it expires at
min(measured_t + LEASE_S, t0 + len(torques) * dt): the earlier of 100 ms after
the source measurement and the end of the plan's torque intervals. measured_t is
when the source state was measured; t0 is the start of the planned trajectory.
Latency compensation may move t0 into the future, but never changes measured_t
or renews the measurement's age allowance. Without compensation, measured_t = t0.

Virtual time defaults to ExecutionSafety(max_plan_age_s=LEASE_S) for the
measurement-age limit. ExecutionSafety(max_plan_age_s=None) disables that age
limit, but the finite plan horizon still bounds the lease.
Both loops check the active plan's lease before evaluating its command law. At
or after expiry, they send no further command from that plan and end the run with
the fault "lease_expired".

Real-time execution checks every plan's effort limits before activation and every
command before sending it. Virtual time applies its selected effort policy after
the control law.

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
from tiago_mpc.laws import (LEASE_S, EffortClip, ExecutionSafety, FrictionFeedforward, PlanFeedback,
                            RealTimeEffort, RealTimeSafety, RejectOnEffort, SampleKnot, decide_command, validate_plan)
from tiago_mpc.interfaces import (Callback, Command, ControlLaw, EffortHandling, Event, Fault, MessageCallback,
                                  MpcInterface, Plan, PlantInterface, Prediction, SolveCompleted, State, TaskInterface,
                                  SafetyInterface, Tick)
from tiago_mpc.prediction import LatencyCompensation, _Predictor, _tick_at_or_after
from tiago_mpc.util import hz_to_s

_TOLERANCE = 1e-9  # absorbs the rounding of summed periods
# Not validated for long runs. Records stored in memory and GC disabled.
# Only raise after verification or handling these issues yourself.
MAX_RUNTIME_MIN = 10

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


# The late-plan event of both loops: knot 0 holds the correction each new plan makes,
# and a plan that first takes effect a knot interval or more after its state never runs
# it (docs/robot.md, Latency). Remedies (thesis workspace, research/tiago-mpc-latency-
# analysis.md, "How other MPC systems apply their plans"): (a) feedback around the plan
# at the command rate, e.g. u = u_plan(t) + Kp (q_plan(t) - q) + Kd (qd_plan(t) - qd) on
# the plan's knot states (OCS2: Riccati gains; BiConMP: joint impedance); (b) latency
# compensation: solve from the state predicted at the plan's application time under the
# committed torques and date the plan from then (Findeisen and Allgöwer; Mastalli et al.).
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
    settings: RealTimeSettings,
    duration: float,
    compensation: LatencyCompensation | None | Literal["default"] = "default",
    safety: SafetyInterface = ExecutionSafety(),
    effort_handling: EffortHandling | None = RejectOnEffort(),
) -> Outcome:
    """Run task on plant on wall time until task.done(state, t) or for duration seconds:
    the robot (RosPlant), or a rehearsal of it with a simulated arm (SimPlant).

    Time is the computer's monotonic clock (WallClock), and everything takes its real time:
    the solve, the plant, the tick callbacks. MpcProcess solves concurrently while the loop
    keeps ticking, and a plan is used at the first tick after it arrives; on the robot
    a state's own age adds to the delay. With settings=RealTimeSettings(update_period=.01), compensation=None,
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

    Ticks are settings.command_period apart. A late tick, one that starts a command period or
    more after its deadline because the work before it ran past the next tick, skips
    the missed ticks and is reported as a late_tick event, unless it follows the first
    command (on the robot, that command switches to effort control through blocking
    service calls, whose duration is the activated event's switch_s).

    The task's time starts at 0 with the first state: done and the duration see the
    tick's time, and at, which gives a request its reference, sees the time of the state
    the request starts from. Before the first tick, mpc.warmup runs
    from the started plant's state, the arm still held, so its time counts against no
    plan's lease. Each plan is checked when it arrives (validate_plan); decide_command
    checks the active plan's lease and refuses commands beyond the raw effort limits.
    A fault ends the run; its tick sends nothing, and a faulting plan never takes effect
    (the tick carries it as Tick.faulted,
    so the record keeps it). Polling result() at every tick also
    notices a dead MPC process within one tick (it raises). Each plan that takes effect a
    knot interval or more after its state, so that its knot 0 never runs, is a late_plan
    event (docs/robot.md, Latency). The last tick, which sees the task done, the time up
    or STOP set, sends no command but reaches the tick callbacks. Garbage collection pauses
    while the loop runs: a full collection over the buffered ticks takes several
    milliseconds. Duration must be finite nonnegative seconds, and a duration over
    MAX_RUNTIME_MIN raises before the plant starts.

    Default compensation forecasts the previous request-to-result delay, seeded
    from the update period or, when zero, the command period. None turns it off.
    Optional compensation predicts the solve's initial condition without changing the
    measured State. Early replies wait for their predicted origin, late replies retain
    their age. The physical propagation model and delay forecast are explicit settings.
    A failed forecast ends as prediction_unavailable; the current valid command and
    callbacks may already have run before that request-time fault is detected.
    User safety and effort policies may add restrictions; the 100 ms lease and
    raw effort checks always apply. EffortClip is refused before startup.
    """

    # This is written as inline code for clarity. The cost is duplication with the virtual-time loop and potentially more maintenance.
    # Time will tell if this decision should be revisited.

    if duration is None or not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite nonnegative seconds")
    if duration > 60 * MAX_RUNTIME_MIN:
        raise ValueError("Not validated for long runs. Records stored in memory and GC disabled.")
    if compensation == "default":
        compensation = default_compensation(settings)
    real_safety, real_effort = RealTimeSafety(safety), RealTimeEffort(effort_handling)
    _check_setup(mpc, law, compensation)
    clock, due = WallClock(), Schedule(settings.update_period)
    plan, solving, start, deadline = None, False, None, None
    predictor = (None if compensation is None else
                 _Predictor(compensation, real_safety, real_effort))
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
            elif t >= deadline + settings.command_period:  # late: the work before it ran past the next tick
                if sent != 1:  # the tick after the first command: the activated event has its delay
                    events.append(Event(t, "late_tick", f"a command tick started {1e3 * (t - deadline):.1f} ms after "
                                                        f"its deadline, {1e3 * settings.command_period:g} ms ticks: the "
                                                        "missed ticks were skipped",
                                        dict(late_s=t - deadline,
                                             skipped=math.floor((t - deadline) / settings.command_period + _TOLERANCE))))
                deadline = t  # skip the missed ticks
            start = s.t if start is None else start
            done, stopped = task.done(s, t - start), STOP.is_set()
            end = done or stopped or t - start >= duration - _TOLERANCE
            new, fault, faulted = mpc.result(), None, None  # a plan that arrived since the last tick
            if new is not None:
                solving, fault = False, validate_plan(new)
                if predictor is not None:
                    predictor.arrived(clock.now() - requested_at)
                    if fault is None:
                        waiting, new = new, None
                if fault is None:
                    if new is not None:
                        plan, plans = new, plans + 1
                else:
                    new, faulted = None, new  # it never takes effect
            if waiting is not None and fault is None:
                # User safety runs only when this plan can supply a command.
                fault = ExecutionSafety().validate(waiting, t, s)
                if fault is not None:
                    faulted, waiting = waiting, None
                elif t >= waiting.t0 - _TOLERANCE:
                    new, waiting = waiting, None
                    plan, plans = new, plans + 1
            command = None
            decision: Command | Fault | None = None
            if plan is not None and not end and fault is None:
                if new is not None and plan.index(t - plan.t0) >= 1:
                    events.append(_late_plan(plan, t, plans - 1))
                decision = decide_command(plan, t, s, law, safety=real_safety,
                                          effort_handling=real_effort)
                if isinstance(decision, Fault):
                    fault = decision
                else:
                    command = decision.applied
                    sent_at = clock.now() if predictor is not None else 0.
                    plant.apply(command)
                    if predictor is not None:
                        predictor.sent(sent_at, command)
                    sent += 1
            update = not end and fault is None and not solving and waiting is None and due(t)
            tick = Tick(t, s, command, plan, new is not None, update, (*events, *plant.take_events()), faulted,
                        request_tick if new is not None or faulted is not None else None,
                        None if decision is None else decision.requested,
                        isinstance(decision, Command) and decision.clipped)
            for callback in tick_callbacks:
                callback(tick)
            if fault is not None:
                return Outcome("fault", fault.reason, t - start)
            if end:
                return _ended(done, stopped, t - start)
            if update:  # the command is out: now ask for a new plan
                request_tick = t
                if predictor is None:
                    mpc.request(s, *task.at(s.t - start, mpc.N, mpc.dt))
                else:
                    now = clock.now()  # after publication and callbacks; prediction work follows
                    next_tick = _tick_at_or_after(now + 1e-9, deadline + settings.command_period,
                                                 settings.command_period)
                    at = _tick_at_or_after(now + predictor.latency_s, next_tick, settings.command_period)
                    predicted = (None if at >= s.t + LEASE_S else
                                 predictor.predict(s, now, at, plan, law, command, next_tick,
                                                   settings.command_period))
                    if predicted is None:
                        return Outcome("fault", "prediction_unavailable", clock.now() - start)
                    requested_at = clock.now()
                    mpc.request(predicted, *task.at(predicted.t - start, mpc.N, mpc.dt))
                solving = True
            deadline += settings.command_period
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
    settings: VirtualTimeSettings,
    duration: float,
    compensation: LatencyCompensation | None = None,
    safety: SafetyInterface = ExecutionSafety(),
    effort_handling: EffortHandling | None = EffortClip(),
    start_time_s: float = 0.,
    message_callbacks: Sequence[MessageCallback] = (),
    before_tick: Callable[[State], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    initial_command: np.ndarray | None = None,
) -> Outcome:
    """Run a controlled simulation on virtual time until the task or duration ends.

    Three quantities determine a run's timing: the command period, the control-update
    schedule and the solve latency. settings holds these separately; none need equal
    the MPC knot interval. The simulator's own computing and observer work consume
    wall time, but do not advance the virtual clock. In run_real_time, all that work
    consumes the actual time on which commands and requests are scheduled.

    This loop advances VirtualClock in settings.command_period increments. A solve
    is requested at a command tick; its plan becomes available at the request time
    plus settings.solve_latency_s. Commands continue from the previous active plan
    until the first tick at or after that availability time. With compensation, the
    plan must also have reached its predicted origin, plan.t0. Before the first
    plan, the plant remains on its initial hold or initial_command.

    For example, an in-process solve requested at virtual time 4 ms might block the
    computer for 3.5 ms. The next command tick is still at virtual time 6 ms:

      computer        [tk][tk][tk][  solve 3.5 ms  ][tk][tk][tk]
      wall clock      0    2    4                   7.5  9.5  11.5
      virtual time    0    2    4                   6    8    10

    The tick work is schematic. The solve's wall duration only determines when its
    plan becomes available if the experiment explicitly selects measured latency.

    settings.solve_latency_s chooses that availability rule:
      a number    Declared seconds, finite and nonnegative. Availability is request
                  time plus that delay, whatever the solve actually took. A solve
                  taking longer than a positive declared delay emits an overrun
                  event without changing its virtual schedule. Zero is the ideal
                  case and never emits an overrun.
      "measured"  The returned plan.stats["wall_s"], in seconds. Each solve's native
                  measured duration determines its own availability; preparation,
                  physics and observer work remain outside this delay. Even a zero
                  measured duration activates no earlier than the next command tick.

    settings.update_period controls requests, independently of solve latency. A
    positive period defines request slots; a slot is used at the first eligible
    command tick. At most one solve or returned pending plan is outstanding, so
    missed request slots are skipped. Zero requests another solve at the tick the
    previous plan becomes active. There is no queue of future control updates.

    VirtualTimeSettings supplies the common presets. The diagrams below use 2 ms
    command ticks, no compensation and an unexpired previous plan. R marks a
    request; P marks an active plan. Before the first plan, "old" is the initial hold.

      ideal(): solve_latency_s=0. Each request's plan is available at its own tick;
      the loop waits for the result without advancing virtual time.

      constant_latency(solve_latency_s=.005): update_period=.01, with each plan
      available 5 ms after its request. The first eligible command tick is at 6 ms:

      time (ms)       0   2   4   6   8   10  12  14  16  18  20
      command ticks   |   |   |   |   |   |   |   |   |   |   |
      control updates R0                  R1                  R2
      solve           [---------]         [---------]
      in effect       old=========P0==================P1=======

      as_fast_as_possible(): update_period=0 and solve_latency_s="measured".
      If the first two solves take 3.5 and 4.1 ms, their plans start at 4 and 10 ms:

      time (ms)       0   2   4   6   8   10  12  14  16
      command ticks   |   |   |   |   |   |   |   |   |
      control updates R0      R1          R2
      solve           [------][-------]
      in effect       old=====P0==========P1===========

      constant_rate(): update_period=solve_latency_s=.01. Requests are 10 ms
      apart and each plan takes effect at the next request slot:

      time (ms)       0   2   4   6   8   10  12  14  16  18  20
      command ticks   |   |   |   |   |   |   |   |   |   |   |
      control updates R0                  R1                  R2
      solve           [-------------------[-------------------]
      in effect       old=================P0==================P1

    like(RealTimeSettings()) copies the real-time periods and declares a solve
    latency of one update period. Compensation is passed separately to this loop.

    An asynchronous MPC can solve while the plant integrates the independent
    interval under the previous command. The loop waits for a result only at its
    first possible activation tick. With unknown measured latency the lower bound
    is the next command tick, or the predicted origin if that is later. With zero
    declared latency and no future origin the loop waits immediately. Measured
    results still wait for their scheduled availability and predicted origin.
    An in-process Mpc stays synchronous; the loop does not create a worker for it.

    SimPlant(sync_real_time=True) can pace the virtual clock for watching. Falling
    behind emits pacing_behind and leaves the numerical timeline unchanged. A
    rehearsal instead uses SimPlant on run_real_time, where work delays commands.

    Warm-up runs before the first tick and is not charged to any plan's latency.
    Normal completion collects an outstanding solve without applying its plan or
    advancing physics, so a caller-owned MPC can be reused.

    effort_handling applies to the complete law's torque (EffortClip by
    default); None leaves effort unchanged. Finite plans still end at their horizon.
    The default ExecutionSafety additionally limits age from the measured observation;
    ExecutionSafety(max_plan_age_s=None) disables that age limit. The compensation
    model is configured independently of the plant.

    Tick callbacks retain command records. Message observers receive queued plant
    Events before the next command, SolveCompleted before activation, and Tick
    after the command decision. before_tick sees every command-boundary state and
    may block before result collection to pause virtual time. An asynchronous solve
    may finish meanwhile; its reply remains unread until the callback returns and
    ordinary result collection resumes. A Tick observer's pause request is seen
    at the next command boundary. Cancellation and STOP are checked once, after
    before_tick; an admitted tick completes its command and physics interval before
    the next checkpoint. Any still-unread reply is left to the caller's teardown.
    A cancelled run is never restarted; construct its replacement with
    a fresh cancellation flag.

    Duration must be finite nonnegative seconds. Runs longer than MAX_RUNTIME_MIN
    minutes are refused. Cyclic GC is disabled during the run, and its original
    policy is restored on exit. start_time_s supports anchored runs;
    task.at/done still receive time relative to this run's initial clock.
    """
    if duration is None or not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite nonnegative seconds")
    if duration > 60*MAX_RUNTIME_MIN:
        raise ValueError("Not validated for long runs. Records stored in memory and GC disabled.")
    if not math.isfinite(start_time_s):
        raise ValueError("start_time_s must be finite")
    if initial_command is not None:
        initial_command = np.array(initial_command, dtype=float)
        if initial_command.shape != (7,) or not np.isfinite(initial_command).all():
            raise ValueError("initial_command requires seven finite native torques")
    _check_setup(mpc, law, compensation)
    clock, due = VirtualClock(start_time_s), Schedule(settings.update_period)
    plan: Plan | None = None
    pending: Plan | None = None
    available, requested_at = start_time_s, start_time_s
    predictor = None if compensation is None else _Predictor(compensation, safety, effort_handling)
    updates, plans = 0, 0
    held_command = initial_command

    def emit(message: Tick | SolveCompleted | Event) -> None:
        for callback in message_callbacks:
            callback(message)

    solving = False
    request_inputs: tuple[State | Prediction, np.ndarray, dict | None] | None = None
    controller_wall_s = 0.
    earliest_result = start_time_s

    def collect_result(events: list[Event]) -> Plan:
        nonlocal solving, available, controller_wall_s
        began = perf_counter()
        solved = mpc.result(block=True)
        controller_wall_s += perf_counter()-began
        if solved is None:
            raise RuntimeError("MPC returned no plan for a blocking result")
        solving = False
        took = solved.stats["wall_s"]
        if not isinstance(took, (float, int)) or not math.isfinite(took) or took < 0:
            raise ValueError("MPC wall_s must be finite nonnegative seconds")
        if message_callbacks:
            assert request_inputs is not None
            captured, captured_reference, changed = request_inputs
            emit(SolveCompleted(requested_at, captured, captured_reference, changed, solved, controller_wall_s))
        latency = took if settings.solve_latency_s == "measured" else settings.solve_latency_s
        available = requested_at+latency
        if predictor is not None:
            predictor.arrived(latency)
        if 0. < latency < took:
            event = Event(requested_at, "overrun", f"a solve took {1e3*took:.1f} ms, longer than "
                          f"solve_latency_s={1e3*latency:g} ms",
                          dict(plan=updates-1, wall_s=took, solve_latency_s=latency))
            events.append(event)
            emit(event)
        return solved

    def take_ready_plan(t: float) -> Plan | None:
        nonlocal pending
        if pending is None or t < available-_TOLERANCE or t < pending.t0-_TOLERANCE:
            return None
        ready, pending = pending, None
        return ready

    plant.start(clock)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        if initial_command is not None:
            plant.apply(initial_command)
            if predictor is not None:
                predictor.sent(start_time_s, initial_command)
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])
        while True:
            s, t = plant.state(), clock.now()
            events = list(plant.take_events())
            # Events retain their physics-endpoint timestamps, but are delivered
            # at this command boundary before another actuator decision.
            for event in events:
                emit(event)
            if before_tick is not None:
                before_tick(s)
            if cancelled() or STOP.is_set():
                break
            if solving and t >= earliest_result-_TOLERANCE:
                pending = collect_result(events)
            done = task.done(s, t-start_time_s)
            end = done or t-start_time_s >= duration-_TOLERANCE
            new = take_ready_plan(t)
            if new is not None:
                plan = new
            fault: str | None = None
            update = False
            plan_request_t = requested_at if new is not None else None
            # Forecast and application reuse this decision unless a new plan replaces it.
            decision: Command | Fault | None = None
            if pending is None and not solving and not end and due(t):
                update, updates = True, updates+1
                solve_started = perf_counter()
                initial: State | Prediction = s
                if predictor is not None:
                    if plan is not None:
                        decision = decide_command(plan, t, s, law, safety=safety,
                                                  effort_handling=effort_handling)
                        if isinstance(decision, Fault):
                            fault = decision.reason
                    at = _tick_at_or_after(t+predictor.latency_s, t, settings.command_period)
                    predicted = (None if fault is not None else predictor.predict(
                        s, t, at, plan, law,
                        decision.applied if isinstance(decision, Command) else held_command,
                        t+settings.command_period, settings.command_period))
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
                    latency_bound = (settings.command_period if settings.solve_latency_s == "measured" else
                                     settings.solve_latency_s)
                    earliest_result = max(t+latency_bound, initial.t)
                    mpc.request(initial, reference, changed)
                    solving = True
                    controller_wall_s = perf_counter()-solve_started
                    if t >= earliest_result-_TOLERANCE:
                        pending = collect_result(events)
                        ready = take_ready_plan(t)
                        if ready is not None:
                            new, plan, plan_request_t = ready, ready, requested_at
                            decision = None
            if new is not None:
                plans += 1
            command: np.ndarray | None = None
            requested = None if decision is None else decision.requested
            clipped = False
            if plan is not None and not end and fault is None:
                if new is not None and plan.index(t-plan.t0) >= 1:
                    event = _late_plan(plan, t, plans-1)
                    events.append(event)
                    emit(event)
                if decision is None:
                    decision = decide_command(plan, t, s, law, safety=safety, effort_handling=effort_handling)
                requested = decision.requested
                if isinstance(decision, Fault):
                    fault = decision.reason
                else:
                    command, clipped = decision.applied, decision.clipped
                    plant.apply(command)
                    held_command = command
                    if predictor is not None:
                        predictor.sent(t, command)
            if solving and (end or fault is not None):
                collect_result(events)
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
            if fault is not None:
                return Outcome("fault", fault, t-start_time_s)
            if end:
                return _ended(done, False, t-start_time_s)
            plant.advance(t+settings.command_period)
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
class VirtualTimeSettings:
    """Virtual command/update periods and solve latency, with common timing presets."""
    command_period: float  # Seconds between command ticks.
    update_period: float  # Seconds between control updates; zero requests whenever ready.
    solve_latency_s: float | Literal["measured"]  # Declared seconds or the solve's native wall time.

    def __post_init__(self) -> None:
        if not (self.solve_latency_s == "measured" or isinstance(self.solve_latency_s, (int, float))
                and math.isfinite(self.solve_latency_s) and self.solve_latency_s >= 0):
            raise ValueError("solve_latency_s must be seconds (finite, >= 0) or 'measured', "
                             f"not {self.solve_latency_s!r}")
        if not math.isfinite(self.command_period) or self.command_period <= 0:
            raise ValueError("command_period must be finite positive seconds")
        if not math.isfinite(self.update_period) or self.update_period < 0:
            raise ValueError("update_period must be finite nonnegative seconds")

    @classmethod
    def ideal(
        cls,
        command_period: float = hz_to_s(500),
        update_period: float = hz_to_s(100),
    ) -> VirtualTimeSettings:
        """Each plan in effect at the tick of its request."""
        return cls(command_period, update_period, 0.)

    @classmethod
    def constant_latency(
        cls,
        solve_latency_s: float = .05,
        command_period: float = hz_to_s(500),
        update_period: float = hz_to_s(100),
    ) -> VirtualTimeSettings:
        """Each plan available solve_latency_s seconds after its request."""
        return cls(command_period, update_period, solve_latency_s)

    @classmethod
    def as_fast_as_possible(cls, command_period: float = hz_to_s(500)) -> VirtualTimeSettings:
        """Each solve takes its measured time, and the next is requested when its plan takes effect."""
        return cls(command_period, 0., "measured")

    @classmethod
    def constant_rate(
        cls,
        period: float = hz_to_s(100),
        command_period: float = hz_to_s(500),
    ) -> VirtualTimeSettings:
        """A request every period, its plan in effect a period later."""
        return cls(command_period, period, period)

    @classmethod
    def like(cls, real: RealTimeSettings) -> VirtualTimeSettings:
        """Copy real's periods and declare each solve to last one control period."""
        return cls(real.command_period, real.update_period, real.update_period)


@dataclass(frozen=True)
class RealTimeSettings:
    """Wall-time command/update periods, defaulting to 500 Hz commands and 50 Hz updates."""
    command_period: float = hz_to_s(500)  # Seconds between command ticks.
    update_period: float = hz_to_s(50)  # Seconds between control updates; zero requests whenever ready.

    def __post_init__(self) -> None:
        if not math.isfinite(self.command_period) or self.command_period <= 0:
            raise ValueError("command_period must be finite positive seconds")
        if not math.isfinite(self.update_period) or self.update_period < 0:
            raise ValueError("update_period must be finite nonnegative seconds")


def default_compensation(settings: RealTimeSettings | VirtualTimeSettings) -> LatencyCompensation:
    """Forecast the declared solve delay or the previous request-to-result delay.

    Measured timing starts with one update period, or one command period when
    updates are requested whenever ready. Loops round the forecast onto their command grid.
    """
    if isinstance(settings, VirtualTimeSettings) and settings.solve_latency_s != "measured":
        return LatencyCompensation(latency_s=settings.solve_latency_s)
    return LatencyCompensation(latency_s="previous",
                               initial_latency_s=settings.update_period or settings.command_period)
