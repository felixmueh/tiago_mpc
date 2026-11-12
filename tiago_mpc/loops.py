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

Both loops apply effort handling after the control law. Real-time execution
always checks the requested and applied command against the raw effort limits.

The two loops are kept separate to improve readability of the real_robot loop.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import gc
import math
import threading
from typing import Literal
import warnings

import numpy as np

from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.laws import (EffortClip, ExecutionSafety, FrictionFeedforward, PlanFeedback,
                            RealTimeEffort, RealTimeSafety, RejectOnEffort, SampleKnot, decide_command)
from tiago_mpc.interfaces import (Command, ControlLaw, Diagnostic, DiagnosticCallback, EffortHandling, Fault,
                                  MpcInterface, Plan, PlantInterface, Prediction, SafetyInterface, SolveCompleted,
                                  SolveCompletedCallback, State, TaskInterface, Tick, TickCallback)
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
    """A periodic time cursor that advances through nominal slots and aligns action times.

    Zero period stays ready without imposing a grid. Alignment can start at an
    actual eligible boundary so floating-point command times retain their cadence.
    """

    def __init__(self, period: float, *, start_at: float = 0.) -> None:
        self.period, self.next_at = period, start_at

    def advance(self, *, past: float | None = None) -> None:
        """Advance the cursor, skipping positive-period slots through past without shifting their phase."""
        if self.period == 0:
            if past is not None:
                self.next_at = max(self.next_at, past)
            return
        slots = (1 if past is None else
                 max(1, math.floor((past-self.next_at+_TOLERANCE)/self.period)+1))
        self.next_at += slots*self.period

    def align(self, at: float) -> float:
        """Round at onto the grid starting at this cursor's next slot."""
        if self.period == 0:
            return max(at, self.next_at)
        slots = max(0, math.ceil((at-self.next_at-_TOLERANCE)/self.period))
        return self.next_at+slots*self.period


def _request_snapshot(
    state: State | Prediction,
    reference: np.ndarray,
    solver_updates: dict | None,
) -> tuple[State | Prediction, np.ndarray, dict | None]:
    """Detach the state, reference and solver updates submitted to MPC."""
    if isinstance(state, Prediction):
        state_snapshot: State | Prediction = Prediction(state.t, state.q, state.qd, measured_t=state.measured_t)
    else:
        state_snapshot = State(state.t, state.q.copy(), state.qd.copy(),
                               deepcopy(state.joint_positions), deepcopy(state.joint_velocities))
        state_snapshot.q.flags.writeable = state_snapshot.qd.flags.writeable = False
    reference_snapshot = np.array(reference, copy=True)
    reference_snapshot.flags.writeable = False
    return state_snapshot, reference_snapshot, deepcopy(solver_updates)


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


# The late-plan diagnostic of both loops: knot 0 holds the correction each new plan makes,
# and a plan that first takes effect a knot interval or more after its state never runs
# it (docs/robot.md, Latency). Remedies (thesis workspace, research/tiago-mpc-latency-
# analysis.md, "How other MPC systems apply their plans"): (a) feedback around the plan
# at the command rate, e.g. u = u_plan(t) + Kp (q_plan(t) - q) + Kd (qd_plan(t) - qd) on
# the plan's knot states (OCS2: Riccati gains; BiConMP: joint impedance); (b) latency
# compensation: solve from the state predicted at the plan's application time under the
# committed torques and date the plan from then (Findeisen and Allgöwer; Mastalli et al.).
def _late_plan(plan: Plan, t: float, plan_index: int) -> Diagnostic:
    """Report a plan that activates after its first knot."""
    return Diagnostic(t, "late_plan", f"a plan took effect {1e3 * (t - plan.t0):.1f} ms after its state, "
                                 f"{1e3 * plan.dt:g} ms knots: its knot 0 never runs (docs/robot.md, Latency)",
                 dict(plan=plan_index, delay_s=t - plan.t0))


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
    """Wall-time command/update periods, defaulting to 500 Hz commands and 100 Hz updates."""
    command_period: float = hz_to_s(500)  # Seconds between command ticks.
    update_period: float = hz_to_s(100)  # Seconds between control updates; zero requests whenever ready.

    def __post_init__(self) -> None:
        if not math.isfinite(self.command_period) or self.command_period <= 0:
            raise ValueError("command_period must be finite positive seconds")
        if not math.isfinite(self.update_period) or self.update_period < 0:
            raise ValueError("update_period must be finite nonnegative seconds")


def run_real_time(
    plant: PlantInterface,
    mpc: MpcInterface,
    task: TaskInterface,
    law: ControlLaw | None = None,  # None: FrictionFeedforward(), its defaults
    tick_callbacks: Sequence[TickCallback] = (),
    *,
    settings: RealTimeSettings = RealTimeSettings(),
    duration: float,
    compensation: LatencyCompensation | None | Literal["default"] = "default",
    safety: SafetyInterface = ExecutionSafety(),
    effort_handling: EffortHandling | None = RejectOnEffort(),
    solve_completed_callbacks: Sequence[SolveCompletedCallback] = (),
    diagnostic_callbacks: Sequence[DiagnosticCallback] = (),
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
    the missed ticks and is reported as a late_tick diagnostic, unless it follows the first
    command (on the robot, that command switches to effort control through blocking
    service calls, whose duration is the activated diagnostic's switch_s).

    The task's time starts at 0 with the first state: done and the duration see the
    tick's time, and at, which gives a request its reference, sees the time of the state
    the request starts from. Before the first tick, mpc.warmup runs
    from the started plant's state, the arm still held, so its time counts against no
    plan's lease. decide_command checks the active plan's lease and refuses requested
    or applied commands beyond the raw effort limits after evaluating the law.
    A fault ends the run and its tick sends nothing. A pending plan whose lease expires
    before activation never takes effect.
    Polling result() at every tick also
    notices a dead MPC process within one tick (it raises). Each plan that takes effect a
    knot interval or more after its state, so that its knot 0 never runs, is a late_plan
    diagnostic (docs/robot.md, Latency). The last tick, which sees the task done, the time up
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

    Command publication precedes all observer callbacks. Each callback group receives
    its own payload once: SolveCompleted for every received plan, Diagnostic for
    observations from the loop and plant, then Tick for the command decision.
    Request inputs are detached at every submission. Native solve latency is
    recorded in plan.stats['wall_s'].
    """

    # This is written as inline code for clarity. The cost is duplication with the virtual-time loop and potentially more maintenance.
    # Time will tell if this decision should be revisited.

    if duration is None or not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite nonnegative seconds")
    if duration > 60 * MAX_RUNTIME_MIN:
        raise ValueError("Not validated for long runs. Records stored in memory and GC disabled.")
    if compensation == "default":
        compensation = default_compensation(settings)
    law = FrictionFeedforward() if law is None else law
    real_safety, real_effort = RealTimeSafety(safety), RealTimeEffort(effort_handling)
    _check_setup(mpc, law, compensation)
    clock, update_schedule = WallClock(), Schedule(settings.update_period)
    active_plan, solve_outstanding, start_time_s, command_deadline = None, False, None, None
    predictor = (None if compensation is None else
                 _Predictor(compensation, real_safety, real_effort))
    # Prediction lookahead in seconds, fixed or updated from each completed request.
    forecast_delay = 0.
    if compensation is not None:
        forecast_delay = (compensation.initial_latency_s if compensation.latency_s == "previous" else
                          compensation.latency_s)
    pending_plan, requested_at, request_tick_time = None, 0., 0.
    last_mpc_request_snapshot: tuple[State | Prediction, np.ndarray, dict | None] | None = None
    commands_sent, plans_activated = 0, 0
    plant.start(clock)  # startup checks on a fresh state; the robot accepts wall time only

    # NOTE: IMPORTANT: Fine for exprs, but for real world use: DON'T use a GC language!
    gc.disable()  # for the whole run: full collections took 7-10 ms at 500 Hz
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        while True:
            s, t = plant.state(), clock.now()  # the newest state, and the tick's time
            diagnostics = plant.take_diagnostics()
            completed_solve: SolveCompleted | None = None
            if command_deadline is None:
                command_deadline = t  # the first tick
                update_schedule.next_at = t
            elif t >= command_deadline + settings.command_period:  # The work before it ran past the next tick.
                if commands_sent != 1:  # The first command's switch delay has its own diagnostic.
                    diagnostic = Diagnostic(t, "late_tick",
                                            f"a command tick started {1e3 * (t - command_deadline):.1f} ms after "
                                            f"its deadline, {1e3 * settings.command_period:g} ms ticks: the "
                                            "missed ticks were skipped",
                                            dict(late_s=t - command_deadline,
                                                 skipped=math.floor((t - command_deadline)
                                                                    / settings.command_period + _TOLERANCE)))
                    diagnostics.append(diagnostic)
                command_deadline = t  # skip the missed ticks
            start_time_s = s.t if start_time_s is None else start_time_s
            done, stopped = task.done(s, t - start_time_s), STOP.is_set()
            end = done or stopped or t - start_time_s >= duration - _TOLERANCE
            plan_activated = False
            received_plan, fault = mpc.result(), None
            if received_plan is not None:
                solve_outstanding = False
                if compensation is not None and compensation.latency_s == "previous":
                    # Include transport and command-tick polling in the observed delay.
                    forecast_delay = clock.now() - requested_at
                assert last_mpc_request_snapshot is not None
                state_snapshot, reference_snapshot, solver_updates = last_mpc_request_snapshot
                completed_solve = SolveCompleted(request_tick_time, state_snapshot, reference_snapshot,
                                                 solver_updates, received_plan)
                if predictor is not None:
                    pending_plan = received_plan
                else:
                    active_plan = received_plan
                    plan_activated = True
            if pending_plan is not None:
                # User safety runs only when this plan can supply a command.
                fault = ExecutionSafety().validate(pending_plan, t, s)
                if fault is None and t >= pending_plan.t0 - _TOLERANCE:
                    active_plan, pending_plan = pending_plan, None
                    plan_activated = True
            if plan_activated:
                plans_activated += 1
            command = None
            decision: Command | Fault | None = None
            if active_plan is not None and not end and fault is None:
                if plan_activated and active_plan.index(t - active_plan.t0) >= 1:
                    diagnostics.append(_late_plan(active_plan, t, plans_activated - 1))
                decision = decide_command(active_plan, t, s, law, safety=real_safety,
                                          effort_handling=real_effort)
                if isinstance(decision, Fault):
                    fault = decision
                else:
                    command = decision.applied
                    plant.apply(command)
                    commands_sent += 1
            request_update = (not end and fault is None and not solve_outstanding
                              and pending_plan is None and t >= update_schedule.next_at-_TOLERANCE)
            if request_update:
                update_schedule.advance(past=t)
            diagnostics.extend(plant.take_diagnostics())
            if completed_solve is not None:
                for solve_completed_callback in solve_completed_callbacks:
                    solve_completed_callback(completed_solve)
            for diagnostic in diagnostics:
                for diagnostic_callback in diagnostic_callbacks:
                    diagnostic_callback(diagnostic)
            tick = Tick(t, s, command, active_plan, plan_activated, request_update,
                        None if decision is None else decision.requested,
                        isinstance(decision, Command) and decision.clipped)
            for tick_callback in tick_callbacks:
                tick_callback(tick)
            if fault is not None:
                return Outcome("fault", fault.reason, t - start_time_s)
            if end:
                return _ended(done, stopped, t - start_time_s)
            if request_update:  # the command is out: now ask for a new plan
                request_tick_time = t  # Notifications retain the tick time even when preparation takes time.
                request_state: State | Prediction = s
                if predictor is not None:
                    now = clock.now()  # after publication and callbacks; prediction work follows
                    next_tick = _tick_at_or_after(now + 1e-9, command_deadline + settings.command_period,
                                                 settings.command_period)
                    prediction_time = _tick_at_or_after(now + forecast_delay, next_tick, settings.command_period)
                    predicted = predictor(s, now, prediction_time, active_plan, law, command, next_tick,
                                          settings.command_period)
                    if predicted is None:
                        return Outcome("fault", "prediction_unavailable", clock.now() - start_time_s)
                    request_state = predicted
                reference, solver_updates = task.at(request_state.t - start_time_s, mpc.N, mpc.dt)
                last_mpc_request_snapshot = _request_snapshot(request_state, reference, solver_updates)
                requested_at = clock.now()  # Measure delay from submission, after prediction and preparation.
                mpc.request(*last_mpc_request_snapshot)
                solve_outstanding = True
            command_deadline += settings.command_period
            plant.advance(command_deadline)  # the plant holds the command until then
    finally:
        try:
            plant.stop()  # zero torque and handback, first
        finally:
            gc.enable()


def run_virtual_time(
    plant: PlantInterface,
    mpc: MpcInterface,
    task: TaskInterface,
    law: ControlLaw | None = None,  # None: FrictionFeedforward(), its defaults
    tick_callbacks: Sequence[TickCallback] = (),
    *,
    settings: VirtualTimeSettings = VirtualTimeSettings.constant_rate(),  # the worst case: a solve takes a period
    duration: float,
    compensation: LatencyCompensation | None | Literal["default"] = "default",
    safety: SafetyInterface = ExecutionSafety(),
    effort_handling: EffortHandling | None = EffortClip(),
    start_time_s: float = 0.,
    solve_completed_callbacks: Sequence[SolveCompletedCallback] = (),
    diagnostic_callbacks: Sequence[DiagnosticCallback] = (),
    before_tick_callbacks: Sequence[Callable[[State], None]] = (),
    cancelled: Callable[[], bool] = lambda: False,
    initial_command: np.ndarray | None = None,
) -> Outcome:
    """Run a controlled simulation on virtual time until the task or duration ends.

    The command period, control-update schedule and solve latency are separate
    settings; none need equal the MPC knot interval. The plant owns physics
    integration and its substeps. The simulator's computing and observer work consume
    wall time, but do not advance the virtual clock. In run_real_time, all that work
    consumes the actual time on which commands and requests are scheduled.

    Each iteration executes one command tick. It samples the plant's state,
    collects and activates eligible earlier work, may submit one request, then
    collects any immediately eligible result before publishing its command.
    The plant then advances to the next command tick under the held command.
    A solve is requested at a command tick; its plan becomes available at the request time
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
                  diagnostic without changing its virtual schedule. Zero is the ideal
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
    latency of one update period. The defaults are constant_rate(), latency
    compensation from default_compensation(settings) (None turns it off) and
    FrictionFeedforward() as the law.

    An asynchronous MPC can solve while the plant integrates the independent
    interval under the previous command. The loop waits for a result only at its
    first possible activation tick. With unknown measured latency the lower bound
    is the next command tick, or the predicted origin if that is later. With zero
    declared latency and no future origin the loop waits immediately. Measured
    results still wait for their scheduled availability and predicted origin.
    An in-process Mpc stays synchronous; the loop does not create a worker for it.
    Collection and activation precede requests at the same command boundary;
    a newly submitted solve may also arrive and activate before that boundary's command.

    SimPlant(sync_real_time=True) can pace the virtual clock for watching. Falling
    behind emits pacing_behind and leaves the numerical timeline unchanged. A
    rehearsal instead uses SimPlant on run_real_time, where work delays commands.

    Warm-up runs before the first tick and is not charged to any plan's latency.
    Completion or a fault collects an outstanding solve after the terminal Tick without
    applying its plan or advancing physics, so a caller-owned MPC can be reused.

    effort_handling applies to the complete law's torque (EffortClip by
    default); None leaves effort unchanged. Finite plans still end at their horizon.
    The default ExecutionSafety additionally limits age from the measured observation;
    ExecutionSafety(max_plan_age_s=None) disables that age limit. The compensation
    model is configured independently of the plant.

    Each callback group receives its own payload once: Diagnostic for queued plant
    diagnostics before the next command, SolveCompleted before activation or after the
    terminal Tick for a drained solve, and Tick after the command decision.
    before_tick_callbacks receive the boundary state in order and may block before
    result collection to pause virtual time. An asynchronous solve
    may finish meanwhile; its reply remains unread until the callbacks return and
    ordinary result collection resumes. A Tick observer's pause request is seen
    at the next command boundary. Cancellation and STOP are checked once, after
    these callbacks; an admitted tick completes its command and physics interval before
    the next checkpoint. Any still-unread reply is left to the caller's teardown.
    A cancelled run is never restarted; construct its replacement with
    a fresh cancellation flag.

    Duration must be finite nonnegative seconds. Runs longer than MAX_RUNTIME_MIN
    minutes are refused. Cyclic GC is disabled during the run, and its original
    policy is restored on exit. start_time_s supports anchored runs;
    task.at/done still receive time relative to this run's initial clock.
    """

    # This is one big function on purpose so the control flow can be read from top to botton.

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
    if compensation == "default":
        compensation = default_compensation(settings)
    law = FrictionFeedforward() if law is None else law
    _check_setup(mpc, law, compensation)
    clock = VirtualClock(start_time_s)
    command_schedule = Schedule(settings.command_period, start_at=start_time_s)
    # Nominal MPC slots remain independent of command timing and outstanding work.
    update_schedule = Schedule(settings.update_period, start_at=start_time_s)
    active_plan: Plan | None = None
    pending_plan: Plan | None = None
    # Scheduled virtual availability from solve latency; the plan's origin may be later.
    plan_available_at, requested_at = start_time_s, start_time_s
    predictor = None if compensation is None else _Predictor(compensation, safety, effort_handling)
    # Prediction lookahead in seconds, fixed or updated from each completed request.
    forecast_delay = 0.
    if compensation is not None:
        forecast_delay = (compensation.initial_latency_s if compensation.latency_s == "previous" else
                          compensation.latency_s)
    requests_sent, plans_activated = 0, 0
    held_command = initial_command

    solve_outstanding = False
    last_mpc_request_snapshot: tuple[State | Prediction, np.ndarray, dict | None] | None = None
    collect_result_at = start_time_s

    def receive_result() -> tuple[Plan, float]:
        """Receive a completed solve and notify observers about it and any latency overrun."""
        solved_plan = mpc.result(block=True)
        if solved_plan is None:
            raise RuntimeError("MPC returned no plan for a blocking result")
        solve_wall_s = solved_plan.stats["wall_s"]
        latency = solve_wall_s if settings.solve_latency_s == "measured" else settings.solve_latency_s
        assert last_mpc_request_snapshot is not None
        state_snapshot, reference_snapshot, solver_updates = last_mpc_request_snapshot
        completed_solve = SolveCompleted(requested_at, state_snapshot, reference_snapshot,
                                         solver_updates, solved_plan)
        for solve_completed_callback in solve_completed_callbacks:
            solve_completed_callback(completed_solve)
        if 0. < latency < solve_wall_s:
            diagnostic = Diagnostic(requested_at, "overrun",
                                    f"a solve took {1e3*solve_wall_s:.1f} ms, longer than "
                                    f"solve_latency_s={1e3*latency:g} ms",
                                    dict(plan=requests_sent-1, wall_s=solve_wall_s, solve_latency_s=latency))
            for diagnostic_callback in diagnostic_callbacks:
                diagnostic_callback(diagnostic)
        return solved_plan, latency

    plant.start(clock)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        if initial_command is not None:
            plant.apply(initial_command)
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])
        while True:
            s, t = plant.state(), clock.now()
            # Physics diagnostics retain their endpoint times and arrive before the next command.
            for diagnostic in plant.take_diagnostics():
                for diagnostic_callback in diagnostic_callbacks:
                    diagnostic_callback(diagnostic)
            for before_tick_callback in before_tick_callbacks:
                before_tick_callback(s)
            if cancelled() or STOP.is_set():
                return Outcome("interrupted", "stop_requested", clock.now()-start_time_s)
            done = task.done(s, t-start_time_s)
            end = done or t-start_time_s >= duration-_TOLERANCE
            plan_activated = False
            fault: str | None = None
            # Forecast and application reuse this decision unless a new plan replaces it.
            decision: Command | Fault | None = None
            if solve_outstanding and collect_result_at-t <= _TOLERANCE:
                pending_plan, latency = receive_result()
                solve_outstanding = False
                plan_available_at = requested_at+latency
                if compensation is not None and compensation.latency_s == "previous":
                    forecast_delay = latency
            if pending_plan is not None and max(plan_available_at, pending_plan.t0)-t <= _TOLERANCE:
                active_plan, pending_plan = pending_plan, None
                plan_activated = True
            request_update = False
            can_request_update = (not solve_outstanding and pending_plan is None and not end
                                  and update_schedule.next_at-t <= _TOLERANCE)
            if can_request_update:
                request_state: State | Prediction = s
                update_schedule.advance(past=t)
                if predictor is not None:
                    if active_plan is not None:
                        decision = decide_command(active_plan, t, s, law, safety=safety,
                                                  effort_handling=effort_handling)
                        if isinstance(decision, Fault):
                            fault = decision.reason
                    prediction_time = command_schedule.align(t+forecast_delay)
                    predicted = (None if fault is not None else predictor(
                        s, t, prediction_time, active_plan, law,
                        decision.applied if isinstance(decision, Command) else held_command,
                        t+settings.command_period, settings.command_period))
                    if predicted is None:
                        fault = fault or "prediction_unavailable"
                    else:
                        request_state = predicted
                if fault is None:
                    requests_sent += 1
                    reference, solver_updates = task.at(request_state.t-start_time_s, mpc.N, mpc.dt)
                    last_mpc_request_snapshot = _request_snapshot(request_state, reference, solver_updates)
                    requested_at = t
                    # Measured latency is unknown until receipt; first integrate one command interval.
                    latency_bound = (settings.command_period if settings.solve_latency_s == "measured" else
                                     settings.solve_latency_s)
                    collect_result_at = max(t+latency_bound, request_state.t)
                    mpc.request(*last_mpc_request_snapshot)
                    solve_outstanding, request_update = True, True
                    # Zero latency can make the new result usable in this tick, before command publication.
                    if collect_result_at-t <= _TOLERANCE:
                        pending_plan, latency = receive_result()
                        solve_outstanding = False
                        plan_available_at = requested_at+latency
                        if compensation is not None and compensation.latency_s == "previous":
                            forecast_delay = latency
                        if max(plan_available_at, pending_plan.t0)-t <= _TOLERANCE:
                            active_plan, pending_plan = pending_plan, None
                            plan_activated = True
                            decision = None
            if plan_activated:
                plans_activated += 1

            command: np.ndarray | None = None
            requested_command = None if decision is None else decision.requested
            clipped = False
            if active_plan is not None and not end and fault is None:
                if plan_activated and active_plan.index(t-active_plan.t0) >= 1:
                    diagnostic = _late_plan(active_plan, t, plans_activated-1)
                    for diagnostic_callback in diagnostic_callbacks:
                        diagnostic_callback(diagnostic)
                if decision is None:
                    decision = decide_command(active_plan, t, s, law, safety=safety,
                                              effort_handling=effort_handling)
                requested_command = decision.requested
                if isinstance(decision, Fault):
                    fault = decision.reason
                else:
                    command, clipped = decision.applied, decision.clipped
                    plant.apply(command)
                    held_command = command
            if fault in ("effort_limit", "invalid_command") and requested_command is not None:
                rejected = Diagnostic(t, "command_rejected", f"command refused: {fault}",
                                      dict(reason=fault, requested_torque=(requested_command.tolist()
                                          if np.isfinite(requested_command).all() else None)))
                for diagnostic_callback in diagnostic_callbacks:
                    diagnostic_callback(rejected)
            tick = Tick(t, s, command, active_plan, plan_activated, request_update,
                        requested_command, clipped)
            for tick_callback in tick_callbacks:
                tick_callback(tick)
            if fault is not None or end:
                break
            command_schedule.advance()
            plant.advance(command_schedule.next_at)

        if solve_outstanding:
            receive_result()
        if fault is not None:
            return Outcome("fault", fault, t-start_time_s)
        return _ended(done, False, t-start_time_s)
    finally:
        try:
            plant.stop()
        finally:
            if gc_enabled:
                gc.enable()
            else:
                gc.disable()


def default_compensation(settings: RealTimeSettings | VirtualTimeSettings) -> LatencyCompensation:
    """Forecast the declared solve delay or the previous request-to-result delay.

    Measured timing starts with one update period, or one command period when
    updates are requested whenever ready. Loops round the forecast onto their command grid.
    """
    if isinstance(settings, VirtualTimeSettings) and settings.solve_latency_s != "measured":
        return LatencyCompensation(latency_s=settings.solve_latency_s)
    return LatencyCompensation(latency_s="previous",
                               initial_latency_s=settings.update_period or settings.command_period)
