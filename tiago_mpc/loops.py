"""The command loops: the execution controller, ticking at the command rate.

MPC runs in a command loop, on the robot and in simulation. On the robot time is
real: states arrive, solves take as long as they take, and plans are used when they
arrive. run_real_time runs on that time (wall time): on the robot, and in the
rehearsal of the robot's architecture with a simulated arm. In simulation we want
controlled experiments: how long a plan takes to arrive and how often plans are
requested are set by the experiment, so that runs are reproducible and comparable,
and can be varied on purpose. run_stepped runs them on time the loop keeps itself
(stepped time).

The command tick is the unit that ties a run together. Three parts run at rates of
their own, and they meet only at command ticks (defaults shown):

  time (ms)       0    2    4    6    8    10   12   14   16   18   20
  command ticks   |    |    |    |    |    |    |    |    |    |    |
                  each tick: the newest state in, one command out (500 Hz)
  control updates R0                       R1                       R2
                  a request to the MPC on every 5th tick (100 Hz)
  plant           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                  between ticks: holds the command; a SimPlant integrates
                  in substeps of its own (at most 1 ms), the robot moves

At each command tick (every command_period) a loop reads the plant's newest state,
takes the MPC's plan if a new one has become available, requests a control update
when one is due and no solve is outstanding, turns the newest plan into a command
with the control law and sends it. Between ticks nothing reaches the plant. A plan
holds knots dt apart (the knot interval); the knot interval, the update period and
the command period may all differ. State.t is when a state was measured, a plan
starts at its state's time (Plan.t0), and the lease, the schedules and the task's
clock all run on the run's time.

Control updates follow two rules, in both loops:
  requests  one at the first tick of each update slot (every update_period from
            the first tick) while no solve is outstanding; slots that pass during
            a solve are skipped. update_period=0: a request at every tick without
            an outstanding solve, so each update starts at the tick the previous
            plan takes effect (back to back).
  plans     a plan takes effect at the first tick at or after it becomes
            available; until then the previous plan stays in effect.
One solve is outstanding at a time.

Typical settings (dt: the knot interval; P: a period), as experiments.common.launch
builds them:
  run_stepped
    ideal                 SimPlant(), Mpc, solve_latency=0, update_period=dt
    constant latency      SimPlant(), Mpc, solve_latency=.05, update_period=dt
    as fast as possible   SimPlant(), Mpc, solve_latency="measured",
                          update_period=0, with the solver's wall-time budget
    constant control rate SimPlant(), Mpc, solve_latency=P, update_period=P
    watch                 SimPlant(sync_real_time=True), any of the above
  run_real_time, with MpcProcess and the solver's wall-time budget
    rehearsal             SimPlant()
    robot                 RosPlant(execute=True)

Both loops return an Outcome: whether the task is done, and the count of what went
wrong with the timing (late ticks, overruns), which run_experiment records.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import gc
import math
from typing import Literal
import warnings

import numpy as np

from .clock import Stepped, Wall
from .laws import sample_knot
from .plan import Callback, Law, MpcInterface, Plan, PlantInterface, TaskInterface, Tick
from .tiago import limits

LEASE_S = .1  # a plan may be commanded at most this long after its state's time
_TOLERANCE = 1e-9  # absorbs the rounding of summed periods
# The late-plan warning of both loops: knot 0 holds the correction each new plan makes,
# and a plan that first takes effect a knot interval or more after its state never runs
# it (docs/robot.md, Latency). Remedies (thesis workspace, research/tiago-mpc-latency-
# analysis.md, "How other MPC systems apply their plans"): (a) feedback around the plan
# at the command rate, e.g. u = u_plan(t) + Kp (q_plan(t) - q) + Kd (qd_plan(t) - qd) on
# the plan's knot states (OCS2: Riccati gains; BiConMP: joint impedance); (b) latency
# compensation: solve from the state predicted at the plan's application time under the
# committed torques and date the plan from then (Findeisen and Allgöwer; Mastalli et al.).


@dataclass(frozen=True)
class Outcome:
    """What a loop returns: whether the task is done, the ticks of run_real_time that
    started a command period or more late, and the solves of run_stepped that took
    longer than their declared solve_latency."""
    done: bool
    late_ticks: int = 0
    overruns: int = 0


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


def checked(plan: Plan | None) -> Plan | None:
    """The plan, if every knot is within the raw effort limits: any knot may be
    commanded before its lease ends, so each plan is checked when it arrives."""
    if plan is not None and np.any(np.abs(plan.torques) > limits().effort):
        raise RuntimeError("MPC plan exceeds the raw effort limit")
    return plan


def guard(command: np.ndarray, plan: Plan, t: float) -> np.ndarray:
    """The command, if finite and within the raw effort limits, while the plan's lease lasts.

    The lease ends at t0 + min(100 ms, the plan's length), so no plan is commanded
    past its end: 31 x 2 ms knots span only 62 ms.
    """
    if t >= plan.t0 + min(LEASE_S, len(plan.torques) * plan.dt):
        raise RuntimeError("MPC command lease expired")
    command, effort = np.asarray(command, dtype=float), limits().effort
    if command.shape != effort.shape or not np.isfinite(command).all() or np.any(np.abs(command) > effort):
        raise RuntimeError("invalid command or raw effort limit exceeded")
    return command


def run_real_time(plant: PlantInterface, mpc: MpcInterface, task: TaskInterface, law: Law = sample_knot,
                  callbacks: Sequence[Callback] = (), *, command_period: float, update_period: float,
                  duration: float) -> Outcome:
    """Run task on plant on wall time until task.done(state, t) or for duration seconds:
    the robot (RosPlant), or a rehearsal of it with a simulated arm (SimPlant).

    Time is the computer's monotonic clock (Wall), and everything takes its real time:
    the solve, the plant, the callbacks. MpcProcess solves concurrently while the loop
    keeps ticking, and a plan is used at the first tick after it arrives; on the robot
    a state's own age adds to the delay. update_period=.01:

      wall time (ms)  0    2    4    6    8    10   12   14   16   18
      command ticks   |    |    |    |    |    |    |    |    |    |
      requests        R0                       R1
      MPC process     [solve  ]                [solve    ]
      in effect       old=======P0============================P1=====
                                ^ arrived at 3.5 ms, used at 4 ms

    Each tick sends its command before it requests a plan, so an in-process Mpc, which
    solves inside the request, delays only the next tick: the ticks it misses
    meanwhile are late.

    Ticks are command_period apart. A late tick, one that starts a command period or
    more after its deadline because the work before it ran past the next tick, skips
    the missed ticks; the first one warns, once per run, unless it follows the first
    command (on the robot, that command switches to effort control through blocking
    service calls). Outcome.late_ticks counts them all.

    The task's t is counted from the first state's time: the tick's time for done and
    the duration, the state's time for at. Before the first tick, mpc.warmup runs
    from the started plant's state, the arm still held, so its time counts against no
    plan's lease. Each plan is checked when it arrives (checked), each command before
    it is sent (guard): within the plan's lease, t0 + min(LEASE_S = 0.1 s, its
    length), never past its last knot. Polling result() at every tick also notices a
    dead MPC process within one tick. The first plan that takes effect a knot
    interval or more after its state, so that its knot 0 never runs, warns, once per
    run (docs/robot.md, Latency). The last tick, which sees the task done or the time up,
    sends no command but reaches the callbacks. Garbage collection pauses while it
    runs: a full collection over the buffered ticks takes several milliseconds.
    """
    clock, due = Wall(), Schedule(update_period)
    plan, solving, start, deadline = None, False, None, None
    sent, late_ticks, warned_tick, warned_plan = 0, 0, False, False
    plant.start(clock)  # startup checks on a fresh state; the robot accepts wall time only
    # Paused for the whole run: full collections took 7-10 ms at 500 Hz. It must be
    # enabled in between (e.g. collecting at safe points); to be discussed (todo.org).
    gc.disable()
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        while True:
            s, t = plant.state(), clock.now()  # the newest state, and the tick's time
            if deadline is None:
                deadline = t  # the first tick
            elif t >= deadline + command_period:  # late: the work before it ran past the next tick
                late_ticks += 1
                if not warned_tick and sent != 1:  # sent == 1: the tick after the first command
                    warned_tick = True
                    warnings.warn(f"a command tick started {1e3 * (t - deadline):.1f} ms after its deadline, "
                                  f"{1e3 * command_period:g} ms ticks: the missed ticks were skipped "
                                  "(warned once per run; the summary counts late ticks)")
                deadline = t  # skip the missed ticks
            start = s.t if start is None else start
            done = task.done(s, t - start)
            end = done or t - start >= duration - _TOLERANCE
            new = checked(mpc.result())  # a plan that arrived since the last tick
            if new is not None:
                plan, solving = new, False
            command = None
            if plan is not None and not end:
                if new is not None and not warned_plan and plan.index(t - plan.t0) >= 1:
                    warned_plan = True
                    warnings.warn(f"a plan took effect {1e3 * (t - plan.t0):.1f} ms after its state, "
                                  f"{1e3 * plan.dt:g} ms knots: its knot 0 never runs "
                                  "(docs/robot.md, Latency; warned once per run)")
                command = guard(law(plan, t, s), plan, t)
                plant.apply(command)
                sent += 1
            tick = Tick(t, s, command, plan, new is not None)
            for callback in callbacks:
                callback(tick)
            if end:
                return Outcome(done, late_ticks=late_ticks)
            if not solving and due(t):  # the command is out: now ask for a new plan
                mpc.request(s, *task.at(s.t - start, mpc.N, mpc.dt))
                solving = True
            deadline += command_period
            plant.advance(deadline)  # the plant holds the command until then
    finally:
        try:
            plant.stop()  # zero torque and handback, first
        finally:
            gc.enable()


def run_stepped(plant: PlantInterface, mpc: MpcInterface, task: TaskInterface, law: Law = sample_knot,
                callbacks: Sequence[Callback] = (), *, command_period: float, update_period: float,
                duration: float, solve_latency: float | Literal["measured"]) -> Outcome:
    """Run task on plant on stepped time until task.done(state, t) or for duration
    seconds: a controlled experiment in simulation (SimPlant).

    Three effects set a simulated run's timing: how long a plan takes to arrive (its
    solve latency), when plans are requested (the update schedule, above), and how
    fast the simulator computes. This loop sets the first two and keeps the third out
    of the results.

    Stepped time. The loop keeps the time (Stepped) and moves it itself, tick by
    tick. Nothing takes time unless the loop writes it in, and it writes in two
    things: the ticks, command_period apart, and the solves, at solve_latency. The
    simulator's own computing and the loop's work take none; they only make the run
    slower or faster than real time. (On wall time, in run_real_time, everything
    takes its real time instead.) The loop waits for each solve, places its plan at
    the request plus the latency, and runs the ticks up to then with the previous
    plan. One in-process solve at the tick at 4 ms; the boxes show what the computer
    does, along its wall clock:

      computer        [tk][tk][tk][  solve 3.5 ms  ][tk][tk][tk]
      wall clock      0    2    4                   7.5  9.5  11.5
      stepped time    0    2    4                   6    8    10
                      the computing took no stepped time; on the timeline the
                      solve lasts solve_latency (its plan is placed at
                      4 + solve_latency), and the ticks at 6 and 8 run after it.

    solve_latency, required, says how long a solve lasts on the timeline:
      a number    declared, in seconds: the plan is available that long after the
                  request, whatever the solve really took. A solve that took longer
                  warns once (0, the explicit ideal, never does); Outcome.overruns
                  counts them.
      "measured"  this computer's own solve time (plan.stats["wall_s"]): the loop
                  waits for the solve, then uses its time as the declared one.

    Three settings, 2 ms command ticks:

      Declared latency, solve_latency=.005, update_period=.01:

      time (ms)       0    2    4    6    8    10   12   14   16   18   20
      command ticks   |    |    |    |    |    |    |    |    |    |    |
      requests        R0                       R1                       R2
      available                   ^ R0+5                    ^ R1+5
      in effect       old============P0=======================P1==========
                                     ^ the first tick at or after 5 ms

      As fast as possible, solve_latency="measured", update_period=0; the solves
      took 3.5 and 4.1 ms of wall time:

      time (ms)       0    2    4    6    8    10   12   14   16
      command ticks   |    |    |    |    |    |    |    |    |
      requests        R0        R1             R2
      available                ^ 3.5      ^ 4+4.1
      in effect       old=======P0=============P1===============
                                ^ R1 at the tick P0 takes effect

      Constant control rate, solve_latency=update_period=.01: a request at every
      slot, its plan in effect at the next (warns once if a solve took longer):

      time (ms)       0    2    4    6    8    10   12   14   16   18   20
      command ticks   |    |    |    |    |    |    |    |    |    |    |
      requests        R0                       R1                       R2
      in effect       old======================P0=======================P1

    Watching. SimPlant(sync_real_time=True) holds each tick until the wall clock has
    caught up, so the run can be watched at real speed; it warns once when it falls
    behind and never changes the results. It is not a rehearsal: a rehearsal runs a
    SimPlant on run_real_time.

    As in run_real_time: the task's t counts from the first state's time, the
    warm-up runs before the first tick, each plan is checked and each command
    guarded, the first plan that never runs its knot 0 warns, the last tick sends no
    command, and garbage collection pauses while it runs.
    """
    if not (solve_latency == "measured"
            or isinstance(solve_latency, (int, float)) and math.isfinite(solve_latency) and solve_latency >= 0):
        raise ValueError(f"solve_latency must be seconds (finite, >= 0) or 'measured', not {solve_latency!r}")
    clock, due = Stepped(), Schedule(update_period)
    plan, pending, available, start = None, None, 0., None  # pending: the solved plan
    overruns, warned_plan = 0, False
    plant.start(clock)
    gc.disable()  # full collections took 7-10 ms at 500 Hz
    try:
        mpc.warmup(plant.state(), task.at(0., mpc.N, mpc.dt)[0])  # the arm is still held
        while True:
            s, t = plant.state(), clock.now()
            start = s.t if start is None else start
            done = task.done(s, t - start)
            end = done or t - start >= duration - _TOLERANCE
            new = None
            if pending is not None and t >= available - _TOLERANCE:  # the first tick at or after it
                new, pending = pending, None
            # Between requests nothing is polled: a dead MpcProcess is noticed at the next request.
            if pending is None and not end and due(t):
                mpc.request(s, *task.at(s.t - start, mpc.N, mpc.dt))
                pending = checked(mpc.result(block=True))  # wait for it: no stepped time passes
                assert pending is not None
                took = pending.stats["wall_s"]
                latency = took if solve_latency == "measured" else solve_latency
                if 0. < latency < took:
                    overruns += 1
                    if overruns == 1:
                        warnings.warn(f"a solve took {1e3 * took:.1f} ms, longer than solve_latency="
                                      f"{1e3 * latency:g} ms (warned once per run; the summary counts overruns)")
                available = t + latency
                if t >= available - _TOLERANCE:  # zero latency: in effect at this tick
                    new, pending = pending, None
            if new is not None:
                plan = new
            command = None
            if plan is not None and not end:
                if new is not None and not warned_plan and plan.index(t - plan.t0) >= 1:
                    warned_plan = True
                    warnings.warn(f"a plan took effect {1e3 * (t - plan.t0):.1f} ms after its state, "
                                  f"{1e3 * plan.dt:g} ms knots: its knot 0 never runs "
                                  "(docs/robot.md, Latency; warned once per run)")
                command = guard(law(plan, t, s), plan, t)
                plant.apply(command)
            tick = Tick(t, s, command, plan, new is not None)
            for callback in callbacks:
                callback(tick)
            if end:
                return Outcome(done, overruns=overruns)
            plant.advance(t + command_period)  # the plant integrates to it, and the clock moves there
    finally:
        try:
            plant.stop()  # zero torque first
        finally:
            gc.enable()
