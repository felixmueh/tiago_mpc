"""The command loop, on the plant's clock.

Once per command tick, run reads the plant's state and execution time, asks the
MPC for plans, turns the latest plan into a command with a control law and sends
it to the plant, which holds it until the next tick. The state's time is when it
was measured; plans, the lease and the schedules run on execution time. run_robot
uses each plan once it arrives (wall clock); run_simulation charges each solve a
latency on the simulation clock, Fixed(s) or Measured().
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import math

import numpy as np

from .laws import sample_knot
from .plan import Tick
from .tiago import limits

LEASE_S = .1  # a plan may be commanded at most this long after its state's time
_TOLERANCE = 1e-9  # absorbs the rounding of summed periods


@dataclass(frozen=True)
class Fixed:
    """Charge every solve s seconds; zero makes the plan take effect at its own tick."""
    s: float = 0.

    def charge(self, plan):
        return self.s


@dataclass(frozen=True)
class Measured:
    """Charge each solve its own wall time (MPCController.step; warm-up solves excluded)."""

    def charge(self, plan):
        return plan.stats["wall_s"]


class Schedule:
    """Control updates every period from the first call: due at the first call at or
    after each slot, at most once per call; missed slots are skipped."""

    def __init__(self, period):
        self.period, self.next = period, None

    def __call__(self, t):
        if self.next is None:
            self.next = t
        slots = math.floor((t - self.next) / self.period + _TOLERANCE)
        if slots < 0:
            return False
        self.next += (slots + 1) * self.period
        return True


class Pending:
    """The one outstanding solve.

    Without latency (wall clock) its plan is used once it arrives. On the simulation
    clock the loop waits for it, and it takes effect at the first tick at or after
    its request time plus the charged latency.
    """

    def __init__(self, mpc, t, latency):
        self.mpc, self.t, self.latency, self.plan = mpc, t, latency, None

    def ready(self, now):
        if self.latency is None:
            return self.mpc.result()
        if self.plan is None:
            self.plan = self.mpc.result(block=True)
            self.at = self.t + self.latency.charge(self.plan)
        return self.plan if now >= self.at - _TOLERANCE else None


def guard(command, plan, t):
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


def run(plant, mpc, task, law=sample_knot, callbacks=(), *,
        command_period, update_period, duration, latency=None):
    """Run task on plant until task.done(state, t) or for duration seconds.

    The task's t is counted from the first state's time: execution time for done
    and the duration, the state's time for its reference. Returns whether the task
    is done. Ticks are command_period apart; a late tick skips the missed ones.
    The last tick, which sees the task done or the time up, sends no command but
    reaches the callbacks. Garbage collection pauses while it runs: a full
    collection over the buffered ticks takes several milliseconds.
    """
    due, plan, pending = Schedule(update_period), None, None
    plant.start()  # startup checks on a fresh state
    # Paused for the whole run: full collections took 7-10 ms at 500 Hz. It must be
    # enabled in between (e.g. collecting at safe points); to be discussed (todo.org).
    gc.disable()
    try:
        start = deadline = None
        while True:
            s, t = plant.state(), plant.now()  # measurement and execution time
            start = s.t if start is None else start
            done = task.done(s, t - start)
            end = done or t - start >= duration - _TOLERANCE
            # Without a pending solve, result() still notices a dead MPC process within one tick.
            new = pending.ready(t) if pending else mpc.result()
            if new is not None:
                plan, pending = new, None
            if pending is None and not end and due(t):
                mpc.request(s, task.reference(s.t - start, mpc.N, mpc.dt),
                            task.updates(s.t - start) if hasattr(task, "updates") else None)
                pending = Pending(mpc, t, latency)
                if (now := pending.ready(t)) is not None:  # zero latency: this tick
                    new, plan, pending = now, now, None
            command = None
            if plan is not None and not end:
                command = guard(law(plan, t, s), plan, t)
                plant.apply(command)
            tick = Tick(t, s, command, plan, new is not None)
            for callback in callbacks:
                callback(tick)
            if end:
                return done
            if deadline is None or deadline + command_period <= t:
                deadline = t  # the first tick, or late: skip the missed ticks
            deadline += command_period
            plant.advance(deadline)  # the plant integrates to it, waits for it, or both
    finally:
        try:
            plant.stop()  # zero torque and handback, first
        finally:
            gc.enable()


def run_robot(plant, *args, **kw):
    """Wall clock: the robot, Gazebo or SimPlant(clock=Clock.wall()); plans are used on arrival."""
    if plant.clock.kind == "simulation":
        raise ValueError("run_robot needs a wall or external clock; use run_simulation")
    return run(plant, *args, latency=None, **kw)


def run_simulation(plant, *args, latency=Fixed(), **kw):
    """Simulated clock: each solve is charged latency, Fixed(s) or Measured()."""
    if plant.clock.kind != "simulation":
        raise ValueError("run_simulation needs a simulation clock; use run_robot")
    return run(plant, *args, latency=latency, **kw)
