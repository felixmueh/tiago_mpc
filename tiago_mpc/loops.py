"""The command loop: one body, two clocks.

Once per command tick, run reads the plant's state, asks the MPC for plans,
turns the latest plan into a command with a control law and sends it to the
plant, which holds it until the next tick. run_robot uses each plan once it
arrives (wall clock); run_simulation charges each solve a latency on the
simulated clock, Fixed(s) or Measured().
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .laws import sample_knot
from .plan import Tick

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

    Without latency (wall clock) its plan is used once it arrives. On the simulated
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


def guard(command, plan, t, effort):
    """The command, if finite and within the raw effort limits, while the plan's lease lasts.

    The lease ends at t0 + min(100 ms, the plan's length), so no plan is commanded
    past its end: 31 x 2 ms knots span only 62 ms.
    """
    if t >= plan.t0 + min(LEASE_S, len(plan.torques) * plan.dt):
        raise RuntimeError("MPC command lease expired")
    command = np.asarray(command, dtype=float)
    if command.shape != np.shape(effort) or not np.isfinite(command).all() or np.any(np.abs(command) > effort):
        raise RuntimeError("invalid command or raw effort limit exceeded")
    return command


def run(plant, mpc, task, law=sample_knot, safety=None, callbacks=(), *,
        command_period, update_period, duration, latency=None):
    """Run task on plant until task.done(state, t) or for duration seconds.

    t is the time since the first state. Returns whether the task is done. The
    last tick, which sees the task done or the time up, sends no command but
    reaches the callbacks.
    """
    effort = np.asarray(mpc.details["effort_limits"], dtype=float)
    due, plan, pending = Schedule(update_period), None, None
    plant.start()  # startup checks on a fresh state
    try:
        start = plant.state().t
        while True:
            s = plant.state()
            if safety:
                safety.check(s)
            t = s.t - start
            done = task.done(s, t)
            end = done or t >= duration - _TOLERANCE
            # Without a pending solve, result() still notices a dead worker within one tick.
            new = pending.ready(s.t) if pending else mpc.result()
            if new is not None:
                plan, pending = new, None
            if pending is None and not end and due(s.t):
                mpc.request(s, task.reference(t, mpc.N, mpc.dt),
                            task.updates(t) if hasattr(task, "updates") else None)
                pending = Pending(mpc, s.t, latency)
                if (now := pending.ready(s.t)) is not None:  # zero latency: this tick
                    new, plan, pending = now, now, None
            command = None
            if plan is not None and not end:
                command = guard(law(plan, s.t, s), plan, s.t, effort)
                plant.apply(command)
            tick = Tick(s, command, plan, new is not None)
            for callback in callbacks:
                callback(tick)
            if end:
                return done
            plant.advance(s.t + command_period)  # robot: wait; simulation: integrate
    finally:
        plant.stop()  # zero torque and handback, first


def run_robot(*args, **kw):
    """Wall clock: the robot, Gazebo or SimPlant(realtime=True); plans are used on arrival."""
    return run(*args, latency=None, **kw)


def run_simulation(*args, latency=Fixed(), **kw):
    """Simulated clock: each solve is charged latency, Fixed(s) or Measured()."""
    return run(*args, latency=latency, **kw)
