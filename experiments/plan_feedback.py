"""Feedback around the plan, as a control law: one remedy for solve latency.

SampleKnot sends only the plan's torques. Once a plan takes effect a knot interval or
more after its state, its knot 0, where each new solve's correction sits, never runs,
and nothing corrects the arm between solves (docs/robot.md, Latency). PlanFeedback adds
a joint-space PD on the plan's own knot states, at the command rate:

    u = u_plan(t) + kp (q_plan(t) - q) + kd (qd_plan(t) - qd)

so the arm is pulled back onto the trajectory the plan expects, whenever the plan took
effect. OCS2 feeds back with Riccati gains here; GATO returns none, hence a PD.
A law needs no loop of its own: it plugs into run_virtual_time, run_real_time and the
experiments (latency compensation, the other remedy, does: latency_compensation.py).

    reach(SimConfig(substep=.003), mpc, timing=VirtualTime.constant_latency(.02), law=PlanFeedback())

The main runs one reach in simulation as experiments/reach.py does, with this law.

Run from the checkout's root: python -m experiments.plan_feedback [--help]
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import sys

import numpy as np
import pinocchio as pin

from experiments.common import arguments, handle_stop_signals, launch
from experiments.reach import Reach, reach
import tiago_mpc as tm


def inertia_gains(bandwidth_rad_s, damping_ratio=1., q=tm.Q_START):
    """Per-joint (kp, kd) = (m w^2, 2 zeta m w), m the diagonal of the URDF arm's
    joint-space inertia at q: each joint alone, as a mass m, corrects its error as a
    second-order system with natural frequency w and damping ratio zeta. One scalar gain
    cannot fit all joints: m spans 0.39 (joint 2) to 0.0008 kg m^2 (joint 7) at Q_START."""
    model = pin.buildModelFromUrdf(str(tm.arm_urdf()))
    m = np.diag(pin.crba(model, model.createData(), np.asarray(q, dtype=float)))
    return tuple(m * bandwidth_rad_s**2), tuple(2 * damping_ratio * m * bandwidth_rad_s)


# 30 rad/s (about 5 Hz), critically damped, untuned: a starting point. Its time constant,
# 33 ms, is of the order of the latencies studied (0-45 ms), so an error is mostly
# corrected before the next plan takes effect, and w h = 0.06 at 2 ms command ticks, so
# the zero-order hold of each command barely delays it.
# TODO: the gains are untuned (30 rad/s, zeta = 1). Follow-ups: a small gain sweep, and the
# friction-plant test with the coefficients at +-50 %.
KP, KD = inertia_gains(30.)


@dataclass(frozen=True)
class PlanFeedback:
    """The plan's knot torque plus kp, kd (seven per-joint values or one scalar) on the
    error to the plan's state at t.

    Between knots the plan's state follows GATO's transcription, which holds each knot's
    torque and so a constant acceleration a = (qd_{k+1} - qd_k) / dt (check_model's step):
    q_plan = q_k + qd_k s + a s^2 / 2 and qd_plan = qd_k + a s, s = t - t_k. qd_plan is
    continuous; q_plan meets q_{k+1} only as far as the solve closed its defects."""
    kp: tuple[float, ...] | float = KP
    kd: tuple[float, ...] | float = KD

    def __call__(self, plan, t, state):
        k = plan.index(t - plan.t0)
        if k >= len(plan.torques):
            raise RuntimeError("time is past the plan's last knot, beyond its lease")
        s = min(max(t - plan.t0 - k * plan.dt, 0.), plan.dt)  # rounding at the knot boundary
        q, qd = plan.states[k, :7], plan.states[k, 7:]
        a = (plan.states[k + 1, 7:] - qd) / plan.dt
        q_plan, qd_plan = q + qd * s + .5 * a * s * s, qd + a * s
        return (plan.torques[k] + np.asarray(self.kp) * (q_plan - state.q)
                + np.asarray(self.kd) * (qd_plan - state.qd))


def main(argv=None):
    handle_stop_signals()
    args = arguments(__doc__, argv)
    if args.plant != "sim":
        raise SystemExit("plan_feedback.py runs in simulation only: the law is untested on the robot")
    task = Reach() if args.target is None else Reach(tuple(args.target))
    record = launch(args, partial(reach, task=task, law=PlanFeedback()), tm.SimConfig(substep=.003), tm.TIAGO,
                    tm.VirtualTime.ideal())
    if isinstance(record, int):
        return record
    return 0 if record.metadata["status"] == "arrived" and record.metadata["results"]["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
