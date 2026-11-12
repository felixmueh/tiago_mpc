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

    record = run_experiment(tm.SimConfig(substep=.003), tm.TIAGO, Reach(), tm.PlanFeedback(),
                            tm.VirtualTimeSettings.constant_latency(.02), compensation=None)
    evaluate(record)  # reach's

The main runs one reach in simulation as experiments/reach.py does, with this law.

Run from the checkout's root: python -m experiments.plan_feedback [--help]
"""
from __future__ import annotations

import json
import sys

from experiments.common import arguments, handle_stop_signals, run_experiment, setup
from experiments.reach import Reach, evaluate
import tiago_mpc as tm


def main(argv=None):
    handle_stop_signals()
    args = arguments(__doc__, argv, plants=("sim",))  # the law is untested on the robot
    task = Reach() if args.target is None else Reach(tuple(args.target))
    plant, settings, loop = setup(args, tm.SimConfig(substep=.003), tm.VirtualTimeSettings.ideal())  # sim: never a status
    record = run_experiment(plant, tm.TIAGO, task, tm.PlanFeedback(), settings, loop=loop,
                            compensation=None, duration=args.duration, out=args.output)
    evaluate(record)
    print(json.dumps(record.metadata, indent=2, default=str))
    print(f"Saved {record.out}")
    return 0 if record.metadata["status"] == "arrived" and record.metadata["results"]["limits_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
