#!/usr/bin/env python3
"""The two remedies for solve latency against stock sample_knot, on the simulation clock:
latency compensation (latency_compensation.py), feedback around the plan (plan_feedback.py)
and both together, over fixed latencies and four tasks. Each run once, serially (one GPU job).

    python experiments/latency_study.py --output _expr_data/latency-examples/<name>
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.figure8 import FIGURE8, figure8
from experiments.latency_compensation import run_compensated
from experiments.plan_feedback import PlanFeedback
from experiments.reach import Reach, reach
from tiago_mpc.laws import sample_knot
from tiago_mpc.loops import Fixed, run_simulation
from tiago_mpc.mpc import TIAGO, Mpc, compile_binary
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import FLANGE_FRAME, Q_START, fk, urdf_friction

COMMAND_PERIOD = .002  # the experiments' default; the control-update period is the knot interval
# "far" is T01 goal 3 of the latency analysis: the flange at Q_START plus this joint offset
# (31.7 cm there; Q_START's joint 4 has since moved from -2.35 to -2.33 rad).
# "reach-friction" runs the loss-free predictor against a plant with the URDF's friction;
# the figure eight predicts tanh friction against a plant with hard-sign friction.
T01_GOAL3 = np.array([.36, 1.2, 0., .45, 0., .15, 0.])
TASKS = {  # name: (experiment, task or None for its default, params, plant)
    "reach": (reach, Reach(), TIAGO, lambda: SimPlant(substep=.003)),
    "far": (reach, Reach(tuple(map(float, fk(Q_START + T01_GOAL3, FLANGE_FRAME))), ee_frame=FLANGE_FRAME),
            replace(TIAGO, ee_frame=FLANGE_FRAME), lambda: SimPlant(substep=.003)),
    "reach-friction": (reach, Reach(), TIAGO, lambda: SimPlant(substep=1e-4, friction=urdf_friction())),
    "figure8": (figure8, None, FIGURE8, lambda: SimPlant(substep=1e-4, friction=urdf_friction())),
}
CASES = {"stock": (False, sample_knot), "compensated": (True, sample_knot),  # name: (compensate, law)
         "feedback": (False, PlanFeedback()), "compensated+feedback": (True, PlanFeedback())}


def study(out, tasks, latencies_ms, cases, prediction_substep_s=None, predictor="gpu"):
    """Each task x latency x case with a fresh in-process Mpc; the uncompensated cases run on
    run_simulation itself. Returns the table rows."""
    rows = []
    for name in tasks:
        experiment, task, params, plant = TASKS[name]
        params = replace(params, max_sqp_time_s=None)  # no wall-time limit on the simulation clock
        binary = compile_binary(params)
        for ms in latencies_ms:
            for case in cases:
                compensate, law = CASES[case]
                run = (partial(run_compensated, latency_s=ms / 1e3, prediction_substep_s=prediction_substep_s,
                               predictor=predictor)
                       if compensate
                       else partial(run_simulation, latency=Fixed(ms / 1e3)))
                case_out = out / name / f"{name}_{case}_{ms:g}ms"
                mpc = Mpc(params, binary)
                try:
                    experiment(plant(), mpc, run=run, law=law, out=case_out, command_period=COMMAND_PERIOD,
                               **({} if task is None else dict(task=task)))
                except Exception as exc:  # a fault is a result: run_experiment saved it as "error"
                    print(f"{case_out.name}: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    mpc.close()
                rows.append(row(case_out, name, case, ms, compensate))
                print(json.dumps(rows[-1]), flush=True)
    return rows


def row(out, task, case, ms, compensate):
    """One run's outcome. The takeover error is the plant's state at the tick each plan took
    effect minus the state it was requested from: the prediction error when compensated
    (the loop's record), otherwise how far the arm moved during the latency. Not the plan's
    knot 0: GATO returns it only as close to the requested state as the solve converged."""
    summary = json.loads((out / "summary.json").read_text())
    with np.load(out / "trajectory.npz") as saved:
        time_s, state, command, t0 = saved["time_s"], saved["state"], saved["command"], saved["plan_t0_s"]
        if compensate:
            error = np.abs(saved["effect_state"] - saved["requested_state"])
        else:
            delay = math.ceil(ms / 1e3 / COMMAND_PERIOD - 1e-9) * COMMAND_PERIOD
            i, k = (np.searchsorted(time_s, t0 + d - 1e-9) for d in (0., delay))
            error = np.abs(state[k[k < len(time_s)]] - state[i[k < len(time_s)]])
    return dict(task=task, latency_ms=ms, case=case, status=summary["status"], end_s=round(float(time_s[-1]), 3),
                max_torque_nm=round(float(np.nanmax(np.abs(command), initial=0.)), 1),
                limits_pass=summary.get("limits_pass"),
                takeover_q_mrad=round(1e3 * float(error[:, :7].max(initial=0.)), 2),
                takeover_qd_rad_s=round(float(error[:, 7:].max(initial=0.)), 3),
                rmse_mm=(round(1e3 * summary["metrics_full_cycle"]["rmse_m"], 1)
                         if "metrics_full_cycle" in summary else None),
                fault=summary["error"].strip().splitlines()[-1] if summary.get("error") else None,
                output=str(out))


def table(rows):
    keys = ["task", "latency_ms", "case", "status", "end_s", "max_torque_nm", "limits_pass",
            "takeover_q_mrad", "takeover_qd_rad_s", "rmse_mm", "fault"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    return "\n".join(lines + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--latencies-ms", nargs="+", type=float, default=[0, 5, 10, 20, 30, 45])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--prediction-substep-s", type=float, help="compensated cases (default: one step per tick)")
    parser.add_argument("--predictor", choices=("gpu", "implicit"), default="gpu", help="compensated cases")
    parser.add_argument("--output", type=Path, required=True, help="a fresh directory")
    args = parser.parse_args(argv)
    rows = study(args.output, args.tasks, args.latencies_ms, args.cases, args.prediction_substep_s, args.predictor)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.output / "table.md").write_text(table(rows) + "\n")
    print(table(rows))


if __name__ == "__main__":
    sys.exit(main())
