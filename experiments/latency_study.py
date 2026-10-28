"""The two remedies for solve latency against stock SampleKnot, on virtual time:
latency compensation (latency_compensation.py), feedback around the plan (plan_feedback.py)
and both together, over fixed latencies and four tasks. Each run once, serially (one GPU job).

Run from the checkout's root:
    python -m experiments.latency_study --output _expr_data/latency-examples/<name>
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np

from experiments import figure8, reach
from experiments.common import handle_stop_signals, run_experiment
from experiments.latency_compensation import Compensated, first_tick_of_each_plan
import tiago_mpc as tm

COMMAND_PERIOD = tm.hz_to_s(500)  # the experiments' default; the control-update period is the knot interval
# "far" reaches the grasp-frame position at T01 goal 3's posture of the latency analysis,
# Q_START plus this joint offset (T01 measured the flange, 31.7 cm away with Q_START's joint 4
# at its former -2.35 rad).
# "reach-friction" runs the loss-free predictor against a plant with the URDF's friction;
# the figure eight predicts Tanh friction against a plant with StickSlip Coulomb friction.
T01_GOAL3 = np.array([.36, 1.2, 0., .45, 0., .15, 0.])
TASKS = {  # name: (the experiment's evaluate, task, params, plant)
    "reach": (reach.evaluate, reach.Reach(), tm.TIAGO, tm.SimConfig(substep=.003)),
    "far": (reach.evaluate, reach.Reach(tuple(map(float, tm.ee_position(tm.Q_START + T01_GOAL3)))), tm.TIAGO,
            tm.SimConfig(substep=.003)),
    "reach-friction": (reach.evaluate, reach.Reach(), tm.TIAGO,
                       tm.SimConfig(friction=tm.urdf_friction(), integrator="euler")),
    "figure8": (figure8.evaluate, figure8.Figure8(), figure8.FIGURE8,
                tm.SimConfig(friction=tm.urdf_friction(), integrator="euler")),
}
CASES = {"stock": (False, tm.SampleKnot()), "compensated": (True, tm.SampleKnot()),  # name: (compensate, law)
         "feedback": (False, tm.PlanFeedback()), "compensated+feedback": (True, tm.PlanFeedback())}


def study(out, tasks, latencies_ms, cases, prediction_substep_s=None, predictor="implicit", *, task_specs=None):
    """Each task x latency x case with a fresh in-process Mpc; the uncompensated cases run on
    run_virtual_time itself, with the latency declared. Returns the table rows, ending
    after an interrupted run (Ctrl-C)."""
    rows = []
    for name in tasks:
        evaluate, task, params, plant = (TASKS if task_specs is None else task_specs)[name]
        params = replace(params, max_sqp_time_s=None)  # declared latencies: no wall-time budget
        binary = tm.compile_binary(params)
        for ms in latencies_ms:
            for case in cases:
                compensate, law = CASES[case]
                timing = (Compensated(solve_latency_s=ms / 1e3, command_period=COMMAND_PERIOD,
                                      prediction_substep_s=prediction_substep_s, predictor=predictor) if compensate
                          else tm.VirtualTime.constant_latency(ms / 1e3, command_period=COMMAND_PERIOD))
                case_out = out / name / f"{name}_{case}_{ms:g}ms"
                mpc = tm.Mpc(params, binary)
                try:
                    evaluate(run_experiment(plant, mpc, task, law, timing, out=case_out))
                except Exception as exc:  # an error is a result: run_experiment saved it as "error"
                    print(f"{case_out.name}: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    mpc.close()
                rows.append(row(case_out, name, case, ms, compensate))
                print(json.dumps(rows[-1]), flush=True)
                if rows[-1]["status"] == "interrupted":
                    return rows
    return rows


def row(out, task, case, ms, compensate):
    """One run's outcome. The takeover error is the plant's state at the tick each plan took
    effect minus the state it was requested from: the prediction error when compensated
    (the plans' requested_state), otherwise how far the arm moved during the latency. Not
    the plan's knot 0: GATO returns it only as close to the requested state as the solve
    converged. fault: the fault's or interruption's reason, else an error's last line."""
    run = tm.load_run(out)
    ticks, plans, metadata = run.ticks, run.plans, run.metadata
    time_s, state, command, t0 = ticks["t"], np.c_[ticks["q"], ticks["qd"]], ticks["command"], plans["t0"]
    if compensate:
        effect = first_tick_of_each_plan(run)
        error = np.abs(state[effect] - plans["requested_state"]) if len(effect) else np.zeros((0, 14))
    else:
        delay = math.ceil(ms / 1e3 / COMMAND_PERIOD - 1e-9) * COMMAND_PERIOD
        i, k = (np.searchsorted(time_s, t0 + d - 1e-9) for d in (0., delay))
        error = np.abs(state[k[k < len(time_s)]] - state[i[k < len(time_s)]])
    results, reason, failure = metadata["results"], metadata["reason"], metadata["error"]
    return dict(task=task, latency_ms=ms, case=case, status=metadata["status"],
                end_s=round(float(time_s[-1]), 3) if len(time_s) else None,
                max_torque_nm=round(float(np.nanmax(np.abs(command), initial=0.)), 1),
                limits_pass=results.get("limits_pass"),
                takeover_q_mrad=round(1e3 * float(error[:, :7].max(initial=0.)), 2),
                takeover_qd_rad_s=round(float(error[:, 7:].max(initial=0.)), 3),
                rmse_mm=(round(1e3 * results["metrics_full_cycle"]["rmse_m"], 1)
                         if "metrics_full_cycle" in results else None),
                fault=reason or (failure.strip().splitlines()[-1] if failure else None),
                output=str(out))


def table(rows):
    keys = ["task", "latency_ms", "case", "status", "end_s", "max_torque_nm", "limits_pass",
            "takeover_q_mrad", "takeover_qd_rad_s", "rmse_mm", "fault"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    return "\n".join(lines + ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows])


def main(argv=None):
    handle_stop_signals()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--latencies-ms", nargs="+", type=float, default=[0, 5, 10, 20, 30, 45])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--prediction-substep-s", type=float, help="compensated cases (default: one step per tick)")
    parser.add_argument("--predictor", choices=("implicit", "gpu"), default="implicit", help="compensated cases")
    parser.add_argument("--output", type=Path, required=True, help="a fresh directory")
    args = parser.parse_args(argv)
    rows = study(args.output, args.tasks, args.latencies_ms, args.cases, args.prediction_substep_s, args.predictor)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    (args.output / "table.md").write_text(table(rows) + "\n")
    print(table(rows))


if __name__ == "__main__":
    sys.exit(main())
