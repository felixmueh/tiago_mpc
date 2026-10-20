#!/usr/bin/env python3
"""Summarize retained configuration attempts and first-stage prediction checks."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    root = args.output.resolve()
    model = pin.buildModelFromUrdf(str(ROOT / "models/tiago_right/tiago_right_arm.urdf"))
    data = model.createData()
    rows, diagnostics, regressions = [], [], []
    for directory in sorted(root.iterdir()):
        file = directory / "summary.json"
        if not file.is_file():
            continue
        summary = json.loads(file.read_text())
        if "cases" not in summary:
            regressions.append(dict(directory=directory.name, **summary))
            continue
        cases = summary["cases"]
        row = dict(profile=directory.name, **summary["counts"])
        stop = json.loads((directory / "STOP.json").read_text()) if (directory / "STOP.json").exists() else {}
        row["interrupted"] = len(stop.get("interrupted_tasks", []))
        row["held"] = len(stop.get("unexecuted_tasks", []))
        for key in ("position_rad", "speed_rad_s", "torque_nm"):
            row["max_"+key] = max((c.get("max_limit_excess", {}).get(key, 0) for c in cases), default=0)
        row["process_wall_s"] = sum(c["wall_s"] for c in cases)
        rows.append(row)
        for c in cases:
            folder = directory / f"pair_{c['task_id']:05d}"
            for file in sorted(folder.glob("plan_*.npz")):
                with np.load(file) as a:
                    signs = np.tile(a["signs"], 2)
                    xu = a["xu"][0]
                    x0, x1 = xu[:14]*signs, xu[21:35]*signs
                    u = xu[14:21]*a["signs"]
                    dt = c["dt_s"]
                    acc = pin.aba(model, data, x0[:7], x0[7:], u)
                    predicted = np.r_[x0[:7]+dt*x0[7:]+.5*dt*dt*acc, x0[7:]+dt*acc]
                    step = int(file.stem.split("_")[1])
                    with np.load(folder / "trajectory.npz") as raw:
                        actual_error = raw["prediction_error_qv"][step]
                    diagnostics.append(dict(profile=directory.name, task=c["task_id"], step=step,
                                            initial_error_q=float(np.max(abs(x0[:7]-a["state"][:7]))),
                                            initial_error_v=float(np.max(abs(x0[7:]-a["state"][7:]))),
                                            discrete_model_defect_q=float(np.max(abs(x1[:7]-predicted[:7]))),
                                            discrete_model_defect_v=float(np.max(abs(x1[7:]-predicted[7:]))),
                                            executed_prediction_error_q=float(actual_error[0]),
                                            executed_prediction_error_v=float(actual_error[1])))
    with (root / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (root / "first_stage_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    (root / "comparison.json").write_text(json.dumps(rows, indent=2)+"\n")
    (root / "regression_checks.json").write_text(json.dumps(regressions, indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
