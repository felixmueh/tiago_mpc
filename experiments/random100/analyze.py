#!/usr/bin/env python3
"""Saved-data audit and compact plots for a completed random100 cohort; no CUDA."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tiago_mpc.tiago import limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    saved = json.loads((out / "summary.json").read_text())
    config = json.loads((out / "configuration.json").read_text())
    rows = saved["cases"]
    if len(config["task_ids"]) != 100:
        raise ValueError("analyze.py expects a complete 100-case run, not a --pairs subset")
    if [r["task_id"] for r in rows] != config["task_ids"]:
        raise ValueError("Incomplete cohort or changed task order")
    raw = limits()
    audits = []
    total_work = Counter()
    inner_modes = Counter()
    native_us, wall_s = [], []
    for row in rows:
        task = row["task_id"]
        file = out / "inputs" / f"random_{task:05d}.npz"
        with np.load(file) as inputs, np.load(out / f"pair_{task:05d}/trajectory.npz") as raw:
            # The command sent at each tick but the last; NaN: none sent yet (zero torque held).
            t, x, u = raw["time_s"], raw["state"], np.nan_to_num(raw["command"][:-1])
            if len(t) == 0 or x.shape != (len(t), 14) or len(u) != len(t)-1:
                raise ValueError(f"Incomplete/invalid trajectory arrays: {task}")
            if not np.allclose(x[0], inputs["initial_state"], atol=1e-12):
                raise ValueError(f"Wrong initial state: {task}")
            finite = bool(np.isfinite(x).all() and np.isfinite(u).all())
            if not finite or np.any(np.diff(t) <= 0):
                raise ValueError(f"Nonfinite retained trajectory or invalid clock: {task}")
            q_ex = np.maximum(raw.lower-x[:, :7], x[:, :7]-raw.upper)
            v_ex = np.abs(x[:, 7:])-raw.velocity
            u_ex = np.abs(u)-raw.effort if len(u) else np.zeros((0, 7))
            maxima = {k: float(max(0, a.max(initial=0))) for k, a in
                      (("joint", q_ex), ("speed", v_ex), ("torque", u_ex))}
            complete = row["status"] in ("arrived", "timeout")
            actual_pass = complete and max(maxima.values()) <= 1e-5
            if complete and actual_pass != row["limits_pass"]:
                raise ValueError(f"Independent physical audit disagrees: {task}")
            if complete and not np.isclose(raw["distance_m"][-1], row["final_distance_m"], atol=1e-12):
                raise ValueError(f"Final error disagrees: {task}")
            first = {}
            for label, a, ts in (("joint", q_ex, t), ("speed", v_ex, t), ("torque", u_ex, t[:-1])):
                bad = np.argwhere(a > 1e-5)
                if len(bad):
                    i, j = bad[0]
                    first[label] = dict(time_s=float(ts[i]), joint=int(j+1), excess=float(a[i, j]))
            audits.append(dict(task_id=task, status=row["status"], reached=bool(row.get("reached")),
                               full_limit_pass=actual_pass, max_excess=maxima,
                               first_violations=first, command_ticks=len(t),
                               final_error_mm=1000*float(raw["distance_m"][-1])))
            native_us.extend(raw["plan_solve_time_us"].tolist())
            wall_s.extend(raw["plan_wall_s"].tolist())
            for key in ("sqp", "pcg", "admm", "admm_linsys", "zero_pcg_rows"):
                total_work[key] += int(raw[f"plan_{key}"].sum())
            inner_modes.update(raw["plan_admm_linsys_mode"].tolist())
    arrivals = [r for r in rows if r.get("reached")]
    valid = [r for r in arrivals if r.get("limits_pass")]
    first = rows[0]
    params = first["params"]
    setting_line = (f"N{params['N']}, B1, {1000*params['dt_s']:g} ms feedback/knots, "
                    f"{1000*first['plant']['substep_s']:g} ms Pinocchio RK4, "
                    f"SQP{params['solver']['max_sqp_iters']}/PCG{params['solver']['max_pcg_iters']}, "
                    f"{(params['N']-1)*params['dt_s']:.3g} s preview, "
                    f"terminal weight {params['solver']['N_cost']:g}, {first['task']['timeout_s']:g} simulated seconds.")
    admm_settings = first["params"]["admm"]
    bound_settings = {k: first["params"][k] for k in ("joint_margin_rad", "velocity_fraction", "effort_fraction")}
    result = dict(counts=saved["counts"], raw_audit_cases=len(audits),
                  violating_episodes={k:sum(a["max_excess"][k] > 1e-5 for a in audits)
                                      for k in ("joint", "speed", "torque")},
                  max_excess={k:max(a["max_excess"][k] for a in audits)
                              for k in ("joint", "speed", "torque")},
                  work_totals=dict(total_work), actual_admm_inner_modes=dict(inner_modes),
                  all_native_solve_time_s=float(np.sum(native_us)/1e6),
                  controller_wall_time_s=float(np.sum(wall_s)),
                  solve_time_median_ms=float(np.median(native_us)/1000),
                  episode_process_wall_sum_s=sum(r["wall_s"] for r in rows),
                  arrival_time_median_s=float(np.median([r["simulated_time_s"] for r in arrivals])) if arrivals else None,
                  timeout_ids=[r["task_id"] for r in rows if r["status"] == "timeout"],
                  error_ids=[r["task_id"] for r in rows if r["status"] not in ("arrived", "timeout")],
                  audits=audits)
    (out / "analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for group, label in ((arrivals, "All arrivals"), (valid, "Arrivals within limits")):
        times = sorted(r["simulated_time_s"] for r in group)
        axes[0].step([0, *times, 8], [0, *range(1, len(times)+1), len(times)], where="post", label=label)
    axes[0].set(xlabel="Simulation time (s)", ylabel="Arrivals / 100 assigned tasks", xlim=(0, 8), ylim=(0, 100))
    axes[0].legend()
    for status, color in (("arrived", "tab:blue"), ("timeout", "tab:orange"), ("error", "tab:red")):
        indices = [i for i, r in enumerate(rows) if r["status"] == status]
        if indices:
            axes[1].scatter([i+1 for i in indices], [audits[i]["final_error_mm"] for i in indices], s=16, label=status, color=color)
    axes[1].axhline(50, color="gray", ls="--", label="50 mm arrival threshold")
    axes[1].set(xlabel="Frozen task order", ylabel="Final position error (mm)", yscale="log")
    axes[1].legend()
    fig.suptitle(f"Frozen random100 — ADMM, N{params['N']}, SQP{params['solver']['max_sqp_iters']}, {1000*params['dt_s']:g} ms fixed clock")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=160); fig.savefig(out / "overview.pdf")
    counts = result["counts"]
    plt.close(fig)
    (out / "REPORT.md").write_text(f"""# Frozen random100 on modern GATO / ADMM

Source `{config['application_commit']}`; exact 100 inputs from `{config['input_source_commit']}`.
All tasks were attempted once in the original order, each from its own random
posture at rest. No comfortable-start substitution, filtering or retries.

| Outcome | Episodes / 100 |
| --- | ---: |
| Arrival within 8 simulated seconds | {counts['arrivals']} |
| Whole-run simulated physical-limit pass | {counts['full_limit_passes']} |
| Both arrival and physical-limit pass | {counts['valid_arrivals']} |
| Finite timeout | {counts['timeouts']} |
| Execution error | {counts['execution_errors']} |

Independent raw-trajectory checks agree with every completed episode's limit
label. Episodes exceeding joint position / speed / torque limits:
{result['violating_episodes']['joint']} / {result['violating_episodes']['speed']} /
{result['violating_episodes']['torque']}; overlaps are possible. Full per-task
excesses and first violations are in `analysis.json`.

![Arrival curve and terminal errors](overview.png)

Settings: {setting_line}
Position tolerance 5 cm, joint-speed L1 <= 1 rad/s, zero dwell. Position-only;
no final approach, collisions, clipping or physical hardware. Fixed simulation
pacing leaves solve latency unpaid. ADMM uses its finite upstream budget;
actual inner modes/work are retained. GATO `{first['gato_commit']}`, package
versions `{json.dumps(first['versions'])}` (the first case's summary).

Effective ADMM settings: `{json.dumps(admm_settings)}`.
Planning bounds: `{json.dumps(bound_settings)}`; the physical
URDF limits used for grading remain unchanged.

{config['differences_from_historical']}

This is a new cohort, not an isolated test of ADMM or an exact historical
reproduction. Arrival stops each successful episode, so this does not establish
post-arrival settling. Native/controller timings are diagnostic, not a quiet
before/after performance comparison. The saved raw histories include every
command tick, torque, solve timing, SQP/PCG/ADMM work and predicted violation.
No KKT residual gate or solver repair was used.

Regenerate from this repository:

```bash
.venv/bin/python experiments/random100/analyze.py {out}
```
""")
    print(json.dumps({k:v for k,v in result.items() if k != "audits"}, indent=2))


if __name__ == "__main__":
    main()
