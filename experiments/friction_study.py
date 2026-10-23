"""Compare the four latency execution controllers with constraint Coulomb friction.

    python -m experiments.friction_study --dry-run
    python -m experiments.friction_study --output _expr_data/controller-friction/<fresh-name>

Each cell runs once on the simulation clock, serially. The plant uses the URDF's
Coulomb and viscous coefficients scaled together by .5, 1 or 1.5, with 1 ms
semi-implicit steps. Predictors and feedback gains stay unchanged: reaching
predicts no friction; figure-eight tracking predicts nominal tanh friction.
Fixed 0/20 ms solve delays isolate delay tolerance; these are not wall-time
performance measurements or real-robot qualifications.
"""
from dataclasses import asdict, replace
import argparse
import fcntl
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from experiments.common import limit_excess
from experiments.latency_study import CASES, COMMAND_PERIOD, TASKS, study, table
import tiago_mpc as tm


def task_specs(scale):
    """Change only the hidden plant coefficients, leaving each predictor fixed."""
    nominal = tm.urdf_friction()
    friction = replace(nominal,
                       coulomb_nm=tuple(scale * x for x in nominal.coulomb_nm),
                       viscous_nm_s_per_rad=tuple(scale * x for x in nominal.viscous_nm_s_per_rad))
    return {name: (*TASKS[name][:3], partial(tm.SimPlant, friction=friction))
            for name in ("reach-friction", "figure8")}


def describe(scales, latencies, tasks, cases):
    return dict(cells=len(scales)*len(latencies)*len(tasks)*len(cases), repeats=1,
                plant_scales=scales, latencies_ms=latencies, tasks=tasks, cases=cases,
                command_period_s=COMMAND_PERIOD, update_period_s=.01,
                time_basis="simulation", predictor="implicit", prediction_substep_s=None,
                repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tm.ROOT, text=True).strip(),
                source_status=subprocess.check_output(["git", "status", "--short"], cwd=tm.ROOT, text=True),
                gato_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tm.ROOT / "external/gato", text=True).strip(),
                urdf_sha256=hashlib.sha256(tm.FULL_URDF.read_bytes()).hexdigest(),
                laws={name: asdict(law) if hasattr(law, "__dataclass_fields__") else law.__name__
                      for name, (_, law) in CASES.items() if name in cases},
                models={str(scale): {name: dict(params=asdict(replace(params, max_sqp_time_s=None)),
                                               plant=make_plant().settings)
                                    for name, (_, _, params, make_plant) in task_specs(scale).items() if name in tasks}
                        for scale in scales})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scales", type=float, nargs="+", default=[.5, 1., 1.5])
    parser.add_argument("--latencies-ms", type=float, nargs="+", default=[0., 20.])
    parser.add_argument("--tasks", choices=("reach-friction", "figure8"), nargs="+", default=["reach-friction", "figure8"])
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--output", type=Path, help="a fresh directory inside this checkout's _expr_data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for values in (args.scales, args.latencies_ms):
        if len(set(values)) != len(values) or not all(np.isfinite(x) and x >= 0 for x in values):
            parser.error("scales and latencies must be distinct, finite and nonnegative")
    for values in (args.tasks, args.cases):
        if len(set(values)) != len(values):
            parser.error("tasks and cases must be distinct")
    config = describe(args.scales, args.latencies_ms, args.tasks, args.cases)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required without --dry-run")
    out = args.output.resolve()
    if not out.is_relative_to(tm.ROOT / "_expr_data"):
        parser.error("--output must be inside this checkout's _expr_data")
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    for source in (Path(__file__), Path(__file__).with_name("latency_study.py"),
                   Path(__file__).with_name("latency_compensation.py"), Path(__file__).with_name("plan_feedback.py")):
        shutil.copyfile(source, out / source.name)
    locks = []
    try:
        # Same locks as random100 and the visualizer's Live scheduler.
        for path in (tm.ROOT.parent / "GATO-upstream-contributions/.gpu.lock", Path("/tmp/a2rlab-timing.lock"),
                     tm.ROOT.parent / "GATO/.git/experiment-runner.lock"):
            if path.parent.exists():
                lock = path.open("a")
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for scale in args.scales:
            batch = study(out / f"plant-{scale:g}x", args.tasks, args.latencies_ms, args.cases,
                          task_specs=task_specs(scale))
            for row in batch:
                row["plant_scale"] = scale
                summary = json.loads((Path(row["output"]) / "summary.json").read_text())
                with np.load(Path(row["output"]) / "trajectory.npz") as saved:
                    excess = limit_excess(saved)
                    row["sampled_limits_pass"] = max(excess.values()) <= 1e-5 if len(saved["state"]) else None
                    row["sampled_limit_excess"] = excess
                    row["finite_states"] = bool(np.isfinite(saved["state"]).all()) if len(saved["state"]) else None
                row["final_distance_mm"] = (None if "final_distance_m" not in summary
                                             else float(summary["final_distance_m"]) * 1000)
            rows.extend(batch)
            (out / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
            (out / "table.md").write_text("\n\n".join(
                f"Plant coefficients {s:g}x\n\n" + table([r for r in rows if r["plant_scale"] == s])
                for s in args.scales if any(r["plant_scale"] == s for r in rows)) + "\n")
    finally:
        for lock in locks:
            lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
