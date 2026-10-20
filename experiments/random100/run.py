#!/usr/bin/env python3
"""Run the historical random100 inputs once with the current TIAGo ADMM example."""
from datetime import datetime, timezone
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def aggregate(out, rows, expected):
    complete = [r for r in rows if r.get("status") in ("arrived", "timeout")]
    counts = dict(assigned=expected, attempted=len(rows), completed=len(complete),
                  arrivals=sum(bool(r.get("reached")) for r in rows),
                  full_limit_passes=sum(bool(r.get("limits_pass")) for r in complete),
                  valid_arrivals=sum(bool(r.get("reached")) and bool(r.get("limits_pass")) for r in complete),
                  timeouts=sum(r.get("status") == "timeout" for r in rows),
                  execution_errors=sum(r.get("status") not in ("arrived", "timeout") for r in rows))
    (out / "summary.json").write_text(json.dumps(dict(counts=counts, cases=rows), indent=2)+"\n")
    fields = ["task_id", "status", "reached", "limits_pass", "simulated_time_s",
              "initial_distance_m", "final_distance_m", "final_speed_l1_rad_s", "wall_s", "output"]
    with (out / "cases.csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config", type=Path, help="Explicit application configuration JSON")
    p.add_argument("--pairs", type=int, nargs="+", help="Declared diagnostic subset; default all100")
    args = p.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    ids = manifest["task_ids"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected the exact ordered 100 distinct historical task IDs")
    if args.pairs:
        if len(set(args.pairs)) != len(args.pairs) or not set(args.pairs).issubset(ids):
            raise ValueError("Diagnostic task IDs must be distinct members of the frozen100")
        ids = args.pairs
    for task in ids:
        name = f"random_{task:05d}.npz"
        file = HERE / "inputs" / name
        if digest(file) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
        with np.load(file, allow_pickle=False) as a:
            if a["initial_state"].shape != (14,) or a["reference"].shape != (1, 192):
                raise ValueError(f"Invalid task dimensions: {name}")
            if not np.allclose(a["reference"].reshape(32, 6)[:, :3], a["target_xyz"], atol=1e-6):
                raise ValueError(f"Nonconstant or inconsistent position target: {name}")
    config = dict(input_source_commit=manifest["source_commit"], task_ids=ids,
                  input_manifest_sha256=digest(HERE / "manifest.json"),
                  application_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  driver_sha256=digest(ROOT / "examples/reach_sim.py"),
                  configuration_overrides=json.loads(args.config.read_text()) if args.config else {},
                  configuration_sha256=digest(args.config) if args.config else None,
                  cohort="upstream-admm-sqp5-position-only-fixed-clock-8s",
                  differences_from_historical="Modern GATO/GRiD, ADMM limit rows, SQP5 (historical1000), u_cost=1e-4 (historical1e-5), rho reset each feedback (historical retained), upstream soft costs/bounds; same native starts/position goals/order.")
    if args.dry_run:
        print(json.dumps(config, indent=2)); return 0
    out = (args.output or ROOT.parent / "experiment_data/tiago_mpc" /
           ("random100-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    # Participate in both the newer benchmark locks and the original experiment lock.
    locks = []
    for path in (ROOT.parent / "GATO-upstream-contributions/.gpu.lock",
                 Path("/tmp/a2rlab-timing.lock"), ROOT.parent / "GATO/.git/experiment-runner.lock"):
        if path.parent.exists():
            lock = path.open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    shutil.copyfile(__file__, out / "run.py")
    shutil.copyfile(HERE / "manifest.json", out / "input_manifest.json")
    if args.config:
        shutil.copyfile(args.config, out / "config.json")
    shutil.copytree(HERE / "inputs", out / "inputs")
    rows = []
    for index, task in enumerate(ids):
        case = out / f"pair_{task:05d}"
        cmd = [sys.executable, str(ROOT / "examples/reach_sim.py"), "--input",
               str(out / "inputs" / f"random_{task:05d}.npz"), "--output", str(case)]
        if args.config:
            cmd.extend(["--config", str(out / "config.json")])
        start = time.perf_counter()
        with (out / f"pair_{task:05d}.log").open("w") as log:
            try:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        path = case / "summary.json"
        row = json.loads(path.read_text()) if path.exists() else dict(status="execution_error", error="Process returned without summary")
        if row.get("status") in ("started", "building_model", "checking_model", "running"):
            row.update(status="execution_error", error=f"Process interrupted/aborted, exit {rc}")
        row.update(task_id=task, process_returncode=rc, wall_s=time.perf_counter()-start, output=str(case))
        rows.append(row)
        counts = aggregate(out, rows, len(ids))
        print(f"{index+1}/{len(ids)} id={task}: {row['status']}, limits={row.get('limits_pass')}; "
              f"arrivals={counts['arrivals']} valid={counts['valid_arrivals']} errors={counts['execution_errors']}", flush=True)
        if row.get("execution_stage") != "episode":
            print("Shared setup failure: remaining tasks held. See case log.", flush=True)
            return 2
    print(json.dumps(dict(output=str(out), counts=counts), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
