"""Count arrivals and joint-position violations in saved reaches; no GPU.

Check every recorded position against the raw joint limits, with 1e-5 rad tolerance.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.random100.run import load
import tiago_mpc as tm


def summarize(study, records, lower, upper):
    """Count each reach once; missing or invalid positions remain unchecked."""
    arrived = sum(record.metadata["status"] == "arrived" for record in records)
    checked, violations = 0, 0
    for record in records:
        q = np.asarray(record.ticks["q"])
        if q.ndim != 2 or q.shape != (len(q), len(lower)) or not len(q) or not np.isfinite(q).all():
            continue
        checked += 1
        violations += bool(np.any((q < lower - 1e-5) | (q > upper + 1e-5)))
    return dict(cases=len(study.cases), arrived=arrived, joint_limit_violations=violations,
                checked=checked, unchecked=len(study.cases) - checked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    study, records = load(args.output)
    limits = tm.limits()
    counts = summarize(study, records, limits.lower, limits.upper)
    (args.output / "analysis.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(f"Arrived: {counts['arrived']}/{counts['cases']}")
    print(f"Joint-position limit violations: {counts['joint_limit_violations']}/{counts['checked']} checked reaches")
    if counts["unchecked"]:
        print(f"Unchecked reaches (missing or invalid positions): {counts['unchecked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
