"""Run 100 fixed simulated reaches and save each motion for analysis.

Use --pairs 8990 for one case or --dry-run to inspect the configuration.
"""
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Literal

import numpy as np

from experiments.common import handle_stop_signals, run_experiment
from experiments.reach import Reach, evaluate
import tiago_mpc as tm

HERE = Path(__file__).resolve().parent
PARAMS = replace(tm.TIAGO, max_sqp_time_s=None)  # a declared latency: no wall-time budget


@dataclass(frozen=True)
class Random100Config:
    """The ordered input IDs, manifest hash and components shared by all reaches."""
    ids: tuple[int, ...]
    inputs_sha256: str
    params: tm.MpcParams = PARAMS
    plant: tm.SimConfig = tm.SimConfig(substep=.003)
    settings: tm.VirtualTimeSettings = tm.VirtualTimeSettings.constant_rate()
    law: tm.ControlLaw = field(default_factory=tm.FrictionFeedforward)
    compensation: tm.LatencyCompensation | None | Literal["default"] = "default"
    safety: tm.ExecutionSafety = tm.ExecutionSafety()
    effort_handling: tm.EffortClip = tm.EffortClip()

    def __post_init__(self):
        if self.compensation == "default":
            object.__setattr__(self, "compensation", tm.default_compensation(self.settings))


@dataclass(frozen=True)
class Case:
    """One saved input's start state and grasp-frame position target."""
    name: str
    initial_state: tuple[float, ...]
    target_m: tuple[float, float, float]


@dataclass(frozen=True)
class Random100Study:
    config: Random100Config
    cases: tuple[Case, ...]


def cases(config):
    """Read start states and convert goal postures into grasp-frame targets."""
    out = []
    for task in config.ids:
        name = f"random_{task:05d}.npz"
        with np.load(HERE / "inputs" / name, allow_pickle=False) as a:
            out.append(Case(f"pair_{task:05d}", tuple(map(float, a["initial_state"])),
                            tuple(map(float, tm.ee_position(a["source_goal_q"])))))
    return tuple(out)


def load(path):
    """The study of a run directory and the run records of its cases that exist, in order."""
    path = Path(path)
    data = json.loads((path / "study.json").read_text())
    study = tm.from_dict(data["study"], [Random100Study, Random100Config, Case])
    return study, [tm.load_run(path / "runs" / c.name, classes=[Reach]) for c in study.cases
                   if (path / "runs" / c.name).exists()]


def main(argv=None):
    handle_stop_signals()  # Ctrl-C: the running case ends as interrupted, and the study stops
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pairs", type=int, nargs="+", help="run only these input IDs (default: all 100)")
    args = p.parse_args(argv)
    manifest = json.loads((HERE / "manifest.json").read_text())
    ids = manifest["task_ids"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected 100 distinct input IDs")
    if args.pairs:
        if len(set(args.pairs)) != len(args.pairs) or not set(args.pairs).issubset(ids):
            raise ValueError("Input IDs must be distinct members of the manifest")
        ids = args.pairs
    config = Random100Config(ids=tuple(ids), inputs_sha256=hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest())
    study = Random100Study(config, cases(config))
    if args.dry_run:
        print(json.dumps(tm.to_dict(config), indent=2))
        return 0
    out = (args.output or tm.ROOT / "_expr_data" /
           ("random100-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    binary = tm.compile_binary(config.params)
    out.mkdir(parents=True, exist_ok=False)
    (out / "study.json").write_text(json.dumps(dict(provenance=tm.provenance(), study=tm.to_dict(study)),
                                               indent=2) + "\n")
    for index, case in enumerate(study.cases):
        mpc, record = None, None
        try:
            mpc = tm.MpcProcess(config.params, binary)  # one process per case
            plant = replace(config.plant, q=case.initial_state[:7], qd=case.initial_state[7:])
            record = run_experiment(plant, mpc, Reach(case.target_m), config.law, config.settings,
                                    compensation=config.compensation, safety=config.safety,
                                    effort_handling=config.effort_handling,
                                    out=out / "runs" / case.name, case=case.name)
            evaluate(record)
        except Exception:  # an error is a result: run_experiment saved it; the other cases continue
            traceback.print_exc()
        finally:
            if mpc is not None:
                mpc.close()
        status = "error" if record is None else record.metadata["status"]
        print(f"{index+1}/{len(study.cases)} {case.name}: {status}", flush=True)
        if tm.STOP.is_set():
            break
    print(json.dumps(dict(output=str(out)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
