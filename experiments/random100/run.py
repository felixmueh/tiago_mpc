"""Run the historical random100 inputs once: one reach and one MpcProcess per case, each to
the grasp-frame position at the case's goal posture.

The study is its own frozen dataclasses: Random100Config, what defines it, and a Case per
input pair, what varies. study.json holds both and the provenance, written once before the
first run; each case's run record is runs/<name>/. load(path) reads them back.

Run from the checkout's root: python -m experiments.random100.run --dry-run (parameters only),
then without --dry-run for all 100 cases or with --pairs 8990 for a subset."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
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
    """What defines the study: the historical input pairs (inputs/random_<id>.npz, in this
    order), the sha256 of manifest.json, which pins the input files, and each run's pieces."""
    ids: tuple[int, ...]
    inputs_sha256: str
    params: tm.MpcParams = PARAMS
    plant: tm.SimConfig = tm.SimConfig(substep=.003)
    settings: tm.VirtualTimeSettings = tm.VirtualTimeSettings.like(tm.RealTimeSettings())
    compensation: tm.LatencyCompensation | None | Literal["default"] = "default"
    safety: tm.ExecutionSafety = tm.ExecutionSafety()
    effort_handling: tm.EffortClip = tm.EffortClip()

    def __post_init__(self):
        if self.compensation == "default":
            object.__setattr__(self, "compensation", tm.default_compensation(self.settings))


@dataclass(frozen=True)
class Case:
    """What varies: its run directory runs/<name>/, the start state (q, qd) at rest and the
    grasp-frame position at the goal posture."""
    name: str
    initial_state: tuple[float, ...]
    target_m: tuple[float, float, float]


@dataclass(frozen=True)
class Random100Study:
    config: Random100Config
    cases: tuple[Case, ...]


def cases(config):
    """One case per id, from its input file, checked: the cases follow from the config."""
    out = []
    for task in config.ids:
        name = f"random_{task:05d}.npz"
        with np.load(HERE / "inputs" / name, allow_pickle=False) as a:
            if a["initial_state"].shape != (14,) or a["reference"].shape != (1, 192):
                raise ValueError(f"Invalid task dimensions: {name}")
            if not np.allclose(a["reference"].reshape(32, 6)[:, :3], a["target_xyz"], atol=1e-6):
                raise ValueError(f"Nonconstant or inconsistent position target: {name}")
            if not np.allclose(a["target_xyz"], tm.fk(a["source_goal_q"], tm.FLANGE_FRAME), atol=1e-5, rtol=1e-4):
                raise ValueError(f"Target differs from the goal posture's flange position: {name}")
            # The inputs' flange targets only check their consistency; each case reaches the
            # grasp-frame position at the same goal posture instead (a new protocol).
            # The historical seed matches the state-hold, zero-torque warm start.
            if not np.allclose(a["first_seed"].ravel(), np.tile(np.r_[a["initial_state"], np.zeros(7)], 32)[:-7],
                               atol=1e-6):
                raise ValueError(f"Frozen task seed differs from the state-hold warm start: {name}")
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
    p.add_argument("--pairs", type=int, nargs="+", help="Declared diagnostic subset; default all100")
    args = p.parse_args(argv)
    manifest = json.loads((HERE / "manifest.json").read_text())
    ids = manifest["task_ids"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected the exact ordered 100 distinct historical task IDs")
    if args.pairs:
        if len(set(args.pairs)) != len(args.pairs) or not set(args.pairs).issubset(ids):
            raise ValueError("Diagnostic task IDs must be distinct members of the frozen100")
        ids = args.pairs
    config = Random100Config(ids=tuple(ids), inputs_sha256=hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest())
    study = Random100Study(config, cases(config))
    if args.dry_run:
        print(json.dumps(tm.to_dict(config), indent=2)); return 0
    out = (args.output or tm.ROOT / "_expr_data" /
           ("random100-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    # Participate in both the newer benchmark locks and the original experiment lock.
    locks = []
    for path in (tm.ROOT.parent / "GATO-upstream-contributions/.gpu.lock",
                 Path("/tmp/a2rlab-timing.lock"), tm.ROOT.parent / "GATO/.git/experiment-runner.lock"):
        if path.parent.exists():
            lock = path.open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
    binary = tm.compile_binary(config.params)
    tm.MpcProcess(config.params, binary).close()  # a shared setup failure (model check) stops here, before any case
    out.mkdir(parents=True, exist_ok=False)
    (out / "study.json").write_text(json.dumps(dict(provenance=tm.provenance(), study=tm.to_dict(study)),
                                               indent=2) + "\n")  # once, before the first run
    shutil.copyfile(__file__, out / "run.py")
    shutil.copyfile(HERE / "manifest.json", out / "input_manifest.json")
    shutil.copytree(HERE / "inputs", out / "inputs")
    for index, case in enumerate(study.cases):
        mpc, record = None, None
        try:
            mpc = tm.MpcProcess(config.params, binary)  # one process per case
            plant = replace(config.plant, q=case.initial_state[:7], qd=case.initial_state[7:])
            record = run_experiment(plant, mpc, Reach(case.target_m), tm.PlanFeedback(), config.settings,
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
        limits = None if record is None else record.metadata["results"].get("limits_pass")
        print(f"{index+1}/{len(study.cases)} {case.name}: {status}, limits={limits}", flush=True)
        if status == "interrupted":
            break
    print(json.dumps(dict(output=str(out)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
