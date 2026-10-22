"""Record: a callback that buffers each command tick in memory; save() writes the run.

- summary.json: commit and source status, status, parameters, plant model, task,
  and what the experiment adds to record.summary;
- trajectory.npz: per command tick its execution time (time_s), the state's time
  (state_time_s), state and command (NaN: none sent); per plan (plan_*) its start
  time, knots and solve stats; plus record.arrays. Times count from the first state's.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from functools import cached_property
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

import numpy as np

from .plan import Plant, Task, Tick
from .tiago import ROOT

if TYPE_CHECKING:
    from .mpc import MpcParams


def _git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


class Record:
    """Record(out, task, params, plant); out=None is _expr_data/<task>/<UTC time>."""

    def __init__(self, out: str | Path | None, task: Task, params: MpcParams, plant: Plant):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.out = Path(out) if out else ROOT / "_expr_data" / type(task).__name__.lower() / stamp
        self.out.mkdir(parents=True, exist_ok=False)
        self.task, self.params, self.plant = task, params, plant
        self.ticks: list[Tick] = []
        self.arrays: dict[str, np.ndarray] = {}  # the experiment's, saved with the trajectory
        self.summary = dict(
            status="running", repository_commit=_git("rev-parse", "HEAD"),
            repository_status=_git("status", "--short"),
            gato_commit=_git("rev-parse", "HEAD", cwd=ROOT / "external/gato"),
            versions={k: importlib.metadata.version(k) for k in ("gato-mpc", "grid-rbd", "pin", "numpy")},
            params=asdict(params), task=dict(type=type(task).__name__, **asdict(task)))

    def __call__(self, tick: Tick) -> None:
        self.ticks.append(tick)

    @cached_property
    def trajectory(self):
        """The buffered ticks as arrays, computed once after the run."""
        ticks = self.ticks
        t0 = ticks[0].state.t if ticks else 0.
        plans = [k.plan for k in ticks if k.new]
        out = dict(time_s=np.array([k.t - t0 for k in ticks]),
                   state_time_s=np.array([k.state.t - t0 for k in ticks]),
                   state=np.array([np.r_[k.state.q, k.state.qd] for k in ticks]).reshape(-1, 14),
                   command=np.array([np.full(7, np.nan) if k.command is None else k.command
                                     for k in ticks]).reshape(-1, 7),
                   plan_t0_s=np.array([p.t0 - t0 for p in plans]),
                   plan_states=np.array([p.states for p in plans]),
                   plan_torques=np.array([p.torques for p in plans]))
        for key in (plans[0].stats if plans else ()):
            values = [p.stats[key] for p in plans]
            if all(v is not None for v in values):
                out[f"plan_{key}"] = np.asarray(values)
        return out

    def save(self):
        trajectory = self.trajectory
        np.savez_compressed(self.out / "trajectory.npz", **trajectory, **self.arrays)
        self.summary.update(ticks=len(self.ticks), plans=len(trajectory["plan_t0_s"]),
                            plant=dict(type=type(self.plant).__name__, **self.plant.settings))  # after stop()
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2, default=_json) + "\n")


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
