"""Recording runs: every run, simulated or on the robot, is kept as a run directory, and
this module is the one that writes and reads it. (Small single purpose serialization library.)

- RunRecord records a run. Created before the run, it writes where the code came from
  (provenance()), the run's configuration, every command tick and completed solve;
  save() writes the files.
- load_run reads a saved run back with the same interface.
- Console prints the first diagnostic of each kind as it happens.
- to_dict and from_dict turn configurations (dataclasses) into JSON and back;
  they also work on their own, e.g. for a study's configuration.

Usually experiments.common.run_experiment creates the record, runs the loop and saves,
also after a fault or an error. By hand, given an mpc, a task (a Reach), a law, a
settings, a plant_config and the run directory out (None: _expr_data/<task type>/<UTC
time>/);

>>> import numpy as np
>>> import tiago_mpc as tm
>>> from experiments.reach import Reach
>>> safety, effort = tm.ExecutionSafety(), tm.EffortClip()
>>> record = tm.RunRecord(out, task=task, params=mpc.params, plant=plant_config, law=law,
...                       settings=settings, compensation=None, safety=safety,
...                       effort_handling=effort, duration=1.)  # every piece the loop gets
>>> plant = plant_config.build()
>>> console = tm.Console()
>>> try:
...     outcome = tm.run_virtual_time(plant, mpc, task, law, tick_callbacks=[record],
...                                   settings=settings, compensation=None, safety=safety,
...                                   effort_handling=effort, duration=1.,
...                                   solve_completed_callbacks=[record.solve_completed],
...                                   diagnostic_callbacks=[record.diagnostic, console])
...     record.metadata.update(status=outcome.status, reason=outcome.reason, end_t=outcome.t)
... finally:
...     for diagnostic in plant.take_diagnostics():
...         record.diagnostic(diagnostic)
...         console(diagnostic)
...     plant.close()
...     record.save()
>>> record.ticks["distance_m"] = np.array([task.distance(q) for q in record.ticks["q"]])
>>> record.metadata["results"]["final_distance_m"] = record.ticks["distance_m"][-1]
>>> record.save()

>>> run = tm.load_run(record.out, classes=[Reach])  # Reach: rebuild the task piece too
>>> list(run.config)
['task', 'params', 'plant', 'law', 'settings', 'compensation', 'safety', 'effort_handling', 'duration']
>>> run.config["settings"] == settings
True
>>> sorted(run.ticks)
['command', 'distance_m', 'plan', 'q', 'qd', 'state_t', 't', 'update']
>>> run.metadata["results"]
{'final_distance_m': ...}

>>> data = tm.to_dict(tm.VirtualTimeSettings.ideal())
>>> data
{'type': 'VirtualTimeSettings', 'command_period': 0.002, 'update_period': 0.01, 'solve_latency_s': 0.0}
>>> tm.from_dict(data) == tm.VirtualTimeSettings.ideal()
True
>>> real = tm.to_dict(tm.RealTimeSettings())
>>> real
{'type': 'RealTimeSettings', 'command_period': 0.002, 'update_period': 0.02}
>>> tm.from_dict(real) == tm.RealTimeSettings()
True
>>> tm.from_dict(tm.to_dict(tm.ExecutionSafety(None)))
ExecutionSafety(max_plan_age_s=None)

A run directory holds (FORMAT 3):
- summary.json: format; provenance (provenance()); config, the named pieces given to
  RunRecord, each to_dict-ed, written at the start; metadata: status, reason, end_t,
  results, error and plant (the plant's run_state);
- ticks.npz, one row per command tick: t, state_t, q, qd, command (NaN when none was
  sent), plan (the index of the plan in effect, -1 before the first), update (the index
  of the control update started at that tick, -1 for none), plus experiment columns;
- plans.npz, one row per completed solve in completion order: t0 (the predicted origin
  when compensated), measured_t (the original observation time), update_t (the request
  time from SolveCompleted),
  states (P, N, 14), torques (P, N - 1, 7), every key of plan.stats that no plan has as
  None, plus experiment columns. Returned plans remain recorded even if the run ends,
  their lease expires or another result supersedes them before activation; no tick's
  plan index points at an unused plan. Plan.stats contains solver statistics only;
- events.json: every diagnostic, a list of {"t", "kind", "message", "data"}.
Times count from the first state's time. The files are plain JSON and npz, readable
without this package. Importing this module loads no ROS and no compiled GATO module.
"""
from __future__ import annotations

from collections.abc import Iterable
import dataclasses
from datetime import datetime, timezone
from functools import cached_property
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.parse
import urllib.request

import gato
from gato.cost_menu import CostMenu, LinearTerm, Term
import numpy as np

from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh
from tiago_mpc.laws import (EffortClip, ExecutionSafety, FrictionFeedforward, HoldFirst, PlanFeedback,
                            RejectOnEffort, SampleKnot)
from tiago_mpc.loops import RealTimeSettings, VirtualTimeSettings
from tiago_mpc.mpc import AdmmLimits, MpcParams, SolverSettings
from tiago_mpc.prediction import LatencyCompensation
from tiago_mpc.interfaces import Diagnostic, Plan, SolveCompleted, Tick
from tiago_mpc.robot import Checks, RobotConfig
from tiago_mpc.safety import CollisionSafetySettings
from tiago_mpc.sim import SimConfig
from tiago_mpc.tiago import ROOT

FORMAT = 3
CLASSES: tuple[type, ...] = (  # the package's configuration classes that from_dict rebuilds
    SolverSettings, MpcParams, AdmmLimits, CostMenu, Term, LinearTerm,
    Friction, Tanh, Sign, StickSlip, SampleKnot, HoldFirst,
    PlanFeedback, FrictionFeedforward, EffortClip, RejectOnEffort, ExecutionSafety,
    VirtualTimeSettings, RealTimeSettings, LatencyCompensation, SimConfig, RobotConfig, Checks, CollisionSafetySettings,
)


def to_dict(obj: Any) -> Any:
    """dataclasses.asdict, adding each dataclass's type name, nested ones too:
    {"type": "Tanh", "width_rad_s": 0.03}. Tuples, lists and numpy arrays become lists,
    numpy scalars Python numbers, Paths strings; plain JSON values pass."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"type": type(obj).__name__, **{f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [to_dict(value) for value in obj]
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def from_dict(data: Any, classes: Iterable[type] = ()) -> Any:
    """The inverse of to_dict: rebuilds every {"type": ...} it finds from CLASSES and the
    classes given; JSON arrays become tuples; a type it does not know stays a plain dict.
    Two different classes of the same name among them raise."""
    known: dict[str, type] = {}
    for cls in (*CLASSES, *classes):
        if known.setdefault(cls.__name__, cls) is not cls:
            raise ValueError(f"two classes named {cls.__name__}: {known[cls.__name__]} and {cls}")
    return _rebuild(data, known)


def _rebuild(data: Any, known: dict[str, type]) -> Any:
    if isinstance(data, list):
        return tuple(_rebuild(value, known) for value in data)
    if not isinstance(data, dict):
        return data
    values = {key: _rebuild(value, known) for key, value in data.items()}
    cls = known.get(values["type"]) if isinstance(values.get("type"), str) else None
    if cls is None:
        return values
    del values["type"]
    return cls(**values)


def _git(*args: str, cwd: str | Path = ROOT) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def _gato_source() -> dict[str, Any]:
    """The GATO this environment runs: its installed version, the distribution's
    direct_url.json (None for a PyPI wheel; the path or URL of a local or git install) and
    gato.__file__; for an editable install also its checkout's commit and source status."""
    dist = importlib.metadata.distribution("gato-mpc")
    text = dist.read_text("direct_url.json")
    direct_url = json.loads(text) if text else None
    source: dict[str, Any] = dict(version=dist.version, direct_url=direct_url, file=gato.__file__)
    if direct_url and direct_url.get("dir_info", {}).get("editable"):
        checkout = urllib.request.url2pathname(urllib.parse.urlparse(direct_url["url"]).path)
        source.update(commit=_git("rev-parse", "HEAD", cwd=checkout), status=_git("status", "--short", cwd=checkout))
    return source


def provenance() -> dict[str, Any]:
    """Where the code came from: this repository's commit and source status, the package
    versions, the GATO this environment runs (_gato_source) and ROS_DISTRO (None without ROS)."""
    return dict(repository_commit=_git("rev-parse", "HEAD"), repository_status=_git("status", "--short"),
                versions={k: importlib.metadata.version(k) for k in ("gato-mpc", "grid-rbd", "pin", "numpy")},
                gato=_gato_source(), ros_distro=os.environ.get("ROS_DISTRO"))


class Console:
    """Print diagnostics to stderr.

    only_first prints only the first diagnostic of each kind.
    """

    def __init__(self, only_first: bool = True) -> None:
        self.only_first = only_first
        self.seen: set[str] = set()

    def __call__(self, diagnostic: Diagnostic) -> None:
        """Print one diagnostic according to only_first."""
        if not self.only_first:
            print(diagnostic.message, file=sys.stderr)
        elif diagnostic.kind not in self.seen:
            self.seen.add(diagnostic.kind)
            print(f"{diagnostic.message} (further {diagnostic.kind} diagnostics are recorded only)", file=sys.stderr)


class RunRecord:
    """Buffer command ticks, completed solves and diagnostics and save the run.

    out=None is _expr_data/<task type>/<UTC time>/ ("run" without a
    task piece). The pieces are the run's configuration, dataclasses or JSON values;
    the record writes provenance() and to_dict of each at once, with status "running".

    The read interface, which load_run gives a saved run too: config (the pieces as
    objects), ticks and plans (columns, built from the buffered ticks and completed
    solves on first read, after the loop), diagnostics (their times counted from the
    first state's) and metadata. An experiment adds columns to ticks and plans and
    results to metadata["results"], then saves again; save() refuses a column of the
    wrong length. Register solve_completed with the loop to record its returned plans.

    Not validated for long runs: every tick stays in memory until the run ends (about
    2 KB per tick in simulation, 4 GB for an hour at 500 Hz).
    """
    out: Path
    provenance: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]

    def __init__(self, out: str | Path | None, **pieces: Any):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        kind = type(pieces["task"]).__name__.lower() if "task" in pieces else "run"
        self.out = Path(out) if out else ROOT / "_expr_data" / kind / stamp
        self.out.mkdir(parents=True, exist_ok=False)
        self.provenance, self.config = provenance(), pieces
        self.metadata = dict(status="running", reason=None, end_t=None, results={}, error=None, plant={})
        self._ticks: list[Tick] = []
        self._plans: list[tuple[float, Plan]] = []
        self._diagnostics: list[Diagnostic] = []
        self._write_summary()

    def __call__(self, tick: Tick) -> None:
        self._ticks.append(tick)

    def solve_completed(self, completed: SolveCompleted) -> None:
        """Buffer a returned plan and its request time in completion order."""
        self._plans.append((completed.request_time_s, completed.plan))

    def diagnostic(self, diagnostic: Diagnostic) -> None:
        """Buffer one diagnostic on the loop's clock."""
        self._diagnostics.append(diagnostic)

    @property
    def diagnostics(self) -> list[Diagnostic]:
        """The diagnostics with times relative to the first recorded state."""
        t0 = self._ticks[0].state.t if self._ticks else 0.
        return [dataclasses.replace(diagnostic, t=diagnostic.t - t0) for diagnostic in self._diagnostics]

    @cached_property
    def ticks(self) -> dict[str, np.ndarray]:
        ticks = self._ticks
        t0 = ticks[0].state.t if ticks else 0.
        updates = np.array([k.update for k in ticks], dtype=bool)
        plan_rows = {id(plan): index for index, (_, plan) in enumerate(self._plans)}
        columns = dict(t=np.array([k.t - t0 for k in ticks]), state_t=np.array([k.state.t - t0 for k in ticks]),
                       q=np.array([k.state.q for k in ticks]).reshape(-1, 7),
                       qd=np.array([k.state.qd for k in ticks]).reshape(-1, 7),
                       command=np.array([np.full(7, np.nan) if k.command is None else k.command
                                         for k in ticks]).reshape(-1, 7),
                       plan=np.array([-1 if k.plan is None else plan_rows[id(k.plan)] for k in ticks], dtype=int),
                       update=np.where(updates, np.cumsum(updates) - 1, -1))
        return columns

    @cached_property
    def plans(self) -> dict[str, np.ndarray]:
        t0 = self._ticks[0].state.t if self._ticks else 0.
        plans = [plan for _, plan in self._plans]
        columns = dict(t0=np.array([p.t0 - t0 for p in plans]),
                       measured_t=np.array([p.measured_t - t0 for p in plans]),
                       update_t=np.array([request_time for request_time, _ in self._plans], dtype=float) - t0,
                       states=np.array([p.states for p in plans]), torques=np.array([p.torques for p in plans]))
        for key in (plans[0].stats if plans else ()):
            values = [p.stats[key] for p in plans]
            if all(v is not None for v in values):
                columns[key] = np.asarray(values)
        return columns

    def save(self) -> None:
        """Write summary.json, ticks.npz, plans.npz and events.json."""
        for name, rows in (("ticks", "t"), ("plans", "t0")):
            table: dict[str, Any] = getattr(self, name)
            for key, value in table.items():
                if len(value) != len(table[rows]):
                    raise ValueError(f"{name}[{key!r}] has {len(value)} rows, not {len(table[rows])}")
            np.savez_compressed(self.out / f"{name}.npz", **table)
        events = [dataclasses.asdict(diagnostic) for diagnostic in self.diagnostics]
        (self.out / "events.json").write_text(json.dumps(events, indent=1, default=_json) + "\n")
        self._write_summary()

    def _write_summary(self) -> None:
        summary = dict(format=FORMAT, provenance=self.provenance,
                       config={name: to_dict(piece) for name, piece in self.config.items()}, metadata=self.metadata)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json) + "\n")


def load_run(path: str | Path, classes: Iterable[type] = ()) -> RunRecord:
    """The run saved in path, with RunRecord's read interface: config rebuilt by from_dict
    from CLASSES and classes (a piece of another class stays a dict, e.g. a task without
    classes=[Reach]). Add columns or results and save() writes them back."""
    out = Path(path)
    summary = json.loads((out / "summary.json").read_text())
    if summary.get("format") != FORMAT:
        raise ValueError(f"{out} is a run of format {summary.get('format')}, not {FORMAT}: "
                         "load it with the code that wrote it (provenance.repository_commit)")
    record = RunRecord.__new__(RunRecord)
    record.out, record.provenance, record.metadata = out, summary["provenance"], summary["metadata"]
    record.config = from_dict(summary["config"], classes)
    with np.load(out / "ticks.npz") as ticks, np.load(out / "plans.npz") as plans:
        record.ticks, record.plans = dict(ticks), dict(plans)
    record._diagnostics = [Diagnostic(**event) for event in json.loads((out / "events.json").read_text())]
    record._ticks = []
    return record


def _json(value: Any) -> Any:
    # Replace with proper json encoder if this ever needs more.
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
