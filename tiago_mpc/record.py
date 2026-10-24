"""The format of a run, the one module that writes and reads it: to_dict and from_dict
for configurations, provenance(), the events (Event, Console), RunRecord, the writer,
and load_run, the reader.

A run directory holds (FORMAT 1):
- summary.json: format; provenance (provenance()); config, the named pieces given to
  RunRecord, each to_dict-ed, written at the start; metadata: status, reason, end_t,
  results, error and plant (the plant's run_state);
- ticks.npz, one row per command tick: t, state_t, q, qd, command (NaN when none was
  sent), plan (the index of the plan in effect, -1 before the first), update (the index
  of the control update started at that tick, -1 for none), plus experiment columns;
- plans.npz, one row per plan: t0, update_t (the t of the tick whose update started it),
  states (P, N, 14), torques (P, N - 1, 7), every key of plan.stats that no plan has as
  None, plus experiment columns; a plan that faulted is the last row (Tick.faulted), and
  no tick's plan index points at it;
- events.json: every event, a list of {"t", "kind", "message", "data"}.
Times count from the first state's time. The files are plain JSON and npz, readable
without this package. Importing this module loads no ROS and no compiled GATO module;
nothing in the package imports it (Event lives in plan.py, beside Tick).
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
from gato import SolverParams
import numpy as np

from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh
from tiago_mpc.laws import HoldFirst, SampleKnot
from tiago_mpc.loops import RealTime, VirtualTime
from tiago_mpc.mpc import AdmmLimits, MpcParams
from tiago_mpc.plan import Event, Tick
from tiago_mpc.robot import Checks, RobotConfig
from tiago_mpc.safety import CollisionSafetySettings
from tiago_mpc.sim import SimConfig
from tiago_mpc.tiago import ROOT

FORMAT = 1
CLASSES: tuple[type, ...] = (  # the package's configuration classes that from_dict rebuilds
    SolverParams, MpcParams, AdmmLimits, Friction, Tanh, Sign, StickSlip, SampleKnot, HoldFirst,
    VirtualTime, RealTime, SimConfig, RobotConfig, Checks, CollisionSafetySettings,
)
TICK_COLUMNS = ("t", "state_t", "q", "qd", "command", "plan", "update")
PLAN_COLUMNS = ("t0", "update_t", "states", "torques")


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
    """A callback that prints an event's message to stderr the first time its kind appears;
    add_events takes events from outside a tick (a plant's after the loop)."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, tick: Tick) -> None:
        self.add_events(tick.events)

    def add_events(self, events: Iterable[Event]) -> None:
        for event in events:
            if event.kind not in self.seen:
                self.seen.add(event.kind)
                print(f"{event.message} (further {event.kind} events are recorded only)", file=sys.stderr)


class RunRecord:
    """RunRecord(out, **pieces): a callback that buffers each command tick and its events;
    save() writes the run. out=None is _expr_data/<task type>/<UTC time>/ ("run" without a
    task piece). The pieces are the run's configuration, frozen dataclasses or JSON values;
    the record writes provenance() and to_dict of each at once, with status "running".

    The read interface, which load_run gives a saved run too: config (the pieces as
    objects), ticks and plans (columns, built from the buffered ticks on first read, after
    the loop), events (their times counted from the first state's) and metadata. An
    experiment adds columns to ticks and plans and results to metadata["results"], then
    saves again; save() refuses a column of the wrong length or one that replaced a
    standard column.

    Not validated for long runs: every tick stays in memory until the run ends (about
    2 KB per tick in simulation, 4 GB for an hour at 500 Hz), and the loops pause
    garbage collection. The loops refuse runs over MAX_RUNTIME_MIN (tiago_mpc.loops);
    a loop of your own is not checked.
    """
    out: Path
    provenance: dict[str, Any]
    config: dict[str, Any]
    events: list[Event]
    metadata: dict[str, Any]

    def __init__(self, out: str | Path | None, **pieces: Any):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        kind = type(pieces["task"]).__name__.lower() if "task" in pieces else "run"
        self.out = Path(out) if out else ROOT / "_expr_data" / kind / stamp
        self.out.mkdir(parents=True, exist_ok=False)
        self.provenance, self.config, self.events = provenance(), pieces, []
        self.metadata = dict(status="running", reason=None, end_t=None, results={}, error=None, plant={})
        self._ticks: list[Tick] = []
        self._standard: dict[str, dict[str, np.ndarray]] = {}  # the columns that save() protects
        self._write_summary()

    def __call__(self, tick: Tick) -> None:
        self._ticks.append(tick)
        self.add_events(tick.events)

    def add_events(self, events: Iterable[Event]) -> None:
        """Store events, their times counted from the first state's."""
        t0 = self._ticks[0].state.t if self._ticks else 0.
        self.events.extend(dataclasses.replace(event, t=event.t - t0) for event in events)

    @cached_property
    def ticks(self) -> dict[str, np.ndarray]:
        ticks = self._ticks
        t0 = ticks[0].state.t if ticks else 0.
        updates = np.array([k.update for k in ticks], dtype=bool)
        columns = dict(t=np.array([k.t - t0 for k in ticks]), state_t=np.array([k.state.t - t0 for k in ticks]),
                       q=np.array([k.state.q for k in ticks]).reshape(-1, 7),
                       qd=np.array([k.state.qd for k in ticks]).reshape(-1, 7),
                       command=np.array([np.full(7, np.nan) if k.command is None else k.command
                                         for k in ticks]).reshape(-1, 7),
                       plan=np.cumsum([k.new for k in ticks], dtype=int) - 1,
                       update=np.where(updates, np.cumsum(updates) - 1, -1))
        self._standard["ticks"] = dict(columns)
        return columns

    @cached_property
    def plans(self) -> dict[str, np.ndarray]:
        t0 = self._ticks[0].state.t if self._ticks else 0.
        plans = [k.plan for k in self._ticks if k.new and k.plan is not None]
        plans += [k.faulted for k in self._ticks if k.faulted is not None]  # on the last tick
        ticks = self.ticks  # control update i yields plan i
        columns = dict(t0=np.array([p.t0 - t0 for p in plans]),
                       update_t=ticks["t"][ticks["update"] >= 0][:len(plans)],
                       states=np.array([p.states for p in plans]), torques=np.array([p.torques for p in plans]))
        for key in (plans[0].stats if plans else ()):
            values = [p.stats[key] for p in plans]
            if all(v is not None for v in values):
                columns[key] = np.asarray(values)
        self._standard["plans"] = dict(columns)
        return columns

    def save(self) -> None:
        """Write summary.json, ticks.npz, plans.npz and events.json."""
        for name, rows in (("ticks", "t"), ("plans", "t0")):
            table: dict[str, Any] = getattr(self, name)
            for key, value in table.items():
                if len(value) != len(table[rows]):
                    raise ValueError(f"{name}[{key!r}] has {len(value)} rows, not {len(table[rows])}")
            for key, value in self._standard[name].items():
                if table.get(key) is not value:
                    raise ValueError(f"{name}[{key!r}] is a standard column; give the experiment's another name")
            np.savez_compressed(self.out / f"{name}.npz", **table)
        events = [dataclasses.asdict(event) for event in self.events]
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
        raise ValueError(f"{out} is not a run of format {FORMAT}")
    record = RunRecord.__new__(RunRecord)
    record.out, record.provenance, record.metadata = out, summary["provenance"], summary["metadata"]
    record.config = from_dict(summary["config"], classes)
    with np.load(out / "ticks.npz") as ticks, np.load(out / "plans.npz") as plans:
        record.ticks, record.plans = dict(ticks), dict(plans)
    record.events = [Event(**event) for event in json.loads((out / "events.json").read_text())]
    record._ticks = []
    record._standard = dict(ticks={k: record.ticks[k] for k in TICK_COLUMNS},
                            plans={k: record.plans[k] for k in PLAN_COLUMNS})
    return record


def _json(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
