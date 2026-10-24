"""The run format (tiago_mpc/record.py): configurations to and from dicts, and a run written
by RunRecord and read back by load_run (CPU only)."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from test_loops import Mpc, Plant, Task
from tiago_mpc.dynamics import Tanh
from tiago_mpc.laws import SampleKnot
from tiago_mpc.loops import VirtualTime, run_virtual_time
from tiago_mpc.mpc import TIAGO
from tiago_mpc.record import FORMAT, Event, RunRecord, from_dict, load_run, to_dict
from tiago_mpc.robot import RobotConfig
from tiago_mpc.safety import CollisionSafetySettings
from tiago_mpc.sim import SimConfig
from tiago_mpc.tiago import urdf_friction


def test_a_run_is_written_and_read_back(tmp_path):
    """A fake loop's run, its states 2 ms old, with the plant's events, an experiment column,
    a stats column and results: the loaded run has the record's read interface and values."""
    class Emitting(Plant):
        def advance(self, t):
            super().advance(t)
            self.events.append(Event(t, "pacing_behind", "behind", dict(behind_s=.001)))

    plant, mpc, task, timing = Emitting(measure=lambda t: t - .002), Mpc(wall=.004), Task(), VirtualTime(.01, .02, .01)
    record = RunRecord(tmp_path / "r", task=task, params=TIAGO, plant=SimConfig(friction=urdf_friction(Tanh(.03))),
                       law=SampleKnot(), timing=timing, duration=.05, case="first")
    written = json.loads((tmp_path / "r" / "summary.json").read_text())  # at once
    assert written["format"] == FORMAT and written["metadata"]["status"] == "running"
    assert written["config"]["timing"] == dict(type="VirtualTime", command_period=.01, update_period=.02,
                                               solve_latency=.01)
    outcome = timing.run(plant, mpc, task, SampleKnot(), [record], duration=.05)
    record.add_events([Event(plant.clock.now(), "handback", "handed back", dict(restored=True, errors=[]))])
    record.ticks["distance_m"] = np.arange(len(record.ticks["t"])) * .1
    record.plans["label"] = np.array([f"plan {i}" for i in range(len(record.plans["t0"]))])
    record.metadata.update(status=outcome.status, end_t=outcome.t, results=dict(final_distance_m=.5))
    record.save()

    ticks, plans = record.ticks, record.plans  # times from the first state's, -2 ms
    np.testing.assert_allclose(ticks["t"], np.arange(6) * .01 + .002, atol=1e-12)
    np.testing.assert_allclose(ticks["state_t"], np.arange(6) * .01, atol=1e-12)
    np.testing.assert_array_equal(ticks["plan"], [-1, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(ticks["update"], [0, -1, 1, -1, 2, -1])
    np.testing.assert_allclose(plans["update_t"], [.002, .022, .042], atol=1e-12)  # the requests' ticks
    np.testing.assert_allclose(plans["t0"], [0., .02, .04], atol=1e-12)  # their states
    assert set(plans) == {"t0", "update_t", "states", "torques", "wall_s", "label"}
    assert [(e.kind, round(e.t, 6)) for e in record.events] == [
        ("late_plan", .012), ("pacing_behind", .012), ("pacing_behind", .022), ("late_plan", .032),
        ("pacing_behind", .032), ("pacing_behind", .042), ("pacing_behind", .052), ("handback", .052)]

    run = load_run(tmp_path / "r", classes=[Task])
    assert run.out == record.out and run.config == record.config and run.config["task"] == task
    assert run.config["case"] == "first" and run.provenance == record.provenance and run.metadata == record.metadata
    assert run.events == record.events
    for loaded, saved in ((run.ticks, ticks), (run.plans, plans)):
        assert set(loaded) == set(saved)
        for key, value in saved.items():
            np.testing.assert_array_equal(loaded[key], value)
            assert loaded[key].dtype == value.dtype
    assert load_run(tmp_path / "r").config["task"] == dict(type="Task", done_at=float("inf"))  # its class not given
    run.ticks["speed"] = np.zeros(len(run.ticks["t"]))  # a metric added later, written back
    run.ticks["distance_m"] = run.ticks["distance_m"] * 2  # recomputed: an experiment column, not a standard one
    run.metadata["results"]["later"] = 1.
    run.save()
    again = load_run(tmp_path / "r")
    assert "speed" in again.ticks and again.metadata["results"] == dict(final_distance_m=.5, later=1.)


def test_save_refuses_a_column_of_the_wrong_length_or_a_replaced_standard_one(tmp_path):
    record = RunRecord(tmp_path / "r", task=Task())
    run_virtual_time(Plant(), Mpc(), Task(), callbacks=[record], command_period=.01, update_period=.01,
                     duration=.03, solve_latency=0.)
    record.ticks["short"] = np.zeros(2)
    with pytest.raises(ValueError, match="rows"):
        record.save()
    del record.ticks["short"]
    record.ticks["q"] = record.ticks["q"] + 1.
    with pytest.raises(ValueError, match="standard column"):
        record.save()


@dataclass(frozen=True)
class Target:
    position_m: tuple[float, float, float] = (0., 0., 0.)


def test_from_dict_rebuilds_known_types_keeps_unknown_ones_and_refuses_two_classes_of_one_name():
    data = json.loads(json.dumps(dict(task=to_dict(Target((1., 2., 3.))), plain=[1, [2, 3]], other={"type": "Other"})))
    assert from_dict(data, [Target]) == dict(task=Target((1., 2., 3.)), plain=(1, (2, 3)), other=dict(type="Other"))
    assert from_dict(data)["task"] == dict(type="Target", position_m=(1., 2., 3.))

    @dataclass(frozen=True)
    class SimConfig:  # an experiment's class with a package class's name
        pass

    with pytest.raises(ValueError, match="two classes named SimConfig"):
        from_dict(data, [SimConfig])
    with pytest.raises(ValueError, match="two classes named Target"):
        from_dict(data, [Target, type("Target", (), {})])


@pytest.mark.parametrize("config", [RobotConfig(execute=True, sea_absolute=True),
                                    CollisionSafetySettings(urdf_path=Path("/tmp/arm.urdf"))])
def test_the_robot_configs_round_trip_with_their_paths(config):
    """from_dict knows no Path: CollisionSafetySettings converts its strings."""
    rebuilt = from_dict(json.loads(json.dumps(to_dict(config))))
    assert rebuilt == config
    safety = rebuilt.safety if isinstance(rebuilt, RobotConfig) else rebuilt
    assert isinstance(safety.blacklist_path, Path)
