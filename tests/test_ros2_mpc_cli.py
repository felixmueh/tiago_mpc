"""The entry point stays observational until execution is explicitly confirmed."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "examples/ros2_mpc.py"
    spec = importlib.util.spec_from_file_location("ros2_mpc_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_is_check_only_and_does_not_prompt(cli, monkeypatch):
    calls = []
    class Runner:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, output, **kwargs):
            calls.append(kwargs)
            return {"status": "checked"}
    monkeypatch.setattr(cli, "RosMpcRunner", Runner)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("check-only must not prompt"))
    assert cli.main([]) == 0
    assert calls == [{"execute": False}]


def test_cancelled_execution_creates_no_ros_client_or_runner(cli, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "cancel")
    monkeypatch.setattr(cli, "RosMpcRunner", lambda *a, **k: pytest.fail("must cancel before construction"))
    assert cli.main(["--execute", "--target", ".5", ".1", ".2"]) == 2


@pytest.mark.parametrize("name", ["reach", "figure8"])
def test_task_selects_its_mpc_parameters_before_ros(cli, monkeypatch, name):
    started = []
    class Runner:
        def __init__(self, config, task, params, *args, **kwargs):
            started.append((task, params))
        def run(self, output, **kwargs):
            return {"status": "checked"}
    monkeypatch.setattr(cli, "RosMpcRunner", Runner)
    assert cli.main(["--task", name, "--target", ".3", "-.2", ".1"]) == 0
    task, params = started[0]
    assert task.kind == name and task.target_xyz_m == (.3, -.2, .1)
    assert params == (cli.FIGURE8 if name == "figure8" else cli.TIAGO)


@pytest.mark.parametrize("args", [
    ["--execute"], ["--duration", "nan"], ["--target", "inf", "0", "0"],
    ["--budget-ms", "-1"], ["--output", "/tmp/not-this-worktree"]])
def test_invalid_launch_fails_before_ros(cli, args):
    with pytest.raises(SystemExit) as result:
        cli.main(args)
    assert result.value.code == 2
