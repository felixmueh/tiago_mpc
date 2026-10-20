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


@pytest.mark.parametrize("args", [
    ["--execute"], ["--duration", "nan"], ["--target", "inf", "0", "0"],
    ["--budget-ms", "-1"], ["--output", "/tmp/not-this-worktree"],
    ["--execute", "--task", "figure8", "--target", ".5", ".1", ".2"]])
def test_invalid_launch_fails_before_ros(cli, args):
    with pytest.raises(SystemExit) as result:
        cli.main(args)
    assert result.value.code == 2
