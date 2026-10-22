"""mypy over tiago_mpc/ and this file (pyproject.toml): the core's annotations agree, and
the plants, MPCs and tasks below, the test fakes among them, satisfy the protocols in
tiago_mpc/plan.py."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tiago_mpc.tiago import ROOT

if TYPE_CHECKING:  # read by mypy only
    from experiments.figure8 import Figure8
    from experiments.reach import Reach
    import test_experiments
    import test_loops
    from tiago_mpc.mpc import Mpc, MpcProcess
    from tiago_mpc.plan import MpcInterface, PlantInterface, TaskInterface
    from tiago_mpc.robot import RosPlant
    from tiago_mpc.sim import SimPlant

    plants: list[type[PlantInterface]] = [SimPlant, RosPlant, test_loops.Plant]
    mpcs: list[type[MpcInterface]] = [Mpc, MpcProcess, test_loops.Mpc, test_experiments.Mpc]
    tasks: list[type[TaskInterface]] = [Reach, Figure8, test_loops.Task]


def test_mypy():
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy is not installed: pip install -r requirements-dev.txt")
    result = subprocess.run([sys.executable, "-m", "mypy"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
