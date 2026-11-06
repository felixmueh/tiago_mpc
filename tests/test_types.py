"""mypy over tiago_mpc/ and this file (pyproject.toml): the core's annotations agree, and
the plants, MPCs, tasks, laws and timings below, the test fakes among them, satisfy the
protocols in tiago_mpc/interfaces.py."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tiago_mpc.tiago import ROOT

if TYPE_CHECKING:  # read by mypy only
    from experiments.figure8 import Figure8
    from experiments.latency_compensation import Compensated
    from experiments.reach import Reach
    import test_experiments
    import test_loops
    from tiago_mpc.laws import EffortClip, FrictionFeedforward, HoldFirst, PlanFeedback, RejectOnEffort, SampleKnot
    from tiago_mpc.dynamics import Friction, Tanh
    from tiago_mpc.loops import RealTime, VirtualTime
    from tiago_mpc.mpc import Mpc, MpcProcess
    from tiago_mpc.interfaces import ControlLaw, EffortHandling, MpcInterface, PlantInterface, TaskInterface, Timing
    from tiago_mpc.robot import RosPlant
    from tiago_mpc.sim import SimPlant

    plants: list[type[PlantInterface]] = [SimPlant, RosPlant, test_loops.Plant]
    mpcs: list[type[MpcInterface]] = [Mpc, MpcProcess, test_loops.Mpc, test_experiments.Mpc]
    tasks: list[type[TaskInterface]] = [Reach, Figure8, test_loops.Task]
    laws: list[ControlLaw] = [SampleKnot(), HoldFirst(), PlanFeedback(),
                              FrictionFeedforward(Friction((1.,)*7, (0.,)*7), PlanFeedback(), Tanh(.03))]
    effort_handling: list[EffortHandling] = [EffortClip(), RejectOnEffort()]
    timings: list[Timing] = [VirtualTime.ideal(), RealTime(), Compensated()]


def test_mypy():
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy is not installed: uv sync (the dev group)")
    result = subprocess.run([sys.executable, "-m", "mypy"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
