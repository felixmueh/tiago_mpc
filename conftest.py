"""The names the doctests use (pytest's doctest_namespace), so the examples in the
docstrings stay clean: tiago_mpc/record.py's example runs with them on the CPU."""
import numpy as np
import pinocchio as pin
import pytest

from experiments.reach import Reach
import tiago_mpc as tm
from tiago_mpc.tiago import arm_urdf


class HoldMpc:
    """A stand-in for Mpc without the GPU: each plan holds the requested state with the
    gravity torques."""

    def __init__(self, params: tm.MpcParams = tm.TIAGO):
        self.params, self.N, self.dt = params, params.N, params.dt_s
        self.model = pin.buildModelFromUrdf(str(arm_urdf()))
        self.data, self.state = self.model.createData(), None

    def warmup(self, state, reference):
        pass

    def request(self, state, reference, updates=None):
        self.state = state

    def result(self, block=False):
        state, self.state = self.state, None
        if state is None:
            return None
        gravity = pin.computeGeneralizedGravity(self.model, self.data, state.q)
        return tm.Plan(state.t, self.dt, np.tile(np.r_[state.q, state.qd], (self.N, 1)),
                       np.tile(gravity, (self.N - 1, 1)), dict(wall_s=0.), measured_t=state.t)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def doctest_names(request, doctest_namespace):
    if isinstance(request.node, pytest.DoctestItem):
        doctest_namespace.update(mpc=HoldMpc(), task=Reach(), law=tm.SampleKnot(), timing=tm.VirtualTime.ideal(),
                                 plant_config=tm.SimConfig(substep=.003),
                                 out=request.getfixturevalue("tmp_path") / "run")
