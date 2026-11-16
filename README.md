# tiago_mpc

Torque MPC for the seven-joint TIAGo Pro right arm using GATO. The package
combines an MPC, a plant, a task and an execution law for simulation or ROS 2
execution. Module docstrings describe the interfaces and their timing rules;
the examples show how to compose them.

Linux, an NVIDIA GPU, CUDA toolkit 12.x, a compatible host compiler, Git LFS
and [uv](https://docs.astral.sh/uv/) are required. uv selects Python 3.10.
The current `pyproject.toml` and lock require local GATO and GRiD sources at
`../GATO/.worktrees/tiago-combined` and `../GRiD`. These supply native extensions
used here; a standalone clone cannot yet install from the lock without them.

```bash
git lfs install
git clone https://github.com/felixmueh/gato_tiago.git tiago_mpc
cd tiago_mpc
git lfs pull
uv sync
uv run python -m experiments.minimal
```

The first run compiles a CUDA module and can take several minutes. Run outputs
are saved under `_expr_data/`; derived models and compiled modules are cached
under `.cache/gato`. CPU checks use `uv run python -m pytest -q` and
`uv run python -m mypy` and do not require ROS.

Read the examples in order: [minimal](experiments/minimal.py),
[reach](experiments/reach.py), [figure eight](experiments/figure8.py), then
[obstacles](experiments/obstacles.py). Run an example with
`uv run python -m experiments.<name>`; the task examples accept `--help`.
[common.py](experiments/common.py) supplies recording and command-line setup.
The [random100 study](experiments/random100/README.md) runs 100 fixed reaching
cases and audits their saved trajectories.

The package documentation is in the source:

- [interfaces.py](tiago_mpc/interfaces.py): states, plans and component protocols.
- [mpc.py](tiago_mpc/mpc.py): solver parameters, compilation and MPC processes.
- [loops.py](tiago_mpc/loops.py): virtual-time and wall-time execution.
- [laws.py](tiago_mpc/laws.py) and [prediction.py](tiago_mpc/prediction.py):
  command decisions and latency compensation.
- [sim.py](tiago_mpc/sim.py), [dynamics.py](tiago_mpc/dynamics.py) and
  [tiago.py](tiago_mpc/tiago.py): simulation, physical models and robot facts.
- [collision.py](tiago_mpc/collision.py): MPC collision constraints.
- [record.py](tiago_mpc/record.py): saved runs and serialization.

[Model provenance](models/README.md), [robot operation](docs/robot.md) and
[collision-safety coverage](docs/safety.md) describe the retained assets and
hardware limitations. The ROS path has CPU fake-client coverage but has not
been qualified against ROS, Gazebo or the physical robot.
