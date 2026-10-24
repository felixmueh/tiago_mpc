# random100: 100 random reaches (simulation only)

100 fixed start/goal pairs: random arm postures at rest, each with one reachable
goal posture. Each case reaches the grasp frame's position at its goal posture
(`ee_position`), like every other task. `manifest.json` lists the task IDs in
their fixed order, and `inputs/` holds one NPZ per task; their stored targets are
the flange's (`arm_right_tool_joint`) positions at the goal postures, which the
runner only checks for the inputs' consistency.

**New protocol (2026-10-24):** earlier random100 runs reached the flange
positions, so their results are not comparable with grasp-frame runs. They keep
their labels; never relabel them as grasp-frame results.

```bash
.venv/bin/python -m experiments.random100.run --dry-run     # parameters only
.venv/bin/python -m experiments.random100.run               # all 100 cases
.venv/bin/python -m experiments.random100.run --pairs 8990  # a subset
.venv/bin/python -m experiments.random100.analyze <output-directory>
```

- **Run:** uses the TIAGo defaults (`TIAGO` in `tiago_mpc/mpc.py`, grasp
  frame) with no solve-time limit (`PARAMS` in `run.py`).
  Each case runs once, in order: `reach` from `experiments/reach.py` with its own
  `MpcProcess` on the GATO module compiled once at the start (`compile_binary`),
  writing to a fresh directory under `_expr_data/`. The runner takes the shared
  GPU locks.
- **Outcomes:** arrival (position error at most 5 cm and joint-speed L1 at most
  1 rad/s within 8 simulated seconds) and simulated physical-limit compliance
  are reported separately. A timeout, limit violation or numerical failure
  keeps its output and the other cases continue; a shared setup failure stops
  the run. There is no torque clipping or retry.
- **Analysis:** `analyze.py` needs no GPU. It checks each case's initial state,
  recomputes joint, speed and torque limit excesses from all saved command
  ticks, and writes a case table, a plot and a short report next to the run.