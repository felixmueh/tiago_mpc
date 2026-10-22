# random100: 100 random reaches (simulation only)

100 fixed start/goal pairs: random arm postures at rest, each with one reachable
position goal (computed by forward kinematics from a goal posture). The goals
refer to the flange (`arm_right_tool_joint`), not the grasp-center frame; the
runner selects that frame. `manifest.json` lists the task IDs in their fixed
order, and `inputs/` holds one NPZ per task.

```bash
.venv/bin/python experiments/random100/run.py --dry-run     # parameters only
.venv/bin/python experiments/random100/run.py               # all 100 cases
.venv/bin/python experiments/random100/run.py --pairs 8990  # a subset
.venv/bin/python experiments/random100/analyze.py <output-directory>
```

- **Run:** uses the TIAGo defaults (`TIAGO` in `tiago_mpc/mpc.py`) with the
  flange as end-effector frame and no solve-time limit (`PARAMS` in `run.py`).
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