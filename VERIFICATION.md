# Initial simulation verification — 20 October 2026

Application source: `638022f1757533257974237c45c206d2b4bd344b`.
GATO: unchanged upstream `aeacf16e7733df1a17dfd6627f3b5d15c2ba4387`.
Hardware/build: GTX 1070 (sm61), CUDA 12.9.86, GCC 11.4.0, CMake 3.31.10,
Python 3.10.12, Pinocchio 3.9.0, GRiD 0.5.1; release float N32 module.

Executed commands, from this checkout:

```bash
.venv/bin/python examples/reach_sim.py --build-only
.venv/bin/python examples/reach_sim.py --hold --timeout 2 \
  --output results/hold-20261020T194021.906203Z
.venv/bin/python examples/reach_sim.py \
  --output results/reach-20261020T194023.881420Z
```

| Check | Result |
| --- | --- |
| Original full and reduced URDF bytes | Identical to supplied sources |
| Assets | All 33 mesh SHA-256 hashes match; LFS object check passes |
| Full URDF geometry | Pinocchio loads 47 visual and 47 collision objects |
| Adapted GPU versus native Pinocchio step, start / target | Maximum absolute state differences 0.0000311 / 0.0000271; tolerance 0.002 |
| Two-second hold | Final/max position error 15.0 mm; speed/position/torque limits pass |
| Reach | Initial error 71.9 mm; arrival at 49.98 mm after 5.82 simulated seconds |
| Reach terminal speed L1 | 0.132 rad/s, below the 1 rad/s arrival threshold |
| Reach simulated physical limits | No excess over URDF position, speed or effort limits at retained plant samples |
| Reach peak speed / torque | 0.244 rad/s / 10.91 Nm |
| Numeric histories | All retained arrays finite; 194 controller solves, 1,941 plant states |

Both episodes returned zero. Each was run once, sequentially with the shared
GPU locks held. Times inside the summaries are diagnostics, not benchmark
comparisons. The reach uses the explicit loose 5 cm arrival criterion and
stops on arrival; it demonstrates basic operation, not centimetre accuracy or
settled tracking. The hold has a 15 mm drift. No orientation, collision,
real-time scheduling or physical robot qualification is claimed.

Raw outputs, source copies, histories, plots and logs remain under `results/`
(ignored). `results/environment.txt` records installed packages. `reach.png`
was inspected for legible axes and complete rendering.

## Retained setup failures

- Initial Pinocchio import failed with unconstrained `cmeel-urdfdom` 6.0.0;
  Pinocchio 3.9 requires the urdfdom 4 / tinyxml2 10 wheel ABI. Both compatible
  versions are now explicit requirements.
- First configure failed because the system CMake did not support `block`;
  `results/build-cmake-failed.log` preserves that configure output. The
  application uses the pinned virtual-environment CMake.
- Registering the native negative-axis URDF directly compiled, then trapped
  in `sim_forward`. `results/build-model-cmake-fixed.log` and
  `results/model-memcheck.log` preserve this failure. GRiD emitted seven
  topology helpers; upstream's fixed-arm adapter supplies a null helper
  pointer. The generated equivalent positive-axis representation emits zero
  helpers and passes the bounded model comparison. The application maps the
  sixth joint's q/v/torque signs; GATO and GRiD sources remain unchanged.
  This is an integration workaround, not a general solver repair.
