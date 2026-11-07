# tiago_scripts

**Experimental: agent-generated (Claude), not reviewed in detail.**

Offline generators of data that `tiago_mpc` loads. Their output is checked in, because
generating it takes minutes. Rerun a script when its inputs change (the full URDF, its
collision meshes, the body posture `body_cover.POSTURE` or the script's settings):
`tests/test_collision.py` fails when a stored cover's recorded inputs no longer match the
model.

Run them in the container from the checkout's root with this checkout's environment
(`.venv-container`; on the host, `.venv`), under the heavy-job lock
(`flock /workspace/.locks/heavy.lock …`). They print their progress and write over the
stored file unless given `--output`. Their robot model is safety's collision model with the
URDF's original meshes (`validate.robot`).

| Script | Regenerates | Time | Run |
|---|---|---|---|
| `arm_cover` | `tiago_mpc/data/collision/arm/exact.json`: the right arm's 28 spheres, per link 2 to 7, an exact (HiGHS MILP) set cover; fails unless every link's cover is proven optimal | about 40 s | `.venv-container/bin/python -m tiago_scripts.arm_cover` |
| `body_cover` | `tiago_mpc/data/collision/body/<choice>.json`: the body's mixed cover (spheres, capsules, boxes) at `POSTURE`, weighted greedy then redundant-shape removal; `balanced` (4 cm bulge, 83 primitives) or `tighter` (3 cm, 155) | about 3 to 4 min and 2.5 GB each | `.venv-container/bin/python -m tiago_scripts.body_cover --choice balanced` |
| `validate` | nothing: prints the validation of a stored cover (both generators store it) | 10 s to 1 min | `.venv-container/bin/python -m tiago_scripts.validate body balanced` |

Each cover file records its primitives (in `tiago_mpc.collision`'s layouts), frame,
posture (body, as named joint positions), inputs (the URDF's and each collision part's
sha256), the generator's settings, the validation (sampled containment, which must find
nothing uncovered, and the bulge's maximum and median) and the library versions. The arm
cover is compiled into GATO's module, so a new arm cover compiles a new module once.

Repeated runs give the same primitives; the files differ only in their timings.
