"""Covers of the TIAGo for the MPC's collision rows.

What GATO offers: collision rows, one per arm sphere and knot, that keep every sphere of the
controlled arm at least one margin away from the nearest obstacle (an exact signed distance).
- The arm side is spheres only. They come from the URDF's collision geometry and are
  compiled into GATO's module, so they move with the arm's kinematics.
- The obstacle side is a runtime environment, set without recompiling: spheres, capsules,
  oriented boxes and half-spaces, in the arm's root frame.
- There are no self-collision rows within the arm: the rows only see the environment.

What this module does for the TIAGo:
- Arm (moving): a sphere cover of the right arm's links and gripper (arm/<name>.json).
  runtime_urdf writes it into the GPU URDF as the arm's only collision geometry, so a
  different arm cover means a different compiled module.
- Rest of the robot (static): everything except the right arm is fixed while it moves
  (torso, head, left arm, base, the fingers at zero). Its cover is fitted once for a single
  pose, which the cover records (body/<name>.json: spheres, capsules and boxes in
  ROOT_FRAME, and the posture as named joint positions), and enters GATO as part of the
  environment.
- Obstacles: a scene's spheres, capsules, boxes and planes (Collision). environment()
  merges them with the body cover into GATO's set_collision_environment arguments.

The covers are stored by name in tiago_mpc/data/collision/. tiago_scripts generates them
offline (tiago_scripts/README.md) and records the inputs. Nothing is fitted at run time, and
loading checks nothing: tests/test_collision.py compares the recorded inputs with the
current model. The covers contain the geometry they were fitted to (up to their
voxelization), so a positive clearance of the covers means clear meshes. They are not
exact: a primitive sticks out of the surface by up to its bulge (3 cm for the arm's).
All covers contain the URDF's original collision meshes; experiments/clearance.py measures
those.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

Sphere = tuple[float, float, float, float]  # centre x, y, z and radius, metres
Capsule = tuple[float, float, float, float, float, float, float]  # segment ends a, b and radius
# Centre, orientation as a unit quaternion (x, y, z, w; as scipy and three.js) and the
# half-extents along the box's own axes.
Box = tuple[float, float, float, float, float, float, float, float, float, float]
Plane = tuple[float, float, float, float]  # unit normal n and offset d: the free side is n.p >= d
# The stored covers (tiago_mpc/data/collision/<kind>/<name>.json; tests check they agree)
ArmCover = Literal["exact"]
BodyCover = Literal["balanced", "tighter"]
Mechanism = Literal["admm", "al", "barrier", "telemetry"]


@dataclass(frozen=True)
class Collision:
    """MpcParams.collision: GATO's collision rows against the body and obstacles.

    arm and body name the stored covers (arm_cover, body_cover; body None: obstacles only,
    the body is not avoided). The body's posture is its cover's (body_posture). The
    obstacles are spheres, capsules, boxes and planes (half-spaces) in ROOT_FRAME, in the
    layouts of Sphere, Capsule, Box and Plane. The rows keep every arm sphere margin_m away
    from the nearest body primitive or obstacle at knots 1..N, with GATO's mechanism
    ("admm", "al", "barrier", or "telemetry", which only reports) and rho.
    """
    # The covers' bulge is the margin; margin_m adds to it.
    arm: ArmCover = "exact"
    body: BodyCover | None = "balanced"
    obstacles: tuple[Sphere, ...] = ()
    capsules: tuple[Capsule, ...] = ()
    boxes: tuple[Box, ...] = ()
    planes: tuple[Plane, ...] = ()
    margin_m: float = .005
    mechanism: Mechanism = "admm"
    rho: float = 5.

    def __post_init__(self):
        def floats(name, width, message):
            values = tuple(tuple(map(float, v)) for v in getattr(self, name))
            if any(len(v) != width or not np.isfinite(v).all() for v in values):
                raise ValueError(f"{name} are {message}")
            object.__setattr__(self, name, values)
            return values

        if any(s[3] <= 0 for s in floats("obstacles", 4, "spheres (x, y, z, radius)")):
            raise ValueError("obstacles are spheres with a positive radius")
        if any(c[6] <= 0 for c in floats("capsules", 7, "capsules (ax, ay, az, bx, by, bz, radius)")):
            raise ValueError("capsules need a positive radius")
        boxes = floats("boxes", 10, "boxes (cx, cy, cz, qx, qy, qz, qw, hx, hy, hz)")
        if any(min(b[7:]) <= 0 for b in boxes):
            raise ValueError("boxes need positive half-extents")
        if any(np.linalg.norm(b[3:7]) < 1e-9 for b in boxes):
            raise ValueError("boxes need a nonzero quaternion")
        planes = floats("planes", 4, "planes (nx, ny, nz, d)")
        if any(np.linalg.norm(p[:3]) < 1e-9 for p in planes):
            raise ValueError("planes need a nonzero normal")
        object.__setattr__(self, "planes", tuple((*(np.array(p[:3]) / np.linalg.norm(p[:3])).tolist(),
                                                  p[3] / np.linalg.norm(p[:3])) for p in planes))
        if self.mechanism not in get_args(Mechanism):
            raise ValueError(f"unknown collision mechanism {self.mechanism}")

    def environment(self) -> dict[str, tuple[tuple[float, ...], ...]]:
        """GATO's set_collision_environment arguments, in ROOT_FRAME: spheres, capsules and
        cuboids (boxes in GATO's layout), each the body cover's first, then the obstacles',
        and planes."""
        body = body_cover(self.body) if self.body is not None else dict(spheres=(), capsules=(), boxes=())
        return dict(spheres=body["spheres"] + self.obstacles, capsules=body["capsules"] + self.capsules,
                    cuboids=tuple(map(cuboid, body["boxes"] + self.boxes)), planes=self.planes)



# ==============================================================================
# The stored covers
# ==============================================================================

DATA = Path(__file__).parent / "data" / "collision"


def cover_file(kind: str, name: str) -> Path:
    """The stored cover of kind ("arm" or "body") and name."""
    return DATA / kind / f"{name}.json"


@cache
def _stored(kind: str, name: str) -> dict[str, Any]:
    path = cover_file(kind, name)
    if not path.is_file():
        stored = sorted(p.stem for p in (DATA / kind).glob("*.json"))
        raise ValueError(f"no stored {kind} cover {name!r} in {path.parent} (stored: {', '.join(stored)})")
    return json.loads(path.read_text())


def arm_cover(name: ArmCover = "exact") -> dict[str, tuple[Sphere, ...]]:
    """The right arm's stored spheres by link (arm_right_<i>_link), centres in the link's
    frame. The flange and gripper (welded at zero opening) belong to arm_right_7_link.

    arm_right_1_link has none: it is mounted on the torso and turns inside the shoulder,
    so its spheres would overlap the body's cover at every posture (the safety
    blacklist's torso / arm 1 pairs).

    The arm covers contain the original meshes only. The safety monitor checks the arm's
    alpha-wrap meshes, which lie about 1 cm outside them (offset 0.01 in
    models/safety/collision_alpha_wrap_a3_o25_cap001/alpha_wrap_results.json) and which the
    spheres do not certainly contain. On the robot the monitor can therefore trip up to
    about 1 cm before the rows' margin is reached; this is accepted until it is observed."""
    return {link: tuple(tuple(s) for s in spheres) for link, spheres in _stored("arm", name)["spheres"].items()}


def body_cover(name: BodyCover = "balanced") -> dict[str, tuple[tuple[float, ...], ...]]:
    """The stored cover of everything but the right arm at its posture (body_posture), in
    ROOT_FRAME: spheres, capsules and boxes in the layouts of Sphere, Capsule and Box."""
    stored = _stored("body", name)
    return {key: tuple(tuple(v) for v in stored[key]) for key in ("spheres", "capsules", "boxes")}


def body_posture(name: BodyCover | None = "balanced") -> dict[str, float]:
    """The named joint positions the body cover was fitted at; every other joint is at zero.
    None gives the default body cover's: it is the same robot, and body None only means
    that GATO does not avoid it."""
    return dict(_stored("body", name or str(Collision.body))["posture"])


def cover_hashes() -> dict[str, str]:
    """The first 8 hex digits of each stored cover file's sha256, by kind/name; records
    store them beside the names (record.provenance)."""
    return {f"{path.parent.name}/{path.stem}": hashlib.sha256(path.read_bytes()).hexdigest()[:8]
            for path in sorted(DATA.glob("*/*.json"))}


def box_rotation(box: tuple[float, ...]) -> np.ndarray:
    """The rotation (3, 3) of a box: its columns are the box's axes in ROOT_FRAME."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(box[3:7]).as_matrix()  # scipy normalizes the quaternion


def cuboid(box: tuple[float, ...]) -> tuple[float, ...]:
    """A box in GATO's cuboid layout: centre, then each unit axis with its half-extent."""
    axes = box_rotation(box).T
    return tuple(float(v) for v in (*box[:3], *(v for axis, h in zip(axes, box[7:]) for v in (*axis, h))))

