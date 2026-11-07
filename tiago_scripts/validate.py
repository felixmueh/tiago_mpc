"""The robot model and inputs of the stored covers, and the validation both generators run.

Experimental: agent-generated (Claude), not reviewed in detail.

The model is safety's collision model with the URDF's original collision meshes and every
joint but the right arm's locked at the body posture.

A cover records the URDF's sha256 and each collision part's mesh sha256, because meshes can
change without the URDF; tests/test_collision.py compares them with the current model.

The validation samples, independently of the fit: points on every part's surface and every
vertex must be inside the cover; the bulge is the distance of points on the cover's boundary
to the parts. Both are discrete checks, not proofs.

Check a stored cover: python -m tiago_scripts.validate arm exact | body balanced
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import platform
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from tiago_mpc import collision as c
from tiago_mpc import safety
from tiago_mpc.tiago import FULL_URDF, RIGHT_ARM_JOINTS, ROOT, ROOT_FRAME


def robot(posture: Mapping[str, float]) -> tuple[pin.Model, pin.GeometryModel]:
    """safety's collision model with the URDF's original meshes and every joint but the right
    arm's locked at posture (joints it does not name at zero; mimic followers named too, so
    the builder does not warn)."""
    locked = {joint.get("name"): posture.get(joint.get("name"), 0.) for joint in ET.parse(FULL_URDF).findall("joint")
              if joint.get("type") != "fixed" and joint.get("name") not in RIGHT_ARM_JOINTS}
    built = safety.build_tiago_collision_model(urdf_path=FULL_URDF, state_fixed_joint_names=locked,
                                               reference_positions=locked)
    return built.model, built.geometry_model


def root_placement(model: pin.Model) -> pin.SE3:
    """ROOT_FRAME in the model's world; it is on a locked part, so constant."""
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    return data.oMf[model.getFrameId(ROOT_FRAME)].copy()


def geometry_mesh(obj, placement: pin.SE3) -> trimesh.Trimesh:
    """A geometry object as a triangle mesh, placed by placement * its own placement."""
    import coal

    shape = obj.geometry
    if isinstance(shape, coal.BVHModelBase):
        mesh = trimesh.load(obj.meshPath, force="mesh")
        mesh.apply_scale(np.asarray(obj.meshScale))
    elif isinstance(shape, coal.Box):
        mesh = trimesh.creation.box(extents=2 * np.asarray(shape.halfSide))
    elif isinstance(shape, coal.Cylinder):
        mesh = trimesh.creation.cylinder(radius=shape.radius, height=2 * shape.halfLength)
    elif isinstance(shape, coal.Sphere):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=shape.radius)
    else:
        raise ValueError(f"unsupported collision geometry {obj.name}: {type(shape).__name__}")
    mesh.apply_transform((placement * obj.placement).homogeneous)
    return mesh


def parts(kind: str, posture: Mapping[str, float]) -> dict[str, tuple[Any, pin.SE3, str]]:
    """The parts a cover of kind ("arm" or "body") covers, by name: the geometry object, the
    placement of its mesh (the arm's in its joint's frame, the body's in ROOT_FRAME) and its
    link. arm_right_1_link has no arm part: it is mounted on the torso (collision.arm_cover)."""
    model, geometry_model = robot(posture)
    root = root_placement(model).inverse()
    result = {}
    for obj in geometry_model.geometryObjects:
        link = model.names[obj.parentJoint].replace("_joint", "_link")
        if kind == "body" and obj.parentJoint == 0:
            result[obj.name] = (obj, root, link)
        elif kind == "arm" and obj.parentJoint != 0 and link != "arm_right_1_link":
            result[obj.name] = (obj, pin.SE3.Identity(), link)
    return result


def inputs(kind: str, posture: Mapping[str, float]) -> dict[str, Any]:
    """The URDF (relative to the checkout) and its sha256, and each part's mesh sha256."""
    hashes = {}
    for name, (obj, placement, _) in parts(kind, posture).items():
        mesh = geometry_mesh(obj, placement)
        hashes[name] = hashlib.sha256(np.asarray(mesh.vertices).tobytes() + np.asarray(mesh.faces).tobytes()).hexdigest()
    return dict(urdf=str(FULL_URDF.relative_to(ROOT)), urdf_sha256=hashlib.sha256(FULL_URDF.read_bytes()).hexdigest(),
                parts_sha256=hashes)


def versions() -> dict[str, str]:
    """The libraries a generator ran with (the MILP's HiGHS is SciPy's bundled one)."""
    from scipy.optimize._highspy import _core as highs

    result = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "trimesh", "pin", "coal")}
    result.update(highs=f"{highs.HIGHS_VERSION_MAJOR}.{highs.HIGHS_VERSION_MINOR}.{highs.HIGHS_VERSION_PATCH}",
                  python=platform.python_version())
    return result


def distance(points: np.ndarray, spheres=(), capsules=(), boxes=()) -> np.ndarray:
    """The signed distance of points (n, 3) to the union of the primitives (Collision's
    layouts); inf without any."""
    result = np.full(len(points), np.inf)
    for x, y, z, r in spheres:
        result = np.minimum(result, np.linalg.norm(points - (x, y, z), axis=1) - r)
    for capsule in capsules:
        a, b, r = np.asarray(capsule[:3]), np.asarray(capsule[3:6]), capsule[6]
        axis = b - a
        t = np.clip((points - a) @ axis / max(axis @ axis, 1e-30), 0., 1.)
        result = np.minimum(result, np.linalg.norm(points - a - t[:, None] * axis, axis=1) - r)
    for box in boxes:
        local = np.abs((points - box[:3]) @ Rotation.from_quat(box[3:7]).as_matrix()) - box[7:]
        result = np.minimum(result, np.linalg.norm(np.maximum(local, 0), axis=1) + np.minimum(local.max(1), 0))
    return result


def _surface(primitive: str, values, rng, density: float) -> np.ndarray:
    """Points on a primitive's surface, about density per square metre."""
    v = np.asarray(values, dtype=float)
    if primitive == "sphere":
        n = max(50, int(density * 4 * np.pi * v[3] ** 2))
        d = rng.normal(size=(n, 3))
        return v[:3] + v[3] * d / np.linalg.norm(d, axis=1, keepdims=True)
    if primitive == "capsule":
        a, b, r = v[:3], v[3:6], v[6]
        length = np.linalg.norm(b - a)
        n = max(50, int(density * (4 * np.pi * r * r + 2 * np.pi * r * length)))
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t = rng.uniform(0, 1, n)[:, None]
        axis = (b - a) / max(length, 1e-12)
        radial = d - (d @ axis)[:, None] * axis  # the side: around the segment
        side = a + t * (b - a) + r * radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        caps = np.where((d @ axis)[:, None] < 0, a, b) + r * d  # the end caps
        return np.vstack([side, caps])
    half = v[7:]
    area = 8 * (half[0] * half[1] + half[1] * half[2] + half[0] * half[2])
    n = max(50, int(density * area))
    local = rng.uniform(-1, 1, (n, 3)) * half
    face = rng.integers(0, 3, n)
    local[np.arange(n), face] = np.where(rng.random(n) < .5, -1, 1) * half[face]
    return local @ Rotation.from_quat(v[3:7]).as_matrix().T + v[:3]


def validate(
    placed_parts: list[tuple[Any, Any]],
    spheres=(),
    capsules=(),
    boxes=(),
    *,
    samples: int = 100_000,
    fill_pitch_m: float = .0025,
    seed: int = 0,
) -> dict[str, Any]:
    """The containment and bulge of a cover of the parts (geometry object and placement)."""
    import coal

    rng = np.random.default_rng(seed)
    meshes = [geometry_mesh(obj, placement) for obj, placement in placed_parts]
    area = sum(mesh.area for mesh in meshes)
    surface = []
    for (obj, placement), mesh in zip(placed_parts, meshes):
        count = max(1000, int(samples * mesh.area / area))
        surface += [trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(2**31)))[0], mesh.vertices]
        if isinstance(obj.geometry, coal.Cylinder):  # its mesh is inscribed: sample the cylinder itself
            theta, z = rng.uniform(0, 2 * np.pi, count), rng.uniform(-1, 1, count) * obj.geometry.halfLength
            local = np.column_stack([obj.geometry.radius * np.cos(theta), obj.geometry.radius * np.sin(theta), z])
            surface.append(local @ (placement * obj.placement).rotation.T + (placement * obj.placement).translation)
    points = np.vstack(surface)
    gaps = distance(points, spheres, capsules, boxes)
    boundary = np.vstack([_surface(kind, v, rng, 6e4) for kind, group in
                          (("sphere", spheres), ("capsule", capsules), ("box", boxes)) for v in group])
    boundary = boundary[distance(boundary, spheres, capsules, boxes) >= -1e-9]  # on the union's boundary
    dense = np.vstack([mesh.sample(max(int(4 * samples * mesh.area / area), 1000), seed=seed + 1)
                       for mesh in meshes] + [mesh.vertices for mesh in meshes])
    bulge, _ = cKDTree(dense).query(boundary)
    inside = np.zeros(len(boundary), dtype=bool)
    for mesh in meshes:
        inside |= mesh.voxelized(fill_pitch_m).fill().is_filled(boundary)
    bulge[inside] = 0.
    return dict(surface_samples=len(points), uncovered_samples=int((gaps > 1e-9).sum()),
                max_gap_m=float(gaps.max()), boundary_samples=len(boundary),
                bulge_max_m=float(bulge.max()), bulge_median_m=float(np.median(bulge)),
                method=f"seed {seed}; bulge 0 inside a part's {fill_pitch_m * 1e3:g} mm voxel fill")


def validate_stored(kind: str, name: str) -> dict[str, Any]:
    """validate() of a stored cover (collision.arm_cover, collision.body_cover)."""
    stored = json.loads(c.cover_file(kind, name).read_text())
    if kind == "arm":  # in the links' frames the arm's parts do not depend on the posture
        cover = c.arm_cover(name)
        by_link: dict[str, list] = {}
        for obj, placement, link in parts("arm", c.body_posture()).values():
            by_link.setdefault(link, []).append((obj, placement))
        return {link: validate(group, cover[link], seed=i) for i, (link, group) in enumerate(sorted(by_link.items()))}
    body = c.body_cover(name)
    placed = [(obj, placement) for obj, placement, _ in parts("body", stored["posture"]).values()]
    return validate(placed, body["spheres"], body["capsules"], body["boxes"], samples=300_000, fill_pitch_m=.005)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("arm", "body"))
    parser.add_argument("name")
    args = parser.parse_args()
    print(json.dumps(validate_stored(args.kind, args.name), indent=1))


if __name__ == "__main__":
    main()
