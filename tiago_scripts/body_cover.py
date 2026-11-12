"""The mixed body covers (spheres, capsules and boxes): tiago_mpc/data/collision/body/<choice>.json.

Experimental: agent-generated (Claude), not reviewed in detail.

The body is everything but the right arm at POSTURE, in ROOT_FRAME, with the URDF's
original collision geometry. Native boxes are kept exactly. Mesh triangles are subdivided
and rasterized by their whole bounding boxes and enclosed voids are filled, so the voxels contain every triangle. The
candidates are primitives enclosing whole voxel cubes: per part and per region of a
recursive split (boxes in three bases, a sphere, capsules along four axes), spheres and
boxes grown inside the solid; each must stick out of the voxel solid by at most the bulge
(measured on a distance grid). A weighted greedy set cover then picks, again and again,
the candidate covering the most still uncovered mesh voxels (whole cubes; interior
voxels weigh interior_weight) per cost, and redundant primitives are removed afterwards.
Every mesh voxel ends up inside one primitive; the native boxes are added back.

Choices (CHOICES): balanced (4 cm bulge; 83 primitives) and tighter (3 cm; 155).
Run in the container from the checkout's root (about 3 to 4 minutes and 2.5 GB each):
    .venv-container/bin/python -m tiago_scripts.body_cover --choice balanced
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import ndimage, optimize, sparse
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from tiago_mpc import collision as c
from tiago_mpc.tiago import ROOT_FRAME
from tiago_scripts.validate import geometry_mesh, inputs, parts, validate, versions

# The joints that stay fixed while the right arm moves, at the robot's measured posture in
# the physical runs of 2026-10-16 (full joint state history of reach_goals goal 3): torso
# lift in metres, head and left arm in radians. Every other joint is at zero.
POSTURE = {"torso_lift_joint": .0995, "head_1_joint": .0690, "head_2_joint": .2869,
           **{f"arm_left_{i + 1}_joint": v for i, v in enumerate((.3592, -1.8222, .4692, -2.3208, -.0002, -1.2002, 0.))}}


@dataclass(frozen=True)
class Settings:
    """How a body cover is fitted."""
    pitch_m: float = .005
    bulge_m: float = .04
    sphere_cost: float = 1.
    capsule_cost: float = 1.6
    box_cost: float = 1.4
    interior_weight: float = .03


CHOICES = {"balanced": Settings(bulge_m=.04), "tighter": Settings(bulge_m=.03)}


@dataclass
class Shape:
    """A primitive in Collision's layout (values), the part it was fitted to (source), its
    volume and its bulge on the distance grid."""
    kind: str
    values: tuple[float, ...]
    source: str
    volume: float
    bulge: float = 0.

    def distance(self, points: np.ndarray) -> np.ndarray:
        v = np.asarray(self.values)
        if self.kind == "sphere":
            return np.linalg.norm(points - v[:3], axis=1) - v[3]
        if self.kind == "capsule":
            axis = v[3:6] - v[:3]
            t = np.clip((points - v[:3]) @ axis / max(axis @ axis, 1e-30), 0., 1.)
            return np.linalg.norm(points - v[:3] - t[:, None] * axis, axis=1) - v[6]
        local = (points - v[:3]) @ Rotation.from_quat(v[3:7]).as_matrix()
        q = np.abs(local) - v[7:]
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(self.values)
        if self.kind == "sphere":
            return v[:3] - v[3], v[:3] + v[3]
        if self.kind == "capsule":
            return np.minimum(v[:3], v[3:6]) - v[6], np.maximum(v[:3], v[3:6]) + v[6]
        extent = np.abs(Rotation.from_quat(v[3:7]).as_matrix()) @ v[7:]
        return v[:3] - extent, v[:3] + extent


def mesh_cells(mesh, pitch: float) -> np.ndarray:
    """All cubes intersecting each microtriangle's bounding box: a conservative raster, with
    enclosed voids filled. Subdivision keeps the original triangles exactly; with each
    microtriangle's edge <= pitch, its bounding box spans at most two cells on each axis."""
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, pitch, max_iter=15)
    triangles = vertices[faces]
    low = np.floor(triangles.min(1) / pitch + .5).astype(np.int32)
    high = np.floor(triangles.max(1) / pitch + .5).astype(np.int32)
    assert np.max(high - low) <= 1, "microtriangle spans more than two cells"
    cells = np.vstack([np.where(offset, high, low) for offset in np.ndindex(2, 2, 2)])
    cells = np.unique(cells, axis=0)
    lo = cells.min(0) - 1
    occupied = np.zeros(cells.max(0) - lo + 2, dtype=bool)
    occupied[tuple((cells - lo).T)] = True
    return np.argwhere(ndimage.binary_fill_holes(occupied)) + lo


def voxel_body(settings: Settings):
    """The body's parts as voxel cells (name, cells, rotation, is a native box), its native
    boxes as Shapes, all cells, the grid's origin, its occupancy and the distance of every
    grid point to the solid."""
    import coal

    parts_, boxes = [], []
    for name, (obj, placement, _) in parts("body", POSTURE).items():
        mesh = geometry_mesh(obj, placement)
        placed = placement * obj.placement
        if isinstance(obj.geometry, coal.Box):
            h = np.asarray(obj.geometry.halfSide)
            boxes.append(Shape("box", tuple(np.r_[placed.translation, Rotation.from_matrix(placed.rotation).as_quat(), h]),
                               name, float(8 * h.prod())))
        cells = mesh_cells(mesh, settings.pitch_m)
        # Mesh approximations of analytic cylinders are inscribed: grow them by their
        # tessellation's sagitta with a voxel halo.
        if isinstance(obj.geometry, coal.Cylinder):
            sections = (len(mesh.vertices) - 2) // 2
            sagitta = obj.geometry.radius * (1 - np.cos(np.pi / sections))
            halo = max(1, int(np.ceil(sagitta / settings.pitch_m)))
            cells = np.unique(np.vstack([cells + d for d in np.ndindex(*((2 * halo + 1,) * 3))]) - halo, axis=0)
        parts_.append((name, cells, placed.rotation, isinstance(obj.geometry, coal.Box)))
    cells = np.unique(np.vstack([p[1] for p in parts_]), axis=0)
    lo, hi = cells.min(0) - 15, cells.max(0) + 15
    occupied = np.zeros(hi - lo + 1, dtype=bool)
    occupied[tuple((cells - lo).T)] = True
    distance = ndimage.distance_transform_edt(~occupied).astype(np.float32) * settings.pitch_m
    return parts_, boxes, cells, lo, occupied, distance


def enclosing_shapes(points, rotation, pitch, name):
    """Candidates enclosing the convex hull of all full voxel cubes."""
    centre = points.mean(0)
    _, axes = np.linalg.eigh((points - centre).T @ (points - centre))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    # Hull vertices suffice: all fitted primitives are convex.
    if len(points) > 30 and np.linalg.matrix_rank(points - points[0]) == 3:
        hull = points[ConvexHull(points).vertices]
    else:
        hull = points
    corners = (np.array(list(np.ndindex(2, 2, 2))) - .5) * pitch
    support = (hull[:, None, :] + corners).reshape(-1, 3)
    fits = []
    for basis in (rotation, np.eye(3), axes):
        local = support @ basis
        low, high = local.min(0), local.max(0)
        mid, half = (low + high) / 2, (high - low) / 2 + 1e-10
        origin = basis @ mid
        fits.append(Shape("box", tuple(np.r_[origin, Rotation.from_matrix(basis).as_quat(), half]),
                          name, float(8 * half.prod())))
    centre = .5 * (support.min(0) + support.max(0))
    radius = np.linalg.norm(support - centre, axis=1).max() + 1e-10
    fits.append(Shape("sphere", tuple(np.r_[centre, radius]), name, float(4 / 3 * np.pi * radius**3)))
    # Each axis defines an enclosing capsule. Optimize its segment length while
    # increasing the radius enough to retain all support points.
    for axis in (rotation[:, 0], rotation[:, 1], rotation[:, 2], axes[:, -1]):
        z = (support - centre) @ axis
        radial2 = np.maximum(np.sum((support - centre)**2, axis=1) - z*z, 0)

        def dimensions(half):
            radius = np.sqrt(np.max(radial2 + np.maximum(np.abs(z) - half, 0)**2)) + 1e-10
            volume = np.pi * radius**2 * (2 * half + 4 / 3 * radius)
            return radius, volume

        maximum = float(np.abs(z).max())
        result = optimize.minimize_scalar(lambda h: dimensions(h)[1], bounds=(0, maximum),
                                         method="bounded", options={"xatol": 1e-5})
        half = float(result.x)
        radius, volume = dimensions(half)
        fits.append(Shape("capsule", tuple(np.r_[centre - half*axis, centre + half*axis, radius]),
                          name, float(volume)))
    return fits, axes


def grid_indices(shape, pitch, origin, grid_shape, halo=0.):
    low, high = shape.bounds()
    low = np.maximum(np.floor(low / pitch - halo).astype(int) - origin, 0)
    high = np.minimum(np.ceil(high / pitch + halo).astype(int) - origin, np.asarray(grid_shape) - 1)
    indices = np.indices(high - low + 1).reshape(3, -1).T + low
    points = (indices + origin) * pitch
    return indices, points


def protrusion(shape, settings, origin, distance):
    low, high = shape.bounds()
    if (low < origin * settings.pitch_m).any() or (high > (origin + np.asarray(distance.shape) - 1) * settings.pitch_m).any():
        return float("inf")
    indices, points = grid_indices(shape, settings.pitch_m, origin, distance.shape, halo=1)
    mask = shape.distance(points) <= np.sqrt(3) / 2 * settings.pitch_m
    return float(distance[tuple(indices[mask].T)].max(initial=0))


def interior_spheres(cells, settings, origin, occupied, distance):
    """Inscribed spheres expanded by the allowed bulge, independent of partitions."""
    depth = ndimage.distance_transform_edt(occupied).astype(np.float32) * settings.pitch_m
    indices = cells - origin
    d = depth[tuple(indices.T)]
    maxima = ndimage.maximum_filter(depth, size=5)
    stride = max(4,round(.032/settings.pitch_m))
    medial_stride = max(2,round(.016/settings.pitch_m))
    sample = ((cells % stride == 0).all(1) | ((d == maxima[tuple(indices.T)]) & (cells % medial_stride == 0).all(1)))
    result = []
    for cell, dep in zip(cells[sample], d[sample]):
        radius = float(dep + .5 * settings.bulge_m - np.sqrt(3) / 2 * settings.pitch_m)
        if radius < settings.pitch_m:
            continue
        shape = Shape("sphere", tuple(np.r_[cell * settings.pitch_m, radius]), "global-solid",
                      float(4 / 3 * np.pi * radius**3))
        shape.bulge = protrusion(shape, settings, origin, distance)
        if shape.bulge <= settings.bulge_m:
            result.append(shape)
    print("interior spheres", len(result), flush=True)
    return result


def interior_boxes(cells, settings, origin, distance):
    """Grow maximal boxes inside the allowed dilation using a summed-volume table.

    Overlapping boxes can follow slabs and concavities without inheriting the
    arbitrary boundaries of a recursive partition.
    """
    # The quality test includes grid centres within a half voxel diagonal of
    # the box. Near a box corner these include diagonal neighbours, so reserve
    # two pitches rather than one before growing the box.
    allowed = distance <= settings.bulge_m - 2 * settings.pitch_m
    bad = np.pad((~allowed).astype(np.int32), ((1, 0),)*3).cumsum(0).cumsum(1).cumsum(2)
    limit = np.asarray(distance.shape) - 1

    def valid(lo, hi):
        x,y,z = lo
        a,b,d = hi + 1
        return (bad[a,b,d] - bad[x,b,d] - bad[a,y,d] - bad[a,b,z]
                + bad[x,y,d] + bad[x,b,z] + bad[a,y,z] - bad[x,y,z]) == 0

    stride = max(4,round(.032/settings.pitch_m))
    seeds = cells[(cells % stride == 0).all(1)] - origin
    result, seen = [], set()
    for seed in seeds:
        for order in itertools.permutations(range(3)):
            lo, hi = seed.copy(), seed.copy()
            for axis in order:
                # Alternate outward growth from each seed, then expand to the
                # largest feasible extent in each direction.
                for sign in (-1, 1):
                    near, far = 0, int(lo[axis] if sign < 0 else limit[axis]-hi[axis])
                    while near < far:
                        step = (near + far + 1)//2
                        a,b = lo.copy(), hi.copy()
                        if sign < 0:
                            a[axis] -= step
                        else:
                            b[axis] += step
                        if valid(a,b):
                            near = step
                        else:
                            far = step-1
                    if sign < 0:
                        lo[axis] -= near
                    else:
                        hi[axis] += near
            key = tuple(np.r_[lo,hi])
            if key in seen:
                continue
            seen.add(key)
            centre = (.5*(lo+hi) + origin) * settings.pitch_m
            half = .5*(hi-lo+1) * settings.pitch_m
            shape = Shape("box", tuple(np.r_[centre, [0.,0.,0.,1.], half]), "global-solid",
                          float(8*half.prod()))
            shape.bulge = protrusion(shape, settings, origin, distance)
            if shape.bulge <= settings.bulge_m:
                result.append(shape)
    print("interior boxes", len(result), flush=True)
    return result


def build_candidates(parts, boxes, settings, origin, distance):
    candidates = list(boxes)
    for name, cells, rotation, native_box in parts:
        if native_box:
            continue
        stack = [(cells, 0)]
        nodes = 0
        while stack:
            region, depth = stack.pop()
            points = region * settings.pitch_m
            fits, axes = enclosing_shapes(points, rotation, settings.pitch_m, name)
            valid = []
            for shape in fits:
                shape.bulge = protrusion(shape, settings, origin, distance)
                if shape.bulge <= settings.bulge_m:
                    valid.append(shape)
            nodes += 1
            if valid:
                candidates.extend(valid)
                # Add a tighter alternative decomposition only when a large region
                # remains. The selector may mix parent and child candidates.
                if depth >= 1 or len(region) < 150:
                    continue
            if len(region) <= 2 or depth >= 16:
                if not valid:
                    raise RuntimeError(f"cannot bound region at {name}, depth {depth}")
                continue
            # Median split along the region's longest PCA direction.
            values = points @ axes[:, -1]
            cut = .5 * (values.min() + values.max())
            mask = values <= cut
            if mask.all() or not mask.any():
                raise RuntimeError("degenerate split")
            stack.extend(((region[mask], depth + 1), (region[~mask], depth + 1)))
        print("candidates", name, nodes, len(candidates), flush=True)
    return candidates


def select_cover(candidates, cells, settings, boxes, origin, occupied):
    """The candidates' full-cube coverage of the required cells, then choose_cover."""
    points = cells * settings.pitch_m
    half_diagonal = np.sqrt(3) / 2 * settings.pitch_m
    tree = cKDTree(points)
    covered, kept = [], []
    for shape in candidates:
        lo, hi = shape.bounds()
        ids = np.asarray(tree.query_ball_point((lo + hi) / 2, np.linalg.norm(hi - lo) / 2 + half_diagonal), dtype=int)
        # A signed distance <= -half_diagonal is a cheap sufficient full-cube test. Boxes
        # use their exact per-axis support, which is tighter.
        v = np.asarray(shape.values)
        if shape.kind == "box":
            basis = Rotation.from_quat(v[3:7]).as_matrix()
            inside = (np.abs((points[ids] - v[:3]) @ basis)
                      + settings.pitch_m / 2 * np.abs(basis).sum(0) <= v[7:] + 1e-9).all(1)
        elif shape.kind == "sphere":
            inside = np.linalg.norm(np.abs(points[ids] - v[:3]) + settings.pitch_m / 2, axis=1) <= v[3] + 1e-9
        else:
            inside = np.ones(len(ids), dtype=bool)
            for d in np.array(list(np.ndindex(2, 2, 2))) - .5:
                inside &= shape.distance(points[ids] + d * settings.pitch_m) <= 1e-9
        ids = ids[inside]
        if len(ids):
            kept.append(shape)
            covered.append(ids)
    lengths = np.array([len(ids) for ids in covered])
    matrix = sparse.csr_matrix((np.ones(lengths.sum(), dtype=np.float32), np.concatenate(covered),
                                np.r_[0, lengths.cumsum()]), shape=(len(kept), len(cells)))
    return choose_cover(kept, matrix, cells, settings, boxes, origin, occupied)


def choose_cover(kept, matrix, cells, settings, boxes, origin, occupied):
    """Weighted greedy set cover of the cells, then redundant-shape removal."""
    covered = [matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]] for i in range(len(kept))]
    costs = np.array([getattr(settings, f"{s.kind}_cost") for s in kept])
    # Native boxes cover their true analytic geometry, including partially intersected
    # raster cells. Only their own raster is exempt; cells from meshes stay required even
    # where they overlap a box's boundary.
    interior = ndimage.binary_erosion(occupied, structure=np.ones((3, 3, 3)))
    weights = np.where(interior[tuple((cells - origin).T)], settings.interior_weight, 1.).astype(np.float32)
    uncovered = weights.copy()
    chosen = []
    costs *= 1 + 1e-7 * np.array([s.volume for s in kept])  # a tiny volume preference breaks ties
    while uncovered.any():
        scores = np.asarray(matrix @ uncovered) / costs
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            missing = cells[uncovered.astype(bool)] * settings.pitch_m
            raise RuntimeError(f"{len(missing)} cells have no covering candidate; first {missing[0]}")
        chosen.append(best)
        uncovered[covered[best]] = 0
    counts = np.asarray(matrix[chosen].sum(0)).ravel()
    for index in reversed(chosen.copy()):
        if (counts[covered[index]] > 1).all():
            chosen.remove(index)
            counts[covered[index]] -= 1
    result = [kept[i] for i in chosen]
    for box in boxes:  # native geometry stays exact even where other primitives explain its raster
        if not any(s.values == box.values for s in result):
            result.append(box)
    return result, dict(solid_cells=len(cells), candidates=len(kept), counts=dict(Counter(s.kind for s in result)),
                        weighted_cost=float(sum(getattr(settings, f"{s.kind}_cost") for s in result)))


def generate(choice: str) -> dict[str, Any]:
    """The cover's file content: primitives by shape in Collision's layouts, the frame,
    posture, inputs, settings, statistics, validation and library versions."""
    settings = CHOICES[choice]
    started = time.perf_counter()
    parts_, boxes, all_cells, origin, occupied, distance = voxel_body(settings)
    # Native boxes are exact; every cell of the other parts must be covered, also where
    # they overlap.
    required = np.unique(np.vstack([p[1] for p in parts_ if not p[3]]), axis=0)
    candidates = build_candidates(parts_, boxes, settings, origin, distance)
    candidates.extend(interior_spheres(all_cells, settings, origin, occupied, distance))
    candidates.extend(interior_boxes(all_cells, settings, origin, distance))
    shapes, stats = select_cover(candidates, required, settings, boxes, origin, occupied)
    stats["max_grid_bulge_m"] = max(s.bulge for s in shapes)
    stats["fit_s"] = time.perf_counter() - started
    primitives = {kind: [[float(v) for v in s.values] for s in shapes if s.kind == kind]
                  for kind in ("sphere", "capsule", "box")}
    body = [(obj, placement) for obj, placement, _ in parts("body", POSTURE).values()]
    validation = validate(body, primitives["sphere"], primitives["capsule"], primitives["box"], samples=300_000,
                          fill_pitch_m=.005)
    if validation["uncovered_samples"]:
        raise RuntimeError(f"the cover misses surface samples: {validation}")
    return dict(kind="body", name=choice, frame=ROOT_FRAME, posture=POSTURE,
                spheres=primitives["sphere"], capsules=primitives["capsule"], boxes=primitives["box"],
                inputs=inputs("body", POSTURE), settings=asdict(settings), stats=stats,
                validation=validation, versions=versions(), elapsed_s=time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--choice", choices=CHOICES, required=True)
    parser.add_argument("--output", type=Path, help="default: tiago_mpc/data/collision/body/<choice>.json")
    args = parser.parse_args()
    cover = generate(args.choice)
    output = args.output or c.cover_file("body", args.choice)
    output.write_text(json.dumps(cover, indent=1) + "\n")
    print(json.dumps(cover["stats"]), f"{cover['elapsed_s']:.0f} s", "->", output)


if __name__ == "__main__":
    main()
