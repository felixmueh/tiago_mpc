"""The right arm's exact sphere cover: tiago_mpc/data/collision/arm/exact.json.

Experimental: agent-generated (Claude), not reviewed in detail.

Spheres centred on a part's surface would stick out by their whole radius. Centred inside
instead, at a depth d below the surface, a sphere of radius d + bulge sticks out by at most
the bulge, and covers a large patch of surface where the solid is thick.

Per arm link (2 to 7; link 7 with the gripper welded at zero opening, collision.arm_cover),
in the link's frame: the parts are voxelized and filled, and each voxel's depth is its
distance to the nearest empty voxel. Each voxel is a candidate centre with radius
min(depth + bulge, max_radius). The voxels that touch empty space are the boundary, which
holds the surface; a candidate covers a boundary voxel when the voxel's whole cube is
inside the sphere. The fewest candidates that cover every boundary voxel are found exactly
by HiGHS (scipy.optimize.milp), proven optimal per link; a time limit or any other status
is an error. Every boundary voxel can cover itself, so a cover exists.

Run in the container from the checkout's root (about a minute; 28 spheres):
    .venv-container/bin/python -m tiago_scripts.arm_cover
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial import cKDTree

from tiago_mpc import collision as c
from tiago_scripts.body_cover import POSTURE
from tiago_scripts.validate import geometry_mesh, inputs, parts, validate, versions

@dataclass(frozen=True)
class Settings:
    """Spheres stick out of the surface by at most bulge_m (plus the voxel pitch) and are at
    most max_radius_m, on a voxel grid of pitch_m; each link's MILP gets time_limit_s."""
    bulge_m: float = .03
    max_radius_m: float = .10
    pitch_m: float = .005
    time_limit_s: float = 300.


def candidates(meshes, settings: Settings) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Candidate centres (n, 3), their radii (n,) and which boundary voxels each covers
    (n, boundary voxels), for the union of meshes (in their common frame)."""
    pitch = settings.pitch_m
    if settings.bulge_m < pitch or settings.max_radius_m < settings.bulge_m + pitch:
        raise ValueError("a cover needs bulge_m >= pitch_m and max_radius_m >= bulge_m + pitch_m")
    filled = np.vstack([mesh.voxelized(pitch).fill().points for mesh in meshes])
    origin = filled.min(0) - 2 * pitch
    cells = np.unique(np.round((filled - origin) / pitch).astype(int), axis=0)
    occupied = np.zeros(cells.max(0) + 3, dtype=bool)
    occupied[tuple(cells.T)] = True
    depth = ndimage.distance_transform_edt(occupied)[tuple(cells.T)] * pitch
    centres = origin + cells * pitch
    boundary = centres[depth <= np.sqrt(3) * pitch]  # face, edge or corner next to empty space
    radii = np.minimum(depth + settings.bulge_m, settings.max_radius_m)
    half_diagonal = .5 * np.sqrt(3) * pitch
    covered_by = cKDTree(boundary).query_ball_point(centres, radii - half_diagonal)
    rows = np.repeat(np.arange(len(centres)), [len(x) for x in covered_by])
    coverage = sparse.csr_matrix((np.ones(len(rows)), (rows, np.concatenate(covered_by).astype(int))),
                                 shape=(len(centres), len(boundary)))
    return centres, radii, coverage


def exact_cover(meshes, settings: Settings) -> tuple[list[c.Sphere], dict[str, Any]]:
    """The fewest candidate spheres covering the meshes' boundary voxels, and the solve's
    record. Raises RuntimeError unless HiGHS proves the cover optimal within the time limit."""
    centres, radii, coverage = candidates(meshes, settings)
    useful = np.flatnonzero(np.diff(coverage.indptr) > 0)  # candidates covering any boundary voxel
    start = time.perf_counter()
    result = milp(np.ones(len(useful)), constraints=LinearConstraint(coverage[useful].T.tocsc(), lb=1, ub=np.inf),
                  integrality=np.ones(len(useful)), bounds=Bounds(0, 1),
                  options=dict(time_limit=settings.time_limit_s, mip_rel_gap=0.))
    record = dict(candidates=len(useful), boundary_voxels=coverage.shape[1], nonzeros=int(coverage[useful].nnz),
                  status=int(result.status), message=result.message, objective=result.fun,
                  dual_bound=getattr(result, "mip_dual_bound", None), solve_s=time.perf_counter() - start)
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"the MILP did not prove an optimal cover: {record}")
    chosen = useful[np.flatnonzero(result.x > .5)]
    return [(float(centres[i, 0]), float(centres[i, 1]), float(centres[i, 2]), float(radii[i])) for i in chosen], record


def generate(settings: Settings = Settings()) -> dict[str, Any]:
    """The arm cover's file content: spheres by link, each link's solve and validation, the
    inputs, settings and library versions."""
    started = time.perf_counter()
    by_link: dict[str, list] = {}
    for obj, placement, link in parts("arm", POSTURE).values():  # in the links' frames: any posture
        by_link.setdefault(link, []).append((obj, placement))
    spheres, solve, validation = {}, {}, {}
    for index, (link, group) in enumerate(sorted(by_link.items())):
        spheres[link], solve[link] = exact_cover([geometry_mesh(obj, placement) for obj, placement in group], settings)
        validation[link] = validate(group, spheres[link], seed=index)
        print(link, len(spheres[link]), "spheres", f"{solve[link]['solve_s']:.1f} s",
              "uncovered", validation[link]["uncovered_samples"], flush=True)
        if validation[link]["uncovered_samples"]:
            raise RuntimeError(f"{link}'s cover misses surface samples: {validation[link]}")
    return dict(kind="arm", frame="each link's own frame", spheres=spheres,
                inputs=inputs("arm", POSTURE),
                settings=asdict(settings), solve=solve, validation=validation, versions=versions(),
                elapsed_s=time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=c.cover_file("arm", "exact"))
    args = parser.parse_args()
    cover = generate()
    args.output.write_text(json.dumps(cover, indent=1) + "\n")
    print(sum(map(len, cover["spheres"].values())), "spheres", f"{cover['elapsed_s']:.0f} s", "->", args.output)


if __name__ == "__main__":
    main()
