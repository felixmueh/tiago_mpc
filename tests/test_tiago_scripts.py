"""The cover generators' pieces on small inputs (CPU only; no regeneration of the stored
covers, which takes minutes: tiago_scripts/README.md)."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from tiago_scripts import arm_cover, body_cover, validate

GROUP = dict(sphere="spheres", capsule="capsules", box="boxes")  # validate.distance's arguments
SMALL = arm_cover.Settings(bulge_m=.02, max_radius_m=.06, pitch_m=.01)


def test_the_exact_arm_cover_covers_a_box_with_no_more_spheres_than_the_greedy():
    box = trimesh.creation.box(extents=(.12, .08, .05))
    spheres, record = arm_cover.exact_cover([box], SMALL)
    assert record["status"] == 0 and record["objective"] == len(spheres)
    s = np.asarray(spheres)
    assert (s[:, 3] <= SMALL.max_radius_m).all()
    inside = np.min(np.linalg.norm(box.vertices[:, None] - s[None, :, :3], axis=2) - s[None, :, 3], axis=1)
    assert (inside <= 1e-9).all()  # the corners are inside the cover
    centres, radii, coverage = arm_cover.candidates([box], SMALL)
    uncovered, greedy = np.ones(coverage.shape[1]), 0
    while uncovered.any():  # the greedy that the exact cover replaced
        uncovered[coverage[int(np.argmax(coverage @ uncovered))].indices] = 0
        greedy += 1
    assert len(spheres) <= greedy


def test_the_exact_arm_cover_fails_when_the_milp_is_not_proven_optimal():
    with pytest.raises(RuntimeError, match="did not prove an optimal cover"):
        arm_cover.exact_cover([trimesh.creation.box(extents=(.12, .08, .05))], replace(SMALL, time_limit_s=0.))


def test_the_conservative_raster_keeps_thin_triangles_between_their_vertices():
    vertices = np.array([[-.071, .012, .007], [.113, .013, .009], [.032, .014, .008]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    pitch = .01
    cells = set(map(tuple, body_cover.mesh_cells(mesh, pitch)))
    points = np.random.default_rng(0).dirichlet([1, 1, 1], size=10000) @ vertices
    assert all(tuple(cell) in cells for cell in np.floor(points / pitch + .5).astype(int))


def test_every_enclosing_primitive_contains_the_full_voxel_cubes():
    rng = np.random.default_rng(1)
    cells = np.unique(rng.integers(-8, 9, size=(200, 3)), axis=0)
    pitch = .008
    points = cells * pitch
    fits, _ = body_cover.enclosing_shapes(points, Rotation.from_rotvec([.4, -.2, .7]).as_matrix(), pitch, "fixture")
    corners = np.array(list(np.ndindex(2, 2, 2))) - .5
    support = (points[:, None, :] + pitch * corners).reshape(-1, 3)
    assert {s.kind for s in fits} == {"sphere", "capsule", "box"}
    for shape in fits:
        assert shape.distance(support).max() <= 1e-8, shape.kind
        # validate's distance agrees with the generator's for each kind
        np.testing.assert_allclose(validate.distance(support, **{GROUP[shape.kind]: [shape.values]}),
                                   shape.distance(support), atol=1e-12)
