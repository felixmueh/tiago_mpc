"""The stored covers for GATO's collision rows and the GPU URDF that carries the arm's (CPU
only)."""
import json
from typing import get_args
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tiago_mpc import collision, tiago

BODY_COUNTS = {"balanced": (38, 4, 41), "tighter": (65, 12, 78)}  # spheres, capsules, boxes


def test_the_stored_covers_load_with_their_counts():
    arm = collision.arm_cover()  # "exact"
    assert {link: len(spheres) for link, spheres in arm.items()} == {
        "arm_right_2_link": 3, "arm_right_3_link": 3, "arm_right_4_link": 3, "arm_right_5_link": 5,
        "arm_right_6_link": 2, "arm_right_7_link": 12}
    for name, counts in BODY_COUNTS.items():
        body = collision.body_cover(name)
        assert tuple(len(body[key]) for key in ("spheres", "capsules", "boxes")) == counts, name
    assert set(collision.cover_hashes()) == {"arm/exact", *(f"body/{name}" for name in BODY_COUNTS)}
    assert set(collision.cover_hashes()) == {*(f"arm/{name}" for name in get_args(collision.ArmCover)),
                                            *(f"body/{name}" for name in get_args(collision.BodyCover))}
    with pytest.raises(ValueError, match="no stored body cover 'missing'"):
        collision.body_cover("missing")  # type: ignore[arg-type]


def test_the_stored_covers_were_made_from_the_current_model_and_posture():
    """The only consistency check: loading checks nothing. On a failure, regenerate the covers
    (tiago_scripts/README.md)."""
    from tiago_scripts.body_cover import POSTURE
    from tiago_scripts.validate import inputs

    for kind, names in (("arm", ("exact",)), ("body", tuple(BODY_COUNTS))):
        for name in names:
            stored = json.loads(collision.cover_file(kind, name).read_text())
            current = inputs(kind, POSTURE)
            assert stored["inputs"] == current, f"{kind}/{name}: python -m tiago_scripts.{kind}_cover"
            if kind == "body":
                assert stored["posture"] == POSTURE and collision.body_posture(name) == POSTURE, name


def test_the_environment_merges_the_body_covers_shapes_with_the_obstacles():
    box = (.1, .2, .3, 0., 0., 0., 1., .1, .2, .3)
    obstacles = dict(obstacles=((0., 0., 0., 1.),), capsules=((0., 0., 0., 1., 1., 1., .1),), boxes=(box,))
    environment = collision.Collision(**obstacles).environment()  # the balanced body cover
    body = collision.body_cover("balanced")
    assert environment["spheres"] == body["spheres"] + obstacles["obstacles"]
    assert environment["capsules"] == body["capsules"] + obstacles["capsules"]
    assert environment["cuboids"] == tuple(map(collision.cuboid, body["boxes"] + (box,)))


def test_the_gpu_urdf_carries_the_spheres_as_its_only_collision_geometry(monkeypatch, tmp_path):
    monkeypatch.setenv("GATO_CACHE_DIR", str(tmp_path))
    spheres = {"arm_right_3_link": ((.01, .02, .03, .04),), "arm_right_7_link": ((0., 0., .1, .05), (0., 0., .2, .05))}
    gpu = tiago.runtime_urdf(None, spheres)
    assert gpu != tiago.runtime_urdf()  # without spheres the GPU URDF is unchanged, another path
    tree = ET.parse(gpu)
    assert not tree.findall("link/collision/geometry/mesh")
    found = {link.get("name"): [(*map(float, c.find("origin").get("xyz").split()),
                                 float(c.find("geometry/sphere").get("radius"))) for c in link.findall("collision")]
             for link in tree.iter("link") if link.findall("collision")}
    assert found == {name: list(s) for name, s in spheres.items()}


def test_a_box_becomes_gatos_cuboid_with_its_axes_from_the_quaternion():
    turned = collision.cuboid((1., 2., 3., 0., 0., 2 * np.sin(np.pi / 4), 2 * np.cos(np.pi / 4), .1, .2, .3))
    np.testing.assert_allclose(turned, (1., 2., 3., 0., 1., 0., .1, -1., 0., 0., .2, 0., 0., 1., .3), atol=1e-12)
    box = (0., 0., 0., *Rotation.random(random_state=0).as_quat(), .1, .2, .3)
    axes = np.array(collision.cuboid(box)[3:]).reshape(3, 4)
    np.testing.assert_allclose(axes[:, :3] @ axes[:, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(axes[:, 3], (.1, .2, .3))


def test_the_environment_holds_every_obstacle_type_in_gatos_layouts():
    box = (.1, .2, .3, 0., 0., 0., 1., .1, .2, .3)
    parameters = collision.Collision(body=None, obstacles=((0, 0, 0, 1),), capsules=((0, 0, 0, 1, 1, 1, .1),),
                                     boxes=(box,), planes=((0, 0, 4, 2),))
    assert parameters.planes == ((0., 0., 1., .5),)  # normalized
    assert parameters.environment() == dict(
        spheres=((0., 0., 0., 1.),), capsules=((0., 0., 0., 1., 1., 1., .1),),
        cuboids=((.1, .2, .3, 1., 0., 0., .1, 0., 1., 0., .2, 0., 0., 1., .3),), planes=((0., 0., 1., .5),))
    assert collision.Collision(body=None).environment() == dict(spheres=(), capsules=(), cuboids=(), planes=())


def test_collision_parameters_reject_malformed_obstacles():
    with pytest.raises(ValueError, match="spheres"):
        collision.Collision(obstacles=((0., 0., 0.),))
    with pytest.raises(ValueError, match="positive radius"):
        collision.Collision(obstacles=((0., 0., 0., 0.),))
    with pytest.raises(ValueError, match="mechanism"):
        collision.Collision(mechanism="hard")
    for malformed, message in ((dict(capsules=((0., 0., 0., 1., 1., 1.),)), "capsules are"),
                               (dict(capsules=((0., 0., 0., 1., 1., 1., 0.),)), "positive radius"),
                               (dict(boxes=((0., 0., 0., 0., 0., 0., 1., .1, .1),)), "boxes are"),
                               (dict(boxes=((0., 0., 0., 0., 0., 0., 1., .1, 0., .1),)), "half-extents"),
                               (dict(boxes=((0., 0., 0., 0., 0., 0., 0., .1, .1, .1),)), "quaternion"),
                               (dict(planes=((0., 0., 0., 1.),)), "normal"),
                               (dict(planes=((0., 0., np.nan, 1.),)), "planes are")):
        with pytest.raises(ValueError, match=message):
            collision.Collision(**malformed)
