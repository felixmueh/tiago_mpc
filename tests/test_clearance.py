"""The exact clearances of experiments/clearance.py (CPU only)."""
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.clearance import Clearance
from tiago_mpc import tiago
from tiago_mpc.collision import Collision


def test_the_clearance_sees_the_body_and_an_obstacle_at_the_grasp_point():
    target = tiago.ee_position(tiago.Q_START)
    check = Clearance(Collision(obstacles=((*target, .04),)))  # the grasp point is between the fingers
    body, obstacle, arm_name, body_name = check.distances(tiago.Q_START)
    assert body > .05 and obstacle.shape == (1,) and obstacle[0] <= 0  # Q_START clears the body by about 7 cm
    assert arm_name.startswith("arm_right") and body_name


def test_the_clearance_measures_capsules_boxes_and_planes():
    q, target = tiago.Q_START, tiago.ee_position(tiago.Q_START)
    sphere = Clearance(Collision(obstacles=((*target + [0, 0, -.3], .03),))).distances(q)[1][0]
    point = (*target + [0, 0, -.3], *target + [0, 0, -.3], .03)  # a capsule of zero length is that sphere
    np.testing.assert_allclose(Clearance(Collision(capsules=(point,))).distances(q)[1], [sphere], atol=1e-6)
    turned = (*target, *Rotation.from_euler("xyz", (.3, -.2, .9)).as_quat(), .05, .05, .05)
    assert Clearance(Collision(boxes=(turned,))).distances(q)[1][0] <= 0  # a cube around the grasp point
    # A plane's clearance is the lowest point of the arm above it; flipped, the highest below it.
    check = Clearance(Collision(planes=((0., 0., 2., -2.), (0., 0., -1., 1.))))  # z >= -1 and z <= -1
    lowest, highest = check.distances(q)[1] + [-1, 1]
    assert check.obstacle_names == ["plane 0", "plane 1"]
    assert -.6 < lowest < target[2] < -highest < .2
    below = Clearance(Collision(planes=((0., 0., 1., lowest + .01),))).distances(q)[1][0]
    np.testing.assert_allclose(below, -.01, atol=1e-9)  # penetration is exact too
