"""The exact clearances of the right arm that the collision covers approximate.

Coal's signed distances between the URDF's original collision meshes in safety's TIAGo
model, at the body cover's posture (tiago_scripts.validate.robot). Pairs of two arm parts are
not measured, because there are no self-collision rows; body pairs with arm_right_1_link
neither: it is mounted on the torso and has no spheres.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from tiago_mpc.collision import Collision, body_posture, box_rotation
from tiago_scripts.validate import geometry_mesh, robot, root_placement


def _z_along(axis: np.ndarray) -> np.ndarray:
    """A rotation whose z column is along axis (any, for a zero axis)."""
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    z = np.asarray(axis) / norm
    x = np.cross(z, [1., 0., 0.] if abs(z[0]) < .9 else [0., 1., 0.])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class Clearance:
    """The arm's signed distances to the body and to each obstacle of a Collision, at native
    arm positions. body None takes the default body cover's posture."""

    def __init__(self, collision: Collision):
        import coal

        self.model, self.geometry_model = robot(body_posture(collision.body))
        self.data, self.gdata = self.model.createData(), pin.GeometryData(self.geometry_model)
        objects = self.geometry_model.geometryObjects
        self.arm = [i for i, o in enumerate(objects)
                    if o.parentJoint != 0 and self.model.names[o.parentJoint] != "arm_right_1_joint"]
        self.body = [i for i, o in enumerate(objects) if o.parentJoint == 0]
        # Bounding spheres in the parent joints' frames: a pair is measured only while their
        # distance, a lower bound of the pair's, is below the smallest distance found so far.
        vertices = [np.asarray(geometry_mesh(o, pin.SE3.Identity()).vertices) for o in objects]
        self.bound_centres = np.array([.5 * (v.min(0) + v.max(0)) for v in vertices])
        self.bound_radii = np.array([np.linalg.norm(v - c, axis=1).max() for v, c in zip(vertices, self.bound_centres)])
        self.vertices = {i: vertices[i] for i in self.arm}  # in the parent joint's frame
        root = root_placement(self.model)

        def placed(rotation, centre):  # a pose in ROOT_FRAME, placed in the world
            return coal.Transform3s(root.rotation @ rotation, root.act(np.asarray(centre, dtype=float)))

        # Coal's capsule is centred on its z axis, with its full length.
        self.obstacles = ([(coal.Sphere(r), placed(np.eye(3), (x, y, z))) for x, y, z, r in collision.obstacles]
                          + [(coal.Capsule(c[6], np.linalg.norm(np.subtract(c[3:6], c[:3]))),
                              placed(_z_along(np.subtract(c[3:6], c[:3])), np.add(c[:3], c[3:6]) / 2))
                             for c in collision.capsules]
                          + [(coal.Box(*(2 * np.asarray(b[7:]))), placed(box_rotation(b), b[:3]))
                             for b in collision.boxes])
        # Planes in the world (n, d). Coal's mesh-halfspace penetration depth is not exact,
        # so distances measures planes itself, from each part's lowest point along n.
        self.planes = [(root.rotation @ np.asarray(p[:3]), p[3] + (root.rotation @ np.asarray(p[:3])) @ root.translation)
                       for p in collision.planes]
        self.obstacle_names = [f"{kind} {i}" for kind, group in
                               (("sphere", collision.obstacles), ("capsule", collision.capsules),
                                ("box", collision.boxes), ("plane", collision.planes))
                               for i in range(len(group))]
        self.request = coal.DistanceRequest()
        self.request.enable_signed_distance = True

    def distances(self, q: np.ndarray) -> tuple[float, np.ndarray, str, str]:
        """The smallest signed distance of the arm to the body, the arm's smallest signed
        distance to each obstacle (in the order of obstacle_names), and the arm and body
        geometry names of the body minimum.

        Coal measures no penetration between two meshes: intersecting meshes are at 0, so
        count contacts as distances <= 0."""
        import coal

        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        pin.updateGeometryPlacements(self.model, self.data, self.geometry_model, self.gdata)
        objects = self.geometry_model.geometryObjects

        def placed(i):
            p = self.gdata.oMg[i]
            return objects[i].geometry, coal.Transform3s(p.rotation, p.translation)

        centres = np.array([self.data.oMi[o.parentJoint].act(c) for o, c in zip(objects, self.bound_centres)])
        body, names = np.inf, ("", "")
        obstacle = np.full(len(self.obstacles) + len(self.planes), np.inf)
        for a in self.arm:
            shape_a, tf_a = placed(a)
            bound = (np.linalg.norm(centres[self.body] - centres[a], axis=1)
                     - self.bound_radii[self.body] - self.bound_radii[a])
            for k in np.argsort(bound):
                if bound[k] >= body:
                    break
                b = self.body[k]
                result = coal.DistanceResult()
                coal.distance(shape_a, tf_a, *placed(b), self.request, result)
                if result.min_distance < body:
                    body, names = result.min_distance, (objects[a].name, objects[b].name)
            for j, (shape_b, tf_b) in enumerate(self.obstacles):
                result = coal.DistanceResult()
                coal.distance(shape_a, tf_a, shape_b, tf_b, self.request, result)
                obstacle[j] = min(obstacle[j], result.min_distance)
            for j, (n, d) in enumerate(self.planes, start=len(self.obstacles)):
                obstacle[j] = min(obstacle[j], self._lowest(a, n) - d)
        return float(body), obstacle, *names

    def _lowest(self, i: int, n: np.ndarray) -> float:
        """The smallest n.p over arm part i: exact for a cylinder, over the vertices for a mesh."""
        import coal

        obj, placement = self.geometry_model.geometryObjects[i], self.gdata.oMg[i]
        if isinstance(obj.geometry, coal.Cylinder):
            along = abs(n @ placement.rotation[:, 2])
            return float(n @ placement.translation - obj.geometry.halfLength * along
                         - obj.geometry.radius * np.sqrt(max(1 - along**2, 0.)))
        joint = self.data.oMi[obj.parentJoint]
        return float(np.min((self.vertices[i] @ joint.rotation.T + joint.translation) @ n))
