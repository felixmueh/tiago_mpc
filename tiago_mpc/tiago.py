"""TIAGo facts: the right arm's joints, postures, frames, models and raw limits,
and the files derived from the full URDF.

models/tiago_pro/tiago_pro.urdf is the only robot description in the repository and
the source of the coefficients and limits. Every other robot file is derived from it
at run time and written to the cache (cache_root): the reduced arm (arm_urdf), its
GPU copy (runtime_urdf) and the safety collision model (safety.py). Nothing derived
is committed, so nothing derived can go stale.

Importing this module loads only NumPy; Pinocchio loads on first use. The native
URDFs stay in ROS coordinates. Only the GPU URDF and the solver's state/control
boundary normalize negative joint axes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cache
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .dynamics import Friction

ROOT = Path(__file__).resolve().parents[1]  # the repository
FULL_URDF = ROOT / "models/tiago_pro/tiago_pro.urdf"  # the whole robot
MESHES = ROOT / "models/tiago_pro/meshes"
RIGHT_ARM_JOINTS = tuple(f"arm_right_{i}_joint" for i in range(1, 8))  # native order
ROOT_FRAME = "torso_lift_link"  # the reduced arm's fixed root; positions are in this frame
GRASP_FRAME = "arm_right_grasp_joint"  # nominal grasp center, the default end effector
FLANGE_FRAME = "arm_right_tool_joint"
# Runs start here at rest (checked on the robot): PAL's comfortable_high_clearance with
# joint 4 at -2.33 instead of -2.35, inside the MPC's planning box (its lower limit -2.4435
# plus MpcParams.joint_margin_rad 0.1).
Q_START = np.array([-0.39, -1.73, -0.38, -2.33, 0.0, -1.21, 0.04])
# A nearby posture: the default reach target's, and the model check's second pose.
Q_REACH = Q_START + np.array([0.18, 0.12, -0.10, 0.18, 0.0, 0.0, 0.0])


def cache_root() -> Path:
    """GATO_CACHE_DIR if set (host runs), else <repo>/.cache/gato; exported for GATO.
    GATO's modules and every derived robot file live here."""
    root = Path(os.environ.get("GATO_CACHE_DIR") or ROOT / ".cache/gato").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GATO_CACHE_DIR"] = str(root)
    return root


def cached(name: str, payload: bytes) -> Path:
    """payload as <cache_root()>/models/<hash of payload>/name. Content-addressed, so
    concurrent processes deriving different files never share a path; written
    atomically, and only when missing or different."""
    path = cache_root() / "models" / digest(payload) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.read_bytes() == payload):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    return path


def digest(*parts: bytes) -> str:
    """Eight hex characters over the parts (length-prefixed, so order and splits matter)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little") + part)
    return h.hexdigest()[:8]


# The reduced right arm. GATO plans for a fixed-base serial chain, and the MPC moves
# only the seven right-arm joints: the torso link becomes the fixed root, and the
# base, torso lift, head and left arm are left out. The gripper's fingers are not
# controlled either. They stay at their neutral posture (all joints at zero), where
# they are rigid parts of the tool link, so their links collapse into it: one link
# with their summed mass and inertia. Their geometry goes with them; the full URDF
# keeps it for safety and display.
def arm_urdf() -> Path:
    """The reduced right arm, derived from FULL_URDF (milliseconds) into the cache: the
    torso as fixed root, the seven arm joints, the flange (tool link) carrying the
    gripper's inertia, and the grasp frame."""
    robot = ET.parse(FULL_URDF).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    chain = [joints[name] for name in (*RIGHT_ARM_JOINTS, "arm_right_tool_joint")]
    arm = ET.Element("robot", name="tiago_pro_right_arm")
    arm.append(deepcopy(links[ROOT_FRAME]))
    arm.extend(deepcopy(links[_link(joint, "child")]) for joint in chain)
    tool = arm[-1]
    _replace_inertial(tool, _subtree_inertia(tool.get("name"), links, joints))
    arm.extend(deepcopy(joint) for joint in chain)
    # The grasp frame: massless, at the nominal grasp center, the midpoint of the two
    # fingertip pads at the neutral posture (the centre of fingertip.stl's flat inner
    # face, x = -11.5 mm, y = 7..47 mm, z = ±10 mm, through the full URDF's finger
    # joints; the tiny asymmetry from PAL's rounded rotations is idealized away). A
    # geometric convention, not a calibrated TCP; it does not move as the gripper opens.
    ET.SubElement(arm, "link", name="arm_right_grasp_link")
    grasp = ET.SubElement(arm, "joint", name=GRASP_FRAME, type="fixed")
    ET.SubElement(grasp, "parent", link="arm_right_tool_link")
    ET.SubElement(grasp, "child", link="arm_right_grasp_link")
    ET.SubElement(grasp, "origin", rpy="0 0 0", xyz="0 0 0.139288457525")
    ET.indent(arm)
    return cached("tiago_right_arm.urdf", ET.tostring(arm, encoding="utf-8", xml_declaration=True))


def _subtree_inertia(name, links, joints):
    """The inertia of link name and every link below it, all joints at zero, in name's frame."""
    import pinocchio as pin

    total = pin.Inertia.Zero()
    pending = [(name, pin.SE3.Identity())]
    while pending:
        link, placement = pending.pop()
        inertial = links[link].find("inertial")
        if inertial is not None:
            i = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
            matrix = np.array([[i["ixx"], i["ixy"], i["ixz"]],
                               [i["ixy"], i["iyy"], i["iyz"]],
                               [i["ixz"], i["iyz"], i["izz"]]])
            local = pin.Inertia(float(inertial.find("mass").get("value")), np.zeros(3), matrix)
            total += (placement * _origin(inertial)).act(local)
        pending.extend((_link(joint, "child"), placement * _origin(joint))
                       for joint in joints.values() if _link(joint, "parent") == link)
    return total


def _link(joint, tag):
    """The link name of joint's <parent> or <child>."""
    return joint.find(tag).get("link")


def _origin(element):
    """The placement of element's <origin> (URDF's fixed-axis roll, pitch, yaw)."""
    import pinocchio as pin

    origin = element.find("origin")
    attributes = {} if origin is None else origin.attrib
    xyz, rpy = (np.fromstring(attributes.get(key, "0 0 0"), sep=" ") for key in ("xyz", "rpy"))
    return pin.SE3(pin.rpy.rpyToMatrix(rpy), xyz)


def _replace_inertial(link, inertia):
    number = "{:.17g}".format
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(number, inertia.lever)), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=number(inertia.mass))
    (xx, xy, xz), (_, yy, yz), (_, _, zz) = inertia.inertia
    ET.SubElement(inertial, "inertia", ixx=number(xx), ixy=number(xy), ixz=number(xz),
                  iyy=number(yy), iyz=number(yz), izz=number(zz))
    for old in link.findall("inertial"):
        link.remove(old)
    link.insert(0, inertial)


@dataclass(frozen=True)
class Limits:
    """Raw joint limits in native order: position (rad), velocity (rad/s), effort (Nm)."""
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray


@cache
def _arm():
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(arm_urdf()))
    return model, model.createData()


@cache
def limits() -> Limits:
    """The full URDF's raw right-arm limits (read-only)."""
    model = _arm()[0]
    arrays = [np.array(a) for a in (model.lowerPositionLimit, model.upperPositionLimit,
                                    model.velocityLimit, model.effortLimit)]
    for a in arrays:
        a.flags.writeable = False
    return Limits(*arrays)


def urdf_friction(width_rad_s: float | None = None) -> Friction:
    """The full URDF's right-arm friction and damping coefficients (the only default)."""
    model = _arm()[0]
    return Friction(tuple(model.friction), tuple(model.damping), width_rad_s)


def fk(q: np.ndarray, frame: str = GRASP_FRAME) -> np.ndarray:
    """Position of frame in ROOT_FRAME, metres."""
    import pinocchio as pin
    model, data = _arm()
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(frame)].translation.copy()


def runtime_urdf(friction=None):
    """The GPU URDF and the GPU coordinate signs.

    A copy of the reduced arm with resolved mesh paths and friction's coefficients, if
    given, whose negative joint axes are turned positive, with their limits; in the
    cache, by its content (cached)."""
    tree = ET.parse(arm_urdf())
    joints = [joint for joint in tree.iter("joint") if joint.get("type") != "fixed"]
    if friction is not None:  # the coefficients survive the axis flips below
        for joint, c, b in zip(joints, friction.coulomb_nm, friction.viscous_nm_s_per_rad):
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("friction", repr(float(c)))
            dynamics.set("damping", repr(float(b)))
    for mesh in tree.iter("mesh"):
        path = MESHES / Path(mesh.attrib["filename"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    signs = []
    for joint in joints:
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if not (np.array_equal(axis, [0, 0, 1]) or np.array_equal(axis, [0, 0, -1])):
            raise ValueError(f"Unsupported arm axis: {axis}")
        sign = float(axis[2])
        signs.append(sign)
        if sign < 0:
            joint.find("axis").set("xyz", "0 0 1")
            limit = joint.find("limit")
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
            limit.set("lower", repr(-hi))
            limit.set("upper", repr(-lo))
            safety = joint.find("safety_controller")
            if safety is not None:
                lo, hi = float(safety.get("soft_lower_limit")), float(safety.get("soft_upper_limit"))
                safety.set("soft_lower_limit", repr(-hi))
                safety.set("soft_upper_limit", repr(-lo))
    payload = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    return cached("tiago_right_grid.urdf", payload), np.asarray(signs)


if __name__ == "__main__":  # the derived files' paths, for debugging
    print(arm_urdf())
    print(runtime_urdf()[0])
