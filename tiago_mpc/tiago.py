"""TIAGo facts: the right arm's joints, postures, frames, models and raw limits,
and the files derived from the full URDF.

models/tiago_pro/tiago_pro.urdf is the only robot description in the repository and
the source of friction coefficients and limits. Armature and stored PD gains live here.
Every other robot file is derived from it
at run time and written to the cache (util.cache_root): the reduced arm (arm_urdf), its
GPU copy (runtime_urdf) and the safety collision model (safety.py). Nothing derived
is committed, so nothing derived can go stale.

Every URDF, the GPU copy included, keeps the native joint axes (joint 6 turns about -z):
the solver works in native TIAGo coordinates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from tiago_mpc.dynamics import Friction
from tiago_mpc.util import cache_meshes, cached

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

# Right arm, native joint order 1–7. Reflected inertia in kg m², fitted on 2026-10-30
# from one real chirp per joint as total inertia minus the URDF's link inertia
# (models/tiago_pro/README.md). It is not part of the URDF and is never folded into
# link inertia.
ARMATURE_KG_M2 = (.43, .43, .19, .17, .21, .17, .18)

# Stored PD gains: kp in Nm/rad, kd in Nm s/rad, native joint order 1–7.
# Retuned on 2026-10-30 with the recipe of the 2026-10-27 simulation tune: effective
# inertia at Q_START with ARMATURE_KG_M2 (now the fitted one), 20 rad/s bandwidth and
# damping ratio 1.4 (util.inertia_gains(20., 1.4, armature=ARMATURE_KG_M2)). These are not
# hardware-qualified gains. Edit these literals directly; util.inertia_gains
# provides estimates to copy here, without recalculating defaults at import.
PD_KP = (
    311.37012743826693, 329.82323220044566, 132.22973559712062, 172.74911197026864,
    91.72976415871804, 76.01463447546624, 72.3090348183072,
)
PD_KD = (
    43.59181784135736, 46.175252508062385, 18.512162983596884, 24.18487567583761,
    12.842166982220524, 10.642048826565272, 10.123264874563008,
)


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
    cache_meshes(arm, lambda filename: MESHES / Path(filename).name)  # package:// by basename
    ET.indent(arm)
    return cached("tiago_right_arm.urdf", ET.tostring(arm, encoding="utf-8", xml_declaration=True))


def _subtree_inertia(name, links, joints):
    """The inertia of link name and every link below it, all joints at zero, in name's frame."""

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


def urdf_friction() -> Friction:
    """The full URDF's right-arm friction and damping coefficients, without a numerical law."""
    model = _arm()[0]
    return Friction(tuple(model.friction), tuple(model.damping))


def fk(q: np.ndarray, frame: str = GRASP_FRAME) -> np.ndarray:
    """Position of frame in ROOT_FRAME, metres."""
    model, data = _arm()
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(frame)].translation.copy()


def ee_position(q: np.ndarray) -> np.ndarray:
    """Position of the grasp frame (GRASP_FRAME) in ROOT_FRAME, metres."""
    return fk(q, GRASP_FRAME)


def ee_orientation(q: np.ndarray) -> np.ndarray:
    """The grasp frame's orientation in ROOT_FRAME as a rotation vector in radians."""
    return pin.log3(_ee_rotation(q))


def orientation_error(q: np.ndarray, target: np.ndarray) -> float:
    """The angle in radians between the grasp frame at q and the rotation-vector target.

    The target is in ROOT_FRAME; the error is the norm of log(R_targetᵀ R(q)).
    """
    return float(np.linalg.norm(pin.log3(pin.exp3(np.asarray(target, dtype=float)).T @ _ee_rotation(q))))


def _ee_rotation(q: np.ndarray) -> np.ndarray:
    model, data = _arm()
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(GRASP_FRAME)].rotation.copy()


def ee_velocity(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Linear velocity of the grasp frame in ROOT_FRAME, m/s."""
    model, data = _arm()
    pin.computeJointJacobians(model, data, np.asarray(q, dtype=float))
    jacobian = pin.getFrameJacobian(model, data, model.getFrameId(GRASP_FRAME),
                                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return jacobian[:3] @ np.asarray(qd, dtype=float)


def runtime_urdf(friction: Friction | None = None) -> Path:
    """The GPU URDF: a copy of the reduced arm with friction's coefficients, if given, in the
    cache by its content (cached). GATO reads only its collision meshes, so the visuals are
    left out."""
    tree = ET.parse(arm_urdf())
    for link in tree.iter("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    if friction is not None:
        joints = [joint for joint in tree.iter("joint") if joint.get("type") != "fixed"]
        for joint, c, b in zip(joints, friction.coulomb_nm, friction.viscous_nm_s_per_rad):
            dynamics = joint.find("dynamics")
            if dynamics is None:
                dynamics = ET.SubElement(joint, "dynamics")
            dynamics.set("friction", repr(float(c)))
            dynamics.set("damping", repr(float(b)))
    payload = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    return cached("tiago_right_grid.urdf", payload)


if __name__ == "__main__":  # the derived files' paths, for debugging
    print(arm_urdf())
    print(runtime_urdf())
