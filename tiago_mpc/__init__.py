"""The package root, for experiments: `import tiago_mpc as tm` and use `tm.<name>`.

It re-exports the names an experiment composes. Modules inside the package import each
other with absolute imports (`from tiago_mpc.plan import State`), never through this
root. Importing the package may load Pinocchio and GATO's Python package, but never ROS
(only creating a RosPlant does) or GATO's compiled module (only the MPC process).
"""
from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh, passive_torque
from tiago_mpc.laws import hold_first, sample_knot
from tiago_mpc.loops import Outcome, Schedule, checked, guard, run_real_time, run_virtual_time
from tiago_mpc.mpc import TIAGO, UPDATES, Mpc, MpcParams, MpcProcess, compile_binary, find_binary
from tiago_mpc.plan import Callback, Law, MpcInterface, Plan, PlanEnded, PlantInterface, State, TaskInterface, Tick
from tiago_mpc.record import Record, gato_source
from tiago_mpc.robot import RosPlant
from tiago_mpc.sim import SimPlant
from tiago_mpc.tiago import (FLANGE_FRAME, FULL_URDF, GRASP_FRAME, Q_REACH, Q_START, RIGHT_ARM_JOINTS, ROOT,
                    ROOT_FRAME, Limits, arm_urdf, ee_position, ee_velocity, fk, limits, urdf_friction)

__all__ = [
    "VirtualClock", "WallClock", "Friction", "Sign", "StickSlip", "Tanh", "passive_torque", "hold_first", "sample_knot",
    "Outcome", "Schedule", "checked", "guard", "run_real_time", "run_virtual_time",
    "TIAGO", "UPDATES", "Mpc", "MpcParams", "MpcProcess", "compile_binary", "find_binary",
    "Callback", "Law", "MpcInterface", "Plan", "PlanEnded", "PlantInterface", "State", "TaskInterface", "Tick",
    "Record", "gato_source", "RosPlant", "SimPlant",
    "FLANGE_FRAME", "FULL_URDF", "GRASP_FRAME", "Q_REACH", "Q_START", "RIGHT_ARM_JOINTS", "ROOT",
    "ROOT_FRAME", "Limits", "arm_urdf", "ee_position", "ee_velocity", "fk", "limits", "urdf_friction",
]
