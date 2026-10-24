"""The package root, for experiments: `import tiago_mpc as tm` and use `tm.<name>`.

It re-exports the names an experiment composes. Modules inside the package import each
other directly (`from .plan import State`), never through this root. Heavy dependencies
(Pinocchio, `gato.common`, ROS, the safety model) are imported where they are used, so
importing `tiago_mpc` stays light.
"""
from .clock import Clock
from .dynamics import Friction, Sign, StickSlip, Tanh, passive_torque
from .laws import hold_first, sample_knot
from .loops import Fixed, Measured, Schedule, checked, guard, run, run_robot, run_simulation
from .mpc import TIAGO, UPDATES, Mpc, MpcParams, MpcProcess, compile_binary, find_binary
from .plan import Callback, Law, MpcInterface, Plan, PlanEnded, PlantInterface, State, TaskInterface, Tick
from .record import Record
from .robot import RosPlant
from .sim import SimPlant
from .tiago import (FLANGE_FRAME, FULL_URDF, GRASP_FRAME, Q_REACH, Q_START, RIGHT_ARM_JOINTS, ROOT,
                    ROOT_FRAME, Limits, arm_urdf, ee_position, fk, limits, urdf_friction)

__all__ = [
    "Clock", "Friction", "Sign", "StickSlip", "Tanh", "passive_torque", "hold_first", "sample_knot",
    "Fixed", "Measured", "Schedule", "checked", "guard", "run", "run_robot", "run_simulation",
    "TIAGO", "UPDATES", "Mpc", "MpcParams", "MpcProcess", "compile_binary", "find_binary",
    "Callback", "Law", "MpcInterface", "Plan", "PlanEnded", "PlantInterface", "State", "TaskInterface", "Tick",
    "Record", "RosPlant", "SimPlant",
    "FLANGE_FRAME", "FULL_URDF", "GRASP_FRAME", "Q_REACH", "Q_START", "RIGHT_ARM_JOINTS", "ROOT",
    "ROOT_FRAME", "Limits", "arm_urdf", "ee_position", "fk", "limits", "urdf_friction",
]
