"""The package root, for experiments: `import tiago_mpc as tm` and use `tm.<name>`.

It re-exports the names an experiment composes. Modules inside the package import each
other with absolute imports (`from tiago_mpc.interfaces import State`), never through this
root. Importing the package may load Pinocchio and GATO's Python package, but never ROS
(only creating a RosPlant does) or GATO's compiled module (only the MPC process).
"""
from tiago_mpc.clock import VirtualClock, WallClock
from tiago_mpc.costs import default_cost, with_reference
from tiago_mpc.dynamics import Friction, Sign, StickSlip, Tanh, passive_torque
from tiago_mpc.laws import (EffortClip, ExecutionSafety, FrictionFeedforward, HoldFirst, PlanFeedback,
                            RealTimeEffort, RealTimeSafety, RejectOnEffort, SampleKnot, decide_command)
from tiago_mpc.loops import (STOP, Outcome, RealTimeSettings, Schedule, VirtualTimeSettings,
                             default_compensation, run_real_time, run_virtual_time)
from tiago_mpc.mpc import TIAGO, UPDATES, Mpc, MpcParams, MpcProcess, SolverSettings, compile_binary, find_binary
from tiago_mpc.prediction import LatencyCompensation
from tiago_mpc.interfaces import (Command, ControlLaw, Diagnostic, DiagnosticCallback, EffortHandling, Fault,
                                  MpcInterface, Plan, PlantInterface, Prediction, SafetyInterface, SolveCompleted,
                                  SolveCompletedCallback, State, TaskInterface, Tick, TickCallback)
from tiago_mpc.record import Console, RunRecord, from_dict, load_run, provenance, to_dict
from tiago_mpc.robot import RobotConfig, RosPlant
from tiago_mpc.sim import SimConfig, SimPlant
from tiago_mpc.tiago import (ARMATURE_KG_M2, FLANGE_FRAME, FULL_URDF, GRASP_FRAME, PD_KD, PD_KP,
                    Q_REACH, Q_START, RIGHT_ARM_JOINTS, ROOT,
                    ROOT_FRAME, Limits, arm_urdf, ee_orientation, ee_position, ee_velocity, fk, limits,
                    orientation_error, urdf_friction)
from tiago_mpc.util import hz_to_s, inertia_gains

__all__ = [
    "VirtualClock", "WallClock", "Friction", "Sign", "StickSlip", "Tanh", "passive_torque",
    "EffortClip", "RejectOnEffort", "EffortHandling", "ExecutionSafety", "RealTimeSafety", "RealTimeEffort", "SafetyInterface",
    "FrictionFeedforward", "HoldFirst", "PlanFeedback", "SampleKnot", "inertia_gains", "decide_command",
    "STOP", "Outcome", "RealTimeSettings", "Schedule", "VirtualTimeSettings",
    "default_compensation", "run_real_time", "run_virtual_time",
    "TIAGO", "UPDATES", "Mpc", "MpcParams", "MpcProcess", "SolverSettings", "default_cost", "with_reference",
    "compile_binary", "find_binary",
    "LatencyCompensation", "Prediction",
    "TickCallback", "SolveCompletedCallback", "DiagnosticCallback", "SolveCompleted", "ControlLaw", "Diagnostic",
    "MpcInterface", "Plan", "PlantInterface",
    "State", "TaskInterface", "Tick", "Command", "Fault",
    "Console", "RunRecord", "from_dict", "load_run", "provenance", "to_dict",
    "RobotConfig", "RosPlant", "SimConfig", "SimPlant",
    "ARMATURE_KG_M2", "FLANGE_FRAME", "FULL_URDF", "GRASP_FRAME", "PD_KD", "PD_KP",
    "Q_REACH", "Q_START", "RIGHT_ARM_JOINTS", "ROOT",
    "ROOT_FRAME", "Limits", "arm_urdf", "ee_orientation", "ee_position", "ee_velocity", "fk", "limits",
    "orientation_error", "urdf_friction",
    "hz_to_s",
]
