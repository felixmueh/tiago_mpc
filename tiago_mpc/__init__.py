"""The package root, for experiments: `import tiago_mpc as tm` and use `tm.<name>`.

It re-exports the names an experiment composes (parameters, MPC, plants, loops, laws,
records, TIAGo facts), each imported on first use: importing the package, or any one of its
modules, loads neither Pinocchio, GATO nor ROS. Modules inside the package import each
other directly (`from .plan import State`), never through this root.
"""
from importlib import import_module

_MODULES = {
    "clock": "Clock",
    "dynamics": "Friction passive_torque",
    "laws": "hold_first sample_knot",
    "loops": "Fixed Measured Schedule checked guard run run_robot run_simulation",
    "mpc": "TIAGO UPDATES Mpc MpcParams MpcProcess compile_binary find_binary",
    "plan": "Callback Law MpcInterface Plan PlanEnded PlantInterface State TaskInterface Tick",
    "record": "Record",
    "robot": "RosPlant",
    "sim": "SimPlant",
    "tiago": "FLANGE_FRAME FULL_URDF GRASP_FRAME Q_REACH Q_START RIGHT_ARM_JOINTS ROOT "
             "ROOT_FRAME Limits arm_urdf ee_position fk limits urdf_friction",
}
_SOURCE = {name: module for module, names in _MODULES.items() for name in names.split()}
__all__ = sorted(_SOURCE)


def __getattr__(name):
    if name not in _SOURCE:
        raise AttributeError(f"module 'tiago_mpc' has no attribute {name!r}")
    return getattr(import_module(f".{_SOURCE[name]}", __name__), name)
