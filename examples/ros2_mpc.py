#!/usr/bin/env python3
"""Run the supervised right-arm ROS 2 MPC application (checks only by default)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/gato/python"))

from experiments.figure8 import FIGURE8, Figure8
from experiments.reach import Reach
from tiago_mpc.mpc import TIAGO
from tiago_mpc.ros2 import TiagoRightArmClient
from tiago_mpc.runner import RosMpcRunner, RunnerConfig

DEFAULT_RUNNER = RunnerConfig()  # Named units and values are printed before use.
REFERENCE_FRAME = "torso_lift_link"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="request effort execution after checks and typed confirmation")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X_M", "Y_M", "Z_M"),
                        help=f"grasp-center target or figure-eight anchor before edge clearance, metres in {REFERENCE_FRAME}")
    parser.add_argument("--task", choices=("reach", "figure8"), default="reach")
    parser.add_argument("--duration", type=float, default=DEFAULT_RUNNER.duration_sec, help="maximum run wall time, seconds")
    parser.add_argument("--budget-ms", type=float, help="optional advisory per-step SQP wall budget, milliseconds")
    parser.add_argument("--sea-absolute", action="store_true", help="use the ported SEA absolute feedback estimator")
    parser.add_argument("--output", type=Path, help="fresh run directory under this checkout's _expr_data")
    args = parser.parse_args(argv)
    if args.execute and args.target is None:
        parser.error("--execute requires an explicit --target in the torso-root frame")
    try:
        config = replace(DEFAULT_RUNNER, duration_sec=args.duration)
        task = Figure8() if args.task == "figure8" else Reach()
        if args.target is not None:
            if not np.isfinite(args.target).all():
                raise ValueError("--target must be finite")
            task = replace(task, **{"start_m" if args.task == "figure8" else "target_m": tuple(args.target)})
        params = replace(FIGURE8 if args.task == "figure8" else TIAGO,
                         budget_s=None if args.budget_ms is None else args.budget_ms / 1000.)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    from tiago_mpc.safety import CollisionSafetySettings
    safety = CollisionSafetySettings()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output or ROOT / "_expr_data/ros2-runner" / stamp
    if not output.resolve().is_relative_to((ROOT / "_expr_data").resolve()):
        parser.error("--output must be inside this checkout's _expr_data")
    print(json.dumps(dict(execute=args.execute, reference_frame=REFERENCE_FRAME,
                         ros_distro=os.environ.get("ROS_DISTRO"),
                         end_effector_frame=params.ee_frame,
                         target_semantics="grasp-center target; figure8 uses anchor plus edge_clearance_m on +X",
                         runner=asdict(config), task=asdict(task), params=asdict(params),
                         safety=asdict(safety), output=str(output)), default=str, indent=2))
    if args.execute:
        print("Requires a stationary base/grippers and the comfortable_high_clearance arm pose.")
        print("No automatic reset or path to that pose is executed. Safety checks remain enabled.")
        print("Effort receiver crash/command-timeout behavior must be established separately.")
        if input("Type EXECUTE to enable supervised effort control: ").strip() != "EXECUTE":
            print("Execution cancelled before creating the ROS client.")
            return 2

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        application = RosMpcRunner(config, task, params, safety,
            arm_factory=lambda: TiagoRightArmClient(sea_absolute=args.sea_absolute))
        summary = application.run(output, execute=args.execute)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(summary, default=str, indent=2))
    return 0 if summary["status"] in {"checked", "arrived", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
