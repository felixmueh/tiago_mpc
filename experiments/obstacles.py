"""Reaches past the robot's own body and past obstacles (spheres, capsules, boxes and
planes), in simulation, with GATO's
collision rows enforced ("on", ADMM) and only reported ("off", telemetry).

Both modes compile the same module and upload the same environment (the body's stored
cover and the scene's obstacles, tiago_mpc/collision.py); only the rows' mechanism differs.
evaluate measures the executed motion against the collision meshes (experiments/clearance.py).
The runs take the real MPC's timing (VirtualTimeSettings.like(RealTimeSettings()): 50 Hz
updates, each solve one control period) with the default latency compensation and
PlanFeedback, as the reach.

Run from the checkout's root: python -m experiments.obstacles [--help]"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import get_args

import numpy as np

from experiments.common import handle_stop_signals, run_experiment
from experiments.reach import Reach
import tiago_mpc as tm
from experiments.clearance import Clearance
from tiago_mpc.collision import BodyCover, Box, Capsule, Collision, Plane, Sphere


@dataclass(frozen=True)
class Scene:
    """A reach to target_m (grasp frame, ROOT_FRAME) among obstacles (Collision's layouts)."""
    target_m: tuple[float, float, float]
    obstacles: tuple[Sphere, ...] = ()
    capsules: tuple[Capsule, ...] = ()
    boxes: tuple[Box, ...] = ()
    planes: tuple[Plane, ...] = ()
    timeout_s: float = 6.


SCENES = {
    # Low in front of the torso: without the rows the wrist hits the mobile base
    # (_expr_data/obstacles-20261029/screen_front.sh).
    "body": Scene((.25, -.10, -.40)),
    # Beside and behind the torso's front: without the rows the upper arm ends inside the
    # torso; with them the arm stops 12 cm short, joint 2 at its planning limit and the upper
    # arm at the margin (_expr_data/obstacles-20261029/find_targets.py, candidate 1).
    "behind": Scene((-.12, -.492, -.278)),
    # Sideways to the right. Two 5 cm spheres that the arm cover of the unconstrained reach
    # (rows off, no obstacles) penetrates by 12 and 9 cm, while the covers of the start
    # and goal postures clear them by at least 2 cm and the body's by 3 cm
    # (_expr_data/obstacles-20261029/grid_obstacles.py on free-side-off).
    "obstacles": Scene((.40, -.65, -.20), ((.342, -.42, -.484, .05), (.362, -.36, -.324, .05))),
    # The same reach past a vertical pole (3 cm radius, 30 cm long), and past a thin wall
    # (20 x 4 x 20 cm, turned 20 degrees about z). The exact meshes of the unconstrained reach
    # (rows off, obstacles-latency-20261101/obstacles-off-sqp1) penetrate them by 5.4 and
    # 6.0 cm, while the start and goal postures clear them by at least 5 cm
    # (_expr_data/obstacles-primitives-20261103/check_candidates.py).
    "capsule": Scene((.40, -.65, -.20), capsules=((.342, -.403, -.628, .342, -.403, -.328, .03),)),
    "box": Scene((.40, -.65, -.20), boxes=((.372, -.403, -.478, 0., 0., .173648, .984808, .10, .02, .10),)),
    # The body reach above a table, the half-space z >= -0.60 m: the start posture clears it
    # by 6.2 cm, the unconstrained reach's goal posture (body-off-sqp1) is 4.9 cm inside it.
    # The plane is infinite, so it also cuts through the mobile base; the rows see only the arm.
    "table": Scene((.25, -.10, -.40), planes=((0., 0., 1., -.60),)),
}


def evaluate(record: tm.RunRecord, stride: int = 5) -> None:
    """The exact clearances of every stride-th tick (and the last), arrival as Reach's, and
    the collision rows' largest reported violation over the plans. Saves the record."""
    params, task, ticks = record.config["params"], record.config["task"], record.ticks
    results = record.metadata["results"]
    if not len(ticks["t"]):
        record.save()
        return
    check = Clearance(params.collision)
    rows = np.r_[np.arange(0, len(ticks["t"]), stride), len(ticks["t"]) - 1]
    measured = [check.distances(ticks["q"][i]) for i in rows]
    body = np.array([m[0] for m in measured])
    per_obstacle = np.array([m[1] for m in measured])  # (samples, obstacles)
    obstacle = per_obstacle.min(1, initial=np.inf)
    worst = int(np.argmin(body))
    distance = np.array([task.distance(q) for q in ticks["q"]])
    arrived = np.flatnonzero(distance <= task.position_tol_m)
    violation = record.plans.get("row_viol")
    results.update(
        min_body_clearance_m=float(body.min()), min_body_clearance_t=float(ticks["t"][rows[worst]]),
        min_body_pair=list(measured[worst][2:]), body_contact_samples=int(np.sum(body <= 0)),
        min_obstacle_clearance_m=float(obstacle.min()), obstacle_contact_samples=int(np.sum(obstacle <= 0)),
        obstacle_clearances_m={name: float(d) for name, d in zip(check.obstacle_names, per_obstacle.min(0))},
        checked_samples=len(rows), final_distance_m=float(distance[-1]),
        first_within_tolerance_s=float(ticks["t"][arrived[0]]) if len(arrived) else None,
        max_collision_row_violation_m=None if violation is None else float(np.max(violation[:, -1])))
    record.ticks["exact_body_clearance_m"] = np.interp(ticks["t"], ticks["t"][rows], body)
    record.save()


def main(argv=None):
    handle_stop_signals()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, default="body")
    parser.add_argument("--mode", choices=("on", "off"), default="on",
                        help="on: ADMM collision rows; off: the same rows as telemetry only")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X_M", "Y_M", "Z_M"),
                        help="replace the scene's target, metres in torso_lift_link")
    parser.add_argument("--no-body", action="store_true", help="rows against the obstacles only, not the body")
    parser.add_argument("--body", choices=get_args(BodyCover), default=Collision.body, help="the stored body cover (tiago_mpc/data/collision/body/)")
    parser.add_argument("--solve-latency", default=None,
                        help="seconds from a request to its plan, or 'measured' (default: one control period)")
    parser.add_argument("--sqp-iters", type=int, help="SQP iterations per solve (default: the solver's)")
    parser.add_argument("--output", type=Path, help="fresh directory (default: under _expr_data/Reach/)")
    args = parser.parse_args(argv)
    scene = SCENES[args.scene]
    if args.target is not None:
        scene = replace(scene, target_m=tuple(args.target))
    collision = Collision(obstacles=scene.obstacles, capsules=scene.capsules, boxes=scene.boxes, planes=scene.planes,
                          body=args.body, mechanism="admm" if args.mode == "on" else "telemetry")
    if args.no_body:
        collision = replace(collision, body=None)
    params = replace(tm.TIAGO, collision=collision)
    if args.sqp_iters is not None:
        params = replace(params, solver=replace(params.solver, max_sqp_iters=args.sqp_iters))
    settings = tm.VirtualTimeSettings.like(tm.RealTimeSettings())
    if args.solve_latency is not None:
        latency = args.solve_latency
        settings = replace(settings, solve_latency_s=latency if latency == "measured" else float(latency))
    task = Reach(scene.target_m, timeout_s=scene.timeout_s)
    record = run_experiment(tm.SimConfig(substep=.003), params, task, tm.PlanFeedback(), settings,
                            compensation=tm.default_compensation(settings), out=args.output, scene=args.scene,
                            mode=args.mode)
    evaluate(record)
    print(json.dumps(record.metadata["results"], indent=2, default=str))
    print(f"Saved {record.out}")
    return 0 if record.metadata["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
