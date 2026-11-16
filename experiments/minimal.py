"""Compose one simulated reach from the plant, MPC parameters, task and timing.

Run with: uv run python -m experiments.minimal
Edit these pieces to change the experiment. run_experiment compiles the MPC,
runs the loop, and saves the configuration, motion and replay under _expr_data/.
"""
import tiago_mpc as tm

from experiments.common import handle_stop_signals, run_experiment
from experiments.reach import Reach, evaluate


def main() -> int:
    handle_stop_signals()
    plant = tm.SimConfig(substep=.003)
    params = tm.TIAGO
    task = Reach()  # Or Reach(target_m=(.4, -.3, -.1)), in torso_lift_link metres.
    settings = tm.VirtualTimeSettings.constant_rate()
    law = tm.FrictionFeedforward()

    record = run_experiment(plant, params, task, law, settings)
    evaluate(record)
    print(f"{record.metadata['status']}: {record.out}")
    print(f"Replay: {record.out / 'replay.npz'}")
    return 0 if record.metadata["status"] == "arrived" and record.metadata["results"]["limits_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
