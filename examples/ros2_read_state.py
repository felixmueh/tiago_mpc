#!/usr/bin/env python3
"""Read-only right-arm observation; no commands or controller service calls."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiago_mpc.ros2 import TiagoRightArmClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=5.)
    parser.add_argument('--max-age', type=float, default=.1)
    parser.add_argument('--sea-absolute', action='store_true')
    args = parser.parse_args()
    if args.samples < 1 or args.timeout <= 0 or args.max_age <= 0:
        parser.error('samples, timeout and max-age must be positive')
    with TiagoRightArmClient(sea_absolute=args.sea_absolute) as arm:
        after_seq = None
        for _ in range(args.samples):
            state = arm.read_state(timeout_sec=args.timeout, after_seq=after_seq,
                                   max_age_sec=args.max_age)
            print(f'seq={state.seq} stamp={state.stamp_sec:.9f} age={state.age_sec:.6f}s '
                  f'q={state.q.tolist()} qd={state.qd.tolist()}')
            after_seq = state.seq


if __name__ == '__main__':
    main()
