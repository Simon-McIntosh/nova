"""Compare a converged state between the two command-buffer arms."""

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True)
    parser.add_argument("--on", required=True)
    parser.add_argument("--width", type=int, default=1)
    arguments = parser.parse_args()
    with np.load(arguments.off) as off, np.load(arguments.on) as on:
        keys = sorted(off.files)
        largest = 0.0
        scale = 0.0
        for key in keys:
            left = np.asarray(off[key], dtype=np.float64)
            right = np.asarray(on[key], dtype=np.float64)
            if arguments.width > 1:
                lane = 0
                right = np.broadcast_to(right, left.shape)
                left, right = left[lane], right[lane]
            difference = np.max(np.abs(left - right))
            largest = max(largest, float(difference))
            scale = max(scale, float(np.max(np.abs(left))))
    relative = largest / scale if scale else 0.0
    print(
        "STATE_COMPARE",
        json.dumps(
            {
                "width": arguments.width,
                "max_abs_difference": largest,
                "max_abs_state": scale,
                "max_relative_difference": relative,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()