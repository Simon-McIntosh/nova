"""Check whether an ablated carrier recovers the split axis read."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np


def check(reference_path, candidate_path):
    reference, candidate = np.load(reference_path), np.load(candidate_path)
    expected = reference["common.read.axis"]
    observed = candidate["common.read.axis"]
    absolute = float(np.max(np.abs(expected - observed)))
    relative = absolute / float(np.max(np.abs(expected)))
    passed = bool(np.isfinite(relative) and relative < 1e-12)
    print(
        "FIRST_STAGE_CONTROL",
        "axis",
        "expected",
        expected.tolist(),
        "observed",
        observed.tolist(),
        "absolute",
        absolute,
        "relative",
        relative,
        "threshold",
        1e-12,
        "passed",
        passed,
        flush=True,
    )
    if not passed:
        print("FAILED single_module_restoration_matches_axis", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd()} "
        f"command={shlex.join([sys.executable, *sys.argv])}",
        flush=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    result = check(args.reference, args.candidate)
    print(f"EXIT={result}", flush=True)
    sys.exit(result)
