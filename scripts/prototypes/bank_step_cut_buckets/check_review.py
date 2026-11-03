"""Check that the overflow test remains tied to live numerical work."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(path: Path, check: str) -> None:
    source = path.read_text()
    checks = {
        "capacity": (
            "capacity = current._NORMAL_CUT_BATCH_WIDTH" in source
            and "chosen = cut[: capacity + 1]" in source
        ),
        "finite": (
            "np.isfinite(expected_values[overflow_cell]).all()" in source
            and "np.isfinite(actual_values[overflow_cell]).all()" in source
            and "assert booked_current != 0" in source
            and "equal_nan=True" not in source
        ),
    }
    passed = checks[check]
    print(f"{check}: {'PASS' if passed else 'FAIL'} in {path}", flush=True)
    print(f"EXIT={0 if passed else 1}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", required=True, choices=("capacity", "finite"))
    main(**vars(parser.parse_args()))
