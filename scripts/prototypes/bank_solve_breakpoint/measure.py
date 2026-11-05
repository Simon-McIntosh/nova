"""Reuse the bank timing and trajectory instruments for a paired solve receipt."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

from scripts.prototypes.bank_step_booking_batch import measure as booking
from scripts.prototypes.bank_solve_linesearch import measure as counted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    with patch.object(
        sys,
        "argv",
        [
            "measure",
            "--run",
            str(args.run),
            "--arm",
            args.arm,
            "--revision",
            args.revision,
        ],
    ):
        booking.main()
    with patch.object(
        sys,
        "argv",
        [
            "measure",
            "--output",
            str(args.run / f"{args.arm}-solve.json"),
            "--revision",
            args.revision,
        ],
    ):
        counted.main()


if __name__ == "__main__":
    main()
