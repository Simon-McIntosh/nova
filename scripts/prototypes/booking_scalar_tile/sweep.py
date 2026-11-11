"""Sweep scalar booking tiles through the common runtime-operand timing driver."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tiles", type=int, nargs="+", choices=[32, 64, 128], default=[32, 64, 128]
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    from scripts.prototypes.booking_scalar_tile.measure import main as measure

    measure(
        [
            str(args.root),
            "--output",
            str(args.output),
            "--tiles",
            *map(str, args.tiles),
            "--widths",
            *map(str, args.widths),
            "--repeats",
            str(args.repeats),
            "--kinds",
            "booking",
        ]
    )


if __name__ == "__main__":
    main()
