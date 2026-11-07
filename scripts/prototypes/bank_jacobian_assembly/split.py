"""Split one batched step_jacobian's stages at the requested lane widths."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import sys

from scripts.prototypes.bank_step_jacobian_split import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lanes", type=int, nargs="+", default=[16, 128])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    from scripts.prototypes.bank_solve_cycle import measure as cycle

    original = cycle.main

    def with_lanes():
        sys.argv = [*sys.argv, "--lanes", *(str(v) for v in args.lanes)]
        return original()

    cycle.main = with_lanes
    sys.argv = ["split", "--output", args.output, "--repeats", str(args.repeats)]
    measure.main()


if __name__ == "__main__":
    main()
