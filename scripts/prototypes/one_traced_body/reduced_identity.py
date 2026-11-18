"""Exercise the unchanged identity assertions on a reduced CPU population.

Only the point supplier is replaced: each check retains its own comparison,
arithmetic and tolerance. Direct invocation exercises the numerical assertions
on CPU without changing the tests' GPU-only collection policy.
"""

import argparse
import json
from pathlib import Path

from tests.biot import test_kernel_compile_cost as jets
from tests.biot import test_moment_program_identity as moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check", choices=("packed", "production", "polygon"), required=True
    )
    args = parser.parse_args()
    points = jets._points()[:100]
    jets._points = moments._points = lambda: points
    checks = {
        "packed": moments.test_packed_moment_primal_and_tangent_identity,
        "production": moments.test_production_source_rows_preserve_jet_identity,
        "polygon": jets.test_point_jet_identity,
    }
    check = checks[args.check]
    receipt = {"check": args.check, "points": len(points), "completed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        check()
    except AssertionError:
        receipt.update(completed=True, passed=False)
        raise
    else:
        receipt.update(completed=True, passed=True)
    finally:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("REDUCED_IDENTITY", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
