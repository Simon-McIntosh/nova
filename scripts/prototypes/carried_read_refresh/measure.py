"""Select fragment refresh while reusing the bank slice profiling instrument."""

import argparse
from functools import partial
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reuse", choices=("off", "on"), required=True)
    args, remaining = parser.parse_known_args()
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    from nova.equilibrium import topology
    from scripts.prototypes.bank_h200_slice_profile import measure

    topology.refresh_at_stationary_points = partial(
        topology.refresh_at_stationary_points, reuse=args.reuse == "on"
    )
    sys.argv = [sys.argv[0], *remaining]
    measure.main()
    output = Path(remaining[remaining.index("--output") + 1])
    receipt = json.loads((output / "receipt.json").read_text())
    receipt["reuse"] = args.reuse
    # Counts exclude debug paths and names, so identical programs cannot pass
    # merely because their archives occupy different directories.
    for name, program in receipt["programs"].items():
        path = output / name / f"solve-{receipt['width']}.hlo"
        if path.is_file():
            count = len(re.findall(r"^\s*%[^\n]* = ", path.read_text(), re.M))
            assert count > 0
            program["optimized_operations"] = count
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
