"""Run a forward regression arm inside the reserved-GPU launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from measure import extract
from regression import gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    # The launcher has already selected the backend and enabled its cache guard.
    with tempfile.TemporaryDirectory(
        prefix="warmup-forward-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary) / "tree"
        extract(tree, args.revision, repository)
        result = gate(
            tree,
            args.revision,
            "tests/test_equilibrium_forward_solve.py",
            args.output,
            platforms=os.environ["JAX_PLATFORMS"],
            extra_pythonpath=str(repository / "scripts/h200_test_lane"),
        )
        raise SystemExit(result["exit_status"])


if __name__ == "__main__":
    main()
