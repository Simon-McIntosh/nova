"""Check the booked route's conductor seams on one row without solving it.

The row pipeline hands the profile to the DIII-D current adapter, which appends
the pf_active circuit groups to the grid and wall conductor responses. The
booked route adds a third target — the sampling nodes it clips the boundary on —
and every target must be widened together, or the solve is called with a longer
conductor vector than the sampling rows were built against.

This driver builds one row exactly as the cohort does, applies the adapter, and
reads the widths back, so a seam that does not close is reported before a row is
queued for a card.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    # A cohort module imports its own siblings by bare name, so the directory
    # it lives in has to be on the path before it is executed.
    sibling = str(path.parent)
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


arms = load(
    "extent_arms_smoke", ROOT / "scripts/prototypes/diiid_xpoint_profile_extent/arms.py"
)
branches = load(
    "branches_smoke",
    ROOT / "scripts/prototypes/diiid_forward_convergence/measure_branches.py",
)

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=137)
    parser.add_argument("--arm", type=str, default="edge")
    args = parser.parse_args()
    configure_dtypes()
    arms.install(bench, args.arm)
    prepared = branches._prepare_frame(args.frame, None)
    operator = prepared["profile"].operator
    sample = operator.sample
    receipt = {
        "frame": args.frame,
        "arm": args.arm,
        "route": bench.FINITE_EDGE_MOMENT_ROUTE,
        "conductor_current_rows": int(np.shape(prepared["current"])[0]),
        "grid_source_columns": int(operator.grid.source_target.shape[1]),
        "wall_source_columns": int(operator.wall.source_target.shape[1]),
        "sample_source_columns": (
            None if sample is None else int(sample.source_target.shape[1])
        ),
        "sample_rows": None if sample is None else int(sample.node_number),
        "sample_target_finite": (
            None
            if sample is None
            else bool(np.all(np.isfinite(np.asarray(sample.source_target))))
        ),
    }
    if (
        sample is not None
        and receipt["sample_source_columns"] != receipt["conductor_current_rows"]
    ):
        raise SystemExit(
            "sampling-node conductor columns do not close: " + json.dumps(receipt)
        )
    print("SMOKE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
