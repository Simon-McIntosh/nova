"""Require the ungated private-cell map to fail the analytic error gate."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    gated_cells = receipt["graph_cells"]
    gated_error = receipt["map_flux_error"]
    ungated_cells = receipt["ungated_graph_cells"]
    ungated_error = receipt["ungated_map_flux_error"]
    print(
        f"gated_cells={gated_cells} gated_error={gated_error:.9f} "
        f"ungated_cells={ungated_cells} ungated_error={ungated_error:.9f}"
    )
    if gated_cells != 179 or gated_error > 0.01:
        raise AssertionError("positive control did not reproduce the passing map")
    if ungated_cells <= gated_cells or ungated_error <= 0.01:
        raise AssertionError("ungated private-cell mutation did not turn the map red")
    print("FAILED test_private_support_gate")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
