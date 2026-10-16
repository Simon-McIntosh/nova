"""Check measured carrier receipts against explicit numerical limits."""

import argparse
import json
import math
from pathlib import Path


MAP_LIMITS = {282: 0.0010866, 550: 0.0005, 2098: 0.0001}
SOLVE_LIMIT = 1e-12
SCAN_CHANGE_LIMIT = 1e-6


def require_map(receipts):
    errors = []
    for receipt in receipts:
        cells = receipt["cells"]
        if cells not in MAP_LIMITS:
            raise ValueError(f"no diverted map limit for {cells} cells")
        error = receipt["centre_map_flux_error"]
        if not math.isfinite(error) or error > MAP_LIMITS[cells]:
            raise ValueError(
                f"{cells}-cell map error {error} exceeds {MAP_LIMITS[cells]}"
            )
        errors.append((cells, error))
    if len(errors) > 1:
        ordered = sorted(errors)
        if any(right[1] >= left[1] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("map error does not fall with resolution")


def require_scan(receipt):
    rows = receipt["scan"][1:]
    if len(rows) < 2:
        raise ValueError("continuity scan has fewer than two samples")
    for left, right in zip(rows, rows[1:]):
        if right["step"] - left["step"] > 1e-7:
            raise ValueError("continuity scan is too sparse")
        if left["support_ids"] != right["support_ids"]:
            raise ValueError("support jumps in continuity scan")
        if left["boundary_class"] != right["boundary_class"]:
            raise ValueError("boundary class jumps in continuity scan")
        for key in ("residual", "membership_71", "cell_current_residual"):
            change = abs(right[key] - left[key])
            if not math.isfinite(change) or change > SCAN_CHANGE_LIMIT:
                raise ValueError(f"{key} jumps by {change}")
        for key in ("wall_candidate", "saddle_candidate"):
            change = abs(right[key] - left[key])
            if not math.isfinite(change) or change > 1e-9:
                raise ValueError(f"{key} jumps by {change} Wb")


def require_solve(receipt):
    history = receipt["history"]
    if not history:
        raise ValueError("cold solve has no corrections")
    residual = history[-1]["residual"]
    if (
        not receipt.get("converged")
        or not math.isfinite(residual)
        or residual > SOLVE_LIMIT
    ):
        raise ValueError(f"cold solve residual {residual} exceeds {SOLVE_LIMIT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("map", "scan", "solve"))
    parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    receipts = [json.loads(path.read_text()) for path in args.receipts]
    if args.kind == "map":
        require_map(receipts)
    elif args.kind == "scan":
        if len(receipts) != 1:
            parser.error("scan needs one receipt")
        require_scan(receipts[0])
    else:
        for receipt in receipts:
            require_solve(receipt)
    print(f"QUALIFIED {args.kind} {len(receipts)}")


if __name__ == "__main__":
    main()
