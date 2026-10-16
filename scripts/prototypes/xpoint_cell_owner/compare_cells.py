"""Compare two quadrature rules over every exported saddle neighbourhood."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.xpoint_cell_owner.conic_integral import integrate_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    rows = []
    result = {"cells": source["cells"], "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for cell in source["rows"]:
        operands = (
            cell["vertices"],
            cell["centre"],
            cell["pitch"],
            cell["field_coefficients"],
            source["boundary_level"],
            source["sigma"],
            source["saddle"],
            source["axis"],
            cell["density_coefficients"],
        )
        adaptive = integrate_cell(*operands)
        dense = integrate_cell(*operands, outer_rule="dense")
        observed = np.array(
            (
                adaptive.area,
                adaptive.current,
                adaptive.radial_moment,
                adaptive.vertical_moment,
            )
        )
        reference = np.array(
            (dense.area, dense.current, dense.radial_moment, dense.vertical_moment)
        )
        relative = np.abs(observed - reference) / np.maximum(np.abs(reference), 1e-14)
        row = {
            "cell": cell["cell"],
            "area_fraction": adaptive.area / cell["full_area"],
            "normal_fraction": cell["normal_fraction"],
            "ordinary_fraction": cell["ordinary_fraction"],
            "booked": cell["booked"],
            "path_weight": cell["path_weight"],
            "adaptive": observed.tolist(),
            "dense": reference.tolist(),
            "relative_error": relative.tolist(),
            "error_estimate": adaptive.error_estimate,
        }
        rows.append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"CELL {json.dumps(row)}", flush=True)
    result["max_relative_error"] = max(max(row["relative_error"]) for row in rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"SUMMARY cells={source['cells']} neighbourhood={len(rows)} "
        f"max_relative={result['max_relative_error']:.12g}",
        flush=True,
    )
    if result["max_relative_error"] > 1e-12:
        raise ValueError("conic integral differs from dense quadrature")


if __name__ == "__main__":
    main()
