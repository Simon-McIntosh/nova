"""Compare constrained Bell support with the inherited cell fractions."""

import argparse
import json
from pathlib import Path

from scripts.prototypes.xpoint_cell_owner.conic_integral import integrate_cell
from scripts.prototypes.xpoint_cell_owner.projection import (
    saddle_constrained_coefficients,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    result = {"cells": source["cells"], "rows": []}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in source["rows"]:
        projected = saddle_constrained_coefficients(
            row["field_coefficients"],
            row["centre"],
            row["pitch"],
            source["saddle"],
            source["boundary_level"],
        )
        integral = integrate_cell(
            row["vertices"],
            row["centre"],
            row["pitch"],
            projected,
            source["boundary_level"],
            source["sigma"],
            source["saddle"],
            source["axis"],
            row["density_coefficients"],
        )
        item = {
            "cell": row["cell"],
            "booked": row["booked"],
            "path_weight": row["path_weight"],
            "projected_fraction": integral.area / row["full_area"],
            "normal_fraction": row["normal_fraction"],
            "ordinary_fraction": row["ordinary_fraction"],
            "current": integral.current,
            "radial_moment": integral.radial_moment,
            "vertical_moment": integral.vertical_moment,
        }
        result["rows"].append(item)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"CELL {json.dumps(item)}", flush=True)
    print(
        f"SUMMARY cells={source['cells']} neighbourhood={len(result['rows'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
