"""Measure connected Bell area on either side of a fitted saddle contact."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.xpoint_cell_owner.conic_integral import integrate_cell


def contact_level(row, boundary_level, sigma):
    """Return the boundary level that passes through the fitted stationary point."""
    field = np.asarray(row["field_coefficients"], dtype=float).copy()
    field[0] -= boundary_level
    field *= sigma
    hessian = np.array(((2 * field[3], field[4]), (field[4], 2 * field[5])))
    stationary = -np.linalg.solve(hessian, field[1:3])
    critical = field[0] + field[1:3] @ stationary
    critical += 0.5 * stationary @ hessian @ stationary
    return boundary_level + critical / sigma, float(critical)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", type=int, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    row = next(item for item in source["rows"] if item["cell"] == args.cell)
    contact, critical = contact_level(row, source["boundary_level"], source["sigma"])
    operands = (
        row["vertices"],
        row["centre"],
        row["pitch"],
        row["field_coefficients"],
    )
    tail = (
        source["sigma"],
        source["saddle"],
        source["axis"],
        row["density_coefficients"],
    )
    result = {
        "cells": source["cells"],
        "cell": args.cell,
        "signed_critical_offset": critical,
        "contact_boundary_level": contact,
        "rows": [],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for signed_offset in (1e-9, 1e-10, 1e-11, -1e-11, -1e-10, -1e-9):
        boundary = contact - signed_offset / source["sigma"]
        integral = integrate_cell(*operands, boundary, *tail, axis_component="bell")
        item = {
            "signed_critical_offset": signed_offset,
            "area_fraction": integral.area / row["full_area"],
            "current": integral.current,
        }
        result["rows"].append(item)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"CONTACT {json.dumps(item)}", flush=True)
    positive = result["rows"][2]["area_fraction"]
    negative = result["rows"][3]["area_fraction"]
    result["near_contact_jump"] = positive - negative
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SUMMARY near_contact_jump={positive - negative:.12g}", flush=True)


if __name__ == "__main__":
    main()
