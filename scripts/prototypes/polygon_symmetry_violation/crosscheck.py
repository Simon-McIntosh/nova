"""Cross-check the packed rows against the un-packed greens reduction.

``polygon_analytic_greens`` builds psi and the field from the polygon's own edge
sum with paired arithmetic; the packed rows are the same integral through the
moment reduction the bank books.  Row 0 of the packed reduction is the uniform
source, which is the polygon itself, so the two routes are comparable row for
row, and the numpy route then supplies a psi difference whose arithmetic is not
the packed one.

For every planned point this reports

* the gap between packed psi and numpy psi, and between the booked Z gradient
  ``-2 pi r B_R`` and the numpy field, for the entry's own cell;
* per entry, the psi central difference and the booked gradient in both routes,
  so a disagreement can be attributed to a route rather than to psi's value.

Reads the row archive a completed ``compare.py`` arm leaves in its scratch
directory.  Run with ``/home/ITER/mcintos/Code/nova/.venv/bin/python
scripts/prototypes/polygon_symmetry_violation/crosscheck.py --split <json>
--geometry <npz> --rows <npz> --output <json>``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from nova.biot.polygonanalytic import polygon_analytic_greens
from scripts.prototypes.polygon_symmetry_violation.compare import plan


def central(ladder):
    """Pair a ladder's values into one central difference per step magnitude."""
    forward = {step: value for step, value in ladder if step > 0.0}
    backward = {step: value for step, value in ladder if step < 0.0}
    return [
        dict(step=step, derivative=(forward[step] - backward[-step]) / (2.0 * step))
        for step in sorted(forward, reverse=True)
        if -step in backward
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.split) as stream:
        split = json.load(stream)
    violations, worst = split["violations_A"], split["worst_B"]
    entries = violations + worst

    geometry = {}
    with np.load(args.geometry) as archive:
        for key in archive:
            geometry[key] = np.asarray(archive[key])
    vertices = np.asarray(geometry["vertices"])
    counts = np.asarray(geometry["vertex_count"])
    centre = np.asarray(geometry["centre"])
    polygons = tuple(vertices[i, : int(n)] for i, n in enumerate(counts))
    cells = sorted({int(entry["cell"]) for entry in entries})
    local_of = {cell: index for index, cell in enumerate(cells)}

    records = plan(violations, worst, centre)
    points = [record["point"] for record in records]
    with np.load(args.rows) as archive:
        packed = np.asarray(archive["rows"])

    numpy_psi = np.zeros(len(points))
    numpy_gradient = np.zeros(len(points))
    for index, (record, point) in enumerate(zip(records, points, strict=True)):
        cell = int(entries[record["entry"]]["cell"])
        one, radial, _ = polygon_analytic_greens(
            np.asarray(point[0]),
            np.asarray(point[1]),
            polygons[cell],
        )
        numpy_psi[index] = one
        numpy_gradient[index] = -2.0 * np.pi * float(point[0]) * radial

    packed_psi = np.zeros(len(points))
    packed_gradient = np.zeros(len(points))
    for index, record in enumerate(records):
        local = local_of[int(entries[record["entry"]]["cell"])]
        packed_psi[index] = packed[record["index"], local, 0, 0]
        packed_gradient[index] = packed[record["index"], local, 0, 2]

    magnitude = np.maximum(np.abs(numpy_psi), 1e-30)
    rows = []
    for entry_index, entry in enumerate(entries):
        local = local_of[int(entry["cell"])]
        direct = next(
            record
            for record in records
            if record["arm"] == "direct" and record["entry"] == entry_index
        )
        packed_ladder = [
            (record["step"], float(packed[record["index"], local, 0, 0]))
            for record in records
            if record["arm"] == "psi" and record["entry"] == entry_index
        ]
        numpy_ladder = [
            (record["step"], float(numpy_psi[record["index"]]))
            for record in records
            if record["arm"] == "psi" and record["entry"] == entry_index
        ]
        packed_cd, numpy_cd = central(packed_ladder), central(numpy_ladder)
        rows.append(
            dict(
                node=int(entry["node"]),
                cell=int(entry["cell"]),
                group="A" if entry_index < len(violations) else "B",
                numpy_direct=float(numpy_gradient[direct["index"]]),
                packed_direct=float(packed_gradient[direct["index"]]),
                numpy_limit=numpy_cd[-1]["derivative"] if numpy_cd else None,
                packed_limit=packed_cd[-1]["derivative"] if packed_cd else None,
            )
        )

    payload = dict(
        points=len(points),
        psi_max_abs_gap=float(np.max(np.abs(packed_psi - numpy_psi))),
        psi_max_relative_gap=float(np.max(np.abs(packed_psi - numpy_psi) / magnitude)),
        psi_scale=float(np.max(np.abs(numpy_psi))),
        gradient_max_abs_gap=float(np.max(np.abs(packed_gradient - numpy_gradient))),
        gradient_scale=float(np.max(np.abs(numpy_gradient))),
        numpy_gradient_max=float(np.max(np.abs(numpy_gradient))),
        rows=rows,
    )
    with open(args.output, "w") as stream:
        json.dump(payload, stream, indent=1)
    print(
        "PSI_PACKED_VS_NUMPY max_abs %.3e max_rel %.3e scale %.3e"
        % (
            payload["psi_max_abs_gap"],
            payload["psi_max_relative_gap"],
            payload["psi_scale"],
        )
    )
    print(
        "GRADIENT_PACKED_VS_NUMPY max_abs %.3e numpy_scale %.3e"
        % (payload["gradient_max_abs_gap"], payload["numpy_gradient_max"])
    )
    print("node group numpy_direct packed_direct numpy_limit packed_limit")
    for row in rows:
        print(
            "%d %s %.6e %.6e %.6e %.6e"
            % (
                row["node"],
                row["group"],
                row["numpy_direct"],
                row["packed_direct"],
                row["numpy_limit"] if row["numpy_limit"] is not None else float("nan"),
                row["packed_limit"]
                if row["packed_limit"] is not None
                else float("nan"),
            )
        )


if __name__ == "__main__":
    main()
