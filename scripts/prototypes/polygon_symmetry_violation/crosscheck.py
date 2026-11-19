"""Cross-check the packed rows against the un-packed greens reduction.

``polygon_analytic_greens`` builds psi and the field from the polygon's own edge
sum with paired arithmetic; the packed rows are the same integral through the
moment reduction the bank books.  Row 0 of the packed reduction is the uniform
source, so the two must agree row for row and the numpy route then supplies a
psi difference whose arithmetic is not the packed one.

For every planned point this reports

* the largest absolute and relative gap between packed psi and numpy psi;
* the same for the booked Z gradient ``-2 pi r B_R``;
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


def greens(points, polygons, cells):
    """Psi and both fields of each selected cell at every point, by the numpy route."""
    psi = np.zeros((len(points), len(cells)))
    radial = np.zeros((len(points), len(cells)))
    vertical = np.zeros((len(points), len(cells)))
    for row, point in enumerate(points):
        for column, cell in enumerate(cells):
            one, two, three = polygon_analytic_greens(
                np.asarray(point[0]),
                np.asarray(point[1]),
                polygons[cell],
            )
            psi[row, column] = one
            radial[row, column] = two
            vertical[row, column] = three
    return psi, radial, vertical


def central(ladder):
    """Pair the psi values into a central difference per step magnitude."""
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

    psi, radial, _ = greens(points, polygons, cells)

    # Row 0 of the moment axis is the uniform source, which is the polygon itself.
    packed_psi = packed[:, :, 0, 0]
    packed_radial = packed[:, :, 0, 2]
    numpy_radial = np.zeros_like(packed_radial)
    for row, point in enumerate(points):
        numpy_radial[row] = 2.0 * np.pi * float(point[0]) * radial[row] * -1.0

    magnitude = np.maximum(np.abs(psi), 1e-30)
    rows = []
    for index, entry in enumerate(entries):
        local = local_of[int(entry["cell"])]
        direct = next(
            record
            for record in records
            if record["arm"] == "direct" and record["entry"] == index
        )
        ladder = [
            (record["step"], float(psi[record["index"], local]))
            for record in records
            if record["arm"] == "psi" and record["entry"] == index
        ]
        numpy_cd = central(ladder)
        packed_cd = central(
            [
                (record["step"], float(packed_psi[record["index"], local]))
                for record in records
                if record["arm"] == "psi" and record["entry"] == index
            ]
        )
        rows.append(
            dict(
                node=int(entry["node"]),
                cell=int(entry["cell"]),
                group="A" if index < len(violations) else "B",
                numpy_direct=float(numpy_radial[direct["index"], local]),
                packed_direct=float(packed_radial[direct["index"], local]),
                numpy_limit=numpy_cd[-1]["derivative"] if numpy_cd else None,
                packed_limit=packed_cd[-1]["derivative"] if packed_cd else None,
                packed_cd=[row["derivative"] for row in packed_cd],
                numpy_cd=[row["derivative"] for row in numpy_cd],
            )
        )

    payload = dict(
        points=len(points),
        cells=len(cells),
        psi_max_abs_gap=float(np.max(np.abs(packed_psi - psi))),
        psi_max_relative_gap=float(np.max(np.abs(packed_psi - psi) / magnitude)),
        radial_max_abs_gap=float(np.max(np.abs(packed_radial - numpy_radial))),
        radial_numpy_max=float(np.max(np.abs(numpy_radial))),
        rows=rows,
    )
    with open(args.output, "w") as stream:
        json.dump(payload, stream, indent=1)
    print(
        "PSI_PACKED_VS_NUMPY max_abs %.3e max_rel %.3e"
        % (payload["psi_max_abs_gap"], payload["psi_max_relative_gap"])
    )
    print(
        "RADIAL_PACKED_VS_NUMPY max_abs %.3e scale %.3e"
        % (payload["radial_max_abs_gap"], payload["radial_numpy_max"])
    )
    print("node group numpy_direct packed_direct numpy_limit packed_limit")
    for row in rows:
        print(
            "%d %s %.3e %.3e %.3e %.3e"
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
