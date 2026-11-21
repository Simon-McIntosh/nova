"""Probe the booked Z gradient through the cell's mirror plane.

A mirrored cell has an exact up-down parity about its centre height, so the
booked Z gradient of the symmetric basis is odd in the height offset and passes
through zero on the plane.  This evaluates that gradient at a ladder of offsets
that straddles the plane, including offsets far below the kernel's smallest
height stencil, and reports the jump the ladder finds at zero.

A gradient that is continuous through the plane approaches zero on both sides.
A branch or limit failure at ``level == 0`` shows as a ladder whose odd part is
smooth while its value at zero is displaced by an even, offset-dependent
amount.  The flux row is evaluated on the same ladder so its parity is reported
beside it.

``--split`` and ``--geometry`` are read as in ``compare.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.polygon_symmetry_violation.compare import evaluate

OFFSETS = (-1e-3, -1e-6, -1e-9, 0.0, 1e-9, 1e-6, 1e-3)


def pair(ladder, magnitude):
    """The booked and flux values one offset apart, above and below the plane."""
    above = next(row for row in ladder if row["offset"] == magnitude)
    below = next(row for row in ladder if row["offset"] == -magnitude)
    return above, below


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--head-tree", default=".")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=4)
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
    polygons = tuple(vertices[i, : int(n)] for i, n in enumerate(counts))
    cells = sorted({int(entry["cell"]) for entry in entries})
    selected = tuple(polygons[cell] for cell in cells)
    local_of = {cell: index for index, cell in enumerate(cells)}

    points = []
    index_of = {}
    for entry_index, entry in enumerate(entries):
        radius = float(entry["coordinates"][0])
        height = float(entry["coordinates"][1])
        for offset in OFFSETS:
            index_of[(entry_index, offset)] = len(points)
            points.append((radius, height + offset))

    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = evaluate(
        points, selected, Path(args.head_tree).resolve(), directory, args.batch
    )

    report = []
    for entry_index, entry in enumerate(entries):
        local = local_of[int(entry["cell"])]
        moment = int(entry["moment"])
        ladder = [
            dict(
                offset=offset,
                booked=float(rows[index_of[(entry_index, offset)], local, moment, 2]),
                flux=float(rows[index_of[(entry_index, offset)], local, moment, 0]),
            )
            for offset in OFFSETS
        ]
        zero = next(row for row in ladder if row["offset"] == 0.0)["booked"]
        jump = {}
        for magnitude in (1e-9, 1e-6, 1e-3):
            above, below = pair(ladder, magnitude)
            mean = 0.5 * (above["booked"] + below["booked"])
            jump[str(magnitude)] = dict(
                mean=mean,
                difference=above["booked"] - below["booked"],
                jump=zero - mean,
            )
        forward = {row["offset"]: row["flux"] for row in ladder if row["offset"] > 0.0}
        backward = {row["offset"]: row["flux"] for row in ladder if row["offset"] < 0.0}
        report.append(
            dict(
                node=int(entry["node"]),
                cell=int(entry["cell"]),
                moment=moment,
                group="A" if entry_index < len(violations) else "B",
                ladder=ladder,
                zero=zero,
                jump=jump,
                flux_central_diff=[
                    dict(
                        step=step,
                        derivative=(forward[step] - backward[-step]) / (2.0 * step),
                    )
                    for step in sorted(forward, reverse=True)
                ],
            )
        )

    def summary(group):
        rows = [row for row in report if row["group"] == group]
        return dict(
            count=len(rows),
            booked_zero_max=float(max(abs(row["zero"]) for row in rows)),
            jump_1e_9_max=float(max(abs(row["jump"]["1e-09"]["jump"]) for row in rows)),
            jump_1e_6_max=float(max(abs(row["jump"]["1e-06"]["jump"]) for row in rows)),
            mean_1e_9_max=float(max(abs(row["jump"]["1e-09"]["mean"]) for row in rows)),
        )

    payload = dict(
        offsets=list(OFFSETS),
        headline=dict(group_a=summary("A"), group_b=summary("B")),
        rows=report,
    )
    with open(args.output, "w") as stream:
        json.dump(payload, stream, indent=1)
    print("HEADLINE " + json.dumps(payload["headline"]))
    print("node group zero mean(+/-1e-9) jump(+/-1e-9)")
    for row in report:
        band = row["jump"]["1e-09"]
        print(
            "%d %s %.6e %.6e %.6e"
            % (row["node"], row["group"], row["zero"], band["mean"], band["jump"])
        )


if __name__ == "__main__":
    main()
