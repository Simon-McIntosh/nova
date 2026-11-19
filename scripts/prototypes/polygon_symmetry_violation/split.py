"""Split the ITER bank's exactly-zero dpsi/dZ entries by up-down geometry.

Reads the bank re-certification receipt and the cell geometry it was built
from, and classifies every entry into two groups:

* group A: the node's Z equals its source cell's centre Z, and that cell is
  mirror-symmetric about that Z.  ``dpsi/dZ`` of moments 0 and 1 must vanish
  there by symmetry, so a nonzero value is a violation.
* group B: everything else -- off the plane of a symmetric cell, or on a cell
  that is not symmetric -- where a nonzero value is expected.

The classification is reproducible from the receipt and geometry alone, with no
kernel evaluation.

Run with ``PYTHONPATH=. python scripts/prototypes/polygon_symmetry_violation/split.py``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

RECEIPT = (
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261117T110237355670-iter-bank-recertify/receipts/zero-entries.json"
)
GEOMETRY = "/home/ITER/mcintos/.cache/nova-hessian-diagnosis/h200/geometry.npz"
PLANE = 1e-9


def polygons(geometry):
    """A tuple of closed cell polygons, read from the geometry archive."""
    vertices = np.asarray(geometry["vertices"])
    counts = np.asarray(geometry["vertex_count"])
    return tuple(vertices[index, : int(count)] for index, count in enumerate(counts))


def mirror_gap(cell, centre_z):
    """How far a cell's vertex set departs from its mirror image about Z."""
    cell = np.asarray(cell, dtype=float)
    reflected = cell.copy()
    reflected[:, 1] = 2.0 * centre_z - reflected[:, 1]
    gaps = [np.min(np.hypot(*(cell - point).T)) for point in reflected]
    return float(np.max(gaps))


def symmetric_cells(geometry):
    """One mirror gap per cell, about that cell's own centre Z."""
    centre = np.asarray(geometry["centre"])
    cells = polygons(geometry)
    return np.array(
        [mirror_gap(cell, centre[index, 1]) for index, cell in enumerate(cells)]
    )


def classify(entries, geometry, *, plane=PLANE):
    """Return {A, B} entry groups and the per-cell mirror gaps."""
    symmetry = symmetric_cells(geometry)
    centre = np.asarray(geometry["centre"])
    groups = {"A": [], "B": []}
    for entry in entries:
        cell = entry["cell"]
        dz = float(entry["coordinates"][1] - centre[cell, 1])
        on_plane = abs(dz) < plane
        symmetric = symmetry[cell] < plane
        record = dict(entry, mirror_gap=float(symmetry[cell]), dz=dz)
        groups["A" if on_plane and symmetric else "B"].append(record)
    return groups, symmetry


def describe(group, key):
    """Absolute-value statistics of one value column over a group."""
    values = np.array(
        [abs(entry[key]) if entry[key] is not None else np.nan for entry in group]
    )
    finite = np.isfinite(values)
    if not finite.any():
        return dict(count=len(group), finite=0, median=None, maximum=None, above_1e12=0)
    return dict(
        count=len(group),
        finite=int(finite.sum()),
        median=float(np.median(values[finite])),
        maximum=float(np.max(values[finite])),
        above_1e12=int((values > 1e-12).sum()),
    )


def audit_entries(groups, geometry):
    """Every entry, carrying its group and whether the height stencil fires.

    ``near_corner`` repeats the kernel's own test -- an edge end within 1e-7 of
    the target Z -- so a report can tell the stencil branch from the ordinary
    one without evaluating the kernel.
    """
    cells = polygons(geometry)
    rows = []
    for name, group in groups.items():
        for entry in group:
            ends = np.asarray(cells[entry["cell"]])[:, 1]
            gap = float(np.min(np.abs(ends - entry["coordinates"][1])))
            rows.append(
                dict(
                    group=name,
                    near_corner=bool(gap <= 1e-7),
                    vertex_z_gap=gap,
                    **{key: entry[key] for key in KEYS},
                )
            )
    return rows


KEYS = (
    "node",
    "cell",
    "moment",
    "repaired_value",
    "base_value",
    "dz",
    "mirror_gap",
    "distance_to_cell_boundary",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", default=RECEIPT)
    parser.add_argument("--geometry", default=GEOMETRY)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.receipt) as stream:
        receipt = json.load(stream)

    geometry = {}
    with np.load(args.geometry) as archive:
        for key in archive:
            geometry[key] = np.asarray(archive[key])

    groups, symmetry = classify(receipt["entries"], geometry)
    report = dict(
        entries=len(receipt["entries"]),
        symmetric_cells=int((symmetry < PLANE).sum()),
        groups={
            name: dict(
                count=len(group),
                max_abs_dz=float(
                    max((abs(entry["dz"]) for entry in group), default=0.0)
                ),
                repaired=describe(group, "repaired_value"),
                base=describe(group, "base_value"),
            )
            for name, group in groups.items()
        },
    )
    violations = sorted(
        (entry for entry in groups["A"] if abs(entry["repaired_value"]) > 1e-12),
        key=lambda entry: -abs(entry["repaired_value"]),
    )
    worst = sorted(groups["B"], key=lambda entry: -abs(entry["repaired_value"]))[:20]
    report["violations_A"] = violations
    report["worst_B"] = worst
    report["entries_annotated"] = audit_entries(groups, geometry)

    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("violations_A", "worst_B")
            },
            indent=1,
        )
    )
    print("VIOLATIONS_A")
    for entry in violations:
        print(json.dumps({key: entry[key] for key in KEYS}))
    print("WORST_B")
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=1)


if __name__ == "__main__":
    main()
