"""Merge per-slice measurement rows and tabulate nova's read against LIUQE.

LIUQE is the reference. Each gap is also expressed as a fraction of the
axis-to-boundary span on that slice, so slices at different plasma sizes and
flux levels are comparable: a position gap is divided by the distance from
LIUQE's axis to its nearest stored X-point, a flux gap by LIUQE's stored
axis-to-boundary flux difference. A gap with no counterpart on the slice -- an
X-point distance where LIUQE stores no X-point -- is left empty rather than
filled with a number the data does not carry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(Path(path).read_text())["rows"])
    rows.sort(key=lambda row: row["index"])
    return rows


def _finite(value):
    return value is not None and bool(np.isfinite(value))


def measure_gaps(row):
    axis_gap = x_gap = boundary_gap = span_length = float("nan")
    liuqe_axis = np.asarray(row["liuqe_axis"], dtype=float)
    if _finite(row.get("nova_axis_psi")):
        axis_gap = float(
            np.linalg.norm(np.asarray(row["nova_axis"], float) - liuqe_axis)
        )
    xpoints = np.asarray(row.get("liuqe_xpoints") or [], dtype=float).reshape(-1, 2)
    if len(xpoints):
        span_length = float(np.min(np.linalg.norm(xpoints - liuqe_axis, axis=1)))
    if len(xpoints) and _finite(row.get("nova_boundary_psi")):
        x_gap = float(
            np.min(
                np.linalg.norm(
                    xpoints - np.asarray(row["nova_boundary"], float), axis=1
                )
            )
        )
    axis_psi = row.get("liuqe_axis_psi_grid")
    boundary_psi = row.get("liuqe_boundary_psi_grid")
    span_flux = float("nan")
    if _finite(axis_psi) and _finite(boundary_psi):
        span_flux = abs(float(axis_psi) - float(boundary_psi))
    if _finite(row.get("nova_boundary_psi")) and _finite(boundary_psi):
        boundary_gap = abs(float(row["nova_boundary_psi"]) - float(boundary_psi))
    axis_psi_gap = float("nan")
    if _finite(row.get("nova_axis_psi")) and _finite(axis_psi):
        axis_psi_gap = abs(float(row["nova_axis_psi"]) - float(axis_psi))
    return dict(
        axis_gap=axis_gap,
        x_gap=x_gap,
        boundary_gap=boundary_gap,
        axis_psi_gap=axis_psi_gap,
        span_length=span_length,
        span_flux=span_flux,
        axis_relative=axis_gap / span_length
        if _finite(axis_gap) and _finite(span_length)
        else float("nan"),
        x_relative=x_gap / span_length
        if _finite(x_gap) and _finite(span_length)
        else float("nan"),
        boundary_relative=boundary_gap / span_flux
        if _finite(boundary_gap) and _finite(span_flux) and span_flux > 0
        else float("nan"),
        axis_psi_relative=axis_psi_gap / span_flux
        if _finite(axis_psi_gap) and _finite(span_flux) and span_flux > 0
        else float("nan"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = merge(args.rows)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))
    header = (
        f"{'idx':>5} {'t[s]':>6} {'LIUQE':>9} {'nova':>9} {'d_axis[m]':>9} "
        f"{'d_axis/s':>8} {'d_xp[m]':>8} {'d_psi':>8} {'d_psi/s':>8}"
    )
    print(header)
    agree = 0
    for row in rows:
        g = measure_gaps(row)
        same = row["liuqe_label"] == row["nova_label"]
        agree += same
        print(
            f"{row['index']:>5} {row['liuqe_time']:>6.3f} {row['liuqe_label']:>9} "
            f"{row['nova_label']:>9} {g['axis_gap']:>9.4f} {g['axis_relative']:>8.3f} "
            f"{g['x_gap']:>8.4f} {g['boundary_gap']:>8.4f} "
            f"{g['boundary_relative']:>8.3f}" + ("" if same else "  DISAGREE")
        )
    print(f"agreement {agree}/{len(rows)}")
    print("merged", args.out)


if __name__ == "__main__":
    main()
