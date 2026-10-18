"""Join the independent spline census to the polarity-aware read receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from census import hessian_type
from field import FastField
from plot import _inside_vessel, _wall_loops_by_role
from records import parse_record


def _case(name):
    return name.replace(":", "-").replace("_", "-").lower()


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def refresh(rows, adapter_rows):
    fixed = {_case(item["case"]): item for item in adapter_rows}
    for row in rows:
        item = fixed.get(_case(row["case"]))
        if item is None:
            raise ValueError(f"fixed adapter receipt lacks {row['case']}")
        record = parse_record(row["path"])
        rings = _wall_loops_by_role(row["path"])
        points = row["census"]
        positions = np.asarray([[p["r"], p["z"]] for p in points]).reshape(-1, 2)
        inside = _inside_vessel(positions, rings)
        row["census"] = []
        for point, admitted in zip(points, inside):
            if admitted:
                point["inside_wall"] = True
                row["census"].append(point)
        row["n_census"] = len(points)
        row["n_in_wall"] = len(row["census"])
        row["axis_extremum"] = None
        if row["census"]:
            selected = (
                max(row["census"], key=lambda p: p["psi"])
                if record.polarity > 0
                else min(row["census"], key=lambda p: p["psi"])
            )
            row["axis_extremum"] = {
                key: selected[key] for key in ("r", "z", "psi", "kind")
            }

        keys = (
            "nova_axis_r",
            "nova_axis_z",
            "nova_axis_psi",
            "nova_saddle_r",
            "nova_saddle_z",
            "nova_saddle_psi",
            "nova_boundary_psi",
            "nova_wall_level",
        )
        row["adapter_a"] = {key: _finite(item.get(key)) for key in keys}
        row["adapter_a"].update(
            nova_label=item["nova_label"], nova_note=item.get("note")
        )
        row["code_points"]["nova_a_axis"] = None
        if row["adapter_a"]["nova_axis_r"] is not None:
            r = row["adapter_a"]["nova_axis_r"]
            z = row["adapter_a"]["nova_axis_z"]
            field = FastField(record.rgrid, record.zgrid, record.psi)
            eig, kind, grad = hessian_type(field, r, z)
            value, _ = field.value_gradient(np.asarray([r]), np.asarray([z]))
            row["code_points"]["nova_a_axis"] = {
                "r": r,
                "z": z,
                "psi": float(value[0]),
                "psi_reported": row["adapter_a"]["nova_axis_psi"],
                "grad_norm": float(np.linalg.norm(grad)),
                "eig": [float(eig[0]), float(eig[1])],
                "kind": kind,
                "agrees_with_polarity": (
                    kind == "maximum" if record.polarity > 0 else kind == "minimum"
                ),
                "source": "polarity-aware nova topology read",
            }
        level = row["adapter_a"]["nova_boundary_psi"]
        for point in row["census"]:
            point["side_nova_a_boundary"] = (
                (
                    "above"
                    if point["psi"] > level
                    else "below"
                    if point["psi"] < level
                    else "on"
                )
                if level is not None
                else None
            )
        row["adapter_source"] = item["case"]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    census_rows = json.loads(args.census.read_text())
    adapter_rows = json.loads(args.adapter.read_text())["rows"]
    rows = refresh(census_rows, adapter_rows)
    args.out.write_text(json.dumps(rows, indent=1, allow_nan=False))
    print(
        f"rows={len(rows)} nonstress={sum(r['machine'] != 'JT-60SA' for r in rows)} "
        f"in_vessel={sum(r['n_in_wall'] for r in rows)} "
        f"total_census={sum(r['n_census'] for r in rows)}"
    )


if __name__ == "__main__":
    main()
