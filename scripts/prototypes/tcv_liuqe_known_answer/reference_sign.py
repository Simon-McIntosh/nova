"""Refuse a reference whose stored flux sign contradicts its declared convention.

Reads the TCV 77549 LIUQE equilibrium through the orientation guard and records,
per slice, the declaration and the measured signs. The reference declares DD
3.41.0 (COCOS 11) with a negative plasma current, which fixes the magnetic axis
as a psi maximum; its stored grid reads the axis as a minimum on every slice.

The field-swap trace records are put through the same guard, because a
sign-guard that refuses a good record is as wrong as one that reads a bad one.
Every record whose record-wall lies inside its own grid agrees with its declared
polarity, and the records carrying no wall are checked on their stored scalars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imas
import numpy as np

from scripts.prototypes.efit_field_swap import crosscheck
from scripts.prototypes.efit_field_swap.orientation import (
    ReferenceSignError,
    check_axis_boundary_sign,
    check_trace_record,
    grid_field,
    grid_value,
    read_convention,
)
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record
from scripts.prototypes.tcv_liuqe_known_answer import measure

DD_COCOS = {3: 11, 4: 17}


def boundary_points(view):
    """The boundary the declared span is measured to: the X-points, else the outline."""
    if view["xpoints"]:
        return np.asarray(view["xpoints"], dtype=np.float64)
    if len(view["outline"]):
        return np.asarray(view["outline"], dtype=np.float64)
    return None


def guard_slice(view, index):
    """Guard one LIUQE slice, returning the refusal or the measured signs."""
    major = int(str(view["dd_version"]).split(".", 1)[0])
    convention = read_convention(
        dd_version=view["dd_version"], plasma_current=view["plasma_current"]
    )
    declared = (
        f"DD {view['dd_version']} (COCOS {DD_COCOS[major]}) with "
        f"Ip = {view['plasma_current']:+.3f} A"
    )
    point = boundary_points(view)
    field = grid_field(view["rgrid"], view["zgrid"], view["psi"])
    axis_value = grid_value(
        view["rgrid"], view["zgrid"], view["psi"], view["axis"], field=field
    )
    boundary_value = (
        None
        if point is None
        else float(
            np.mean(
                [
                    grid_value(
                        view["rgrid"], view["zgrid"], view["psi"], p, field=field
                    )
                    for p in point
                ]
            )
        )
    )
    row = dict(
        index=index,
        dd_version=view["dd_version"],
        cocos=DD_COCOS[major],
        plasma_current=view["plasma_current"],
        declared_sigma=int(np.sign(float(convention.sigma))),
        scalar_span=view["psi_axis"] - view["psi_boundary"],
        grid_axis_psi=axis_value,
        grid_boundary_psi=boundary_value,
        grid_span=None if boundary_value is None else axis_value - boundary_value,
        boundary_kind="xpoint" if view["xpoints"] else "outline",
        refused=False,
        reason="",
    )
    try:
        signs = check_axis_boundary_sign(
            convention,
            view["rgrid"],
            view["zgrid"],
            view["psi"],
            view["axis"],
            point,
            scalar_axis_psi=view["psi_axis"],
            scalar_boundary_psi=view["psi_boundary"],
            slice_index=index,
            declared=declared,
            field=field,
        )
    except ReferenceSignError as error:
        row["refused"] = True
        row["reason"] = str(error)
        return row
    row["measured_signs"] = signs
    return row


def field_swap_rows():
    """Every trace record through the guard, as (case, row)."""
    rows = []
    for case, source, path in crosscheck.cases():
        record = read_trace_record(crosscheck.resolve_trace(path))
        row = dict(
            case=case,
            source=source,
            polarity=record.polarity,
            slice=record.slice,
            refused=False,
            reason="",
        )
        try:
            row["measured_signs"] = check_trace_record(record)
        except ReferenceSignError as error:
            row["refused"] = True
            row["reason"] = str(error)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liuqe", required=True)
    parser.add_argument("--slices", required=True, help="comma-separated indices")
    parser.add_argument("--out", required=True)
    parser.add_argument("--field-swap-out")
    args = parser.parse_args()
    print("nova.__file__", __import__("nova").__file__, flush=True)

    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    db = imas.DBEntry(f"imas:hdf5?path={args.liuqe}", "r", dd_version="3.41.0")
    eq = db.get("equilibrium")
    if int(eq.ids_properties.homogeneous_time) != 1:
        raise ValueError("LIUQE equilibrium must use its IDS-level time base")
    rows = []
    for index in slices:
        row = guard_slice(measure.slice_view(eq, index), index)
        rows.append(row)
        print(
            "slice",
            index,
            "refused" if row["refused"] else "read",
            f"grid_span={row['grid_span']!r}",
            f"scalar_span={row['scalar_span']!r}",
            flush=True,
        )
    db.close()
    refused = sum(row["refused"] for row in rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"LIUQE_ROWS {len(rows)} REFUSED {refused}", flush=True)

    if args.field_swap_out:
        swap = field_swap_rows()
        Path(args.field_swap_out).write_text(
            json.dumps({"rows": swap}, indent=2) + "\n"
        )
        swap_refused = sum(row["refused"] for row in swap)
        print(f"FIELD_SWAP_ROWS {len(swap)} REFUSED {swap_refused}", flush=True)
    return rows


if __name__ == "__main__":
    main()
