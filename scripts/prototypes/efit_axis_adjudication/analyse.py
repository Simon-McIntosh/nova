"""Build the census for every imas-efit trace record and stage the results.

Reads the records in place, rebuilds each one's own field with scipy, checks
the rebuild at imas-efit's recorded axis, runs the exhaustive stationary-point
census, and writes one JSON staging file that the figure and page steps read.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from records import parse_record
from census import census, hessian_type
from field import FastField

REPORT = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission"
)
CANONICAL = pathlib.Path(
    "/home/ITER/mcintos/Code/imas-efit/tests/unit/fortran/data/boundary_level"
)
ADAPTER_A = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261018T063236394668-efit-field-swap/exports/disagreement.json"
)
ADAPTER_B = pathlib.Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261018T063236394831-efit-field-swap/exports/measurements_final"
)


def finite(value):
    return value if value is not None and math.isfinite(value) else None


MACHINES = (
    ("jt60sa", "JT-60SA"),
    ("aug", "AUG"),
    ("iter", "ITER"),
    ("mastu", "MAST-U"),
    ("synthetic", "synthetic"),
)


def machine_of(case_id: str) -> str:
    for key, label in MACHINES:
        if case_id.startswith(key):
            return label
    return "other"


def build_cases():
    """Yield (case_id, path, source_note) for 24 pair records + 2 synthetic."""
    cases = []
    for line in (REPORT / "pairs.tsv").read_text().splitlines():
        if not line.strip():
            continue
        pair, left, right = line.split("\t")
        for side, raw in (("A", left), ("B", right)):
            path = pathlib.Path(raw)
            note = "pairs.tsv trace path"
            if not path.is_file():
                path = CANONICAL / path.name
                note = "canonical imas-efit tracked fixture (stale worktree path)"
            if not path.is_file():
                raise FileNotFoundError(raw)
            cases.append((f"{pair}-{side}", path, note))
    for side in ("a", "b"):
        path = REPORT / "crossing" / f"synthetic_crossing_{side}.txt"
        cases.append(
            (f"synthetic-crossing-{side.upper()}", path, "synthetic crossing trace")
        )
    return cases


def load_adapter_a():
    data = json.loads(ADAPTER_A.read_text())
    out = {}
    for row in data["rows"]:
        case = row["case"].replace(":", "-").lower()
        # adapter A carries nan geometry where nova admits no axis or saddle;
        # carry those as absent rather than as a point at nan
        for key in (
            "nova_axis_r",
            "nova_axis_z",
            "nova_axis_psi",
            "nova_saddle_r",
            "nova_saddle_z",
            "nova_saddle_psi",
            "nova_boundary_psi",
            "field_grad_at_nova_axis",
        ):
            row[key] = finite(row.get(key))
        out[case] = row
    return out


def load_adapter_b():
    out = {}
    for path in sorted(ADAPTER_B.glob("*.json")):
        row = json.loads(path.read_text())
        out[path.stem.lower()] = row
    return out


def point_side(value: float, level: float) -> str:
    if value > level:
        return "above"
    if value < level:
        return "below"
    return "on"


def analyse(case_id, path, note, ad_a, ad_b, n_seed):
    record = parse_record(path)
    field = FastField(record.rgrid, record.zgrid, record.psi)

    # (1) rebuild check at imas-efit's recorded axis
    axis_value, grad = field.value_gradient(
        np.array([record.axis_r]), np.array([record.axis_z])
    )
    axis_value = float(axis_value[0])
    grad_norm = float(np.linalg.norm(grad[0]))
    value_gap = float(abs(axis_value - record.axis_psi))
    rebuild_ok = value_gap <= 1e-10 and grad_norm <= 1e-9
    contour_levels = np.linspace(float(record.psi.min()), float(record.psi.max()), 22)

    points = census(
        field,
        record,
        n_seed=n_seed,
        wall_polygon=record.wall_polygon if len(record.wall_polygon) >= 3 else None,
    )

    a = ad_a.get(case_id.lower())
    b = ad_b.get(case_id.lower())

    def near_level(level):
        if level is None:
            return None
        best = min(points, key=lambda p: abs(p.psi - level))
        return best

    census_rows = []
    for p in points:
        census_rows.append(
            {
                "r": float(p.position[0]),
                "z": float(p.position[1]),
                "psi": float(p.psi),
                "eig": [float(p.eigenvalues[0]), float(p.eigenvalues[1])],
                "kind": p.kind,
                "converged": bool(p.converged),
                "inside_wall": bool(p.inside_wall),
                "grad_norm": float(np.linalg.norm(p.gradient)),
                "side_efit_boundary": point_side(p.psi, record.wall_level),
                "side_nova_a_boundary": (
                    point_side(p.psi, a["nova_boundary_psi"])
                    if a and a["nova_boundary_psi"] is not None
                    else None
                ),
                "side_nova_b_boundary": point_side(p.psi, b["nova"]["boundary_level"])
                if b
                else None,
            }
        )

    in_wall = [p for p in points if p.inside_wall]
    extremum = None
    if in_wall:
        extremum = (
            max(in_wall, key=lambda p: p.psi)
            if record.polarity > 0
            else min(in_wall, key=lambda p: p.psi)
        )

    def describe_code_point(r, z, level, source):
        eig, kind, g = hessian_type(field, r, z)
        val = float(field.value_gradient(np.array([r]), np.array([z]))[0][0])
        agrees = (record.polarity > 0 and kind == "maximum") or (
            record.polarity < 0 and kind == "minimum"
        )
        return {
            "r": float(r),
            "z": float(z),
            "psi": float(val),
            "psi_reported": float(level) if level is not None else None,
            "grad_norm": float(np.linalg.norm(g)),
            "eig": [float(eig[0]), float(eig[1])],
            "kind": kind,
            "agrees_with_polarity": bool(agrees),
            "source": source,
        }

    # (2) Hessian type for every axis either code reports
    efit_axis = describe_code_point(
        record.axis_r, record.axis_z, record.axis_psi, "imas-efit axis (record)"
    )
    nova_a_axis = None
    if a is not None and a["nova_axis_r"] is not None:
        nova_a_axis = describe_code_point(
            a["nova_axis_r"],
            a["nova_axis_z"],
            a["nova_axis_psi"],
            "adapter A nova axis (nova topology read)",
        )
    nova_b_axis = None
    if b is not None:
        level = b["nova"]["axis_level"]
        match = near_level(level)
        if match is not None:
            nova_b_axis = describe_code_point(
                match.position[0],
                match.position[1],
                level,
                "adapter B axis (census point nearest nova axis level)",
            )
            nova_b_axis["matched_by"] = "level"

    result = {
        "case": case_id,
        "path": str(path),
        "source_note": note,
        "machine": machine_of(case_id),
        "slice": record.slice,
        "iteration": record.iteration,
        "verdict": record.verdict,
        "efit_label": record.label,
        "polarity": record.polarity,
        "contour_levels": contour_levels.tolist(),
        "efit": {
            "axis_r": float(record.axis_r),
            "axis_z": float(record.axis_z),
            "axis_psi": float(record.axis_psi),
            "saddle_refined": float(record.saddle_refined),
            "saddle_contested": float(record.saddle_contested),
            "saddle_source_node": int(record.saddle_source_node),
            "saddle_source_position": (
                record.saddle_source_position.tolist()
                if record.saddle_source_position is not None
                else None
            ),
            "boundary_level": float(record.wall_level),
            "fed_back_psi": float(record.fed_back_psi),
            "label": record.label,
        },
        "rebuild": {
            "axis_psi_reported": float(record.axis_psi),
            "axis_psi_spline": float(axis_value),
            "axis_psi_abs_diff": value_gap,
            "grad_norm_at_axis": float(grad_norm),
            "status": "pass" if rebuild_ok else "rebuild failed",
        },
        "adapter_a": (
            {
                "nova_axis_r": a["nova_axis_r"],
                "nova_axis_z": a["nova_axis_z"],
                "nova_axis_psi": a["nova_axis_psi"],
                "nova_saddle_r": a["nova_saddle_r"],
                "nova_saddle_z": a["nova_saddle_z"],
                "nova_saddle_psi": a["nova_saddle_psi"],
                "nova_boundary_psi": a["nova_boundary_psi"],
                "nova_wall_level": a["nova_wall_level"],
                "nova_label": a["nova_label"],
                "nova_note": a.get("note"),
            }
            if a
            else None
        ),
        "adapter_b": (
            {
                "nova_axis_level": float(b["nova"]["axis_level"]),
                "nova_saddle_position": b["nova"]["saddle_position"],
                "nova_saddle_level": b["nova"]["saddle_level"],
                "nova_boundary_level": float(b["nova"]["boundary_level"]),
                "nova_label": b["nova"]["label"],
                "nova_wall_contact_level": float(b["nova"]["wall_contact_level"]),
            }
            if b
            else None
        ),
        "census": census_rows,
        "n_census": len(points),
        "n_in_wall": len(in_wall),
        "axis_extremum": (
            {
                "r": float(extremum.position[0]),
                "z": float(extremum.position[1]),
                "psi": float(extremum.psi),
                "kind": extremum.kind,
            }
            if extremum is not None
            else None
        ),
        "code_points": {
            "efit_axis": efit_axis,
            "nova_a_axis": nova_a_axis,
            "nova_b_axis": nova_b_axis,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-seed", type=int, default=120)
    args = parser.parse_args()

    ad_a = load_adapter_a()
    ad_b = load_adapter_b()
    cases = build_cases()
    results = []
    for case_id, path, note in cases:
        results.append(analyse(case_id, path, note, ad_a, ad_b, args.n_seed))
        print(f"analysed {case_id}: {len(results[-1]['census'])} census points")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent.joinpath("results.json").write_text(json.dumps(results, indent=1))
    print("wrote", out.parent / "results.json")


if __name__ == "__main__":
    main()
