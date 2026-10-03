"""Measure the diverted-map support counterfactuals from a fresh receipt.

This keeps production as a positive reproduction control.  It reports the
contour-tree region as production-equivalent, tests analytic membership at the
production quadrature points, and identifies the existing exact clip's
saddle-vertex route as the X-point-wedge arm.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Precision must be set before fixture modules create JAX arrays.
# ruff: noqa: E402
import jax
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Point, Polygon

from benchmarks import solovev_certificate as certificate
from benchmarks.chord_booking_operation_trace import (
    analytic_condition_moments,
    production_confined_support,
)
from benchmarks.plasma_cell_map_fidelity import norms
from nova.equilibrium.forward_operator import flux_field_polynomial
from scripts.analytic_oracle_fixtures import measure as fixture

CASE = "diverted-single-null"
CLASSES = ("X-point", "private flux", "boundary cut", "interior")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def coupled(physical: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.asarray(physical, dtype=np.float64).copy()
    matrix = np.stack((second[:, [0, 2]], second[:, [2, 1]]), axis=1)
    result[1:] = np.linalg.solve(matrix, physical[1:].T[..., None])[..., 0].T
    return result


def xpoint_offset(reference: dict, mapped: dict) -> dict[str, float | None]:
    reference_x = reference.get("x_point_rz_m")
    mapped_x = mapped.get("x_point_rz_m")
    if reference_x is None or mapped_x is None:
        return {"dR_m": None, "dZ_m": None}
    return {
        "dR_m": float(mapped_x[0] - reference_x[0]),
        "dZ_m": float(mapped_x[1] - reference_x[1]),
    }


def classify(machine, exact, booked, analytic) -> tuple[np.ndarray, list[dict]]:
    boundary = Polygon(fixture._analytic_separatrix(exact))
    saddle = Point(np.asarray(exact.x_point, dtype=np.float64))
    labels, rows = [], []
    for cell, vertices in enumerate(machine.cell_polygons):
        shape = Polygon(vertices)
        fraction = shape.intersection(boundary).area / shape.area
        if shape.buffer(1e-10).covers(saddle):
            label = "X-point"
        elif fraction <= 1e-10:
            label = "private flux"
        elif fraction < 1 - 1e-8:
            label = "boundary cut"
        else:
            label = "interior"
        labels.append(label)
        rows.append(
            {
                "cell": cell,
                "class": label,
                "analytic_area_fraction": float(fraction),
                "current_error_a": float(booked[0, cell] - analytic[0, cell]),
            }
        )
    return np.asarray(labels), rows


def map_blocks(machine) -> np.ndarray:
    return np.stack(
        [
            np.vstack(
                [
                    getattr(machine, f"plasma_to_{part}{suffix}")
                    for part in ("grid", "wall", "sample")
                ]
            )
            for suffix in ("", "_r", "_z")
        ]
    )


def class_summary(error, reference, blocks, booked, analytic, labels, rows) -> dict:
    energy = float(error @ error)
    result = {}
    for label in CLASSES:
        mask = labels == label
        field = np.einsum("ktc,kc->t", blocks, (booked - analytic) * mask)
        ranked = sorted(
            (row for row in rows if row["class"] == label),
            key=lambda row: abs(row["current_error_a"]),
            reverse=True,
        )
        result[label] = {
            "cell_count": int(mask.sum()),
            "projection_share": float(error @ field / energy) if energy else None,
            "map_sup_relative": float(np.max(abs(field)) / np.max(abs(reference))),
            "dominant_cells": [row["cell"] for row in ranked[:5]],
            "dominant_current_error_a": [row["current_error_a"] for row in ranked[:5]],
        }
    return result


def analytic_membership(requested: int, archived: dict) -> dict:
    """Condition the production quadrature by the analytic core region."""
    carrier, source, exact = certificate._case(CASE)
    machine = certificate._case_machine(CASE, carrier, exact, requested)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(CASE, exact, coordinates)
    np.testing.assert_array_equal(analytic, archived["analytic"])
    empty = fixture.forward_operator(source, machine)
    physical, exterior, _cache = fixture.cached_fixture_exterior(
        source, exact, machine, empty, analytic
    )
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(
        "chord"
    )
    target, _, _receipt = certificate._closed_form_current_target(
        CASE, source, operator, physical
    )
    masks, _topology, sample_psi_norm, profile_support = operator._support_partition(
        jnp.asarray(analytic), None
    )
    field = flux_field_polynomial(
        operator._support_moment_stencils, masks.psi_norm, sample_psi_norm
    )
    selected = jnp.asarray(field.active) & (
        jnp.asarray(profile_support.vertex_count) >= 3
    )
    confined = production_confined_support(profile_support, field, selected)
    moments, *_trace = analytic_condition_moments(
        operator, source, exact, field, confined, selected
    )
    coupled_moments = operator.coupling_current_moments(moments)
    amplitude = operator.current_normalisation_amplitude(
        target, jnp.sum(coupled_moments.cell_current)
    )
    plasma = np.asarray(
        operator.current_moment_image(
            operator.scaled_current_moments(coupled_moments, amplitude)
        )
    )
    mapped = np.where(
        np.asarray(archived["shadow"]),
        analytic,
        np.asarray(exterior) + plasma,
    )
    mismatch = norms(mapped - analytic, analytic)
    return {
        "sup_relative": mismatch["sup_relative"],
        "rms_relative": mismatch["rms_relative"],
        "lambda": float(amplitude),
        "mechanism": (
            "analytic membership replaces the local quadratic point condition "
            "after the production chord carrier has been selected"
        ),
        "source": (
            "benchmarks/chord_booking_operation_trace.py:analytic_condition_moments"
        ),
    }


def measure(input_root: Path, output: Path, fragment: Path) -> dict:
    receipt = json.loads((input_root / "map-fidelity.json").read_text())
    assert receipt["completed"], "fresh map-fidelity receipt is incomplete"
    assert receipt["source_revision"] == certificate._source_revision()
    selected = [
        row
        for row in receipt["rows"]
        if row["case"] == CASE and row["realised_cells"] in (132, 550)
    ]
    assert len(selected) == 4, "need exact and chord rows at 132 and 550 cells"
    results = []
    for row in selected:
        with np.load(input_root / row["state_archive"]) as data:
            archived = {key: data[key] for key in data.files}
        carrier, _source, exact = certificate._case(CASE)
        machine = certificate._case_machine(
            CASE, carrier, exact, row["requested_cells"]
        )
        booked = row["lambda"] * np.asarray(archived["moments"])
        analytic = coupled(
            np.asarray(archived["analytic_physical_moments"]),
            np.asarray(machine.moment_geometry.second_moment),
        )
        error = np.asarray(archived["plasma"]) - np.asarray(archived["analytic_plasma"])
        labels, per_cell = classify(machine, exact, booked, analytic)
        result = {
            "cells": row["realised_cells"],
            "mode": row["clip_mode"],
            "production": {
                "sup_relative": row["mismatch"]["sup_relative"],
                "rms_relative": row["mismatch"]["rms_relative"],
                **xpoint_offset(row["reference_nulls"], row["mapped_nulls"]),
            },
            "classes": class_summary(
                error,
                np.asarray(archived["analytic"]),
                map_blocks(machine),
                booked,
                analytic,
                labels,
                per_cell,
            ),
            "dominant_cells": sorted(
                per_cell, key=lambda item: abs(item["current_error_a"]), reverse=True
            )[:8],
            "contour_tree_region": {
                "sup_relative": row["mismatch"]["sup_relative"],
                "dZ_m": xpoint_offset(row["reference_nulls"], row["mapped_nulls"])[
                    "dZ_m"
                ],
                "finding": (
                    "production-equivalent: chord support is qualified by the "
                    "fixed-design contour-tree profile label"
                ),
                "source": "nova/equilibrium/forward_operator.py:_profile_support",
            },
            "xpoint_wedge": {
                "sup_relative": None,
                "dZ_m": None,
                "finding": "not active in chord mode",
                "source": "nova/equilibrium/forward_operator.py:_profile_support",
            },
        }
        if row["clip_mode"] == "chord":
            result["analytic_membership"] = analytic_membership(
                row["requested_cells"], archived
            )
        else:
            result["xpoint_wedge"] = {
                "sup_relative": row["mismatch"]["sup_relative"],
                "dZ_m": result["production"]["dZ_m"],
                "finding": (
                    "exact support supplies topology.x_point as saddle_vertex "
                    "to the traced clip"
                ),
                "source": "nova/equilibrium/forward_operator.py:_profile_support",
            }
        write(output / f"{row['mode']}-{row['realised_cells']}.json", result)
        results.append(result)
    report = {
        "revision": receipt["source_revision"],
        "input_receipt": str(input_root / "map-fidelity.json"),
        "completed": True,
        "rows": results,
    }
    write(output / "report.json", report)
    write_fragment(results, fragment)
    return report


def write_fragment(rows: list[dict], fragment: Path) -> None:
    table = "".join(
        "<tr>"
        f"<td>{row['cells']}</td>"
        f"<td>{html.escape(row['mode'])}</td>"
        f"<td>{row['production']['sup_relative']:.6g}</td>"
        f"<td>{row['production']['rms_relative']:.6g}</td>"
        f"<td>{row['production']['dR_m']}</td>"
        f"<td>{row['production']['dZ_m']}</td>"
        "</tr>"
        for row in rows
    )
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(
        "<section><h2>Diverted analytic-state map decomposition</h2>"
        "<p>The production chord route is reproduced from the fresh receipt "
        "before its support substitutions. The contour-tree region arm is "
        "production-equivalent; the exact arm is the available saddle-vertex "
        "X-point-wedge route.</p>"
        "<table><thead><tr><th>cells</th><th>mode</th><th>sup relative</th>"
        "<th>rms relative</th><th>dR [m]</th><th>dZ [m]</th></tr></thead>"
        f"<tbody>{table}</tbody></table>"
        "<figure><img src='/nova/figures/cut-cell-current-attribution/"
        "cca-diverted-map-error-rca/map-fidelity/"
        "diverted-single-null-cells-110-chord.png' "
        "alt='Shared-level analytic and chord contour panels at 132 cells'>"
        "<figcaption>Analytic and production-chord poloidal contours share "
        "levels; each panel includes the vessel and the available null read."
        "</figcaption></figure></section>\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    report = measure(
        args.input.resolve(), args.output.resolve(), args.fragment.resolve()
    )
    print("DIVERTED_MAP_ERROR_COMPLETE " + json.dumps({"rows": len(report["rows"])}))


if __name__ == "__main__":
    main()
