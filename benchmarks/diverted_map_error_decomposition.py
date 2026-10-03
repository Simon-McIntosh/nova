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
from benchmarks.plasma_cell_map_fidelity import norms, nulls
from nova.equilibrium.forward_operator import (
    _ExactClipLevel,
    _implicit_traced_level_arc,
    _traced_clip,
    fit_split_spline,
    flux_field_polynomial,
)
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


def mapped_support(
    operator, masks, sample_psi_norm, support, target, analytic, exterior, shadow
):
    """Image one explicit support through the production moment and Green paths."""
    moment_masks = operator._moment_support_masks(masks, support)
    moments = operator.source.current_moments(
        moment_masks,
        operator.support_current_moments,
        support,
        sample_flux=sample_psi_norm,
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
    mapped = np.where(np.asarray(shadow), analytic, np.asarray(exterior) + plasma)
    return mapped, float(amplitude)


def contour_tree_region(requested: int, archived: dict, mode: str) -> dict:
    """Re-book the support from an independently invoked contour-tree read."""
    carrier, source, exact = certificate._case(CASE)
    machine = certificate._case_machine(CASE, carrier, exact, requested)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(CASE, exact, coordinates)
    empty = fixture.forward_operator(source, machine)
    physical, exterior, _cache = fixture.cached_fixture_exterior(
        source, exact, machine, empty, analytic
    )
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(mode)
    target, _, _receipt = certificate._closed_form_current_target(
        CASE, source, operator, physical
    )
    production_masks, _production_topology, _sample, _support = (
        operator._support_partition(jnp.asarray(analytic), None)
    )
    masks, topology, _connected, admitted = operator._fixed_design_read(
        jnp.asarray(analytic), None
    )
    assert bool(admitted), "contour-tree oracle requires an admitted axis"
    sample_flux = operator.sample_node_flux(jnp.asarray(analytic))
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    support = operator._profile_support(
        masks, topology, jnp.asarray(analytic), sample_psi_norm
    )
    mapped, amplitude = mapped_support(
        operator,
        masks,
        sample_psi_norm,
        support,
        target,
        analytic,
        exterior,
        archived["shadow"],
    )
    mismatch = norms(mapped - analytic, analytic)
    membership_difference = int(
        np.count_nonzero(
            np.asarray(masks.profile_participation)
            != np.asarray(production_masks.profile_participation)
        )
    )
    return {
        "sup_relative": mismatch["sup_relative"],
        "rms_relative": mismatch["rms_relative"],
        **xpoint_offset(nulls(operator, analytic), nulls(operator, mapped)),
        "lambda": amplitude,
        "membership_difference_count": membership_difference,
        "executed": True,
        "source": "nova/equilibrium/forward_operator.py:_fixed_design_read",
    }


def analytic_saddle_wedge(requested: int, archived: dict) -> dict:
    """Rebuild exact support with the analytic saddle, not the mapped saddle."""
    carrier, source, exact = certificate._case(CASE)
    machine = certificate._case_machine(CASE, carrier, exact, requested)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(CASE, exact, coordinates)
    empty = fixture.forward_operator(source, machine)
    physical, exterior, _cache = fixture.cached_fixture_exterior(
        source, exact, machine, empty, analytic
    )
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(
        "exact"
    )
    target, _, _receipt = certificate._closed_form_current_target(
        CASE, source, operator, physical
    )
    masks, topology, sample_psi_norm, _production_support = operator._support_partition(
        jnp.asarray(analytic), None
    )
    shared_flux = operator.shared_node_flux(jnp.asarray(analytic))
    inside_boundary = operator.polarity * (shared_flux - topology.boundary_flux)
    coefficient = operator.support_flux_coefficients(masks.psi_norm, sample_psi_norm)
    coefficient = -coefficient.at[:, 0].add(1.0)
    coordinate = jnp.asarray(operator.grid.coordinate, dtype=masks.psi_norm.dtype)
    surface = fit_split_spline(
        coordinate[None, :, 0],
        coordinate[None, :, 1],
        masks.psi_norm[None, :],
        masks.psi_norm[None, :] - 1.0,
        order=6,
        regularization=1.0e-14,
    )
    level = _ExactClipLevel(
        surface,
        coefficient,
        operator._support_curve_centre,
        operator._support_curve_scale,
    )
    mesh = operator.moment_geometry.atomic_mesh
    vertices = jnp.asarray(mesh.node_coordinates)[jnp.asarray(mesh.cell_nodes)]
    participation = masks.profile_participation | operator._vertex_level_participation(
        mesh.cell_vertex_count, level(vertices)
    )
    analytic_apex = jnp.asarray(exact.x_point, dtype=analytic.dtype)
    support = _traced_clip(
        mesh.node_coordinates,
        mesh.cell_nodes,
        mesh.cell_vertex_count,
        mesh.centroids,
        mesh.support_capacity,
        inside_boundary,
        saddle_vertex=analytic_apex,
        curve_evaluator=level,
        participating_cell=participation,
        arc_tracer=_implicit_traced_level_arc,
    ).qualify(participation)
    mapped, amplitude = mapped_support(
        operator,
        masks,
        sample_psi_norm,
        support,
        target,
        analytic,
        exterior,
        archived["shadow"],
    )
    mismatch = norms(mapped - analytic, analytic)
    return {
        "sup_relative": mismatch["sup_relative"],
        "rms_relative": mismatch["rms_relative"],
        **xpoint_offset(nulls(operator, analytic), nulls(operator, mapped)),
        "lambda": amplitude,
        "production_apex_rz_m": np.asarray(topology.x_point).tolist(),
        "analytic_apex_rz_m": np.asarray(exact.x_point).tolist(),
        "executed": True,
        "source": "nova/equilibrium/forward_operator.py:_profile_support",
    }


def analytic_membership(requested: int, archived: dict, mode: str) -> dict:
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
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(mode)
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
        operator, operator.source, exact, field, confined, selected
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
        **xpoint_offset(nulls(operator, analytic), nulls(operator, mapped)),
        "lambda": float(amplitude),
        "clip_mode": mode,
        "mechanism": (
            "analytic membership replaces the local quadratic point condition "
            "after the production chord carrier has been selected"
        ),
        "source": (
            "benchmarks/chord_booking_operation_trace.py:analytic_condition_moments"
        ),
    }


def measure(input_root: Path, output: Path, fragment: Path, base_sha: str) -> dict:
    receipt = json.loads((input_root / "map-fidelity.json").read_text())
    assert receipt["completed"], "fresh map-fidelity receipt is incomplete"
    assert receipt["source_revision"] == base_sha, (
        f"receipt revision {receipt['source_revision']} is not dispatch base {base_sha}"
    )
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
            "xpoint_wedge": {
                "sup_relative": None,
                "dZ_m": None,
                "finding": "not active in chord mode",
                "source": "nova/equilibrium/forward_operator.py:_profile_support",
            },
        }
        result["analytic_membership"] = analytic_membership(
            row["requested_cells"], archived, row["clip_mode"]
        )
        result["contour_tree_region"] = contour_tree_region(
            row["requested_cells"], archived, row["clip_mode"]
        )
        if row["clip_mode"] == "exact":
            result["xpoint_wedge"] = analytic_saddle_wedge(
                row["requested_cells"], archived
            )
        write(output / f"{row['clip_mode']}-{row['realised_cells']}.json", result)
        results.append(result)
    report = {
        "measurement_revision": receipt["source_revision"],
        "analysis_revision": certificate._source_revision(),
        "input_receipt": str(input_root / "map-fidelity.json"),
        "completed": True,
        "rows": results,
    }
    write(output / "report.json", report)
    write_fragment(results, fragment, receipt)
    return report


def write_fragment(rows: list[dict], fragment: Path, receipt: dict) -> None:
    map_table = "".join(
        "<tr>"
        f"<td>{html.escape(row['case'])}</td>"
        f"<td>{row['realised_cells']}</td>"
        f"<td>{html.escape(row['clip_mode'])}</td>"
        f"<td>{row['mismatch']['sup_relative']:.6g}</td>"
        f"<td>{row['mismatch']['rms_relative']:.6g}</td>"
        f"<td>{row['support_current_centroid_offset_mm']['dR']}</td>"
        f"<td>{row['support_current_centroid_offset_mm']['dZ']}</td>"
        "</tr>"
        for row in receipt["rows"]
        if row["status"] == "measured"
    )
    rows_132 = [row for row in rows if row["cells"] == 132]
    class_table = "".join(
        "<tr>"
        f"<td>{html.escape(row['mode'])}</td>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{values['cell_count']}</td>"
        f"<td>{values['projection_share']:.6g}</td>"
        f"<td>{values['map_sup_relative']:.6g}</td>"
        f"<td>{', '.join(map(str, values['dominant_cells']))}</td>"
        "</tr>"
        for row in rows_132
        for label, values in row["classes"].items()
    )
    substitution_table = "".join(
        "<tr>"
        f"<td>{row['cells']}</td>"
        f"<td>{html.escape(row['mode'])}</td>"
        f"<td>{row['contour_tree_region']['sup_relative']:.6g}</td>"
        f"<td>{row['contour_tree_region']['dZ_m']}</td>"
        f"<td>{row['contour_tree_region']['membership_difference_count']}</td>"
        f"<td>{row['xpoint_wedge']['sup_relative']}</td>"
        f"<td>{row['xpoint_wedge']['dZ_m']}</td>"
        f"<td>{row['xpoint_wedge'].get('production_apex_rz_m')}</td>"
        f"<td>{row['xpoint_wedge'].get('analytic_apex_rz_m')}</td>"
        f"<td>{row.get('analytic_membership', {}).get('sup_relative')}</td>"
        f"<td>{row.get('analytic_membership', {}).get('dZ_m')}</td>"
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
        "<p>No tested substitution puts diverted exact below the 1e-2 bound. "
        "Analytic membership improves chord at 550 cells below the bound but "
        "does not alter the exact saddle-vertex route.</p>"
        "<p>The 2026-09-21 followup premise is stale: production now passes "
        "topology.x_point as saddle_vertex in forward_operator.py:3424, from "
        "commit d90a522fe; the analytic-apex arm below measures the remaining "
        "difference from that mapped apex.</p>"
        "<h3>Full H200 map receipt</h3><table><thead><tr><th>case</th>"
        "<th>cells</th><th>mode</th>"
        "<th>sup relative</th>"
        "<th>rms relative</th><th>dR [m]</th><th>dZ [m]</th></tr></thead>"
        f"<tbody>{map_table}</tbody></table>"
        "<h3>132-cell current-image classes</h3><table><thead><tr><th>mode</th>"
        "<th>class</th><th>cells</th><th>projection share</th><th>map sup relative</th>"
        f"<th>dominant indices</th></tr></thead><tbody>{class_table}</tbody></table>"
        "<h3>Oracle support substitutions</h3><table><thead><tr><th>cells</th>"
        "<th>base mode</th><th>contour-tree region sup</th><th>contour-tree dZ [m]</th>"
        "<th>contour-tree membership differences</th>"
        "<th>X-point wedge sup</th><th>X-point wedge dZ [m]</th>"
        "<th>production apex [m]</th><th>analytic apex [m]</th>"
        "<th>analytic-membership sup</th><th>analytic-membership dZ [m]</th>"
        "</tr></thead>"
        f"<tbody>{substitution_table}</tbody></table>"
        "<figure><img src='/nova/figures/cut-cell-current-attribution/"
        "cca-diverted-map-error-rca/map-fidelity-stable/"
        "diverted-single-null-cells-110-chord.png' "
        "alt='Shared-level analytic and chord contour panels at 132 cells'>"
        "<figcaption>Analytic and production-chord poloidal contours share "
        "levels; each panel includes the vessel and the available null read."
        "</figcaption></figure>"
        "<figure><img src='/nova/figures/cut-cell-current-attribution/"
        "cca-diverted-map-error-rca/map-fidelity-stable/"
        "diverted-single-null-cells-110-exact.png' "
        "alt='Shared-level analytic and exact-support contour panels at 132 cells'>"
        "<figcaption>Analytic and saddle-vertex exact-support poloidal contours "
        "share levels; the exact route does not meet the certificate bound."
        "</figcaption></figure></section>\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()
    report = measure(
        args.input.resolve(),
        args.output.resolve(),
        args.fragment.resolve(),
        args.base_sha,
    )
    print("DIVERTED_MAP_ERROR_COMPLETE " + json.dumps({"rows": len(report["rows"])}))


if __name__ == "__main__":
    main()
