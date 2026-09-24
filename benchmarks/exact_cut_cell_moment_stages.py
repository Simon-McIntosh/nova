"""Attribute simple cut-cell current error inside the exact moment path.

The incoming exact support, the secondary local-flux reclip, the density
evaluated by the production profile, and the final polynomial moment reduction
are measured separately at the analytic state.  No equilibrium solve or
production source change is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmarks import exact_support_floor_attribution as base


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT = ROOT / "docs/figures/cut-cell-current-attribution/exact-support-floor"
OUTPUT = ROOT / "docs/figures/cut-cell-current-attribution/exact-moment-stages"
ROWS = (
    ("weak-rotation-reactor-static", 110),
    ("weak-rotation-reactor-static", 300),
    ("diverted-single-null", 110),
)
TERMS = (
    "secondary_geometry",
    "density_evaluation",
    "reclip_self_intersection_loss",
    "moment_reduction",
)


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n")


def _polygon(vertices, count):
    from shapely.geometry import Polygon

    if int(count) < 3:
        return Polygon()
    return Polygon(np.asarray(vertices)[: int(count)])


def _vertex_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the symmetric largest nearest-vertex distance in metres."""
    if not len(left) and not len(right):
        return 0.0
    if not len(left) or not len(right):
        return float("inf")
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    return float(max(distance.min(axis=0).max(), distance.min(axis=1).max()))


def _ring_rule(vertices, order, *, positive_orientation):
    """Return a signed triangle-fan rule for one closed polygonal chain."""
    vertices = np.asarray(vertices)
    if len(vertices) < 3:
        return np.empty((0, 2)), np.empty(0)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1.0) / 2.0, weights / 2.0
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    wu, wv = np.meshgrid(weights, weights, indexing="ij")
    u = u.reshape(-1)
    v = v.reshape(-1)
    rule_weight = (wu * wv).reshape(-1)
    triangle = np.arange(1, len(vertices) - 1)
    first = np.broadcast_to(vertices[:1], (len(vertices) - 2, 2))
    second = vertices[triangle]
    third = vertices[triangle + 1]
    edge_first = second - first
    edge_second = third - first
    points = (
        first[:, None, :]
        + u[None, :, None] * edge_first[:, None, :]
        + (1.0 - u)[None, :, None] * v[None, :, None] * edge_second[:, None, :]
    )
    cross = (
        edge_first[..., 0] * edge_second[..., 1]
        - edge_first[..., 1] * edge_second[..., 0]
    )
    if positive_orientation and cross.sum() < 0.0:
        cross *= -1.0
    rule = cross[:, None] * (1.0 - u)[None, :] * rule_weight[None, :]
    return points.reshape(-1, 2), rule.reshape(-1)


def _geometry_rings(geometry):
    """Return positively oriented exteriors and negatively oriented holes."""
    from shapely.geometry.polygon import orient

    if geometry.is_empty:
        return []
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [ring for part in geometry.geoms for ring in _geometry_rings(part)]
    if geometry.geom_type != "Polygon":
        return []
    polygon = orient(geometry, sign=1.0)
    return [
        np.asarray(polygon.exterior.coords)[:-1],
        *(np.asarray(ring.coords)[:-1] for ring in polygon.interiors),
    ]


def _padded_rule(rings_by_cell, centres, order, *, positive_orientation):
    """Pad independent signed ring rules to one batch without changing weights."""
    rules = []
    for rings in rings_by_cell:
        parts = [
            _ring_rule(ring, order, positive_orientation=positive_orientation)
            for ring in rings
        ]
        points = [part[0] for part in parts if len(part[0])]
        weights = [part[1] for part in parts if len(part[1])]
        rules.append(
            (
                np.concatenate(points) if points else np.empty((0, 2)),
                np.concatenate(weights) if weights else np.empty(0),
            )
        )
    width = max((len(weight) for _point, weight in rules), default=0)
    width = max(width, 1)
    padded_points = np.broadcast_to(
        np.asarray(centres)[:, None, :], (len(rules), width, 2)
    ).copy()
    padded_weights = np.zeros((len(rules), width))
    for index, (points, weights) in enumerate(rules):
        padded_points[index, : len(points)] = points
        padded_weights[index, : len(weights)] = weights
    return padded_points, padded_weights


def _profile_reference(
    field,
    profile,
    rings_by_cell,
    centres,
    cells,
    order,
    *,
    positive_orientation,
):
    """Numerically integrate the production profile on signed polygon rings."""
    import jax
    import jax.numpy as jnp

    field = jax.tree.map(jnp.asarray, field)
    points, weights = _padded_rule(
        rings_by_cell,
        centres,
        order,
        positive_orientation=positive_orientation,
    )

    @jax.jit
    def integrate(point, weight, cell, centre):
        psi_norm, _radial, _vertical = field.sample(point, cell)
        density = profile.current_density(point[..., 0], psi_norm)
        weighted = density * weight
        first = jnp.sum(weighted[..., None] * (point - centre[:, None, :]), axis=1)
        return jnp.column_stack((jnp.sum(weighted, axis=1), first[:, 0], first[:, 1]))

    return np.asarray(
        jax.device_get(
            integrate(
                jnp.asarray(points),
                jnp.asarray(weights),
                jnp.asarray(cells, dtype=jnp.int32),
                jnp.asarray(centres),
            )
        )
    )


def _analytic_reference(polygons, density, centres, order=16):
    return np.stack(
        [
            base.polygon_integral(polygon, density, centre, order=order)[:3]
            for polygon, centre in zip(polygons, centres, strict=True)
        ]
    )


def _instrument_controls():
    """Prove the reference distinguishes signed cancellation from lobe area."""
    bow = np.asarray(((1.0, 0.0), (3.0, 2.0), (1.0, 2.0), (3.0, 0.0)))
    region, faces = base.lobe_region(bow)
    _points, signed_weight = _ring_rule(bow, 16, positive_orientation=True)
    union_rules = [
        _ring_rule(ring, 16, positive_orientation=False)
        for ring in _geometry_rings(region)
    ]
    signed_area = float(signed_weight.sum())
    union_area = float(sum(weight.sum() for _point, weight in union_rules))
    np.testing.assert_allclose(signed_area, 0.0, atol=1e-13)
    np.testing.assert_allclose(union_area, 2.0, rtol=1e-13)
    assert sorted(face["winding"] for face in faces) == [-1, 1]
    return {
        "bow_tie_signed_winding_area_m2": signed_area,
        "bow_tie_nonzero_winding_union_area_m2": union_area,
        "winding_census": sorted(face["winding"] for face in faces),
    }


def _evaluate_path(operator, state):
    import jax
    import jax.numpy as jnp
    from nova.equilibrium.clip_quadrature import _quadratic_support
    from nova.equilibrium.source import _FluxSelectedProfile
    from nova.equilibrium.stencil_mesh import flux_field_polynomial

    profile = _FluxSelectedProfile(operator.source.core, operator.source.common_sol)

    @jax.jit
    def evaluate(value, active):
        masks, topology, samples, support = active._support_partition(value, None)
        moments = active.source.current_moments(
            active._moment_support_masks(masks, support),
            active.support_current_moments,
            support,
            sample_flux=samples,
        )
        field = flux_field_polynomial(
            active._support_moment_stencils, masks.psi_norm, samples
        )
        selected = field.active & (jnp.asarray(support.vertex_count) >= 3)
        coefficient = -jnp.asarray(field.coefficient)
        coefficient = coefficient.at[:, 0].add(1.0)
        effective = _quadratic_support(
            jnp.asarray(support.support_vertices),
            jnp.asarray(support.vertex_count),
            jnp.asarray(support.centroids),
            coefficient,
            field.centre,
            field.scale,
            selected,
        )
        return support, effective, moments, field, topology, selected

    support, effective, moments, field, topology, selected = jax.device_get(
        evaluate(jnp.asarray(state), operator)
    )
    return support, effective, np.asarray(moments).T, field, topology, selected, profile


def _stage_census(
    cells,
    incoming,
    effective,
    production,
    field,
    profile,
    density,
    centres,
):
    cells = np.asarray(cells, dtype=np.int32)
    incoming_vertices = np.asarray(incoming.support_vertices)[cells]
    incoming_count = np.asarray(incoming.vertex_count)[cells]
    effective_vertices = np.asarray(effective.support_vertices)[cells]
    effective_count = np.asarray(effective.vertex_count)[cells]
    selected_centres = centres[cells]
    incoming_polygons = [
        _polygon(vertices, count)
        for vertices, count in zip(incoming_vertices, incoming_count, strict=True)
    ]
    effective_chains = [
        _polygon(vertices, count)
        for vertices, count in zip(effective_vertices, effective_count, strict=True)
    ]
    effective_regions = []
    lobe_faces = []
    self_intersection = []
    for vertices, count, chain in zip(
        effective_vertices, effective_count, effective_chains, strict=True
    ):
        if chain.is_valid:
            effective_regions.append(chain)
            lobe_faces.append([])
            self_intersection.append(False)
        else:
            region, faces = base.lobe_region(np.asarray(vertices)[: int(count)])
            effective_regions.append(region)
            lobe_faces.append(faces)
            self_intersection.append(True)
    self_intersection = np.asarray(self_intersection, dtype=bool)
    analytic_incoming = _analytic_reference(
        incoming_polygons, density, selected_centres
    )
    analytic_effective = _analytic_reference(
        effective_regions, density, selected_centres
    )
    signed_rings = [
        [np.asarray(vertices)[: int(count)]]
        for vertices, count in zip(effective_vertices, effective_count, strict=True)
    ]
    union_rings = [_geometry_rings(region) for region in effective_regions]
    if profile.open_field_line is not None:
        raise ValueError("the stage attribution requires an undeclared open closure")
    confined_profile = profile.confined
    signed_profile_order_8 = _profile_reference(
        field,
        confined_profile,
        signed_rings,
        selected_centres,
        cells,
        8,
        positive_orientation=True,
    )
    signed_profile_order_16 = _profile_reference(
        field,
        confined_profile,
        signed_rings,
        selected_centres,
        cells,
        16,
        positive_orientation=True,
    )
    union_profile_order_8 = _profile_reference(
        field,
        confined_profile,
        union_rings,
        selected_centres,
        cells,
        8,
        positive_orientation=False,
    )
    union_profile_order_16 = _profile_reference(
        field,
        confined_profile,
        union_rings,
        selected_centres,
        cells,
        16,
        positive_orientation=False,
    )
    booked = production[cells]
    terms = {
        "secondary_geometry": analytic_effective - analytic_incoming,
        "density_evaluation": union_profile_order_16 - analytic_effective,
        "reclip_self_intersection_loss": (
            signed_profile_order_16 - union_profile_order_16
        ),
        "moment_reduction": booked - signed_profile_order_16,
    }
    closure = sum(terms.values())
    np.testing.assert_allclose(
        closure,
        booked - analytic_incoming,
        rtol=2e-12,
        atol=2e-8,
    )
    details = []
    for position, cell in enumerate(cells):
        left = incoming_vertices[position, : incoming_count[position]]
        right = effective_vertices[position, : effective_count[position]]
        chain_area = 0.5 * abs(
            np.sum(
                right[:, 0] * np.roll(right[:, 1], -1)
                - right[:, 1] * np.roll(right[:, 0], -1)
            )
        )
        details.append(
            {
                "cell": int(cell),
                "incoming_vertex_count": int(incoming_count[position]),
                "effective_vertex_count": int(effective_count[position]),
                "incoming_area_m2": incoming_polygons[position].area,
                "effective_chain_signed_area_m2": chain_area,
                "effective_lobe_union_area_m2": effective_regions[position].area,
                "effective_minus_incoming_area_m2": (
                    effective_regions[position].area - incoming_polygons[position].area
                ),
                "effective_chain_is_self_intersecting": bool(
                    self_intersection[position]
                ),
                "nonzero_winding_faces": lobe_faces[position],
                "symmetric_vertex_distance_m": _vertex_distance(left, right),
                "incoming_vertices_rz_m": left,
                "effective_vertices_rz_m": right,
                "booked_moments": booked[position],
                "analytic_incoming_moments": analytic_incoming[position],
                "analytic_lobe_union_moments": analytic_effective[position],
                "production_signed_winding_order_8_moments": (
                    signed_profile_order_8[position]
                ),
                "production_signed_winding_order_16_moments": (
                    signed_profile_order_16[position]
                ),
                "nonzero_winding_lobe_union_order_8_moments": (
                    union_profile_order_8[position]
                ),
                "nonzero_winding_lobe_union_order_16_moments": (
                    union_profile_order_16[position]
                ),
                "terms": {key: value[position] for key, value in terms.items()},
                "closure_moments": closure[position],
            }
        )
    return {
        "cells": details,
        "cell_count": len(cells),
        "booked_moments": booked.sum(axis=0),
        "analytic_incoming_moments": analytic_incoming.sum(axis=0),
        "analytic_lobe_union_moments": analytic_effective.sum(axis=0),
        "production_signed_winding_order_8_moments": signed_profile_order_8.sum(axis=0),
        "production_signed_winding_order_16_moments": signed_profile_order_16.sum(
            axis=0
        ),
        "nonzero_winding_lobe_union_order_8_moments": union_profile_order_8.sum(axis=0),
        "nonzero_winding_lobe_union_order_16_moments": union_profile_order_16.sum(
            axis=0
        ),
        "terms": {key: value.sum(axis=0) for key, value in terms.items()},
        "signed_winding_order_doubling_l1_current_a": float(
            np.abs(signed_profile_order_8[:, 0] - signed_profile_order_16[:, 0]).sum()
        ),
        "lobe_union_order_doubling_l1_current_a": float(
            np.abs(union_profile_order_8[:, 0] - union_profile_order_16[:, 0]).sum()
        ),
        "self_intersection_count": int(self_intersection.sum()),
        "self_intersection_cells": cells[self_intersection],
        "max_vertex_distance_m": max(
            (item["symmetric_vertex_distance_m"] for item in details), default=0.0
        ),
        "changed_vertex_set_count": sum(
            item["symmetric_vertex_distance_m"] > 1e-12 for item in details
        ),
    }


def measure(case_name, requested_cells, output):
    import jax
    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.forward_operator import set_support_clip_mode
    from scripts.analytic_oracle_fixtures import measure as fixture

    started = time.monotonic()
    label = f"{case_name}-cells-{requested_cells}"
    print(f"BUILD {label}", flush=True)
    base_row = json.loads((BASE_OUTPUT / f"{label}.json").read_text())
    base_exact = base_row["modes"]["exact"]
    carrier, source, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier, exact, -requested_cells)
    operator = fixture.forward_operator(source, machine)
    with np.load(
        ROOT
        / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
        / f"{label}-exact.npz"
    ) as bank:
        state = bank["analytic"]
    set_support_clip_mode("exact")
    jax.clear_caches()
    incoming, effective, production, field, topology, selected, profile = (
        _evaluate_path(operator, state)
    )
    selected = np.asarray(selected, dtype=bool)
    centres = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    base_cells = base_exact["cells"]
    simple_cut = np.asarray(
        [
            item["class"] == "separatrix-cut"
            and not item["non_simple"]
            and selected[item["cell"]]
            for item in base_cells
        ]
    )
    interior = np.asarray(
        [item["class"] == "interior" and selected[item["cell"]] for item in base_cells]
    )
    simple_result = _stage_census(
        np.flatnonzero(simple_cut),
        incoming,
        effective,
        production,
        field,
        profile,
        source.toroidal_current_density,
        centres,
    )
    interior_result = _stage_census(
        np.flatnonzero(interior),
        incoming,
        effective,
        production,
        field,
        profile,
        source.toroidal_current_density,
        centres,
    )
    base_error = base_exact["classes"]["separatrix-cut"][
        "simple_moment_integration_error_a"
    ]
    stage_total = sum(simple_result["terms"][name][0] for name in TERMS)
    closure_relative = abs(stage_total - base_error) / max(abs(base_error), 1e-30)
    if closure_relative > 1e-6:
        raise AssertionError(
            f"{label} closure {closure_relative} exceeds 1e-6: "
            f"stages={stage_total} base={base_error}"
        )
    dominant = max(TERMS, key=lambda name: abs(simple_result["terms"][name][0]))
    dominant_value = float(simple_result["terms"][dominant][0])
    dominant_share = abs(dominant_value) / max(abs(base_error), 1e-30)
    if dominant_share < 0.9:
        raise AssertionError(
            f"{label} no stage carries 90 percent: {dominant}={dominant_share}"
        )
    reduction_value = float(simple_result["terms"]["moment_reduction"][0])
    corrected_booked = base_exact["booked_current_a"] - reduction_value
    corrected_fraction = corrected_booked / base_exact["archived_target_current_a"]
    if case_name == "weak-rotation-reactor-static" and requested_cells == 110:
        np.testing.assert_allclose(
            reduction_value,
            -1009496.0723696492,
            rtol=1e-10,
            atol=1e-5,
        )
    if case_name == "weak-rotation-reactor-static" and corrected_fraction < 0.999:
        raise AssertionError(
            f"{label} reference substitution reaches only {corrected_fraction}"
        )
    interior_current = np.abs(
        np.asarray([item["booked_moments"][0] for item in interior_result["cells"]])
    )
    interior_ratios = {}
    for name in TERMS:
        values = np.abs(
            np.asarray([item["terms"][name][0] for item in interior_result["cells"]])
        )
        ratios = values / np.maximum(interior_current, np.finfo(float).tiny)
        interior_ratios[name] = float(ratios.max(initial=0.0))
        if interior_ratios[name] >= 1e-9:
            raise AssertionError(
                f"{label} interior {name} ratio {interior_ratios[name]} >= 1e-9"
            )
    row = {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(centres),
        "base_report": str(BASE_OUTPUT / f"{label}.json"),
        "base_report_sha256": hashlib.sha256(
            (BASE_OUTPUT / f"{label}.json").read_bytes()
        ).hexdigest(),
        "simple_separatrix_cut": simple_result,
        "interior_positive_control": {
            "cell_count": interior_result["cell_count"],
            "max_abs_stage_term_over_cell_current": interior_ratios,
            "all_below_1e_9": all(value < 1e-9 for value in interior_ratios.values()),
        },
        "base_simple_integration_error_a": base_error,
        "stage_sum_a": float(stage_total),
        "closure_relative": float(closure_relative),
        "dominant_stage": dominant,
        "dominant_stage_share": float(dominant_share),
        "negative_control": {
            "declaration": (
                "substitute the independent reference at the named stage and show "
                "the deficit removed"
            ),
            "substituted_stage": "moment_reduction",
            "substituted_term_a": reduction_value,
            "original_exact_support_fraction": base_exact[
                "support_fraction_of_archived_target"
            ],
            "corrected_booked_current_a": float(corrected_booked),
            "corrected_exact_support_fraction": float(corrected_fraction),
            "weak_rotation_reaches_0_999": bool(
                case_name != "weak-rotation-reactor-static"
                or corrected_fraction >= 0.999
            ),
        },
        "read_axis_rz_m": np.asarray(topology.axis),
        "seconds": time.monotonic() - started,
    }
    _write_json(output / f"{label}.json", row)
    print(
        f"ATTRIBUTED {label} stage={dominant} share={dominant_share:.9g} "
        f"closure={closure_relative:.3g} corrected_fraction={corrected_fraction:.12g}",
        flush=True,
    )
    return row


def render(rows, output):
    import matplotlib.pyplot as plt

    labels = [
        f"{'weak' if row['case'].startswith('weak') else 'diverted'} "
        f"{row['requested_cells']}"
        for row in rows
    ]
    values = {
        name: np.asarray(
            [
                row["simple_separatrix_cut"]["terms"][name][0]
                / row["base_simple_integration_error_a"]
                for row in rows
            ]
        )
        for name in TERMS
    }
    figure, axis = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    x = np.arange(len(rows))
    width = 0.24
    colours = ("steelblue", "indianred", "goldenrod", "seagreen")
    for offset, (name, colour) in enumerate(zip(TERMS, colours, strict=True)):
        axis.bar(
            x + (offset - (len(TERMS) - 1) / 2) * width,
            values[name],
            width,
            label=name,
            color=colour,
        )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xticks(x, labels)
    axis.set_ylabel("signed share of base simple-cut error")
    axis.set_title("Exact moment path attribution at the analytic state")
    axis.legend(
        frameon=False,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    axis.spines[["top", "right"]].set_visible(False)
    path = output / "stage-attribution.svg"
    figure.savefig(path)
    plt.close(figure)
    return path.name


def summarize(rows, output):
    complete = len(rows) == len(ROWS)
    figure = render(rows, output) if complete else None
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree": str(ROOT),
        "command": sys.argv,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "completed": complete,
        "rows_requested": ROWS,
        "sign_convention": (
            "booked minus incoming-polygon analytic = secondary_geometry + "
            "density_evaluation + reclip_self_intersection_loss + moment_reduction"
        ),
        "reference_rule": (
            "the production confined closure is integrated independently after "
            "the secondary reclip with Duffy order 16; order 8 is retained as "
            "the doubled-order convergence control"
        ),
        "instrument_controls": _instrument_controls(),
        "figure": figure,
        "rows": rows,
    }
    _write_json(output / "report.json", report)
    lines = [
        "# Exact cut-cell moment stage attribution",
        "",
        report["sign_convention"],
        "",
        "The incoming exact support is compared with the effective polygon after "
        "the production quadratic normalized-flux reclip. Density evaluation is "
        "then separated from the polynomial moment reduction on that same polygon.",
        "",
        "| Case | Cells | Simple cuts | Geometry A | Density A "
        "| Reclip self-intersection A "
        "| Moment reduction A | Named stage | Share | Closure relative "
        "| Corrected exact fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        terms = row["simple_separatrix_cut"]["terms"]
        lines.append(
            f"| {row['case']} | {row['requested_cells']} "
            f"| {row['simple_separatrix_cut']['cell_count']} "
            f"| {terms['secondary_geometry'][0]:.9g} "
            f"| {terms['density_evaluation'][0]:.9g} "
            f"| {terms['reclip_self_intersection_loss'][0]:.9g} "
            f"| {terms['moment_reduction'][0]:.9g} "
            f"| {row['dominant_stage']} | {row['dominant_stage_share']:.9f} "
            f"| {row['closure_relative']:.3g} "
            f"| {row['negative_control']['corrected_exact_support_fraction']:.9f} |"
        )
    if figure is not None:
        lines += [
            "",
            f"![Signed stage shares](/nova/figures/cut-cell-current-attribution/"
            f"exact-moment-stages/{figure})",
        ]
    lines += [
        "",
        "## Controls",
        "",
        "| Case | Cells | Self-intersections | Changed vertex sets "
        "| Max vertex distance m | Signed order 8 to 16 L1 A "
        "| Union order 8 to 16 L1 A | Interior max geometry/current "
        "| Interior max density/current | Interior max self-intersection/current "
        "| Interior max reduction/current |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cut = row["simple_separatrix_cut"]
        interior = row["interior_positive_control"][
            "max_abs_stage_term_over_cell_current"
        ]
        lines.append(
            f"| {row['case']} | {row['requested_cells']} "
            f"| {cut['self_intersection_count']} "
            f"| {cut['changed_vertex_set_count']} | {cut['max_vertex_distance_m']:.9g} "
            f"| {cut['signed_winding_order_doubling_l1_current_a']:.9g} "
            f"| {cut['lobe_union_order_doubling_l1_current_a']:.9g} "
            f"| {interior['secondary_geometry']:.3g} "
            f"| {interior['density_evaluation']:.3g} "
            f"| {interior['reclip_self_intersection_loss']:.3g} "
            f"| {interior['moment_reduction']:.3g} |"
        )
    lines += [
        "",
        "The declared negative control replaces the moment-reduction term with its "
        "signed-winding independent reference. Both weak-rotation rows must then "
        "reach an exact "
        "support fraction of at least 0.999. Every cell's incoming and effective "
        "vertices, area, three physical moments, stage terms, and closure are "
        "retained in report.json. No solve and no production source change ran.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(f"revision={revision} tree={ROOT} command={sys.argv!r}", flush=True)
    assert jax.default_backend() == "gpu", jax.devices()
    print(f"devices={jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)
    rows = []
    for case_name, requested_cells in ROWS:
        rows.append(measure(case_name, requested_cells, args.output))
        summarize(rows, args.output)
    report = summarize(rows, args.output)
    assert report["completed"]
    assert all(row["closure_relative"] <= 1e-6 for row in rows)
    assert all(row["dominant_stage_share"] >= 0.9 for row in rows)
    assert all(row["interior_positive_control"]["all_below_1e_9"] for row in rows)
    assert all(row["negative_control"]["weak_rotation_reaches_0_999"] for row in rows)
    print(
        "ATTRIBUTION_COMPLETE: all stage closures, controls and substitutions pass",
        flush=True,
    )


if __name__ == "__main__":
    main()
