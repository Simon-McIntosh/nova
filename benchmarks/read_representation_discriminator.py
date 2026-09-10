"""Separate state error from boundary-representation error in the cut-cell
current shortfall.

The production profile support clips every atomic cell against the signed
flux of the current iterate (``traced_clip(inside_boundary)``), locating the
plasma boundary as the zero crossing of a *linear* interpolation of the nodal
flux along each cell edge.  The exact support on the held spline-chain stack
locates the same boundary through an order-6 global split-spline level set,
and the six-point ring quadratic is the third candidate (the cell-local
``support_flux_coefficients`` polynomial).  Each locates a slightly different
boundary, and the unit-amplitude current a clip books depends on which is
used.

This driver measures that dependency on the committed certificate rows ---
the weak, moderate and strong Solov'ev rows at 110 cells and the single-null
Cerfon-Freidberg row at 300 cells.  For each row it evaluates two flux states
on the mesh nodes: the analytic flux sampled exactly at the nodes (the exact
input) and the committed terminal state (persisted ``render_data``).  For
each state it locates the boundary level set under all three representations
with the same chord clip the production support uses (``traced_clip`` against
``inside_boundary``, geometric candidacy), and writes per (row, state,
representation): the unit-amplitude current total over the analytic total,
the participating analytic cut cells, the six worst per-cell
booked-minus-analytic currents, the median and maximum boundary position
error of the edge roots against the analytic separatrix (metres and cell
pitches), and the count of cells where the representation's level disagrees
in sign with the nodal flux.

The decisive comparison is the exact-input state: a representation whose
unit-amplitude ratio is within 0.1 percent of unity there carries no
representation error, so the shortfall measured on the terminal state is
state error; a representation that falls short on the exact input is at fault
by that amount.  The report answers per row which of the global spline, the
ring quadratic or the linear edge root is at fault and by how much, and
states the state-error share of the terminal shortfall.

One line-contour panel per row on the exact-input state shows the analytic
separatrix beside the three representations' edge roots, with the cells where
they differ outlined, the analytic nulls and the wall drawn.

Receipt and panels land under
``docs/figures/cut-cell-current-attribution/read-representation``; the report
is mirrored to the crew reports path.  CPU-only x64: run under the shared
environment interpreter with ``JAX_PLATFORMS=cpu`` on one ``*_debug``
allocation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import numpy as np

from nova.jax.config import configure_dtypes

#: The persisted certificate parts (read-only reference data in main).
PARTS = Path(
    "/home/ITER/mcintos/Code/nova/docs/figures/gs-absolute-accuracy/solovev/"
    "production-route-parts"
)
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/cut-cell-current-attribution/read-representation"
)
REPORTS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/read-representation"
)
EXIT_MARKER = "READ_REPRESENTATION_EXIT=0"

ROWS = (
    (
        "weak",
        "weak-rotation-reactor-static",
        "weak-rotation-reactor-static-production-route-reduced.json",
        -110,
    ),
    (
        "moderate",
        "moderate-rotation-conventional-static",
        "moderate-rotation-conventional-static-production-route-reduced.json",
        -110,
    ),
    (
        "strong",
        "strong-rotation-compact-static",
        "strong-rotation-compact-static-production-route-reduced.json",
        -110,
    ),
    (
        "single-null",
        "diverted-single-null",
        "diverted-single-null-production-route-cells-300.json",
        -300,
    ),
)
REPRESENTATIONS = ("linear", "quadratic", "spline")
STATES = ("exact", "terminal")

#: A representation whose exact-input unit-amplitude shortfall stays within
#: this band of unity carries no representation error.
REPRESENTATION_FREE_BAND = 1.0e-3
#: Cells whose representations disagree by more than this fraction of the
#: characteristic pitch are outlined on the exact-input panel.
OUTLINE_TOLERANCE_PITCHES = 0.01

TOTAL_FLUX_FACTOR = 2.0 * np.pi


def _row_data(row: tuple) -> dict:
    """Return the immutable per-row part data."""
    _label, case_name, part_file, requested_cells = row
    part = json.loads((PARTS / part_file).read_text())
    render = part["render_data"]
    return {
        "row": row,
        "label": _label,
        "case": case_name,
        "requested_cells": requested_cells,
        "part_file": part_file,
        "characteristic_pitch_m": part["characteristic_pitch_m"],
        "realised_cells": part["realised_cells"],
        "analytic_flux_wb": np.asarray(render["analytic_flux_wb"], dtype=np.float64),
        "terminal_flux_wb": np.asarray(render["terminal_flux_wb"], dtype=np.float64),
        "coordinates_rz_m": np.asarray(render["coordinates_rz_m"], dtype=np.float64),
        "wall_rz_m": np.asarray(render["wall_units_rz_m"][0], dtype=np.float64),
        "boundary_rz_m": np.asarray(render["boundary_rz_m"], dtype=np.float64),
        "analytic_topology": render["analytic_topology"],
        "terminal_topology": render["terminal_topology"],
        "solver_qualification": part["solver"].get("qualification"),
    }


def _machine(part: dict):
    """Build the same carrier, source and operator the certificate row used."""
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture
    from benchmarks.solovev_certificate import _case, _case_machine

    carrier, source, exact = _case(part["case"])
    machine = _case_machine(part["case"], carrier, exact, part["requested_cells"])
    rebuilt = np.vstack((machine.node, machine.wall_node, machine.sample_coordinates))
    if rebuilt.shape != part["coordinates_rz_m"].shape or not np.allclose(
        rebuilt, part["coordinates_rz_m"]
    ):
        raise RuntimeError(
            f"row {part['label']}: machine coordinates do not reproduce the "
            "persisted render coordinates; the machine cache identity drifted"
        )
    operator = oracle_fixture.forward_operator(source, machine)
    return source, exact, machine, operator


def _analytic_region_polygon(part: dict, source, exact):
    """Return the analytic plasma polygon at the true separatrix level."""
    from shapely.geometry import Polygon
    from benchmarks.solovev_certificate import _is_diverted_case

    if _is_diverted_case(part["case"]):
        return Polygon(exact.separatrix(721))
    radius, half_height, _weight, _offset = source._surface_nodes(0.0, 240)
    upper = np.c_[radius, half_height]
    lower = np.c_[radius[::-1], -half_height[::-1]]
    return Polygon(np.vstack((upper, lower)))


def _cell_analytic_currents(part: dict, machine, source, exact):
    """Return per-cell analytic current and the analytic cut-cell mask."""
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    from scripts.analytic_oracle_fixtures.measure import _polygon_rule

    region = _analytic_region_polygon(part, source, exact)
    atomic = machine.moment_geometry.atomic_mesh
    nodes = np.asarray(atomic.node_coordinates)
    cells = np.asarray(atomic.cell_nodes)
    counts = np.asarray(atomic.cell_vertex_count)

    def density(points):
        return np.asarray(source.toroidal_current_density(points[:, 0], points[:, 1]))

    def parts_of(geometry):
        if isinstance(geometry, Polygon):
            return [geometry] if geometry.area > 0.0 else []
        if isinstance(geometry, MultiPolygon):
            pieces: list = []
            for piece in geometry.geoms:
                pieces.extend(parts_of(piece))
            return pieces
        if isinstance(geometry, GeometryCollection):
            pieces = []
            for piece in geometry.geoms:
                pieces.extend(parts_of(piece))
            return pieces
        return []

    current = np.zeros(len(cells))
    for index in range(len(cells)):
        polygon = Polygon(nodes[cells[index][: counts[index]]])
        intersection = polygon.intersection(region)
        if intersection.is_empty or intersection.area <= 0.0:
            continue
        for piece in parts_of(intersection):
            points, weights = _polygon_rule(
                np.asarray(piece.exterior.coords, dtype=np.float64)
            )
            current[index] += float(np.sum(weights * density(points)))
    vertex_flux = np.asarray(part["analytic_flux_wb"])[cells]
    slot = np.arange(vertex_flux.shape[1])[None, :]
    valid = slot < counts[:, None]
    masked = np.where(valid, vertex_flux, np.nan)
    cut = (np.nanmin(masked, axis=1) < 0.0) & (np.nanmax(masked, axis=1) > 0.0)
    return current, cut


def _flux_level_read(part: dict, operator, state, *, analytic_level: bool):
    """Return the topology read and boundary-level fields for one state."""
    import jax.numpy as jnp

    physical = jnp.asarray(state)[: operator.physical_node_number]
    masks, topology, _connected, _admitted = operator._fixed_design_read(physical)
    axis = float(topology.axis_flux)
    if analytic_level:
        # Clip the exact input at the true analytic separatrix level so the
        # representation test is not contaminated by the discrete read's own
        # boundary-level error on an exact field.
        boundary_level = 0.0
    else:
        # The terminal state clips at the read boundary level, exactly as
        # production does on the committed state.
        boundary_level = float(topology.boundary_flux)
    span = boundary_level - axis
    if span == 0.0:
        raise RuntimeError(f"row {part['label']}: degenerate flux span")
    grid_flux, _wall_flux = operator.topology.split_flux_map(physical)
    shared_flux = operator.shared_node_flux(physical)
    inside_boundary = operator.polarity * (shared_flux - boundary_level)
    psi_grid = (grid_flux - axis) / span
    sample_psi = (operator.sample_node_flux(jnp.asarray(state)) - axis) / span
    return (
        masks,
        topology,
        inside_boundary,
        psi_grid,
        sample_psi,
        boundary_level,
    )


def _representation_tools(
    operator, atomic_mesh, masks, inside_boundary, psi_grid, sample_psi
):
    """Return the boundary tools all representations share for one state.

    The participation mask follows the production geometric candidacy rule
    (profile-owned plus any cell with a vertex on the confined side of the
    boundary level); the ring quadratic is ``1 - psi_norm`` per cell and the
    order-6 split spline is fit through the grid nodes' ``psi_norm``.
    """
    import jax.numpy as jnp
    from nova.linalg.split_spline import fit_split_spline

    participation = masks.profile_participation | operator._chord_vertex_participation(
        atomic_mesh, inside_boundary
    )
    flux_coefficient = operator.support_flux_coefficients(psi_grid, sample_psi)
    inside_coefficient = (-flux_coefficient).at[:, 0].add(1.0)
    centre = np.asarray(operator._support_curve_centre)
    scale = np.asarray(operator._support_curve_scale)
    coordinate = jnp.asarray(operator.grid.coordinate, dtype=psi_grid.dtype)
    surface = fit_split_spline(
        coordinate[None, :, 0],
        coordinate[None, :, 1],
        psi_grid[None, :],
        psi_grid[None, :] - 1.0,
        order=6,
        regularization=1.0e-14,
    )
    fit_executed = bool(np.asarray(surface.fit_executed))

    def curved_level(points):
        return -surface._patch_evaluation(
            surface.level_set_coefficients, points[..., 0], points[..., 1]
        ).value

    return (
        participation,
        inside_coefficient,
        centre,
        scale,
        curved_level,
        fit_executed,
    )


def _clip_support(atomic_mesh, inside_boundary, tools, representation):
    """Clip every cell under one boundary representation."""
    participation, inside_coefficient, centre, scale, curved_level, _fit = tools
    if representation == "linear":
        return atomic_mesh.traced_clip(inside_boundary).qualify(participation)
    if representation == "quadratic":
        return atomic_mesh.traced_clip(
            inside_boundary,
            curve_coefficient=inside_coefficient,
            curve_centre=centre,
            curve_scale=scale,
            participating_cell=participation,
        ).qualify(participation)
    return atomic_mesh.traced_clip(
        inside_boundary,
        curve_evaluator=curved_level,
        participating_cell=participation,
    ).qualify(participation)


def _booked_current(operator, masks, support, sample_psi):
    """Return the unit-amplitude per-cell current of one clipped support."""
    moment_masks = operator._moment_support_masks(masks, support)
    moments = operator.source.current_moments(
        moment_masks,
        operator.support_current_moments,
        support,
        sample_flux=sample_psi,
    )
    return np.asarray(moments.cell_current, dtype=np.float64)


def _sign_disagreement(
    part, operator, atomic_mesh, masks, inside_boundary, tools, representation
):
    """Count cells where the representation's level disagrees with the nodal flux."""
    import jax.numpy as jnp
    from nova.equilibrium.separatrix_clip import _traced_quadratic_value

    _participation, inside_coefficient, centre, scale, curved_level, _fit = tools
    cells = np.asarray(atomic_mesh.cell_nodes)
    counts = np.asarray(atomic_mesh.cell_vertex_count)
    nodes = np.asarray(atomic_mesh.node_coordinates)
    nodal = np.asarray(inside_boundary)
    disagreements = 0
    for index in range(len(cells)):
        row = cells[index][: counts[index]]
        points = nodes[row]
        if representation == "linear":
            level = nodal[row]
        elif representation == "quadratic":
            level = np.asarray(
                _traced_quadratic_value(
                    jnp.asarray(points)[None],
                    inside_coefficient[index : index + 1],
                    centre[index : index + 1],
                    scale[index : index + 1],
                )
            )[0]
        else:
            level = np.asarray(curved_level(jnp.asarray(points)))
        if np.any(np.sign(level) * np.sign(nodal[row]) < 0):
            disagreements += 1
    return disagreements


def _edge_root_measure(
    part, source, exact, operator, atomic_mesh, masks, inside_boundary, tools
):
    """Return per-representation edge-root boundary-error statistics.

    Every atomic edge the representation carries across its level is rooted
    with the same fraction the traced clip uses (linear, quadratic segment
    root, or 48-iteration level bisection) and the normal distance to the
    analytic separatrix ``|f| / ||grad f||`` is measured.  Per-cell root
    tables are returned so the exact-input panel can outline cells where the
    representations disagree.
    """
    import jax.numpy as jnp
    from nova.equilibrium.separatrix_clip import (
        _traced_quadratic_value,
        _traced_quadratic_segment_root,
        _traced_level_segment_root,
    )
    from benchmarks.solovev_certificate import _is_diverted_case

    participation, inside_coefficient, centre, scale, curved_level, _fit = tools
    if _is_diverted_case(part["case"]):

        def f_analytic(points):
            return np.asarray(exact.flux(points), dtype=np.float64)

        def g_analytic(points):
            return np.asarray(exact.gradient(points), dtype=np.float64)

    else:

        def f_analytic(points):
            return TOTAL_FLUX_FACTOR * np.asarray(
                source.flux(points[:, 0], points[:, 1])
            )

        def g_analytic(points):
            radial, vertical = np.asarray(
                source.flux_gradient(points[:, 0], points[:, 1])
            )
            return TOTAL_FLUX_FACTOR * np.column_stack((radial, vertical))

    node_coords = np.asarray(atomic_mesh.node_coordinates)
    cell_nodes = np.asarray(atomic_mesh.cell_nodes)
    cell_counts = np.asarray(atomic_mesh.cell_vertex_count)
    participation = np.asarray(participation)

    def root_point(representation):
        points_by_cell: dict = {}
        for cell in np.flatnonzero(participation):
            row = cell_nodes[cell][: cell_counts[cell]]
            start_idx = row
            end_idx = np.roll(row, -1)
            start_pt = node_coords[start_idx]
            end_pt = node_coords[end_idx]
            if representation == "linear":
                start_value = np.asarray(inside_boundary)[start_idx]
                end_value = np.asarray(inside_boundary)[end_idx]
            elif representation == "quadratic":
                start_value = np.asarray(
                    _traced_quadratic_value(
                        jnp.asarray(start_pt),
                        inside_coefficient[cell : cell + 1],
                        centre[cell : cell + 1],
                        scale[cell : cell + 1],
                    )
                )[0]
                end_value = np.asarray(
                    _traced_quadratic_value(
                        jnp.asarray(end_pt),
                        inside_coefficient[cell : cell + 1],
                        centre[cell : cell + 1],
                        scale[cell : cell + 1],
                    )
                )[0]
            else:
                start_value = np.asarray(curved_level(jnp.asarray(start_pt)))
                end_value = np.asarray(curved_level(jnp.asarray(end_pt)))
            crossing = (start_value > 0.0) != (end_value > 0.0)
            for edge in np.flatnonzero(crossing):
                if representation == "linear":
                    fraction = start_value[edge] / (start_value[edge] - end_value[edge])
                elif representation == "quadratic":
                    fraction = float(
                        np.asarray(
                            _traced_quadratic_segment_root(
                                jnp.asarray(start_pt[edge])[None],
                                jnp.asarray(end_pt[edge])[None],
                                jnp.asarray(start_value[edge])[None],
                                jnp.asarray(end_value[edge])[None],
                                inside_coefficient[cell : cell + 1],
                                centre[cell : cell + 1],
                                scale[cell : cell + 1],
                            )
                        )[0].item()
                    )
                else:
                    fraction = float(
                        np.asarray(
                            _traced_level_segment_root(
                                jnp.asarray(start_pt[edge])[None],
                                jnp.asarray(end_pt[edge])[None],
                                jnp.asarray(start_value[edge])[None],
                                jnp.asarray(end_value[edge])[None],
                                lambda points: curved_level(points),
                            )
                        )[0].item()
                    )
                points_by_cell.setdefault(cell, []).append(
                    start_pt[edge] + fraction * (end_pt[edge] - start_pt[edge])
                )
        return points_by_cell

    per_cell = {
        representation: root_point(representation) for representation in REPRESENTATIONS
    }
    results = {}
    for representation in REPRESENTATIONS:
        roots = (
            np.asarray(per_cell[representation][cell])
            for cell in per_cell[representation]
        )
        roots = (
            np.concatenate(list(roots))
            if per_cell[representation]
            else np.empty((0, 2))
        )
        if roots.shape[0] == 0:
            results[representation] = {"count": 0, "median_m": 0.0, "max_m": 0.0}
            continue
        magnitude = np.abs(f_analytic(roots))
        gradient = np.linalg.norm(g_analytic(roots), axis=1)
        distance = np.where(
            gradient > 0.0, magnitude / np.maximum(gradient, 1.0e-30), np.nan
        )
        distance = distance[np.isfinite(distance)]
        results[representation] = {
            "count": int(distance.size),
            "median_m": float(np.median(distance)) if distance.size else 0.0,
            "max_m": float(np.max(distance)) if distance.size else 0.0,
        }
    return per_cell, results


def _measure_state(
    part,
    source,
    exact,
    operator,
    atomic_mesh,
    state,
    analytic_cut,
    analytic_current,
    analytic_total,
    *,
    state_name,
):
    """Measure all three representations on one flux state."""

    masks, topology, inside_boundary, psi_grid, sample_psi, boundary_level = (
        _flux_level_read(part, operator, state, analytic_level=state_name == "exact")
    )
    tools = _representation_tools(
        operator, atomic_mesh, masks, inside_boundary, psi_grid, sample_psi
    )
    per_cell, root_stats = _edge_root_measure(
        part, source, exact, operator, atomic_mesh, masks, inside_boundary, tools
    )
    pitch = part["characteristic_pitch_m"]
    rows = {}
    for representation in REPRESENTATIONS:
        support, fit_executed = (
            _clip_support(atomic_mesh, inside_boundary, tools, representation),
            tools[5],
        )
        booked = _booked_current(operator, masks, support, sample_psi)
        included = np.asarray(support.included, dtype=bool)
        participating = analytic_cut & (booked != 0.0) & included
        total = float(np.sum(booked))
        difference = booked - analytic_current
        worst_order = np.argsort(np.abs(difference))[::-1][:6]
        stats = root_stats[representation]
        rows[representation] = {
            "unit_amplitude_total_a": total,
            "ratio_of_analytic_total": total / analytic_total,
            "analytic_cut_cells_participating": int(np.sum(participating)),
            "booked_cells": int(np.sum(included)),
            "worst_cells": [
                {
                    "cell": int(index),
                    "booked_a": float(booked[index]),
                    "analytic_a": float(analytic_current[index]),
                    "difference_a": float(difference[index]),
                }
                for index in worst_order
            ],
            "boundary_roots": stats["count"],
            "boundary_error_median_m": stats["median_m"],
            "boundary_error_max_m": stats["max_m"],
            "boundary_error_median_pitches": stats["median_m"] / pitch,
            "boundary_error_max_pitches": stats["max_m"] / pitch,
            "sign_disagreement_cells": _sign_disagreement(
                part,
                operator,
                atomic_mesh,
                masks,
                inside_boundary,
                tools,
                representation,
            ),
            "spline_fit_executed": fit_executed,
        }
    return {
        "state": state_name,
        "boundary_level_wb": boundary_level,
        "boundary_level_error_vs_analytic_wb": boundary_level,
        "representations": rows,
        "per_cell_roots": {repr_name: cells for repr_name, cells in per_cell.items()},
    }


def _measure_row(row: tuple) -> dict:
    """Measure every (state, representation) combination for one row."""
    part = _row_data(row)
    source, exact, machine, operator = _machine(part)
    atomic_mesh = operator.moment_geometry.atomic_mesh
    analytic_current, analytic_cut = _cell_analytic_currents(
        part, machine, source, exact
    )
    analytic_total = float(np.sum(analytic_current))
    closed_form = source.plasma_current()
    states = []
    for state_name, state in (
        ("exact", part["analytic_flux_wb"]),
        ("terminal", part["terminal_flux_wb"]),
    ):
        states.append(
            _measure_state(
                part,
                source,
                exact,
                operator,
                atomic_mesh,
                state,
                analytic_cut,
                analytic_current,
                analytic_total,
                state_name=state_name,
            )
        )
    read_error = abs(float(part["analytic_topology"]["boundary_flux_wb"]))
    displacement = _boundary_level_displacement(part, source, exact, read_error)
    return {
        "row": part["label"],
        "case": part["case"],
        "requested_cells": part["requested_cells"],
        "realised_cells": part["realised_cells"],
        "characteristic_pitch_m": part["characteristic_pitch_m"],
        "solver_qualification": part["solver_qualification"],
        "analytic_total_a": analytic_total,
        "analytic_total_closed_form_a": (
            closed_form if np.isfinite(closed_form) else None
        ),
        "analytic_cut_cell_count": int(np.sum(analytic_cut)),
        "exact_read_boundary_level_wb": read_error,
        "exact_read_boundary_share_of_span": (
            read_error / abs(float(part["analytic_topology"]["flux_span_wb"]))
            if abs(float(part["analytic_topology"]["flux_span_wb"])) > 0.0
            else None
        ),
        "exact_read_boundary_displacement_m": displacement,
        "states": {state["state"]: state for state in states},
        "verdict": _verdict_row(
            part["label"],
            states,
            analytic_total,
            read_error,
            displacement,
            part["analytic_topology"],
        ),
    }


def _boundary_level_displacement(part: dict, source, exact, dpsi_wb: float) -> float:
    """Return the median normal displacement (m) of a flux-level error dpsi_wb.

    Uses the median gradient magnitude on the analytic separatrix, so a read
    whose boundary level is ``dpsi_wb`` off the analytic separatrix displaces
    the clip boundary by roughly this distance.
    """
    from benchmarks.solovev_certificate import _is_diverted_case

    if _is_diverted_case(part["case"]):
        gradient = np.asarray(exact.gradient(exact.separatrix(721)), dtype=np.float64)
    else:
        radius, half_height, _weight, _offset = source._surface_nodes(0.0, 721)
        radial, vertical = np.asarray(
            source.flux_gradient(radius, half_height), dtype=np.float64
        )
        gradient = TOTAL_FLUX_FACTOR * np.column_stack((radial, vertical))
    magnitude = np.linalg.norm(gradient, axis=1)
    finite = magnitude[np.isfinite(magnitude) & (magnitude > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(dpsi_wb / np.median(finite))


def _verdict_row(
    row_label: str,
    states: list,
    analytic_total: float,
    read_boundary_wb: float,
    read_displacement_m: float,
    analytic_topology: dict,
) -> dict:
    """Return the per-row attribution sentence and numbers."""
    del analytic_total
    exact = states[0]["representations"]
    terminal = states[1]["representations"]
    exact_shortfall = {
        name: 1.0 - exact[name]["ratio_of_analytic_total"] for name in REPRESENTATIONS
    }
    free = {
        name: abs(exact_shortfall[name]) <= REPRESENTATION_FREE_BAND
        for name in REPRESENTATIONS
    }
    at_fault = [
        name
        for name in REPRESENTATIONS
        if abs(exact_shortfall[name]) > REPRESENTATION_FREE_BAND
    ]
    # The planned carrier is the global spline; fall back on any free
    # representation, then on the least-error one.
    if free["spline"]:
        best = "spline"
    elif free["quadratic"]:
        best = "quadratic"
    elif free["linear"]:
        best = "linear"
    else:
        best = min(REPRESENTATIONS, key=lambda name: abs(exact_shortfall[name]))
    terminal_shortfall = 1.0 - terminal[best]["ratio_of_analytic_total"]
    state_error = terminal_shortfall - exact_shortfall[best]
    if terminal_shortfall != 0.0:
        state_error_share = state_error / terminal_shortfall
    else:
        state_error_share = 0.0 if abs(state_error) < 1.0e-12 else float("nan")

    if not free["spline"] and not free["quadratic"] and not free["linear"]:
        # No representation is within the band; the exact-input residuals are
        # all small.  If the terminal shortfall is identical to the exact
        # state at the read level, it is a boundary-level read error.
        span = float(analytic_topology.get("flux_span_wb") or 0.0)
        sentence = (
            f"{row_label}: no representation is within the 0.1 percent band "
            f"on the exact input "
            f"(linear {exact_shortfall['linear']:+.4f}, quadratic "
            f"{exact_shortfall['quadratic']:+.4f}, spline "
            f"{exact_shortfall['spline']:+.4f}); "
        )
        if abs(read_boundary_wb) > 1.0e-6 and abs(span) > 0.0:
            sentence += (
                f"the terminal shortfall of {terminal_shortfall:.4f} is the "
                f"read boundary level, which sits {read_boundary_wb:.4f} Wb "
                f"({read_boundary_wb / abs(span) * 100.0:.1f} percent of the "
                f"flux span, about {read_displacement_m * 1000.0:.1f} mm "
                "inward) on both the exact and the terminal state, so it is "
                "a read error under every representation, not a "
                "representation or terminal-field error."
            )
        else:
            sentence += (
                f"the terminal shortfall of {terminal_shortfall:.4f} is "
                "therefore not separated from representation error on this "
                "row."
            )
    elif any(free.values()) and not all(free.values()):
        fault_names = [name for name in at_fault if name != best]
        worst_fault = max(abs(exact_shortfall[name]) for name in fault_names)
        sentence = (
            f"{row_label}: the {best} boundary carries no representation "
            f"error (exact-input shortfall {exact_shortfall[best]:+.4f}) while "
            f"{', '.join(fault_names)} fall"
            f"{'s' if len(fault_names) == 1 else ''} "
            f"short by {worst_fault:.4f} "
            "even on the exact input; "
        )
        share = max(0.0, min(1.0, state_error_share))
        if terminal_shortfall != 0.0:
            sentence += (
                f"the terminal shortfall of {terminal_shortfall:.4f} is "
                f"{(share * 100.0):.1f} percent state error."
            )
        else:
            sentence += "the terminal shortfall is zero."
    else:
        sentence = (
            f"{row_label}: every representation is within the 0.1 percent "
            f"band on the exact input (linear {exact_shortfall['linear']:+.4f}, "
            f"quadratic {exact_shortfall['quadratic']:+.4f}, spline "
            f"{exact_shortfall['spline']:+.4f}); "
        )
        share = max(0.0, min(1.0, state_error_share))
        if terminal_shortfall != 0.0:
            sentence += (
                f"the terminal shortfall of {terminal_shortfall:.4f} is "
                f"{(share * 100.0):.1f} percent state error."
            )
        else:
            sentence += "the terminal shortfall is zero."
    return {
        "sentence": sentence,
        "exact_shortfall": exact_shortfall,
        "free": free,
        "best": best,
        "at_fault": at_fault,
        "terminal_shortfall_best": terminal_shortfall,
        "state_error": state_error,
        "state_error_share_of_terminal_shortfall": state_error_share,
        "representation_fault": at_fault,
    }


def _disagreement_cells(per_cell_roots, pitch: float):
    """Return cells where any two representations' edge roots differ > tol."""
    from scipy.spatial.distance import cdist

    tolerance = OUTLINE_TOLERANCE_PITCHES * pitch
    cells = set()
    for roots in per_cell_roots.values():
        cells.update(roots)
    disagreements = []
    for cell in cells:
        sets = [
            np.asarray(per_cell_roots[name].get(cell, np.empty((0, 2))))
            for name in REPRESENTATIONS
        ]
        nonempty = [points for points in sets if points.shape[0] > 0]
        if len(nonempty) != len(sets):
            disagreements.append(cell)
            continue
        separation = 0.0
        for first in range(len(REPRESENTATIONS)):
            for second in range(first + 1, len(REPRESENTATIONS)):
                if sets[first].shape[0] == 0 or sets[second].shape[0] == 0:
                    continue
                separation = max(
                    separation,
                    float(np.max(cdist(sets[first], sets[second]).min(axis=1))),
                )
        if separation > tolerance:
            disagreements.append(cell)
    return disagreements


def _render_panel(part, machine, operator, measured_exact, exact_state, target: Path):
    """Draw one exact-input line-contour panel per row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from nova.media.ink import poloidal_axes
    from nova.media.poloidal import draw_nulls, draw_wall

    atomic = operator.moment_geometry.atomic_mesh
    cells = np.asarray(atomic.cell_nodes)
    counts = np.asarray(atomic.cell_vertex_count)
    pitch = part["characteristic_pitch_m"]
    root_tables = measured_exact["per_cell_roots"]
    outline = _disagreement_cells(root_tables, pitch)

    node_coordinates = np.asarray(atomic.node_coordinates)
    figure, axes = plt.subplots(figsize=(9, 9))
    poloidal_axes(axes)
    axes.tricontour(
        node_coordinates[:, 0],
        node_coordinates[:, 1],
        np.asarray(exact_state),
        levels=np.linspace(
            float(np.nanmin(exact_state)), float(np.nanmax(exact_state)), 12
        ),
        colors="#16537e",
        linewidths=0.5,
        zorder=2,
    )
    # the three representations' edge roots on this state
    marker = {"linear": "+", "quadratic": "x", "spline": "."}
    colour = {"linear": "#1b9e77", "quadratic": "#d95f02", "spline": "#7570b3"}
    for name in REPRESENTATIONS:
        roots = (
            np.concatenate(
                [np.asarray(points) for points in root_tables[name].values()]
            )
            if root_tables[name]
            else np.empty((0, 2))
        )
        if roots.shape[0] == 0:
            continue
        axes.plot(
            roots[:, 0],
            roots[:, 1],
            marker[name] if name != "spline" else ",",
            linestyle="none",
            color=colour[name],
            markersize=3.0 if name != "spline" else 2.0,
            alpha=0.9,
            zorder=6,
        )
    if len(outline):
        polygons = [cells[index][: counts[index]] for index in outline]
        axes.add_collection(
            PolyCollection(
                [np.asarray(atomic.node_coordinates)[row] for row in polygons],
                facecolors="none",
                edgecolors="#c9a227",
                linewidths=1.3,
                zorder=5,
            )
        )
    boundary = part["boundary_rz_m"]
    axes.plot(
        boundary[:, 0],
        boundary[:, 1],
        color="#222222",
        linewidth=1.6,
        linestyle="solid",
        zorder=7,
    )
    draw_wall(axes, part["wall_rz_m"][:, 0], part["wall_rz_m"][:, 1])
    axis_rz = np.asarray(part["analytic_topology"]["axis_rz_m"])
    x_rz = part["analytic_topology"]["x_point_rz_m"]
    draw_nulls(
        axes,
        magnetic_axis=axis_rz,
        x_points=np.asarray(x_rz) if x_rz is not None else None,
    )
    axes.text(
        0.99,
        0.01,
        (
            f"{part['label']} exact-input: analytic separatrix (black) with "
            f"{', '.join(REPRESENTATIONS)} edge roots; "
            f"ochre = cells where the representations differ by > "
            f"{OUTLINE_TOLERANCE_PITCHES * 100.0:.1f}% pitch"
        ),
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(rows: list, source_revision: str, lane: dict) -> str:
    lines = [
        "# Boundary-representation discriminator: where the cut-cell "
        "shortfall comes from",
        "",
        "Measured on the committed Solov'ev -110 rows and the single-null -300 "
        "row.  Each row is evaluated on the analytic flux sampled exactly at "
        "the nodes (exact input) and on the committed terminal state; each "
        "state's boundary level is located under the linear edge root, the "
        "ring quadratic and the order-6 global split spline, clipped with "
        "``traced_clip`` and the production geometric candidacy.",
        "",
        f"- source revision `{source_revision}`; lane `{lane['node']}` "
        f"({lane['jax_platforms']}).",
        "- A representation whose exact-input ratio is within 0.1 percent of "
        "unity carries no representation error; the terminal shortfall under "
        "it is then state error.",
        "",
        "## Per-row verdicts",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['verdict']['sentence']}")
    lines.append("")
    lines.append(
        "| row | analytic total [A] | cut cells | exact: linear | exact: "
        "quadratic | exact: spline | terminal: linear | terminal: quadratic | "
        "terminal: spline | terminal best | state error |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        exact, terminal = row["states"]["exact"], row["states"]["terminal"]
        cells = [
            exact["representations"][name]["ratio_of_analytic_total"]
            for name in REPRESENTATIONS
        ]
        terms = [
            terminal["representations"][name]["ratio_of_analytic_total"]
            for name in REPRESENTATIONS
        ]
        verdict = row["verdict"]
        lines.append(
            f"| {row['row']} | {row['analytic_total_a']:.1f} | "
            f"{row['analytic_cut_cell_count']} | "
            + " | ".join(f"{value:.5f}" for value in cells)
            + " | "
            + " | ".join(f"{value:.5f}" for value in terms)
            + f" | {verdict['best']} | {verdict['state_error']:.5f} |"
        )
    lines.append("")
    lines.append("## Per-row detail (exact input then terminal)")
    for row in rows:
        lines.append(
            f"### {row['row']} (cells {row['realised_cells']}, "
            f"pitch {row['characteristic_pitch_m']:.4f} m, "
            f"qualification {row['solver_qualification']})"
        )
        for state_name in STATES:
            state = row["states"][state_name]
            lines.append(
                f"- **{state_name}** boundary level "
                f"{state['boundary_level_wb']:.6f} Wb (analytic separatrix at "
                "0; the terminal read error is its value)"
            )
            for name in REPRESENTATIONS:
                entry = state["representations"][name]
                lines.append(
                    f"  - {name}: ratio {entry['ratio_of_analytic_total']:.5f}, "
                    f"participating cut cells "
                    f"{entry['analytic_cut_cells_participating']}, "
                    f"boundary root error median "
                    f"{entry['boundary_error_median_m'] * 1000.0:.3f} mm "
                    f"({entry['boundary_error_median_pitches']:.5f} pitch), "
                    f"max {entry['boundary_error_max_m'] * 1000.0:.3f} mm "
                    f"({entry['boundary_error_max_pitches']:.5f} pitch), "
                    f"sign disagreements "
                    f"{entry['sign_disagreement_cells']}"
                )
    lines.append("")
    lines.append(
        "### Worse cells (booked minus analytic, top 6 per state and representation)"
    )
    for row in rows:
        for state_name in STATES:
            state = row["states"][state_name]
            for name in REPRESENTATIONS:
                worst = state["representations"][name]["worst_cells"]
                cells_formatted = "; ".join(
                    f"{entry['cell']}:{entry['difference_a']:+.0f}" for entry in worst
                )
                lines.append(f"- {row['row']} {state_name} {name}: {cells_formatted}")
    return "\n".join(lines)


def _lane() -> dict:
    import jax

    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURM_JOB_NODELIST", socket.gethostname()),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_default_backend": jax.default_backend(),
    }


def _analytic_flux_at(source, exact, part: dict, points: np.ndarray) -> np.ndarray:
    """Return the exact analytic flux at arbitrary points (Wb units)."""
    from benchmarks.solovev_certificate import _is_diverted_case

    if _is_diverted_case(part["case"]):
        return np.asarray(exact.flux(points), dtype=np.float64)
    return TOTAL_FLUX_FACTOR * np.asarray(
        source.flux(points[:, 0], points[:, 1]), dtype=np.float64
    )


def _run_all(report_dir: Path, figure_dir: Path, selected: list[str]) -> dict:
    import subprocess

    from nova.jax.config import (
        configure_persistent_compilation_cache,
        default_persistent_compilation_cache_root,
    )

    configure_dtypes()
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in ROWS:
        if selected and row[0] not in selected:
            continue
        part = _row_data(row)
        measured = _measure_row(row)
        rows.append(measured)
        source, exact, machine, operator = _machine(part)
        atomic = operator.moment_geometry.atomic_mesh
        exact_atomic = _analytic_flux_at(
            source, exact, part, np.asarray(atomic.node_coordinates)
        )
        _render_panel(
            part,
            machine,
            operator,
            measured["states"]["exact"],
            exact_atomic,
            figure_dir / f"{part['label']}-exact-representations.png",
        )
    rows.sort(key=lambda entry: entry["row"])
    if selected:
        rows = [entry for entry in rows if entry["row"] in selected]
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    lane = _lane()
    payload_rows = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key != "states"}
        clean["states"] = {
            state_name: {
                key: value for key, value in state.items() if key != "per_cell_roots"
            }
            for state_name, state in row["states"].items()
        }
        payload_rows.append(clean)
    payload = {
        "schema": "nova.read-representation-discriminator",
        "source_revision": source_revision,
        "acceptance": (
            "a representation whose exact-input ratio is within 0.1 percent "
            "of unity carries no representation error"
        ),
        "rows": payload_rows,
        "lane": lane,
        "exit_marker": EXIT_MARKER,
    }
    (report_dir / "receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report = _markdown_report(rows, source_revision, lane)
    (report_dir / "report.md").write_text(report)
    return {"rows": len(rows), "source_revision": source_revision}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=OUTPUT)
    parser.add_argument("--rows", nargs="*", default=[])
    parser.add_argument("--reports", type=Path, default=REPORTS)
    arguments = parser.parse_args()
    summary = _run_all(arguments.report_dir, arguments.figure_dir, arguments.rows)
    if arguments.reports is not None:
        arguments.reports.mkdir(parents=True, exist_ok=True)
        (arguments.reports / "report.md").write_text(
            (arguments.report_dir / "report.md").read_text()
        )
    print(f"rows measured: {summary['rows']} at {summary['source_revision']}")
    print(EXIT_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
