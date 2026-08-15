"""Measure the plasma round trip with conservative linear current moments.

The source mesh is fixed.  Full interior cells use the degree-three
centroid-and-corner stencil, cells cut by the separatrix use exact polygon
moments of a cellwise-linear density, and all three changing current vectors
contract against exact fixed-section flux blocks.  The centroid production
path is evaluated beside it as a control; neither result changes the identity
bound.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import zarr
from scipy import stats
from scipy.constants import mu_0
from scipy.interpolate import LinearNDInterpolator

from benchmarks.efit_analytic_roundtrip_floor import (
    _coupled_flux,
    _hex_mesh,
    _recovered_density,
)
from benchmarks.efit_boundary_method_comparison import (
    _cell_polygons,
    _current_moments,
)
from benchmarks.efit_constant_current_attribution import (
    ANALYTIC_CASE,
    GRID_SEQUENCE,
    REFERENCE_COMPOSITION_FRACTIONS,
)
from benchmarks.efit_flux_decomposition import (
    _density_from_flux,
    _evaluate_on_grid,
    _interpolator,
    _lcfs_mask,
    _read_stored_slice,
)
from benchmarks.efit_roundtrip_identity import account_round_trip
from nova.biot.greens import hybrid_greens
from nova.biot.polygonanalytic import polygon_analytic_flux_moments
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import reference_cases

IDENTITY_FRACTION = 1.0e-6
DEFAULT_SHOT = 21978
DEFAULT_SLICE = 46
DEFAULT_ARTIFACT_CACHE = Path.home() / ".cache" / "mast-artifact-ef"
DEFAULT_ARTIFACT_DIGEST = (
    "sha256:b41c076e1fb7e16dabe3bada2f5d890125a857c400ce7599dfa488e8ebef90e4"
)
EFIT_CONTROL_FRACTION = 5.152548851e-3
CONTROL_RELATIVE_TOLERANCE = 2.0e-7
EFIT_MASKED_CONTROL_RATIO = 2.08758329834
EFIT_UNMASKED_CONTROL_RATIO = 0.686454739323
COARSE_PRODUCTION_FORWARD_FRACTION = 6.542230468e-2
COARSE_LINEAR_FORWARD_FRACTION = 1.010285525e-3


def _rectangles(
    coordinate: np.ndarray, width: float, height: float
) -> list[np.ndarray]:
    """Return counter-clockwise fixed source sections."""
    offset = (
        np.asarray(
            [[-width, -height], [width, -height], [width, height], [-width, height]]
        )
        / 2.0
    )
    return [centre + offset for centre in coordinate]


def _shared_corners(cells: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate rectangle corners for the interior gather."""
    scale = max(float(np.max(np.abs(np.vstack(cells)))), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale
    lookup: dict[tuple[int, int], int] = {}
    nodes: list[np.ndarray] = []
    rows = np.empty((len(cells), 4), dtype=np.intp)
    for row, cell in zip(rows, cells, strict=True):
        for slot, point in enumerate(cell):
            key = tuple(np.rint(point / tolerance).astype(np.int64))
            if key not in lookup:
                lookup[key] = len(nodes)
                nodes.append(point)
            row[slot] = lookup[key]
    return np.asarray(nodes), rows


def _rectangular_mesh(
    radius: np.ndarray, height: np.ndarray
) -> tuple[StencilMesh, float, float]:
    """Build a quadratic-ring mesh on a tensor-product grid."""
    width = float(np.mean(np.diff(radius)))
    vertical_extent = float(np.mean(np.diff(height)))
    rr, zz = np.meshgrid(radius, height)
    coordinate = np.column_stack((rr.ravel(), zz.ravel()))
    rows = []
    nr = len(radius)
    nz = len(height)
    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            centre = iz * nr + ir
            neighbours = [
                (iz + dz) * nr + ir + dr
                for dz, dr in (
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                )
            ]
            rows.append([centre, *neighbours])
    return (
        StencilMesh(
            coordinate=coordinate,
            stencil=np.asarray(rows, dtype=np.intp),
            area=np.full(len(coordinate), width * vertical_extent),
        ),
        width,
        vertical_extent,
    )


def _block_chunk(arguments):
    """Build exact moment blocks and the production control for source indices."""
    indices, target_r, target_z, coordinate, cells, width, height = arguments
    rows = np.empty((4, len(target_r), len(indices)))
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for column, source in enumerate(indices):
            rows[:3, :, column] = polygon_analytic_flux_moments(
                target_r,
                target_z,
                cells[source],
                expansion_point=coordinate[source],
            )
            rows[3, :, column] = hybrid_greens(
                target_r,
                target_z,
                float(coordinate[source, 0]),
                float(coordinate[source, 1]),
                width,
                height,
            )[0]
    return indices, rows


def _coupling_blocks(
    mesh: StencilMesh,
    cells: list[np.ndarray],
    width: float,
    height: float,
    workers: int,
) -> np.ndarray:
    """Build the three exact blocks and production control matrix once."""
    source_chunks = [
        chunk
        for chunk in np.array_split(np.arange(mesh.node_count), workers)
        if len(chunk)
    ]
    arguments = [
        (
            chunk,
            mesh.coordinate[:, 0],
            mesh.coordinate[:, 1],
            mesh.coordinate,
            cells,
            width,
            height,
        )
        for chunk in source_chunks
    ]
    blocks = np.empty((4, mesh.node_count, mesh.node_count))
    if workers == 1:
        results = map(_block_chunk, arguments)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_block_chunk, arguments)
    try:
        for indices, rows in results:
            for local_column, source in enumerate(indices):
                blocks[:, :, source] = rows[:, :, local_column]
    finally:
        if workers != 1:
            pool.shutdown()
    return blocks


def _full_cell_vectors(
    mesh: StencilMesh,
    centroid_density: np.ndarray,
    shared_density: np.ndarray,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble degree-three current moments on every stencil-valid cell."""
    shared_nodes, cell_node = _shared_corners(
        _rectangles(mesh.coordinate, width, height)
    )
    if shared_density.shape != (len(shared_nodes),):
        raise ValueError("shared density does not match the interior node pool")
    second_mean = np.broadcast_to(
        np.asarray([width**2 / 12.0, height**2 / 12.0]),
        (mesh.node_count, 2),
    )
    interior_operator = mesh.current_moment_stencil(cell_node, second_mean)
    interior = interior_operator(centroid_density, shared_density)
    return (
        np.asarray(interior.cell_current),
        12.0 * np.asarray(interior.radial_moment) / width**2,
        12.0 * np.asarray(interior.vertical_moment) / height**2,
    )


def _linear_vectors(
    mesh: StencilMesh,
    clipped,
    centroid_density: np.ndarray,
    shared_density: np.ndarray,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble full-cell stencil and clipped-boundary current moments."""
    interior = _full_cell_vectors(mesh, centroid_density, shared_density, width, height)
    gradient = np.column_stack(mesh.gradient(centroid_density))
    boundary = clipped.linear_current_moments(centroid_density, gradient)
    use_boundary = clipped.boundary
    use_interior = clipped.included & ~use_boundary
    current = np.where(use_boundary, boundary.current, interior[0])
    radial_coefficient = np.where(
        use_boundary, 12.0 * boundary.radial / width**2, interior[1]
    )
    vertical_coefficient = np.where(
        use_boundary, 12.0 * boundary.vertical / height**2, interior[2]
    )
    current = np.where(use_boundary | use_interior, current, 0.0)
    radial_coefficient = np.where(use_boundary | use_interior, radial_coefficient, 0.0)
    vertical_coefficient = np.where(
        use_boundary | use_interior, vertical_coefficient, 0.0
    )
    return current, radial_coefficient, vertical_coefficient


def _contract(blocks: np.ndarray, vectors: tuple[np.ndarray, ...]) -> np.ndarray:
    return sum(
        block @ vector for block, vector in zip(blocks[:3], vectors, strict=True)
    )


def _restrict_vectors(
    vectors: tuple[np.ndarray, ...], admitted: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Retain fixed-section moment vectors only on admitted source cells."""
    return tuple(np.where(admitted, vector, 0.0) for vector in vectors)


def _fit_order(cell_size: np.ndarray, error: np.ndarray) -> dict[str, float]:
    fit = stats.linregress(np.log(cell_size), np.log(error))
    return {
        "observed_order": float(fit.slope),
        "order_standard_error": float(fit.stderr),
        "r_squared": float(fit.rvalue**2),
        "coefficient": float(np.exp(fit.intercept)),
    }


def _errors(difference: np.ndarray, span: float) -> dict[str, float]:
    return {
        "sup_fraction_of_span": float(np.max(np.abs(difference)) / span),
        "rms_fraction_of_span": float(np.sqrt(np.mean(difference**2)) / span),
    }


def _snap_analytic_level(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Represent analytically zero contour nodes as exact endpoint crossings."""
    level = np.asarray(values, dtype=float).copy()
    tolerance = (
        128.0 * np.finfo(level.dtype).eps * max(float(np.max(np.abs(level))), 1.0)
    )
    snapped = np.abs(level) <= tolerance
    level[snapped] = 0.0
    return level, int(np.count_nonzero(snapped))


def _interpolate_density(
    coordinate: np.ndarray, density: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    values = np.asarray(LinearNDInterpolator(coordinate, density)(targets), dtype=float)
    return np.nan_to_num(values)


def _source_vector_audit(
    case,
    mesh: StencilMesh,
    width: float,
    height: float,
    clipped,
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Compare the coarse-grid source vectors with the established reference."""
    polygons, classes = _cell_polygons(case, mesh.coordinate, width, height)
    current, radial, vertical, _centres = _current_moments(
        case, mesh.coordinate, polygons
    )
    computed_radial = vectors[1] * width**2 / 12.0
    computed_vertical = vectors[2] * height**2 / 12.0
    selected = vectors[0] != 0.0
    boundary = classes == "boundary"

    def compare(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "cell_count": int(np.count_nonzero(mask)),
            "current_sup_difference_a": float(
                np.max(np.abs(vectors[0][mask] - current[mask]))
            ),
            "current_rms_difference_a": float(
                np.sqrt(np.mean((vectors[0][mask] - current[mask]) ** 2))
            ),
            "radial_moment_sup_difference_a_m": float(
                np.max(np.abs(computed_radial[mask] - radial[mask]))
            ),
            "vertical_moment_sup_difference_a_m": float(
                np.max(np.abs(computed_vertical[mask] - vertical[mask]))
            ),
        }

    return {
        "reference": (
            "converged clipped-current quadrature from the boundary-method comparison"
        ),
        "boundary_cell_count_matches": bool(
            np.count_nonzero(clipped.boundary) == np.count_nonzero(boundary)
        ),
        "all_nonzero_cells": compare(selected),
        "boundary_cells": compare(boundary),
        "signed_total_current_difference_a": float(
            np.sum(vectors[0]) - np.sum(current)
        ),
    }


def _analytic_resolution(radial_count: int, vertical_count: int, workers: int):
    configure_dtypes()
    case = reference_cases()[ANALYTIC_CASE]
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    mesh, width, height = _hex_mesh(
        radial_count, vertical_count, case.major_radius, half_height
    )
    cells = _rectangles(mesh.coordinate, width, height)
    atomic = AtomicCellMesh.from_cells(cells, centroids=mesh.coordinate)
    signed_level, snapped_level_nodes = _snap_analytic_level(atomic.sample(case.flux))
    clipped = atomic.clip(signed_level)
    corner, _cell_node = _shared_corners(cells)
    centroid_density = np.asarray(
        case.toroidal_current_density(mesh.coordinate[:, 0], mesh.coordinate[:, 1])
    )
    corner_density = np.asarray(
        case.toroidal_current_density(corner[:, 0], corner[:, 1])
    )
    vectors = _linear_vectors(
        mesh, clipped, centroid_density, corner_density, width, height
    )
    blocks = _coupling_blocks(mesh, cells, width, height, workers)
    linear_flux = _contract(blocks, vectors)
    recovered = _recovered_density(mesh, linear_flux)
    recovered_corner = _interpolate_density(mesh.coordinate, recovered, corner)
    repeated_vectors = _full_cell_vectors(
        mesh, recovered, recovered_corner, width, height
    )
    repeated_linear_flux = _contract(blocks, repeated_vectors)
    remasked_vectors = _linear_vectors(
        mesh, clipped, recovered, recovered_corner, width, height
    )
    remasked_linear_flux = _contract(blocks, remasked_vectors)

    plasma = case.contains(mesh.coordinate[:, 0], mesh.coordinate[:, 1])
    driven_density = np.where(plasma, centroid_density, 0.0)
    production_current = driven_density * mesh.cell_area
    production_flux = _coupled_flux(mesh, driven_density, width, height)
    production_recovered = _recovered_density(mesh, production_flux)
    production_repeat = _coupled_flux(mesh, production_recovered, width, height)
    dense_production_flux = blocks[3] @ production_current
    dense_production_repeat = blocks[3] @ (production_recovered * mesh.cell_area)
    span = float(TOTAL_FLUX_FACTOR * case.axis_flux)
    linear_error = _errors(linear_flux - repeated_linear_flux, span)
    remasked_error = _errors(linear_flux - remasked_linear_flux, span)
    production_error = _errors(production_flux - production_repeat, span)
    banked = REFERENCE_COMPOSITION_FRACTIONS[
        GRID_SEQUENCE.index((radial_count, vertical_count))
    ]
    control_diagnostic = {
        "grid": [radial_count, vertical_count],
        "computed_sup_fraction_of_span": production_error["sup_fraction_of_span"],
        "banked_sup_fraction_of_span": banked,
        "relative_reproduction_error": float(
            production_error["sup_fraction_of_span"] / banked - 1.0
        ),
        "dense_initial_sup_difference_wb": float(
            np.max(np.abs(dense_production_flux - production_flux))
        ),
        "dense_repeat_sup_difference_wb": float(
            np.max(np.abs(dense_production_repeat - production_repeat))
        ),
    }
    print(
        json.dumps({"analytic_control_smoke": control_diagnostic}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    if (
        abs(production_error["sup_fraction_of_span"] / banked - 1.0)
        > CONTROL_RELATIVE_TOLERANCE
    ):
        raise AssertionError(
            "analytic production control did not reproduce its banked value"
        )
    return {
        "grid": [radial_count, vertical_count],
        "cell_count": mesh.node_count,
        "characteristic_cell_size_m": float(np.sqrt(mesh.cell_area[0])),
        "boundary_cell_count": int(np.count_nonzero(clipped.boundary)),
        "analytic_level_nodes_snapped_to_zero": snapped_level_nodes,
        "clip_relative_area_residual": float(
            abs(clipped.patch_area_sum - clipped.contour_area) / clipped.contour_area
        ),
        "linear_representation": linear_error,
        "repeat_support_diagnostic": {
            "recovery_semantics": (
                "carry delta-star recovery on every stencil-valid cell without "
                "a second LCFS mask, matching the analytic production control"
            ),
            "double_remask_counterfactual": remasked_error,
            "corrected_over_double_remask_sup_ratio": float(
                linear_error["sup_fraction_of_span"]
                / remasked_error["sup_fraction_of_span"]
            ),
            "initial_total_current_a": float(np.sum(vectors[0])),
            "corrected_repeat_total_current_a": float(np.sum(repeated_vectors[0])),
            "double_remask_repeat_total_current_a": float(np.sum(remasked_vectors[0])),
            "initial_boundary_current_a": float(np.sum(vectors[0][clipped.boundary])),
            "corrected_repeat_boundary_current_a": float(
                np.sum(repeated_vectors[0][clipped.boundary])
            ),
            "double_remask_repeat_boundary_current_a": float(
                np.sum(remasked_vectors[0][clipped.boundary])
            ),
            "corrected_repeat_exterior_current_a": float(
                np.sum(repeated_vectors[0][~clipped.included])
            ),
        },
        "source_vector_audit": (
            _source_vector_audit(case, mesh, width, height, clipped, vectors)
            if (radial_count, vertical_count) == GRID_SEQUENCE[0]
            else None
        ),
        "centroid_production_control": {
            **production_error,
            "banked_sup_fraction_of_span": banked,
            "relative_reproduction_error": float(
                production_error["sup_fraction_of_span"] / banked - 1.0
            ),
            "dense_contraction_comparison": {
                "initial_sup_difference_wb": control_diagnostic[
                    "dense_initial_sup_difference_wb"
                ],
                "repeat_sup_difference_wb": control_diagnostic[
                    "dense_repeat_sup_difference_wb"
                ],
            },
        },
    }


def _efit_measurement(
    shot: int,
    slice_index: int,
    store: Path,
    artifact_cache: Path,
    artifact_digest: str,
    workers: int,
) -> dict[str, Any]:
    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    requested_time = float(group["time"][slice_index])
    stored = _read_stored_slice(group, requested_time)
    chain = build_mast_parity_chain(
        shot,
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
    )
    profile = chain.profile_solver
    radius = np.asarray(profile.grid_r, dtype=float)
    height_axis = np.asarray(profile.grid_z, dtype=float)
    mesh, width, height = _rectangular_mesh(radius, height_axis)
    cells = _rectangles(mesh.coordinate, width, height)
    atomic = AtomicCellMesh.from_cells(cells, centroids=mesh.coordinate)
    nova_flux = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, stored.total_flux_wb),
        radius,
        height_axis,
    )
    stored_lattice = FluxLattice(stored.radius_m, stored.height_m)
    stored_density, stored_valid = _density_from_flux(
        stored_lattice, stored.total_flux_wb
    )
    stored_lcfs = _lcfs_mask(
        stored.radius_m,
        stored.height_m,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    stored_density = np.where(stored_valid & stored_lcfs, stored_density, 0.0)
    density_interpolator = _interpolator(
        stored.radius_m, stored.height_m, stored_density.T
    )
    centroid_density = _evaluate_on_grid(
        density_interpolator, radius, height_axis
    ).ravel()
    lcfs_points = np.column_stack((stored.lcfs_height_m, stored.lcfs_radius_m))
    boundary_flux = float(
        np.median(
            _interpolator(stored.radius_m, stored.height_m, stored.total_flux_wb)(
                lcfs_points
            )
        )
    )
    centre_flux = nova_flux.ravel()
    nova_lcfs = _lcfs_mask(
        radius,
        height_axis,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    ).T.ravel()
    axis_index = np.flatnonzero(nova_lcfs)[
        np.argmax(np.abs(centre_flux[nova_lcfs] - boundary_flux))
    ]
    sign = np.sign(centre_flux[axis_index] - boundary_flux)
    signed_node_flux = sign * (
        atomic.sample(
            lambda r, z: _interpolator(
                stored.radius_m, stored.height_m, stored.total_flux_wb
            )(np.column_stack((z, r)))
        )
        - boundary_flux
    )
    finite_node_flux = np.isfinite(signed_node_flux)
    exterior_level = -max(
        float(np.max(np.abs(signed_node_flux[finite_node_flux]))), 1.0
    )
    signed_node_flux = np.where(finite_node_flux, signed_node_flux, exterior_level)
    stored_component = inside_polygon(
        atomic.node_coordinates[:, 0],
        atomic.node_coordinates[:, 1],
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    suppressed_positive_nodes = int(
        np.count_nonzero((signed_node_flux > 0.0) & ~stored_component)
    )
    signed_node_flux = np.where(
        stored_component, signed_node_flux, -np.abs(signed_node_flux)
    )
    clipped = atomic.clip(signed_node_flux)
    if not clipped.contour_closed or not np.isfinite(clipped.contour_area):
        raise AssertionError("EFIT clip did not produce a closed finite contour")
    corner, _cell_node = _shared_corners(cells)
    corner_density = density_interpolator(np.column_stack((corner[:, 1], corner[:, 0])))
    vectors = _linear_vectors(
        mesh, clipped, centroid_density, corner_density, width, height
    )
    blocks = _coupling_blocks(mesh, cells, width, height, workers)
    linear_flux = _contract(blocks, vectors)
    recovered = np.asarray(
        -mesh.delta_star(linear_flux) / (TOTAL_FLUX_FACTOR * mu_0 * mesh.node_radius)
    )
    recovered = np.where(np.asarray(mesh.interior(2)), recovered, 0.0)
    recovered_corner = _interpolate_density(mesh.coordinate, recovered, corner)
    repeated_vectors = _linear_vectors(
        mesh, clipped, recovered, recovered_corner, width, height
    )
    repeated_linear_flux = _contract(blocks, repeated_vectors)
    unmasked_vectors = _full_cell_vectors(
        mesh, recovered, recovered_corner, width, height
    )
    unmasked_linear_flux = _contract(blocks, unmasked_vectors)
    admitted_full_vectors = _restrict_vectors(unmasked_vectors, clipped.included)
    admitted_full_flux = _contract(blocks, admitted_full_vectors)
    centroid_admitted_vectors = _restrict_vectors(repeated_vectors, nova_lcfs)
    centroid_admitted_flux = _contract(blocks, centroid_admitted_vectors)
    boundary_dropped_vectors = _restrict_vectors(repeated_vectors, ~clipped.boundary)
    boundary_dropped_flux = _contract(blocks, boundary_dropped_vectors)

    banked_control = account_round_trip(
        shot=shot,
        slice_index=slice_index,
        store=store,
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
    )
    span = float(np.ptp(stored.total_flux_wb))
    control_fraction = float(
        banked_control["operator_identity"][
            "measured_sup_error_fraction_of_stored_span"
        ]
    )
    control_error = {
        "sup_fraction_of_span": control_fraction,
        "rms_fraction_of_span": None,
    }
    print(
        json.dumps(
            {
                "efit_control_smoke": {
                    "computed_sup_fraction_of_span": control_fraction,
                    "banked_sup_fraction_of_span": EFIT_CONTROL_FRACTION,
                    "relative_reproduction_error": (
                        control_fraction / EFIT_CONTROL_FRACTION - 1.0
                    ),
                }
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    if (
        abs(control_error["sup_fraction_of_span"] / EFIT_CONTROL_FRACTION - 1.0)
        > CONTROL_RELATIVE_TOLERANCE
    ):
        raise AssertionError(
            "EFIT production control did not reproduce its banked value"
        )
    linear_error = _errors(linear_flux - repeated_linear_flux, span)
    unmasked_error = _errors(linear_flux - unmasked_linear_flux, span)
    admitted_full_error = _errors(linear_flux - admitted_full_flux, span)
    exterior_cut = admitted_full_flux - unmasked_linear_flux
    boundary_reclip = repeated_linear_flux - admitted_full_flux
    admission_change = centroid_admitted_flux - repeated_linear_flux
    boundary_discard = boundary_dropped_flux - repeated_linear_flux
    candidates = {
        "exterior_stencil_current_cut": {
            **_errors(exterior_cut, span),
            "removed_cell_count": int(np.count_nonzero(~clipped.included)),
            "removed_current_a": float(np.sum(unmasked_vectors[0][~clipped.included])),
            "role": "exact masked-minus-unmasked term",
        },
        "boundary_cells_clipped_twice": {
            **_errors(boundary_reclip, span),
            "affected_cell_count": int(np.count_nonzero(clipped.boundary)),
            "removed_current_a": float(
                np.sum(admitted_full_vectors[0][clipped.boundary])
                - np.sum(repeated_vectors[0][clipped.boundary])
            ),
            "role": "exact masked-minus-unmasked term",
        },
        "centroid_instead_of_any_intersection_admission": {
            **_errors(admission_change, span),
            "discarded_boundary_cell_count": int(
                np.count_nonzero(clipped.boundary & ~nova_lcfs)
            ),
            "removed_current_a": float(
                np.sum(repeated_vectors[0][clipped.boundary & ~nova_lcfs])
            ),
            "role": "alternative admission-rule counterfactual",
        },
        "all_clipped_boundary_support_discarded": {
            **_errors(boundary_discard, span),
            "discarded_boundary_cell_count": int(np.count_nonzero(clipped.boundary)),
            "removed_current_a": float(np.sum(repeated_vectors[0][clipped.boundary])),
            "role": "boundary-support-loss counterfactual",
        },
    }
    exact_terms = (
        "exterior_stencil_current_cut",
        "boundary_cells_clipped_twice",
    )
    dominant_term = max(
        exact_terms, key=lambda name: candidates[name]["sup_fraction_of_span"]
    )
    closure = (
        repeated_linear_flux - unmasked_linear_flux - exterior_cut - boundary_reclip
    )
    masked_ratio = float(
        linear_error["sup_fraction_of_span"] / control_error["sup_fraction_of_span"]
    )
    unmasked_ratio = float(
        unmasked_error["sup_fraction_of_span"] / control_error["sup_fraction_of_span"]
    )
    for measured, banked, label in (
        (masked_ratio, EFIT_MASKED_CONTROL_RATIO, "masked"),
        (unmasked_ratio, EFIT_UNMASKED_CONTROL_RATIO, "unmasked"),
    ):
        if abs(measured / banked - 1.0) > CONTROL_RELATIVE_TOLERANCE:
            raise AssertionError(f"EFIT {label} linear ratio did not reproduce")
    return {
        "shot": shot,
        "slice": slice_index,
        "time_s": stored.time_s,
        "grid": [len(radius), len(height_axis)],
        "stored_flux_span_wb": span,
        "boundary_cell_count": int(np.count_nonzero(clipped.boundary)),
        "level_component_selection": {
            "authority": "stored LCFS polygon",
            "suppressed_positive_nodes_outside_lcfs": suppressed_positive_nodes,
            "contour_closed": bool(clipped.contour_closed),
        },
        "clip_relative_area_residual": float(
            abs(clipped.patch_area_sum - clipped.contour_area) / clipped.contour_area
        ),
        "linear_representation": linear_error,
        "ranking_against_control": {
            "masked_linear_over_centroid_sup_ratio": masked_ratio,
            "banked_masked_linear_over_centroid_sup_ratio": (EFIT_MASKED_CONTROL_RATIO),
            "masked_ratio_relative_reproduction_error": float(
                masked_ratio / EFIT_MASKED_CONTROL_RATIO - 1.0
            ),
            "unmasked_linear_over_centroid_sup_ratio": unmasked_ratio,
            "banked_unmasked_linear_over_centroid_sup_ratio": (
                EFIT_UNMASKED_CONTROL_RATIO
            ),
            "unmasked_ratio_relative_reproduction_error": float(
                unmasked_ratio / EFIT_UNMASKED_CONTROL_RATIO - 1.0
            ),
            "reading": (
                "the mandatory stored-LCFS remask reverses the EFIT ranking; "
                "without that remask the fixed linear representation remains "
                "below the centroid control"
            ),
        },
        "repeat_support_diagnostic": {
            "recovery_semantics": (
                "reapply the LCFS support after delta-star recovery, matching the "
                "mandatory remask in the EFIT production control"
            ),
            "unmasked_counterfactual": unmasked_error,
            "initial_total_current_a": float(np.sum(vectors[0])),
            "masked_repeat_total_current_a": float(np.sum(repeated_vectors[0])),
            "unmasked_repeat_total_current_a": float(np.sum(unmasked_vectors[0])),
            "initial_boundary_current_a": float(np.sum(vectors[0][clipped.boundary])),
            "masked_repeat_boundary_current_a": float(
                np.sum(repeated_vectors[0][clipped.boundary])
            ),
            "unmasked_repeat_boundary_current_a": float(
                np.sum(unmasked_vectors[0][clipped.boundary])
            ),
            "unmasked_repeat_exterior_current_a": float(
                np.sum(unmasked_vectors[0][~clipped.included])
            ),
        },
        "remask_mechanism": {
            "candidate_effects": candidates,
            "exact_masked_minus_unmasked_closure_fraction_of_span": float(
                np.max(np.abs(closure)) / span
            ),
            "dominant_exact_term": dominant_term,
            "dominant_exact_term_sup_fraction_of_span": candidates[dominant_term][
                "sup_fraction_of_span"
            ],
            "dominant_over_exterior_cut_sup_ratio": float(
                candidates[dominant_term]["sup_fraction_of_span"]
                / candidates["exterior_stencil_current_cut"]["sup_fraction_of_span"]
            ),
            "masked_once_composition_error": admitted_full_error,
            "masked_once_over_centroid_sup_ratio": float(
                admitted_full_error["sup_fraction_of_span"]
                / control_error["sup_fraction_of_span"]
            ),
            "finding": (
                "the ranking inversion is caused primarily by clipping recovered "
                "boundary moments a second time, not by the required removal of "
                "exterior stencil current"
            ),
            "recommendation": (
                "compare the composition masked, because the plasma term must not "
                "admit exterior delta-star current; apply that support restriction "
                "once by removing exterior cells while retaining full recovered "
                "boundary moments, rather than clipping the boundary moments again"
            ),
            "bound_changed_or_applied": False,
        },
        "centroid_production_control": {
            **control_error,
            "banked_sup_fraction_of_span": EFIT_CONTROL_FRACTION,
            "relative_reproduction_error": float(
                control_error["sup_fraction_of_span"] / EFIT_CONTROL_FRACTION - 1.0
            ),
        },
    }


def _analytic_bank(workers: int) -> dict[str, Any]:
    """Return the analytic grid bank and the prebank assembly check."""
    analytic = [_analytic_resolution(nr, nz, workers) for nr, nz in GRID_SEQUENCE]
    cell_size = np.asarray([row["characteristic_cell_size_m"] for row in analytic])
    sup = np.asarray(
        [row["linear_representation"]["sup_fraction_of_span"] for row in analytic]
    )
    rms = np.asarray(
        [row["linear_representation"]["rms_fraction_of_span"] for row in analytic]
    )
    sup_fit = _fit_order(cell_size, sup)
    rms_fit = _fit_order(cell_size, rms)
    identity_size = float(
        (IDENTITY_FRACTION / sup_fit["coefficient"])
        ** (1.0 / sup_fit["observed_order"])
    )
    return {
        "method": {
            "interior": "degree-three centroid-and-shared-corner current moments",
            "boundary": "shared-crossing conservative clip with exact linear moments",
            "coupling": "exact fixed polygon flux blocks G0, GR, GZ",
            "analytic_repeat_support": (
                "unmasked delta-star recovery on all stencil-valid cells, matching "
                "the analytic production control"
            ),
            "efit_repeat_support": "mandatory LCFS remask, matching the EFIT control",
            "identity_bound": IDENTITY_FRACTION,
            "bound_changed_or_applied": False,
        },
        "prebank_assembly_check": {
            "grid": analytic[0]["grid"],
            "source_vectors": analytic[0]["source_vector_audit"],
            "forward_ranking_from_boundary_method_comparison": {
                "production_centroid_sup_fraction_of_span": (
                    COARSE_PRODUCTION_FORWARD_FRACTION
                ),
                "fixed_linear_sup_fraction_of_span": (COARSE_LINEAR_FORWARD_FRACTION),
                "linear_improvement_factor": float(
                    COARSE_PRODUCTION_FORWARD_FRACTION / COARSE_LINEAR_FORWARD_FRACTION
                ),
            },
            "roundtrip_ranking": {
                "centroid_production_sup_fraction_of_span": analytic[0][
                    "centroid_production_control"
                ]["sup_fraction_of_span"],
                "fixed_linear_sup_fraction_of_span": analytic[0][
                    "linear_representation"
                ]["sup_fraction_of_span"],
                "linear_over_centroid_ratio": float(
                    analytic[0]["linear_representation"]["sup_fraction_of_span"]
                    / analytic[0]["centroid_production_control"]["sup_fraction_of_span"]
                ),
            },
            "repeat_support": analytic[0]["repeat_support_diagnostic"],
            "conclusion": (
                "the apparent ranking inversion was a second-mask assembly defect; "
                "the corrected analytic repeat retains recovered boundary and "
                "exterior stencil currents exactly as its control does"
            ),
        },
        "analytic": {
            "case": ANALYTIC_CASE,
            "resolutions": analytic,
            "sup_convergence": sup_fit,
            "rms_convergence": rms_fit,
            "identity_extrapolation": {
                "characteristic_cell_size_m": identity_size,
                "linear_refinement_from_finest": float(cell_size[-1] / identity_size),
                "finest_sup_over_bound": float(sup[-1] / IDENTITY_FRACTION),
            },
        },
    }


def _efit_bank(arguments: argparse.Namespace) -> dict[str, Any]:
    """Return the requested stored-slice measurement."""
    return _efit_measurement(
        arguments.shot,
        arguments.slice,
        arguments.store,
        arguments.artifact_cache,
        arguments.artifact_digest,
        arguments.workers,
    )


def _read_bank(path: Path) -> dict[str, Any]:
    """Read one completed partial bank for assembly without recomputation."""
    return json.loads(path.read_text())


def measure(arguments: argparse.Namespace) -> dict[str, Any]:
    configure_dtypes()
    analytic = (
        _read_bank(arguments.analytic_results)
        if arguments.analytic_results is not None
        else _analytic_bank(arguments.workers)
    )
    efit = (
        _read_bank(arguments.efit_results)
        if arguments.efit_results is not None
        else _efit_bank(arguments)
    )
    return {**analytic, "efit": efit}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--slice", type=int, default=DEFAULT_SLICE)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--artifact-cache", type=Path, default=DEFAULT_ARTIFACT_CACHE)
    parser.add_argument("--artifact-digest", default=DEFAULT_ARTIFACT_DIGEST)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--efit-only", action="store_true")
    parser.add_argument("--analytic-results", type=Path)
    parser.add_argument("--efit-results", type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.analytic_only and arguments.efit_only:
        raise ValueError("analytic-only and efit-only are mutually exclusive")
    if arguments.analytic_only:
        result = _analytic_bank(arguments.workers)
    elif arguments.efit_only:
        result = _efit_bank(arguments)
    else:
        result = measure(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
