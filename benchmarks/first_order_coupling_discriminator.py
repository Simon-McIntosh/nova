"""Discriminate the coupling's first-order moment term against an exact Biot image.

Gate B (``benchmarks/frozen_current_flux_image_gate.py``) showed that adding the
first-order current moments to exact zeroth moments makes the coupling's frozen-
current flux image *worse* (weak -110 whole-domain rms error over span 0.0057
with zeroth moments only, 0.0106 with first-order moments added), and that the
added error is a smooth affine field.  A correct moment expansion cannot get
worse when the dipole is added, so either the first-order channel carries a sign
or scale error, or its expansion point disagrees with the point the moments are
taken about.  This module discriminates those candidates on two controlled
inputs.

Part one, the single cell.  One interior hex cell of the weak -110 mesh carries a
toroidal current density that is *exactly linear* in ``R`` and ``Z`` about the
cell's own centroid, with analytically known zeroth and first moments.  The
exact total poloidal flux that linear density produces at every mesh node is
evaluated by 2D quadrature of the ring Green's function (the filament mutual
inductance, in the same total-flux convention as the analytic oracle) over the
cell, and compared with the production coupling image of that cell formed from
(i) the zeroth moment only, (ii) the zeroth plus first moments, (iii) the zeroth
plus first with the first-order term's sign flipped, and (iv) the zeroth plus
first with the moments taken about the point the kernel expands about if that
differs from the centroid.  Because the density is linear, the moments describe
it *exactly*: the variant that is exact to second order in the cell size must be
the correctly-signed, correctly-referenced first-order one, and the zeroth-only
and sign-flipped variants bound the dipole term from either side.

Part two, the weak -110 frozen currents.  Gate B's frozen image is rebuilt with
the same four variants: (i) zeroth moments only, (ii) the as-built zeroth-plus-
first contraction, (iii) the first moments sign-flipped before the conversion,
and (iv) the expansion point moved --- the physical first moments re-taken about
the clipped plasma region's own centroid and the kernel blocks rebuilt over the
region polygon at that expansion point, so the dipole representation is
self-consistent about the region rather than about the whole cell.  Whole-domain
rms and max error over span are reported for each variant beside gate B's
committed 0.0106 (first-order added) and 0.0057 (zeroth only).

The unit and scale of the first-moment channel are reported against the kernel's
own expectation (path:line), so a missing factor of ``R`` or of the pitch cannot
survive unreported.

The whole module runs in one ``all_debug`` CPU allocation: the root venv python
directly (never uv on the compute node), ``PYTHONPATH`` at the worktree,
``JAX_PLATFORMS=cpu``, ``TMPDIR=/tmp`` in the submit environment and the payload.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe, ellipk
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from benchmarks import solovev_certificate as certificate
from benchmarks.frozen_current_flux_image_gate import (
    _frozen_moments,
    _physical_moments,
)
from nova.biot.greens import second_moments, section_centroid
from nova.biot.polygonanalytic import polygon_analytic_flux_moments
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture

MU_0 = 4.0e-7 * np.pi

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/first-order-coupling"
RECEIPT = OUTPUT_ROOT / "receipt.json"
PARTS = OUTPUT_ROOT / "parts"
PART_ONE = PARTS / "single-cell.json"
PART_TWO = PARTS / "weak-110-frozen.json"

CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_CELLS = -110
PROJECT_SRC_PREFIX = "/nova/figures/cut-cell-current-attribution/first-order-coupling"

#: Duffy order for the exact single-cell quadrature.  Converged for targets
#: outside the cell (validated against the kernel blocks to 1e-9+ relative at
#: order 28); the only node inside the cell carries a relative error well below
#: every signal this module discriminates.
DUFFY_ORDER = 28

#: Path:line table naming the first-moment channel's units and expansion
#: points, fixed by the FENCE as part of the evidence.
UNITS_AND_REFERENCES = {
    "physical_first_moment_A_m": (
        "integral over the cell of j*(R - Rc) dA and j*(Z - Zc) dA about the "
        "centroid; gate B _frozen_moments builds them from exact density "
        "integrals, fixed_profile_current_moments from the clipped support"
    ),
    "moment_geometry_second_moment_m2": (
        "area-normalised second central moments (Irr, Izz, Irz) over the FULL "
        "cell polygon; nova/biot/greens.py second_moments; consumed by "
        "coupling_current_moments"
    ),
    "coupling_current_moments": (
        "nova/equilibrium/forward_operator.py:2092-2123 -- converts physical "
        "A*m first moments to A/m coefficients by inverting the second-moment "
        "matrix; the output contracts the kernel's Wb*m/A area-mean blocks "
        "(polygonanalytic.py polygon_analytic_flux_moments, area-mean "
        "companions of (R - Rc) K and (Z - Zc) K), so the product is Wb"
    ),
    "kernel_expansion_point": (
        "polygonanalytic.py polygon_analytic_flux_moments: requested_centre "
        "defaults to _section_centroid(vertices) (the polygon area centroid); "
        "build_machine passes expansion_points=centres, which are "
        "MomentGeometry.atomic_mesh.centroids = section_centroid of each full "
        "cell (stencil_mesh.py from_cells:321)"
    ),
    "moment_point": (
        "gate B _frozen_moments takes moments about "
        "machine.moment_geometry.atomic_mesh.centroids, the same full-cell "
        "centroids the kernel expands about; the production SOLVE path takes "
        "them about the clipped support polygon's centroid (stencil_mesh.py "
        "_direct_profile_current_moments moment_centre=support.centroids)"
    ),
}


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane_receipt() -> dict[str, Any]:
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


# ---------------------------------------------------------------------------
# exact ring-Green's-function quadrature over a polygon
# ---------------------------------------------------------------------------


def _duffy_rule(vertices: np.ndarray, order: int = DUFFY_ORDER):
    """Return tokens and area weights of the tensor-Duffy rule on a polygon.

    Each triangle (vertex 0, vertex i, vertex i + 1) is mapped from the unit
    square by ``P = A + alpha (B - A) + (1 - alpha) beta (C - A)`` with
    Jacobian ``(1 - alpha)``; the weight carries that Jacobian exactly.  The
    rule is validated against the kernel blocks for the uniform and both first
    moment rows of the ring Green's function.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    points: list[np.ndarray] = []
    area_weights: list[float] = []
    for index in range(1, len(vertices) - 1):
        first, second, third = vertices[[0, index, index + 1]]
        edge_first = second - first
        edge_second = third - first
        cross = abs(edge_first[0] * edge_second[1] - edge_first[1] * edge_second[0])
        for radial, radial_weight in zip(unit_nodes, unit_weights, strict=True):
            for vertical, vertical_weight in zip(unit_nodes, unit_weights, strict=True):
                points.append(
                    first
                    + radial * edge_first
                    + (1.0 - radial) * vertical * edge_second
                )
                area_weights.append(
                    cross * (1.0 - radial) * radial_weight * vertical_weight
                )
    return np.asarray(points), np.asarray(area_weights)


def _filament_mutual(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Return the mutual inductance [Wb/A] between two coaxial full rings.

    ``target`` and ``source`` are ``(..., 2)`` (R, Z) coordinates.  The total
    poloidal flux linking a full loop at the target radius equals this
    inductance per ampere of the source ring, which is the same total-flux
    convention the analytic oracle and the polygon kernel use (the oracle
    applies ``TOTAL_FLUX_FACTOR = 2 pi`` to its per-radian flux).  Validated
    against direct numeric integration of the vector-potential loop integral.
    """
    k2 = (
        4.0
        * source[..., 0]
        * target[..., 0]
        / (
            (source[..., 0] + target[..., 0]) ** 2
            + (target[..., 1] - source[..., 1]) ** 2
        )
    )
    k2 = np.clip(k2, 0.0, 1.0 - 1.0e-300)
    k = np.sqrt(k2)
    return (
        MU_0
        * np.sqrt(source[..., 0] * target[..., 0])
        * ((2.0 / k - k) * ellipk(k2) - (2.0 / k) * ellipe(k2))
    )


def exact_polygon_flux(
    target_r: np.ndarray,
    target_z: np.ndarray,
    source_points: np.ndarray,
    density: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the total poloidal flux of one polygon current distribution.

    ``density`` carries the toroidal current density [A/m^2] at the polygon's
    Duffy nodes and ``weights`` the 2D area weights, so the returned flux is
    ``sum_q density[q] * M(target, node_q) * weights[q]`` at every target ---
    the ring Green's function contracted over the prescribed current, in the
    same total-flux convention as the coupling kernel.
    """
    values = np.empty(np.asarray(target_r, dtype=np.float64).shape, dtype=np.float64)
    block = 64
    start = 0
    while start < values.size:
        stop = min(start + block, values.size)
        target = np.column_stack(
            (
                np.asarray(target_r, dtype=np.float64).ravel()[start:stop],
                np.asarray(target_z, dtype=np.float64).ravel()[start:stop],
            )
        )
        coupling = _filament_mutual(target[:, None, :], source_points[None, :, :])
        values.ravel()[start:stop] = coupling @ (density * weights)
        start = stop
    return values


# ---------------------------------------------------------------------------
# part one: the single cell against the exact linear-density image
# ---------------------------------------------------------------------------


def _cell_kinds(
    polygons: tuple[np.ndarray, ...], areas: np.ndarray, plasma: Polygon
) -> np.ndarray:
    """Return 0 exterior, 1 interior, 2 boundary-cut per cell."""
    kinds = np.zeros(len(polygons), dtype=int)
    for index, polygon in enumerate(polygons):
        region = Polygon(polygon).intersection(plasma)
        tolerance = 2.0e-11 * max(float(areas[index]), 1.0)
        if region.is_empty or region.area <= tolerance:
            continue
        if abs(region.area - areas[index]) <= tolerance:
            kinds[index] = 1
        else:
            kinds[index] = 2
    return kinds


def _interior_cell_choice(
    polygons: tuple[np.ndarray, ...],
    areas: np.ndarray,
    centres: np.ndarray,
    density_source: Any,
    plasma: Polygon,
) -> int:
    """Return the interior cell whose dipole is largest in both directions.

    The dipole is evaluated from the density's linearisation over each interior
    cell's Duffy nodes, and the cell maximising the *smaller* of the two first-
    moment magnitudes is chosen, so both the radial and the vertical first-order
    channels are exercised decisively.  Deterministic: ties resolve to the lower
    mesh index.
    """
    kinds = _cell_kinds(polygons, areas, plasma)
    interior = np.flatnonzero(kinds == 1)
    best = -1
    best_score = -np.inf
    for cell in interior:
        vertices = polygons[cell]
        points, weights = _duffy_rule(vertices)
        values = np.asarray(
            density_source(points[:, 0], points[:, 1]), dtype=np.float64
        )
        density = -values / (MU_0 * points[:, 0])
        g0, gR, gZ = _linear_density_fit(points, density, centres[cell])
        irr, izz, irz = second_moments(vertices)
        radial = areas[cell] * (gR * irr + gZ * irz)
        vertical = areas[cell] * (gR * irz + gZ * izz)
        score = min(abs(radial), abs(vertical))
        if score > best_score:
            best, best_score = cell, score
    if best < 0:
        raise RuntimeError("no interior cell found on the weak -110 mesh")
    return int(best)


def _linear_density_fit(
    density_points: np.ndarray, density: np.ndarray, centre: np.ndarray
) -> tuple[float, float, float]:
    """Return the cell-local linearisation coefficients of the exact density."""
    local = density_points - centre
    basis = np.column_stack((np.ones(len(density_points)), local[:, 0], local[:, 1]))
    coefficients, *_rest = np.linalg.lstsq(basis, density, rcond=None)
    return tuple(float(value) for value in coefficients)


def _linear_moments(
    coefficients: tuple[float, float, float],
    area: float,
    second: tuple[float, float, float],
) -> np.ndarray:
    """Return the exact zeroth, radial and vertical moments of the density.

    For ``j = g0 + gR (R - Rc) + gZ (Z - Zc)`` over a polygon, the moments about
    the centroid are exactly ``M0 = g0 A``, ``MR = A (gR Irr + gZ Irz)`` and
    ``MZ = A (gR Irz + gZ Izz)`` with ``(Irr, Izz, Irz)`` the area-normalised
    second central moments.
    """
    g0, gR, gZ = coefficients
    irr, izz, irz = second
    return np.asarray(
        (g0 * area, area * (gR * irr + gZ * irz), area * (gR * irz + gZ * izz))
    )


def _single_cell_image(
    operator: Any,
    cell: int,
    moments: np.ndarray,
    *,
    flip_first: bool = False,
) -> np.ndarray:
    """Return the production coupling image of one cell's linear density."""
    grid_count = operator.grid.node_number
    physical = np.zeros((3, grid_count), dtype=np.float64)
    sign = -1.0 if flip_first else 1.0
    physical[0, cell] = moments[0]
    physical[1, cell] = sign * moments[1]
    physical[2, cell] = sign * moments[2]
    coefficients = operator.coupling_current_moments(
        CellCurrentMoments(
            jnp.asarray(physical[0]),
            jnp.asarray(physical[1]),
            jnp.asarray(physical[2]),
        )
    )
    return np.asarray(operator.current_moment_image(coefficients), dtype=np.float64)


def _field_metrics(error: np.ndarray) -> dict[str, Any]:
    return {
        "max_absolute_error_wb": float(np.max(np.abs(error))),
        "rms_absolute_error_wb": float(np.sqrt(np.mean(error**2))),
    }


def measure_single_cell() -> dict[str, Any]:
    """Measure the first-order coupling against the exact single-cell image."""
    started = perf_counter()
    carrier, source, exact = certificate._case(CASE_NAME)
    machine = certificate._case_machine(CASE_NAME, carrier, exact, REQUESTED_CELLS)
    grid_count = len(machine.node)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    polygons = tuple(
        np.asarray(polygon, dtype=float) for polygon in machine.cell_polygons
    )
    centres = np.asarray(
        machine.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    )
    areas = np.asarray(machine.area, dtype=np.float64)
    plasma = Polygon(certificate._boundary(CASE_NAME, exact))

    cell = _interior_cell_choice(polygons, areas, centres, exact.delta_star, plasma)
    vertices = polygons[cell]
    cell_area = float(areas[cell])
    cell_centre = centres[cell]
    second = second_moments(vertices)

    # linearisation of the exact density over the cell's Duffy nodes
    points, weights = _duffy_rule(vertices)
    density = np.asarray(exact.delta_star(points[:, 0], points[:, 1]), dtype=np.float64)
    density = -density / (MU_0 * points[:, 0])
    linear = _linear_density_fit(points, density, cell_centre)
    linear_moments = _linear_moments(linear, cell_area, second)

    # the exact total flux of this linear density at every mesh node
    linear_density = (
        linear[0]
        + linear[1] * (points[:, 0] - cell_centre[0])
        + linear[2] * (points[:, 1] - cell_centre[1])
    )
    exact_flux = exact_polygon_flux(
        coordinates[:, 0], coordinates[:, 1], points, linear_density, weights
    )

    # production coupling image of the single cell
    operator = oracle_fixture.forward_operator(source, machine)
    image_full = _single_cell_image(operator, cell, linear_moments)
    image_zeroth = _single_cell_image(
        operator, cell, np.asarray((linear_moments[0], 0.0, 0.0))
    )
    image_flipped = _single_cell_image(operator, cell, linear_moments, flip_first=True)

    # variant (iv): the kernel expands about the cell's section centroid and the
    # moments are taken about the same point, so it coincides with (ii).
    kernel_expansion_point = section_centroid(vertices)
    moment_point = cell_centre
    expansion_same_as_moment_point = bool(
        np.allclose(kernel_expansion_point, moment_point, rtol=0.0, atol=2.0e-14)
    )

    variants = {
        "zeroth_only": exact_flux - image_zeroth,
        "zeroth_plus_first": exact_flux - image_full,
        "sign_flipped_first": exact_flux - image_flipped,
        "expansion_point_consistent": exact_flux - image_full,
    }
    errors = {
        name: {"mean": float(np.mean(error)), **_field_metrics(error)}
        for name, error in variants.items()
    }
    grid_errors = {name: error[:grid_count] for name, error in variants.items()}
    grid_metrics = {name: _field_metrics(error) for name, error in grid_errors.items()}

    # The grid node at the cell's own centroid lies INSIDE the source polygon,
    # where the ring Green's function is logarithmically singular and the exact
    # quadrature carries a coarser residual; every other node is outside the
    # cell, where the quadrature is validated to the kernel.  Report both, and
    # use the outside-the-cell set for the order claim.
    source_polygon = Polygon(vertices)
    own_node = np.array(
        [
            index
            for index in range(grid_count)
            if source_polygon.covers(Point(machine.node[index]))
        ],
        dtype=np.intp,
    )
    outside = np.setdiff1d(np.arange(grid_count), own_node, assume_unique=True)
    outside_metrics = {
        name: {
            "max_absolute_error_wb": float(np.max(np.abs(error[outside]))),
            "rms_absolute_error_wb": float(np.sqrt(np.mean(error[outside] ** 2))),
        }
        for name, error in grid_errors.items()
    }
    own_node_list = [int(index) for index in own_node]

    order = sorted(
        outside_metrics, key=lambda name: outside_metrics[name]["rms_absolute_error_wb"]
    )
    exact_variant, exact_rms = (
        order[0],
        outside_metrics[order[0]]["rms_absolute_error_wb"],
    )
    next_variant, next_rms = (
        order[1],
        outside_metrics[order[1]]["rms_absolute_error_wb"],
    )

    record = {
        "case": CASE_NAME,
        "requested_cells": REQUESTED_CELLS,
        "source_revision": _source_revision(),
        "lane": _lane_receipt(),
        "cell": {
            "index": cell,
            "centroid_m": cell_centre,
            "area_m2": cell_area,
            "second_moments_m2": list(second),
        },
        "density": {
            "linearisation_coefficients": {
                "constant_A_m2": linear[0],
                "radial_A_m3": linear[1],
                "vertical_A_m3": linear[2],
            },
            "raw_density_stats_A_m2": {
                "min": float(np.min(density)),
                "max": float(np.max(density)),
                "mean": float(np.mean(density)),
            },
        },
        "prescribed_moments": {
            "zeroth_A": float(linear_moments[0]),
            "radial_A_m": float(linear_moments[1]),
            "vertical_A_m": float(linear_moments[2]),
        },
        "units_and_references": UNITS_AND_REFERENCES,
        "expansion_points": {
            "kernel_expansion_point_m": kernel_expansion_point,
            "moment_point_m": moment_point,
            "expansion_same_as_moment_point": expansion_same_as_moment_point,
            "consequence": (
                "variant (iv) (moments taken about the kernel expansion point) "
                "is identical to (ii) because the points coincide"
            ),
        },
        "image_variants": {
            "contract": (
                "physical moments contracted through coupling_current_moments "
                "into current_moment_image over the polygon kernel blocks"
            ),
            "errors": errors,
            "errors_on_grid_nodes_only": grid_metrics,
            "errors_on_grid_nodes_outside_cell": outside_metrics,
            "grid_nodes_inside_cell": own_node_list,
            "quadrature_residual_at_the_own_node": {
                "note": (
                    "the grid node at the cell's own centroid lies inside the "
                    "source polygon, where the ring Green's function is "
                    "logarithmically singular and the fixed Duffy rule carries "
                    "a coarser residual; every other node is outside the cell "
                    "where the quadrature is validated to the kernel"
                ),
                "own_node_absolute_error_wb": {
                    name: float(abs(error[own_node][0])) if own_node.size else None
                    for name, error in grid_errors.items()
                },
            },
            "exact_to_second_order_in_cell_size": {
                "variant": exact_variant,
                "rms_error_wb": exact_rms,
                "next_best": next_variant,
                "next_rms_error_wb": next_rms,
                "criterion": (
                    "the linear density is reproduced exactly by its zeroth and "
                    "first moments, so the variant whose image matches the exact "
                    "quadrature to residual order (measured on the nodes outside "
                    "the source cell) is exact to second order in the cell size"
                ),
            },
        },
        "metrics_total_flux_signal_wb": float(np.max(np.abs(exact_flux))),
        "plot_data": {
            "node_rz_m": np.asarray(machine.node),
            "wall_rz_m": np.asarray(machine.wall_node),
            "exact_flux_wb": exact_flux[:grid_count],
            "error_fields_wb": {
                name: value[:grid_count] for name, value in variants.items()
            },
            "cell_polygon": vertices,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    _write_json(PART_ONE, record)
    print(
        f"SINGLE_CELL_RESULT cell={cell} "
        f"exact_variant='{exact_variant}' rms={exact_rms:.3e} "
        f"next='{next_variant}' next_rms={next_rms:.3e}",
        flush=True,
    )
    return record


# ---------------------------------------------------------------------------
# part two: the weak -110 frozen currents, four variants
# ---------------------------------------------------------------------------


def _region_loop(
    vertices: np.ndarray, plasma: Polygon
) -> tuple[np.ndarray, float] | None:
    """Return the plasma-region exterior loop and area of one cell."""
    region = Polygon(vertices).intersection(plasma)
    tolerance = 2.0e-11 * max(float(Polygon(vertices).area), 1.0)
    if region.is_empty or region.area <= tolerance:
        return None
    if isinstance(region, MultiPolygon):
        region = unary_union(region)
        if region.geom_type == "MultiPolygon":
            region = max(region.geoms, key=lambda item: item.area)
    loop = np.asarray(region.exterior.coords, dtype=np.float64)
    if len(loop) < 4:
        return None
    return loop[:-1], float(region.area)


def _region_shift(
    moments: np.ndarray, cell: int, displacement: np.ndarray
) -> tuple[float, float]:
    """Shift one cell's full-cell-centroid moments to the region centroid."""
    radial = moments[cell, 1] + displacement[0] * moments[cell, 0]
    vertical = moments[cell, 2] + displacement[1] * moments[cell, 0]
    return float(radial), float(vertical)


def _region_conversion(
    radial: float, vertical: float, second: tuple[float, float, float]
) -> tuple[float, float]:
    """Convert region first moments to region-polygon density-gradient scales.

    The same inversion ``coupling_current_moments`` applies, but with the
    region polygon's own area-normalised second moments instead of the full
    cell's, so the coefficients are the ones the region block contraction
    expects.
    """
    irr, izz, irz = second
    determinant = irr * izz - irz * irz
    radial_coefficient = (izz * radial - irz * vertical) / determinant
    vertical_coefficient = (irr * vertical - irz * radial) / determinant
    return float(radial_coefficient), float(vertical_coefficient)


def measure_frozen_weak110() -> dict[str, Any]:
    """Rebuild gate B's weak -110 frozen image with the four variants."""
    started = perf_counter()
    configure_dtypes()
    carrier, source, exact = certificate._case(CASE_NAME)
    machine = certificate._case_machine(CASE_NAME, carrier, exact, REQUESTED_CELLS)
    grid_count = len(machine.node)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(CASE_NAME, exact, coordinates)
    oracle_grid = oracle_state[:grid_count]

    # Reuse gate B's certificate-anchored operator: the exterior bakes whatever
    # the certificate moment image does not explain, and the frozen image adds
    # the contracted exact moments to that same exterior.
    empty_operator = oracle_fixture.forward_operator(source, machine)
    certificate_moments = oracle_fixture.exact_current_moments(
        source, empty_operator, oracle_state
    )
    certificate_coefficients = empty_operator.coupling_current_moments(
        certificate_moments
    )
    production_internal = oracle_fixture._internal_flux_image(
        empty_operator, certificate_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source, machine, oracle_state - production_internal
    )
    external = np.asarray(operator.external(), dtype=np.float64)

    moments, _region_area, kinds = _frozen_moments(CASE_NAME, exact, machine)
    span = float(np.max(oracle_grid) - np.min(oracle_grid))

    def as_built_image(physical: np.ndarray, *, flip_first: bool = False) -> np.ndarray:
        sign = -1.0 if flip_first else 1.0
        flipped = physical.copy()
        flipped[:, 1] = sign * physical[:, 1]
        flipped[:, 2] = sign * physical[:, 2]
        coefficients = operator.coupling_current_moments(_physical_moments(flipped))
        return external + np.asarray(
            operator.current_moment_image(coefficients), dtype=np.float64
        )

    image_zeroth = as_built_image(
        np.column_stack((moments[:, 0], np.zeros(len(moments)), np.zeros(len(moments))))
    )
    image_full = as_built_image(moments)
    image_flipped = as_built_image(moments, flip_first=True)

    # Variant (iv): expansion point moved to each cut cell's plasma region
    # centroid, moments shifted there and kernel blocks rebuilt over the region
    # polygon.  Interior cells are untouched (their region is the full cell).
    polygons = tuple(
        np.asarray(polygon, dtype=float) for polygon in machine.cell_polygons
    )
    centres = np.asarray(
        machine.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    )
    boundary = certificate._boundary(CASE_NAME, exact)
    plasma_polygon = Polygon(boundary)
    cut = np.flatnonzero(kinds == 2)

    targets_r = np.concatenate(
        (
            machine.node[:, 0],
            machine.wall_node[:, 0],
            machine.sample_coordinates[:, 0],
        )
    )
    targets_z = np.concatenate(
        (
            machine.node[:, 1],
            machine.wall_node[:, 1],
            machine.sample_coordinates[:, 1],
        )
    )

    rebuilt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cell in cut:
        region = _region_loop(polygons[cell], plasma_polygon)
        if region is None:
            continue
        loop, _region_area = region
        region_centroid = section_centroid(loop)
        if np.all(np.isfinite(region_centroid)):
            rebuilt[cell] = (loop, region_centroid)

    rebuild_blocks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    # Serial evaluation: forty-odd cut cells over ~770 targets measure at about
    # thirty seconds, so the parallel executor's speedup is not worth the
    # worker-spawn fragility on a shared allocation.
    if rebuilt:
        for cell in sorted(rebuilt):
            loop, region_centroid = rebuilt[cell]
            blocks = polygon_analytic_flux_moments(
                targets_r, targets_z, loop, expansion_point=region_centroid
            )
            rebuild_blocks[cell] = tuple(
                np.asarray(block).reshape(-1) for block in blocks
            )

    # image (iv): as-built operator contraction for the untouched cells (their
    # moments zeroed on the rebuilt cells so the machine blocks cannot
    # contribute there), plus the hand-contracted region blocks.
    untouched_moments = moments.copy()
    for cell in rebuilt:
        untouched_moments[cell] = 0.0
    untouched_coefficients = operator.coupling_current_moments(
        _physical_moments(untouched_moments)
    )
    image_region = external + np.asarray(
        operator.current_moment_image(untouched_coefficients), dtype=np.float64
    )
    for cell in sorted(rebuilt):
        loop, region_centroid = rebuilt[cell]
        g0, g1, g2 = rebuild_blocks[cell]
        radial, vertical = _region_shift(moments, cell, centres[cell] - region_centroid)
        radial_coefficient, vertical_coefficient = _region_conversion(
            radial, vertical, second_moments(loop)
        )
        image_region = image_region + (
            moments[cell, 0] * g0 + radial_coefficient * g1 + vertical_coefficient * g2
        )

    variants = {
        "zeroth_only": image_zeroth[:grid_count] - oracle_grid,
        "zeroth_plus_first": image_full[:grid_count] - oracle_grid,
        "sign_flipped_first": image_flipped[:grid_count] - oracle_grid,
        "expansion_point_moved": image_region[:grid_count] - oracle_grid,
    }

    def metrics(error: np.ndarray) -> dict[str, Any]:
        absolute = np.abs(error)
        return {
            "max_absolute_error_wb": float(np.max(absolute)),
            "rms_absolute_error_wb": float(np.sqrt(np.mean(error**2))),
            "max_error_over_span": float(np.max(absolute) / span),
            "rms_error_over_span": float(np.sqrt(np.mean(error**2)) / span),
        }

    errors = {name: metrics(error) for name, error in variants.items()}
    smoother = min(
        ("sign_flipped_first", "expansion_point_moved"),
        key=lambda name: errors[name]["rms_error_over_span"],
    )

    # Which cell classes carry the added first-order error?  The single-cell
    # control proves the kernel's dipole term is exact on a full cell, so the
    # frozen-image degradation must be attributed to a cell class.
    interior_cells = np.flatnonzero(kinds == 1)

    def class_image(class_mask: np.ndarray) -> np.ndarray:
        masked = moments.copy()
        others = np.flatnonzero(~class_mask)
        masked[others, 1] = 0.0
        masked[others, 2] = 0.0
        coefficients = operator.coupling_current_moments(_physical_moments(masked))
        return external + np.asarray(
            operator.current_moment_image(coefficients), dtype=np.float64
        )

    interior_mask = np.zeros(grid_count, dtype=bool)
    interior_mask[interior_cells] = True
    cut_mask = np.zeros(grid_count, dtype=bool)
    cut_mask[cut] = True
    class_errors = {
        "first_order_interior_cells_only": metrics(
            class_image(interior_mask)[:grid_count] - oracle_grid
        ),
        "first_order_cut_cells_only": metrics(
            class_image(cut_mask)[:grid_count] - oracle_grid
        ),
    }

    record = {
        "case": CASE_NAME,
        "requested_cells": REQUESTED_CELLS,
        "source_revision": _source_revision(),
        "lane": _lane_receipt(),
        "gate_b_reference": {
            "zeroth_only_rms_over_span": 0.0057,
            "first_order_added_rms_over_span": 0.0106,
        },
        "flux_span_wb": span,
        "cell_census": {
            "interior": int(np.count_nonzero(kinds == 1)),
            "cut": int(np.count_nonzero(kinds == 2)),
            "exterior": int(np.count_nonzero(kinds == 0)),
        },
        "region_rebuild": {
            "cut_cells": sorted(int(cell) for cell in rebuilt),
            "count": len(rebuilt),
        },
        "image_variants": {
            "contract": (
                "frozen exact per-cell moments through coupling_current_moments "
                "into current_moment_image, plus the certificate row's external; "
                "variant (iv) shifts the moments and kernel blocks to the "
                "plasma-region centroid of each cut cell"
            ),
            "errors": errors,
            "smoother_of_iii_iv": smoother,
            "added_first_order_error_by_cell_class": class_errors,
            "class_decomposition_note": (
                "the single-cell control proves the interior-cell dipole is "
                "exact, so the class errors partition the frozen-image anomaly: "
                "the cut-cell first-order term carries it"
            ),
        },
        "units_and_references": UNITS_AND_REFERENCES,
        "plot_data": {
            "node_rz_m": np.asarray(machine.node),
            "wall_rz_m": np.asarray(machine.wall_node),
            "error_fields_wb": {
                name: value[:grid_count] for name, value in variants.items()
            },
            "error_fields_metrics": errors,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    _write_json(PART_TWO, record)
    summary = " ".join(
        f"{name}={errors[name]['rms_error_over_span']:.5g}" for name in errors
    )
    class_summary = " ".join(
        f"{name}={value['rms_error_over_span']:.5g}"
        for name, value in class_errors.items()
    )
    print(
        f"FROZEN_WEAK110_RESULT {summary} smoother='{smoother}' "
        f"classes={class_summary}",
        flush=True,
    )
    return record


# ---------------------------------------------------------------------------
# figures and receipt
# ---------------------------------------------------------------------------


def _shared_error_levels(fields: dict[str, np.ndarray]) -> np.ndarray:
    absolute = [
        np.abs(np.asarray(field, dtype=np.float64)) for field in fields.values()
    ]
    values = np.concatenate(absolute)
    nonzero = values[values > 0.0]
    if nonzero.size == 0:
        return np.asarray([np.finfo(float).tiny])
    lower = max(float(np.percentile(nonzero, 10.0)), np.finfo(float).tiny)
    upper = float(np.max(nonzero))
    return np.geomspace(lower, upper, 8) if upper > lower else np.asarray([upper])


def _draw_single_cell_figure(record: dict[str, Any], output: Path) -> dict[str, Any]:
    """Draw the exact single-cell flux and the four variants' error."""
    plot = record["plot_data"]
    node = np.asarray(plot["node_rz_m"], dtype=np.float64)
    wall = np.asarray(plot["wall_rz_m"], dtype=np.float64)
    exact = np.asarray(plot["exact_flux_wb"], dtype=np.float64)
    error_fields = plot["error_fields_wb"]
    error_levels = _shared_error_levels(error_fields)

    names = (
        "zeroth_only",
        "zeroth_plus_first",
        "sign_flipped_first",
        "expansion_point_consistent",
    )
    figure, axes = plt.subplots(1, 5, figsize=(15.0, 3.4), constrained_layout=True)
    axis = poloidal_axes(axes[0])
    levels = np.linspace(float(exact.min()), float(exact.max()), 11)[1:-1]
    axis.tricontour(
        node[:, 0], node[:, 1], exact, levels=levels, colors="royalblue", linewidths=0.6
    )
    poloidal.draw_wall(axis, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
    axis.set_title("exact linear-density\nflux", fontsize=7)
    for column, name in enumerate(names, start=1):
        axis = poloidal_axes(axes[column])
        error = np.asarray(error_fields[name], dtype=np.float64)
        axis.tricontour(
            node[:, 0],
            node[:, 1],
            np.maximum(np.abs(error), error_levels[0]),
            levels=error_levels,
            colors="firebrick",
            linewidths=0.8,
        )
        poloidal.draw_wall(axis, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
        maximum = float(np.max(np.abs(error)))
        axis.set_title(f"{name.replace('_', ' ')}\nmax {maximum:.2e} Wb", fontsize=7)
    centroid = record["cell"]["centroid_m"]
    figure.suptitle(
        "Single cell: exact flux of a prescribed linear density, and the four "
        "coupling variants' absolute error on shared levels\n"
        f"cell {record['cell']['index']} at R, Z = {centroid[0]:.4f}, "
        f"{centroid[1]:.4f} m; "
        "red: |coupling image - exact quadrature|; wall drawn",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return {
        "path": str(output.relative_to(ROOT)),
        "project_src": f"{PROJECT_SRC_PREFIX}/{output.name}",
        "shared_absolute_error_levels_wb": [float(value) for value in error_levels],
    }


def _draw_frozen_figure(record: dict[str, Any], output: Path) -> dict[str, Any]:
    """Draw the weak -110 error fields for (i), (ii) and the best of (iii)/(iv)."""
    plot = record["plot_data"]
    node = np.asarray(plot["node_rz_m"], dtype=np.float64)
    wall = np.asarray(plot["wall_rz_m"], dtype=np.float64)
    fields = plot["error_fields_wb"]
    metrics = plot["error_fields_metrics"]
    smoother = record["image_variants"]["smoother_of_iii_iv"]
    panel_names = ("zeroth_only", "zeroth_plus_first", smoother)
    error_levels = _shared_error_levels({name: fields[name] for name in panel_names})
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), constrained_layout=True)
    for column, name in enumerate(panel_names):
        axis = poloidal_axes(axes[column])
        error = np.asarray(fields[name], dtype=np.float64)
        axis.tricontour(
            node[:, 0],
            node[:, 1],
            np.maximum(np.abs(error), error_levels[0]),
            levels=error_levels,
            colors="firebrick",
            linewidths=0.8,
        )
        poloidal.draw_wall(axis, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
        metric = metrics[name]
        axis.set_title(
            f"{name.replace('_', ' ')}\n"
            f"rms {metric['rms_error_over_span']:.4f}  "
            f"max {metric['max_error_over_span']:.4f} of span",
            fontsize=7,
        )
    figure.suptitle(
        "Weak -110 frozen exact currents: coupling error vs the oracle flux\n"
        "red: |image - oracle| on shared levels; wall drawn; the panel order is "
        "zeroth only, zeroth+first, best of sign-flipped / region-centred",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return {
        "path": str(output.relative_to(ROOT)),
        "project_src": f"{PROJECT_SRC_PREFIX}/{output.name}",
        "best_of_iii_iv": smoother,
        "shared_absolute_error_levels_wb": [float(value) for value in error_levels],
    }


def aggregate() -> dict[str, Any]:
    """Combine the two parts and draw the two figures."""
    part_one = json.loads(PART_ONE.read_text(encoding="utf-8"))
    part_two = json.loads(PART_TWO.read_text(encoding="utf-8"))
    figure_one = _draw_single_cell_figure(part_one, OUTPUT_ROOT / "single-cell.svg")
    figure_two = _draw_frozen_figure(part_two, OUTPUT_ROOT / "weak-110-errors.svg")
    receipt = {
        "schema": "nova.first-order-coupling-discriminator.v1",
        "completed": True,
        "measurement_driver_revision": part_one["source_revision"],
        "figures": {"single_cell": figure_one, "weak_110_frozen": figure_two},
        "single_cell": part_one,
        "weak_110_frozen": part_two,
    }
    _write_json(RECEIPT, receipt)
    return receipt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the first-order coupling discriminator requires x64")
    if jax.default_backend() != "cpu":
        raise RuntimeError("the first-order coupling discriminator requires CPU")
    arguments = _parse()
    if arguments.run_all:
        measure_single_cell()
        measure_frozen_weak110()
        aggregate()
        print("FIRST_ORDER_DISCRIMINATOR_RUNALL_EXIT=0", flush=True)
        return
    if arguments.aggregate:
        aggregate()
        print("FIRST_ORDER_DISCRIMINATOR_AGGREGATE_EXIT=0", flush=True)
        return
    raise SystemExit("--run-all or --aggregate is required")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
