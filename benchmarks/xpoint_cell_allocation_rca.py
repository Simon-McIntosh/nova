"""Root-cause production chord-clip current allocation in multi-crossing cells.

The production support partition in chord mode (``set_support_clip_mode`` on
:mod:`nova.equilibrium.forward_operator`) hands every profile-participating
cell its FULL atomic hexagon, never clipping against the separatrix at all: a
cell whose analytic separatrix crosses its boundary more than twice is either
fully attributed or not attributed, with no partial region.  In an X-point
cell, where the analytic separatrix self-crosses at the saddle, this
single-polygon chord allocation is at its furthest from the analytic core-side
region.  The committed ``exact`` mode replaces the chord with the
spline-traced boundary and makes every cut cell participate.

This driver measures, per row and per boundary-cut cell:

* the analytic separatrix crossing count across the cell boundary,
* the analytic core-side current moments (exact density over the exact
  plasma-intersection of the cell, adapted to ~1e-12),
* the chord attribution (chord-mode support, the prior committed clip),
* the exact attribution (the committed spline-chain clip),
* the three relative moment errors of each clip against the analytic integral.

The X-point cell of the single-null certificate case and cells 101 and 102 of
the weak -110 case are itemised geometries: the analytic X-point position, the
chord support endpoints, the exact chain vertices, and the nearest
topology-read saddle candidate with its containment verdict and distance.

A self-test verifies the analytic density against the production density for
both case families and reproduces three banked cut-cell exact moments from the
cut-cell-moments receipt before any row is measured.

One invocation measures one row; ``--aggregate`` renders the figures and writes
the shared receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.path as mpl_path
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from benchmarks import solovev_certificate as certificate
from nova.equilibrium.analytic_single_null import CerfonFreidbergSingleNull
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture

MU_0 = 4.0e-7 * np.pi

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/uniform-cell-clip-and-coupling/xpoint-cell-rca"
RECEIPT = OUTPUT_ROOT / "receipt.json"
CASE_REQUESTS = (
    ("diverted-single-null", -110),
    ("diverted-single-null", -300),
    ("weak-rotation-reactor-static", -110),
)
CHORD_REVERTED_CELLS = (101, 102)  # the two cells the discriminator reverts
ADAPTIVE_RELATIVE_TOLERANCE = 5.0e-13
ADAPTIVE_ABS_ERROR = 1.0e-12
FIXED_INTERIOR_POINTS = 9  # Gauss points per axis on the fixed interior rule
BANKED_WEAK_110_CUT_RECEIPT = (
    ROOT / "docs/figures/uniform-cell-clip-and-coupling/cut-cell-moments/receipt.json"
)
_ORDER_SLICES = {
    "zeroth": slice(0, 1),
    "first": slice(1, 3),
    "second": slice(3, 6),
}


# ---------------------------------------------------------------------------
# serialisation and lane helpers
# ---------------------------------------------------------------------------


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
        "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _array_digest(values: np.ndarray) -> str:
    packed = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _case_key(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-{abs(requested_cells)}"


def _part_path(case_name: str, requested_cells: int) -> Path:
    return OUTPUT_ROOT / "parts" / f"{_case_key(case_name, requested_cells)}.json"


# ---------------------------------------------------------------------------
# analytic separatrix geometry
# ---------------------------------------------------------------------------


def _is_diverted(case: Any) -> bool:
    """Return whether the exact reference is the Cerfon-Freidberg single null."""
    return isinstance(case, CerfonFreidbergSingleNull)


def _analytic_boundary_level(exact: Any) -> float:
    """Return the physical per-radian flux of the analytic separatrix."""
    if _is_diverted(exact):
        return float(np.asarray(exact.flux(certificate.X_POINT_M[None, :]))[0])
    return 0.0


def _closed_plasma_polygon(case_name: str, exact: Any) -> np.ndarray:
    """Return the closed reference-boundary polyline around the magnetic axis."""
    if _is_diverted(exact):
        return np.asarray(exact.separatrix(1441), dtype=np.float64)
    return certificate._boundary(case_name, exact)


def _analytic_separatrix_branches(
    case_name: str, exact: Any
) -> dict[str, list[np.ndarray]]:
    """Return the analytic separatrix branch polylines at the boundary level.

    The limited static family contributes one closed boundary.  The diverted
    case separates its closed core lobe from the two open divertor legs with
    the certificate's contour extraction and lobe selection.
    """
    if not _is_diverted(exact):
        return {
            "core_lobe": certificate._boundary(case_name, exact),
            "legs": [],
        }
    boundary = np.asarray(certificate._boundary(case_name, exact), dtype=np.float64)
    wall = np.asarray(certificate._diverted_wall(exact), dtype=np.float64)
    pitch = float(
        np.sqrt(
            np.median(
                np.asarray(
                    oracle_fixture.cached_machine(
                        oracle_fixture.analytic_case(),
                        -110,
                        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
                    ).area,
                    dtype=np.float64,
                )
            )
        )
    )
    radial = np.linspace(
        min(
            float(np.min(boundary[:, 0])),
            float(np.max(boundary[:, 0])),
            float(np.min(wall[:, 0])),
        )
        - 2.0 * pitch,
        max(
            float(np.min(boundary[:, 0])),
            float(np.max(boundary[:, 0])),
            float(np.max(wall[:, 0])),
        )
        + 2.0 * pitch,
        241,
    )
    vertical = np.linspace(
        min(float(np.min(boundary[:, 1])), float(np.min(wall[:, 1]))) - 2.0 * pitch,
        max(float(np.max(boundary[:, 1])), float(np.max(wall[:, 1]))) + 2.0 * pitch,
        241,
    )
    radial_grid, vertical_grid = np.meshgrid(radial, vertical)
    coordinates = np.column_stack((radial_grid.ravel(), vertical_grid.ravel()))
    field = np.asarray(exact.flux(coordinates), dtype=np.float64).reshape(
        vertical_grid.shape
    )
    boundary_level = _analytic_boundary_level(exact)
    figure, axis = plt.subplots()
    contour_set = axis.contour(radial, vertical, field, levels=[boundary_level])
    plt.close(figure)
    components = [
        np.asarray(component, dtype=np.float64)
        for component in contour_set.allsegs[0]
        if len(component) >= 3
    ]
    if not components:
        raise RuntimeError("analytic boundary contour extraction returned no component")
    grid_spacing = float(np.hypot(radial[1] - radial[0], vertical[1] - vertical[0]))
    selected, divertor_legs, _index = certificate._single_null_core_lobe(
        components,
        magnetic_axis=np.asarray(certificate.AXIS_M),
        x_point=np.asarray(certificate.X_POINT_M),
        grid_spacing=grid_spacing,
    )
    return {
        "core_lobe": np.asarray(selected, dtype=np.float64),
        "legs": [np.asarray(leg, dtype=np.float64) for leg in divertor_legs],
    }


# ---------------------------------------------------------------------------
# exact density and analytic moment integrals
# ---------------------------------------------------------------------------


def _flux_at(case: Any, points: np.ndarray) -> np.ndarray:
    """Evaluate the exact per-radian flux at ``(R, Z)`` points."""
    points = np.asarray(points, dtype=np.float64)
    if _is_diverted(case):
        return np.asarray(case.flux(points), dtype=np.float64)
    return np.asarray(case.flux(points[:, 0], points[:, 1]), dtype=np.float64)


def _exact_density(case: Any, points: np.ndarray) -> np.ndarray:
    """Return the exact toroidal current density at physical ``(R, Z)`` points.

    Both families are exact Grad-Shafranov solutions, so ``J_phi = -DeltaStar
    (psi) / (mu0 R)`` with the strong-form source of each reference.  The
    self-test asserts both agree with the production density before any row is
    measured.
    """
    points = np.asarray(points, dtype=np.float64)
    if _is_diverted(case):
        source = np.asarray(case.grad_shafranov_source(points), dtype=np.float64)
    else:
        source = np.asarray(
            case.delta_star(points[:, 0], points[:, 1]), dtype=np.float64
        )
    return -source / (MU_0 * points[:, 0])


def _polygon_z_intervals(
    vertices: np.ndarray, radius: float
) -> list[tuple[float, float]]:
    """Return the vertical intervals of a polygon at one radius value."""
    crossings: list[float] = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        span = second[0] - first[0]
        if span == 0.0:
            continue
        fraction = (radius - first[0]) / span
        if 0.0 < fraction < 1.0:
            crossings.append(float(first[1] + fraction * (second[1] - first[1])))
    crossings.sort()
    return [
        (lower, upper)
        for lower, upper in zip(crossings[0::2], crossings[1::2])
        if upper > lower
    ]


def _region_moment_integrals(
    region_vertices: np.ndarray,
    centre: np.ndarray,
    case: Any,
    *,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the exact density moments over one region polygon.

    Nested adaptive quadrature: inner vertical integral of the exact density
    times the moment monomial, outer radial integral split at every region
    vertex.  Returns (six density moments about ``centre``, error bounds).
    """
    region_vertices = np.asarray(region_vertices, dtype=np.float64)
    lower = float(np.min(region_vertices[:, 0]))
    upper = float(np.max(region_vertices[:, 0]))
    if upper <= lower:
        return np.zeros(6), np.zeros(6)
    breaks = sorted(
        {lower, upper}
        | {float(value) for value in region_vertices[:, 0] if lower < value < upper}
    )
    moment_values = np.zeros(6)
    moment_errors = np.zeros(6)
    powers = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))

    def inner(radius: float, radial_power: int, vertical_power: int) -> float:
        total = 0.0
        for z_lower, z_upper in _polygon_z_intervals(region_vertices, radius):
            if z_upper <= z_lower:
                continue

            def vertical_integrand(z: float) -> float:
                density = _exact_density(case, np.asarray([[radius, z]]))[0]
                return density * (z - centre[1]) ** vertical_power

            value, _error = quad(
                vertical_integrand,
                z_lower,
                z_upper,
                epsabs=ADAPTIVE_ABS_ERROR,
                epsrel=relative_tolerance,
                limit=200,
            )
            total += value
        return total * (radius - centre[0]) ** radial_power

    for first, second in zip(breaks, breaks[1:]):
        if second <= first:
            continue
        for index, (radial_power, vertical_power) in enumerate(powers):
            value, error = quad(
                lambda radius: inner(radius, radial_power, vertical_power),
                first,
                second,
                epsabs=ADAPTIVE_ABS_ERROR,
                epsrel=relative_tolerance,
                limit=300,
            )
            moment_values[index] += value
            moment_errors[index] += error
    return moment_values, moment_errors


def _duffy_rule(
    vertices: np.ndarray, points_per_axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed product rule over a convex polygon by vertex fan."""
    nodes, weights = np.polynomial.legendre.leggauss(points_per_axis)
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


def _fixed_cell_moments(
    case: Any, vertices: np.ndarray, centre: np.ndarray
) -> np.ndarray:
    """Return exact density moments over a full convex cell by a fixed rule."""
    points, weights = _duffy_rule(np.asarray(vertices), FIXED_INTERIOR_POINTS)
    density = _exact_density(case, points)
    offset = points - centre
    weighted = weights * density
    return np.asarray(
        (
            np.sum(weighted),
            np.sum(weighted * offset[:, 0]),
            np.sum(weighted * offset[:, 1]),
            np.sum(weighted * offset[:, 0] ** 2),
            np.sum(weighted * offset[:, 0] * offset[:, 1]),
            np.sum(weighted * offset[:, 1] ** 2),
        )
    )


def _region_loop_vertices(region: Polygon) -> np.ndarray | None:
    """Return the ordered outer boundary loop of a region polygon."""
    if isinstance(region, MultiPolygon):
        region = unary_union(region)
        if region.geom_type == "MultiPolygon":
            region = max(region.geoms, key=lambda item: item.area)
    if region.is_empty or region.area <= 0.0:
        return None
    loop = np.asarray(region.exterior.coords, dtype=np.float64)
    if len(loop) < 4:
        return None
    return loop[:-1]


def _exact_cell_integrals(
    case: Any,
    polygon: np.ndarray,
    centre: np.ndarray,
    plasma_polygon: Polygon,
    cell_area: float,
) -> tuple[int, float, np.ndarray, np.ndarray | None]:
    """Classify a cell and return its exact core-side moments.

    Returns ``(kind, region_area, moments, region_loop)`` where ``kind`` is
    0 exterior, 1 interior (full cell), 2 boundary-cut.
    """
    cell_shape = Polygon(polygon)
    region = cell_shape.intersection(plasma_polygon)
    tolerance = 2.0e-11 * max(cell_area, 1.0)
    if region.is_empty or region.area <= tolerance:
        return 0, 0.0, np.zeros(6), None
    if abs(region.area - cell_area) <= tolerance:
        moments = _fixed_cell_moments(case, polygon, centre)
        return 1, cell_area, moments, np.asarray(polygon, dtype=np.float64)
    loop = _region_loop_vertices(region)
    if loop is None:
        return 0, 0.0, np.zeros(6), None
    region_vertices = np.asarray(loop, dtype=np.float64)
    moments, _errors = _region_moment_integrals(
        region_vertices,
        centre,
        case,
        relative_tolerance=ADAPTIVE_RELATIVE_TOLERANCE,
    )
    return 2, float(region.area), moments, loop


# ---------------------------------------------------------------------------
# separatrix crossing count
# ---------------------------------------------------------------------------


def _separatrix_crossing_count(
    case: Any, boundary_level: float, polygon: np.ndarray
) -> int:
    """Count analytic separatrix crossings over a cell's boundary edges.

    Each cell edge is subdivided and the analytic per-radian flux sampled, so
    a branch entering and leaving counts twice: the number of sign changes of
    ``flux - boundary_level`` around the closed cell.
    """
    count = 0
    vertices = np.asarray(polygon, dtype=np.float64)
    subdivisions = 96
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        parameters = np.linspace(0.0, 1.0, subdivisions + 1)
        points = first[None, :] + parameters[:, None] * (second - first)[None, :]
        flux = _flux_at(case, points)
        signs = np.sign(flux - boundary_level)
        count += int(np.count_nonzero(np.diff(signs) != 0))
    return count


# ---------------------------------------------------------------------------
# production support attribution
# ---------------------------------------------------------------------------


def _production_flux_coefficients(
    operator: Any, centroid_flux: np.ndarray, sample_flux: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the exact coefficient arrays used by the production route."""

    cell_count = len(operator.grid.coordinate)
    coefficient = np.full((cell_count, 6), np.nan, dtype=np.float64)
    centre = np.full((cell_count, 2), np.nan, dtype=np.float64)
    scale = np.full((cell_count, 2), np.nan, dtype=np.float64)
    value_pool = np.concatenate((centroid_flux, sample_flux))
    for stencil in operator._support_moment_stencils:
        ring = np.asarray(stencil.ring_centre, dtype=np.intp)
        gathered = value_pool[np.asarray(stencil.ring_gather_index, dtype=np.intp)]
        coefficient[ring] = np.einsum(
            "rps,rs->rp", np.asarray(stencil.ring_flux_weight), gathered
        )
        centre[ring] = np.asarray(stencil.ring_sampling_centre)
        scale[ring] = np.asarray(stencil.ring_coordinate_scale)
    if np.any(~np.isfinite(coefficient)):
        missing = np.flatnonzero(~np.all(np.isfinite(coefficient), axis=1))
        raise RuntimeError(f"production flux coefficients missing for cells {missing}")
    return coefficient, centre, scale


def _quadratic_values(
    points: np.ndarray,
    coefficient: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    local = (points - centre) / scale
    radial, vertical = local[:, 0], local[:, 1]
    design = np.column_stack(
        (
            np.ones(len(points)),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        )
    )
    return design @ coefficient


def _mode_production_integrals(
    operator: Any, state: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Return per-cell production moments and the support under one clip mode.

    Mirrors the prior cut-cell-moments measurement: the ring-quadratic fitted
    flux is evaluated at the fixed Duffy nodes over each cell's support polygon
    and the source density integrated there.
    """
    set_support_clip_mode(mode)
    masks, _topology, sample_flux, support = operator._support_partition(
        jnp.asarray(state)
    )
    centroid_flux = np.asarray(masks.psi_norm, dtype=np.float64)
    sample_flux_array = np.asarray(sample_flux, dtype=np.float64)
    coefficient, sampling_centre, coordinate_scale = _production_flux_coefficients(
        operator, centroid_flux, sample_flux_array
    )
    counts = np.asarray(support.vertex_count, dtype=np.intp)
    support_vertices = np.asarray(support.support_vertices, dtype=np.float64)
    moment_centre = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    values = np.zeros((len(counts), 6), dtype=np.float64)
    for cell, count in enumerate(counts):
        if count < 3:
            continue
        points, weights = _duffy_rule(support_vertices[cell, :count], 8)
        fitted_flux = _quadratic_values(
            points,
            coefficient[cell],
            sampling_centre[cell],
            coordinate_scale[cell],
        )
        density = np.asarray(
            operator.source.core.current_density(
                jnp.asarray(points[:, 0]), jnp.asarray(fitted_flux)
            ),
            dtype=np.float64,
        )
        offset = points - moment_centre[cell]
        weighted = weights * density
        values[cell] = (
            np.sum(weighted),
            np.sum(weighted * offset[:, 0]),
            np.sum(weighted * offset[:, 1]),
            np.sum(weighted * offset[:, 0] ** 2),
            np.sum(weighted * offset[:, 0] * offset[:, 1]),
            np.sum(weighted * offset[:, 1] ** 2),
        )
    participation = np.asarray(masks.profile_participation, dtype=bool)
    return values, participation, support


def _support_loop(support: Any, cell: int) -> np.ndarray | None:
    count = int(np.asarray(support.vertex_count, dtype=np.intp)[cell])
    if count < 3:
        return None
    return np.asarray(support.support_vertices[cell, :count], dtype=np.float64)


# ---------------------------------------------------------------------------
# topology-read candidate table
# ---------------------------------------------------------------------------


def _candidate_table(operator: Any, physical: np.ndarray) -> dict[str, Any]:
    """Return the retained O and X candidate rows the production read emits."""
    grid_flux = operator.topology.split_flux_map(jnp.asarray(physical))[0]
    status = jax.device_get(
        operator._fixed_design_topology.grid.candidate_table_status(grid_flux)
    )
    candidates = np.asarray(status["retained_candidate"], dtype=np.float64)
    valid = np.asarray(status["retained_valid"], dtype=bool)
    return {
        "o_candidates": candidates[0][valid[0]],
        "x_candidates": candidates[1][valid[1]],
        "o_candidate_count": int(np.asarray(status["candidate_count"])[0]),
        "x_candidate_count": int(np.asarray(status["candidate_count"])[1]),
    }


def _production_read(operator: Any, state: np.ndarray) -> dict[str, Any]:
    """Run the production topology read on the analytic state and record it."""
    physical = jnp.asarray(state)
    try:
        _masks, topology = operator.read(physical)
    except Exception as error:  # noqa: BLE001 - the RCA records the refusal
        return {
            "read_status": "refused",
            "class": None,
            "axis_rz_m": None,
            "x_point_rz_m": None,
            "boundary_flux_wb": None,
            "axis_flux_wb": None,
            "exception_text": str(error)[:300],
        }
    x_point = np.asarray(topology.x_point, dtype=np.float64)
    polarity = (
        1.0 if float(topology.boundary_flux) <= float(topology.axis_flux) else -1.0
    )
    margins = certificate.candidate_flux_margins(operator, physical, polarity=polarity)
    return {
        "read_status": "qualified",
        "class": "diverted" if bool(topology.diverted) else "limited",
        "axis_rz_m": np.asarray(topology.axis, dtype=np.float64).tolist(),
        "x_point_rz_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
        "boundary_flux_wb": float(topology.boundary_flux),
        "axis_flux_wb": float(topology.axis_flux),
        "flux_span_wb": float(topology.flux_span),
        "o_candidate_count": margins["o_candidate_count"],
        "x_candidate_count": margins["x_candidate_count"],
        "x_second_best_flux_margin_wb": margins["x_second_best_flux_margin_wb"],
        "exception_text": None,
    }


# ---------------------------------------------------------------------------
# analytic self-test
# ---------------------------------------------------------------------------


def _analytic_self_test() -> dict[str, Any]:
    """Assert density conventions and reproduce banked cut-cell exact moments."""
    records: dict[str, Any] = {"density_checks": [], "banked_cells": []}
    static = certificate.reference_cases()["weak-rotation-reactor"].static_limit()
    diverted = CerfonFreidbergSingleNull()
    for name, exact in (("static", static), ("diverted", diverted)):
        source = (
            certificate._diverted_source(diverted) if name == "diverted" else static
        )
        machine = oracle_fixture.cached_machine(
            oracle_fixture.analytic_case(),
            -110,
            wall_nodes=oracle_fixture.WALL_POINT_COUNT,
        )
        operator = oracle_fixture.forward_operator(source, machine)
        radius = np.linspace(
            1.12 * exact.major_radius,
            1.28 * exact.major_radius,
            6,
        )
        height = np.linspace(-0.3 * exact.minor_radius, 0.3 * exact.minor_radius, 6)
        points = np.column_stack((_as_flat(radius), _as_flat(height)))
        exact_density = _exact_density(exact, points)
        if name == "diverted":
            topology = certificate._analytic_diverted_topology(exact)
            psi_norm = (_flux_at(exact, points) - topology["boundary_flux_wb"]) / (
                topology["axis_flux_wb"] - topology["boundary_flux_wb"]
            )
        else:
            psi_norm = _flux_at(exact, points) / float(exact.axis_flux)
        production_density = np.asarray(
            operator.source.core.current_density(
                jnp.asarray(points[:, 0]), jnp.asarray(psi_norm)
            ),
            dtype=np.float64,
        )
        relative = float(
            np.max(np.abs(exact_density - production_density))
            / max(np.max(np.abs(exact_density)), np.finfo(float).tiny)
        )
        records["density_checks"].append(
            {
                "family": name,
                "max_relative_density_disagreement": relative,
                "passed": relative < 1.0e-9,
            }
        )

    if BANKED_WEAK_110_CUT_RECEIPT.exists():
        banked = json.loads(BANKED_WEAK_110_CUT_RECEIPT.read_text(encoding="utf-8"))
        row = [
            item
            for item in banked["rows"]
            if item["case"] == "weak-rotation-reactor-static"
            and item["requested_cells"] == -110
        ][0]
        machine = oracle_fixture.cached_machine(
            static, -110, wall_nodes=oracle_fixture.WALL_POINT_COUNT
        )
        polygons = tuple(
            np.asarray(polygon, dtype=np.float64) for polygon in machine.cell_polygons
        )
        centres = np.asarray(machine.moment_geometry.atomic_mesh.centroids)
        plasma = Polygon(certificate._boundary("weak-rotation-reactor-static", static))
        for cell in (2, 3, 4):
            banked_cell = next(
                item for item in row["cut_cells"] if item["cell"] == cell
            )
            exact_value = float(banked_cell["moments"]["zeroth"]["exact"][0])
            _kind, _region_area, moments, _loop = _exact_cell_integrals(
                static, polygons[cell], centres[cell], plasma, float(machine.area[cell])
            )
            denominator = max(abs(exact_value), 1.0)
            relative = abs(float(moments[0]) - exact_value) / denominator
            records["banked_cells"].append(
                {
                    "cell": cell,
                    "banked_exact_zeroth": exact_value,
                    "measured_exact_zeroth": float(moments[0]),
                    "relative_disagreement": relative,
                    "passed": relative < 1.0e-6,
                }
            )
    records["all_passed"] = all(
        item["passed"] for item in records["density_checks"]
    ) and all(item["passed"] for item in records["banked_cells"])
    return records


def _as_flat(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# per-row measurement
# ---------------------------------------------------------------------------


def _find_x_point_cell(
    polygons: tuple[np.ndarray, ...], x_point: np.ndarray | None
) -> int | None:
    if x_point is None:
        return None
    for index, polygon in enumerate(polygons):
        if mpl_path.Path(polygon).contains_point(x_point):
            return index
    return None


def _edge_sharing_neighbours(
    polygons: tuple[np.ndarray, ...], focus: set[int], tolerance: float
) -> list[int]:
    """Return cells sharing an atomic edge with any focus cell."""
    neighbours: set[int] = set()
    for cell in focus:
        own = np.asarray(polygons[cell], dtype=np.float64)
        for other, candidate in enumerate(polygons):
            if other in focus or other in neighbours:
                continue
            candidate = np.asarray(candidate, dtype=np.float64)
            if _polygons_share_edge(own, candidate, tolerance):
                neighbours.add(other)
    return sorted(neighbours)


def _polygons_share_edge(
    first: np.ndarray, second: np.ndarray, tolerance: float
) -> bool:
    for first_a, first_b in zip(first, np.roll(first, -1, axis=0), strict=True):
        for second_a, second_b in zip(second, np.roll(second, -1, axis=0), strict=True):
            if (
                np.linalg.norm(first_a - second_a) <= tolerance
                and np.linalg.norm(first_b - second_b) <= tolerance
            ) or (
                np.linalg.norm(first_a - second_b) <= tolerance
                and np.linalg.norm(first_b - second_a) <= tolerance
            ):
                return True
    return False


def _moment_record(actual: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    return {
        order: _relative_record(actual[slice_], exact[slice_], order)
        for order, slice_ in _ORDER_SLICES.items()
    }


def _relative_record(
    actual: np.ndarray, exact: np.ndarray, order: str
) -> dict[str, Any]:
    denominator = float(np.linalg.norm(exact))
    return {
        "production": np.asarray(actual).tolist(),
        "exact": np.asarray(exact).tolist(),
        "relative_error": (
            None
            if denominator == 0.0
            else float(np.linalg.norm(actual - exact) / denominator)
        ),
        "absolute_error_norm": float(np.linalg.norm(actual - exact)),
        "exact_norm": denominator,
    }


def _physical_moments(values: np.ndarray):
    from nova.equilibrium.stencil_mesh import CellCurrentMoments

    return CellCurrentMoments(
        jnp.asarray(values[:, 0]),
        jnp.asarray(values[:, 1]),
        jnp.asarray(values[:, 2]),
    )


def measure_row(case_name: str, requested_cells: int) -> dict[str, Any]:
    started = perf_counter()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the X-point cell RCA requires JAX extended precision")
    if jax.default_backend() != "cpu":
        raise RuntimeError("the X-point cell RCA requires the CPU backend")
    carrier_case, source_case, exact_case = certificate._case(case_name)
    machine = certificate._case_machine(
        case_name, carrier_case, exact_case, requested_cells
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact_state = certificate._exact_state(case_name, exact_case, coordinates)
    grid_count = len(machine.node)
    operator = oracle_fixture.forward_operator(source_case, machine)
    centres = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    cell_areas = np.asarray(machine.area, dtype=np.float64)
    polygons = tuple(
        np.asarray(cell, dtype=np.float64) for cell in machine.cell_polygons
    )
    boundary_level = _analytic_boundary_level(exact_case)
    axis = np.asarray(
        certificate.AXIS_M if _is_diverted(exact_case) else exact_case.magnetic_axis,
        dtype=np.float64,
    )
    x_point = (
        np.asarray(certificate.X_POINT_M, dtype=np.float64)
        if _is_diverted(exact_case)
        else None
    )
    if _is_diverted(exact_case):
        topology_analytic = certificate._analytic_diverted_topology(exact_case)
        axis_flux = topology_analytic["axis_flux_wb"]
        boundary_flux = topology_analytic["boundary_flux_wb"]
    else:
        axis_flux = float(exact_case.axis_flux)
        boundary_flux = boundary_level

    branches = _analytic_separatrix_branches(case_name, exact_case)
    core_lobe = np.asarray(branches["core_lobe"], dtype=np.float64)
    plasma_polygon = Polygon(core_lobe)

    read = _production_read(operator, exact_state)
    candidate_table = _candidate_table(operator, exact_state)

    # production chord and exact clip attributions (two mode runs)
    chord_values, chord_participation, chord_support = _mode_production_integrals(
        operator, exact_state, "chord"
    )
    exact_values, exact_participation, exact_support = _mode_production_integrals(
        operator, exact_state, "exact"
    )

    # analytic core-side moments over every cell
    cell_kind = np.zeros(len(polygons), dtype=int)
    exact_region_area = np.zeros(len(polygons))
    exact_moments = np.zeros((len(polygons), 6))
    analytic_crossing = np.zeros(len(polygons), dtype=int)
    region_loops: list[np.ndarray | None] = [None] * len(polygons)

    with warnings.catch_warnings():
        warnings.simplefilter("always", IntegrationWarning)
        for cell, polygon in enumerate(polygons):
            kind, region_area, moments, loop = _exact_cell_integrals(
                exact_case,
                polygon,
                centres[cell],
                plasma_polygon,
                float(cell_areas[cell]),
            )
            cell_kind[cell] = kind
            exact_region_area[cell] = region_area
            exact_moments[cell] = moments
            region_loops[cell] = loop
            if kind == 2:
                analytic_crossing[cell] = _separatrix_crossing_count(
                    exact_case, boundary_level, polygon
                )

    cut = cell_kind == 2
    interior = cell_kind == 1
    area_tolerance = 2.0e-11 * np.maximum(cell_areas, 1.0)
    cut = (exact_region_area > area_tolerance) & (
        exact_region_area < cell_areas - area_tolerance
    )
    interior = exact_region_area >= cell_areas - area_tolerance

    x_point_cell = _find_x_point_cell(polygons, x_point)

    # X-point cell geometry: chord endpoints, exact chain vertices
    chord_endpoints: dict[str, Any] = {"cell": None, "vertices_rz_m": None}
    exact_chain: dict[str, Any] = {"cell": None, "vertices_rz_m": None}
    if x_point_cell is not None:
        chord_poly = _support_loop(chord_support, x_point_cell)
        chain_poly = _support_loop(exact_support, x_point_cell)
        chord_endpoints = {
            "cell": int(x_point_cell),
            "vertex_count": int(
                np.asarray(chord_support.vertex_count, dtype=np.intp)[x_point_cell]
            ),
            "vertices_rz_m": chord_poly.tolist() if chord_poly is not None else None,
        }
        exact_chain = {
            "cell": int(x_point_cell),
            "vertex_count": int(
                np.asarray(exact_support.vertex_count, dtype=np.intp)[x_point_cell]
            ),
            "vertices_rz_m": chain_poly.tolist() if chain_poly is not None else None,
        }

    # nearest topology-read saddle candidate to the analytic X-point
    x_rows = np.asarray(candidate_table["x_candidates"], dtype=np.float64)
    saddle_candidate: dict[str, Any] | None = None
    x_candidate_details: list[dict[str, Any]] = []
    if x_point is not None:
        for index, row in enumerate(x_rows):
            x_candidate_details.append(
                {
                    "candidate_rz_m": row[:2].tolist(),
                    "candidate_flux_wb": float(row[2]),
                    "distance_to_analytic_x_point_m": float(
                        np.linalg.norm(row[:2] - x_point)
                    ),
                }
            )
        if x_rows.shape[0]:
            distances = np.linalg.norm(x_rows[:, :2] - x_point[None, :], axis=1)
            nearest = int(np.argmin(distances))
            candidate = x_rows[nearest]
            inside_wall = bool(
                mpl_path.Path(np.asarray(machine.wall_node)).contains_point(
                    candidate[:2]
                )
            )
            inside_lobe = bool(mpl_path.Path(core_lobe).contains_point(candidate[:2]))
            saddle_candidate = {
                "candidate_rz_m": candidate[:2].tolist(),
                "candidate_flux_wb": float(candidate[2]),
                "distance_to_analytic_x_point_m": float(distances[nearest]),
                "inside_wall_polygon": inside_wall,
                "inside_analytic_core_lobe": inside_lobe,
                "containment_verdict": (
                    "inside_wall_and_lobe"
                    if inside_wall and inside_lobe
                    else "outside_analytic_lobe"
                    if not inside_lobe
                    else "outside_wall"
                ),
            }
        else:
            saddle_candidate = {
                "candidate_rz_m": None,
                "distance_to_analytic_x_point_m": None,
                "inside_wall_polygon": None,
                "inside_analytic_core_lobe": None,
                "containment_verdict": "no_candidate_retained",
            }

    # per-cut-cell records
    cut_records = []
    for cell in np.flatnonzero(cut):
        cut_records.append(
            {
                "cell": int(cell),
                "centre_rz_m": centres[cell].tolist(),
                "separatrix_crossing_count": int(analytic_crossing[cell]),
                "chord_reverted_cell": int(cell) in CHORD_REVERTED_CELLS,
                "chord_participation": bool(chord_participation[cell]),
                "exact_participation": bool(exact_participation[cell]),
                "analytics": {
                    "core_side_current_a": float(exact_moments[cell, 0]),
                    "area_fraction": float(
                        exact_region_area[cell] / max(cell_areas[cell], 1.0)
                    ),
                    "region_area_m2": float(exact_region_area[cell]),
                    "cell_area_m2": float(cell_areas[cell]),
                },
                "moments": {
                    "analytic": exact_moments[cell].tolist(),
                    "chord": chord_values[cell].tolist(),
                    "exact": exact_values[cell].tolist(),
                    "chord_relative_error": _moment_record(
                        chord_values[cell], exact_moments[cell]
                    ),
                    "exact_relative_error": _moment_record(
                        exact_values[cell], exact_moments[cell]
                    ),
                },
                "figure_polygons": {
                    "chord_support_vertices_rz_m": (
                        _support_loop(chord_support, cell).tolist()
                        if _support_loop(chord_support, cell) is not None
                        else None
                    ),
                    "exact_support_vertices_rz_m": (
                        _support_loop(exact_support, cell).tolist()
                        if _support_loop(exact_support, cell) is not None
                        else None
                    ),
                    "analytic_region_vertices_rz_m": (
                        region_loops[cell].tolist()
                        if region_loops[cell] is not None
                        else None
                    ),
                    "cell_vertices_rz_m": polygons[cell].tolist(),
                },
            }
        )

    # which cells dominate the attribution error (chord against analytic zeroth)
    dominated: list[dict[str, Any]] = []
    if cut_records:
        squared = np.asarray(
            [
                (
                    float(record["moments"]["chord"][0])
                    - float(record["analytics"]["core_side_current_a"])
                )
                ** 2
                for record in cut_records
            ],
            dtype=np.float64,
        )
        order = np.argsort(squared)[::-1]
        total_squared = float(np.sum(squared))
        for rank, cell_index in enumerate(order[:6]):
            record = cut_records[int(cell_index)]
            dominated.append(
                {
                    "rank": rank + 1,
                    "cell": record["cell"],
                    "crossing_count": record["separatrix_crossing_count"],
                    "chord_current_a": record["moments"]["chord"][0],
                    "analytic_current_a": record["analytics"]["core_side_current_a"],
                    "fraction_of_squared_error": float(
                        squared[cell_index] / max(total_squared, np.finfo(float).tiny)
                    ),
                }
            )
    else:
        dominated = []

    # exact-state image flux deviation (the "flux error at 1e-17" claim)
    exact_coefficients = operator.coupling_current_moments(
        _physical_moments(exact_moments)
    )
    internal = np.asarray(
        operator.current_moment_image(exact_coefficients), dtype=np.float64
    )
    external = np.asarray(operator.external(), dtype=np.float64)
    deviated = (external + internal)[:grid_count] - exact_state[:grid_count]
    flux_error = {
        "max_abs_wb": float(np.max(np.abs(deviated))),
        "rms_wb": float(np.sqrt(np.mean(deviated**2))),
    }

    # focus cells: the X-point cell on the diverted rows, cells 101 and 102 on weak
    focus_cells: list[int] = []
    if x_point_cell is not None:
        focus_cells.append(x_point_cell)
    elif not _is_diverted(exact_case):
        focus_cells.extend(
            int(cell) for cell in CHORD_REVERTED_CELLS if cell < len(polygons)
        )
    scale = max(float(np.max(np.abs(np.vstack(polygons)))), 1.0)
    edge_tolerance = 256.0 * np.finfo(np.float64).eps * scale
    neighbours = _edge_sharing_neighbours(polygons, set(focus_cells), edge_tolerance)

    print(
        f"XPOINT_ROW case={case_name} requested={requested_cells} "
        f"cut={len(cut_records)} x_cell={x_point_cell} "
        f"x_candidates={candidate_table['x_candidate_count']} "
        f"image_max_abs_flux_error={flux_error['max_abs_wb']:.3e}",
        flush=True,
    )
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "source_revision": _source_revision(),
        "lane": _lane_receipt(),
        "analytic": {
            "boundary_flux_wb": boundary_flux,
            "axis_flux_wb": axis_flux,
            "flux_span_wb": abs(axis_flux - boundary_flux),
            "axis_rz_m": axis.tolist(),
            "x_point_rz_m": x_point.tolist() if x_point is not None else None,
            "separatrix_branch_count": (
                1 + len(branches["legs"]) if _is_diverted(exact_case) else 1
            ),
            "core_lobe_point_count": len(core_lobe),
            "core_lobe_rz_m": core_lobe.tolist(),
            "divertor_legs_rz_m": [leg.tolist() for leg in branches["legs"]],
        },
        "topology_read": read,
        "candidate_table": {
            "o_candidate_count": candidate_table["o_candidate_count"],
            "x_candidate_count": candidate_table["x_candidate_count"],
            "x_candidates_detail": x_candidate_details,
            "nearest_x_candidate": saddle_candidate,
        },
        "flux_error": flux_error,
        "cell_census": {
            "boundary_cut": int(np.count_nonzero(cut)),
            "interior": int(np.count_nonzero(interior)),
            "exterior": int(len(cut) - np.count_nonzero(cut | interior)),
            "x_point_cell": x_point_cell,
            "focus_cells": focus_cells,
            "edge_sharing_neighbours": neighbours,
            "chord_participating_boundary_cut": int(
                np.count_nonzero(cut & chord_participation)
            ),
            "exact_participating_boundary_cut": int(
                np.count_nonzero(cut & exact_participation)
            ),
            "max_separatrix_crossing": (
                int(np.max(analytic_crossing[cut])) if np.any(cut) else 0
            ),
            "multi_crossing_cells": sorted(
                int(cell) for cell in np.flatnonzero(cut & (analytic_crossing > 2))
            ),
        },
        "x_point_cell_detail": (
            {
                "cell": int(x_point_cell),
                "analytic_x_point_rz_m": x_point.tolist(),
                "centre_rz_m": centres[x_point_cell].tolist(),
                "crossing_count": int(analytic_crossing[x_point_cell]),
                "chord_endpoints": chord_endpoints,
                "exact_chain_vertices": exact_chain,
                "analytic_current_a": float(exact_moments[x_point_cell, 0]),
                "chord_current_a": float(chord_values[x_point_cell, 0]),
                "exact_current_a": float(exact_values[x_point_cell, 0]),
            }
            if x_point_cell is not None
            else None
        ),
        "cut_cells": cut_records,
        "dominated_cells": dominated,
        "plot_data": {
            "node_rz_m": np.asarray(machine.node),
            "wall_rz_m": np.asarray(machine.wall_node),
            "exact_state_wb": exact_state[:grid_count],
            "cell_polygons_rz_m": [polygon.tolist() for polygon in polygons],
            "cell_centres_rz_m": centres,
            "cell_areas_m2": cell_areas,
            "analytic_crossing_count": analytic_crossing,
            "cut_mask": cut,
            "analytic_current_a": exact_moments[:, 0],
            "chord_current_a": chord_values[:, 0],
            "exact_current_a": exact_values[:, 0],
            "analytic_region_loops": region_loops,
            "chord_support_loops": [
                _support_loop(chord_support, cell) for cell in range(len(polygons))
            ],
            "exact_support_loops": [
                _support_loop(exact_support, cell) for cell in range(len(polygons))
            ],
            "core_lobe_rz_m": core_lobe,
            "divertor_legs_rz_m": branches["legs"],
            "x_point_cell": x_point_cell,
            "focus_cells": focus_cells,
            "edge_sharing_neighbours": neighbours,
            "x_point_rz_m": x_point.tolist() if x_point is not None else None,
            "analytic_axis_rz_m": axis.tolist(),
        },
        "selftest": None,
        "elapsed_seconds": perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _draw_figure_one(
    axis: Any,
    row: dict[str, Any],
    cell: int,
    *,
    neighbours: list[int],
) -> None:
    plot = row["plot_data"]
    records = {record["cell"]: record for record in row["cut_cells"]}
    polygons = [
        np.asarray(loop, dtype=np.float64) for loop in plot["cell_polygons_rz_m"]
    ]
    axis = poloidal_axes(axis)
    polygon = polygons[cell]
    for other in neighbours:
        neighbour = polygons[other]
        axis.plot(neighbour[:, 0], neighbour[:, 1], color="0.85", linewidth=0.6)
    axis.plot(polygon[:, 0], polygon[:, 1], color="0.45", linewidth=1.2)
    cell_min = np.min(polygon, axis=0)
    cell_max = np.max(polygon, axis=0)
    span = np.max(cell_max - cell_min) * 0.12
    bounds = np.array([cell_min - span, cell_max + span])
    axis.set_xlim(bounds[:, 0])
    axis.set_ylim(bounds[:, 1])
    # analytic separatrix branches through the cell (clipped by the axes)
    lobe = np.asarray(plot["core_lobe_rz_m"], dtype=np.float64)
    for branch in [lobe] + [
        np.asarray(leg, dtype=np.float64) for leg in plot["divertor_legs_rz_m"]
    ]:
        axis.plot(
            branch[:, 0], branch[:, 1], color="firebrick", linewidth=1.2, zorder=4
        )
    # production chord clip polygon (chord-mode support) and exact chain
    record = records.get(cell)
    if record is not None:
        chord = record["figure_polygons"]["chord_support_vertices_rz_m"]
        exact = record["figure_polygons"]["exact_support_vertices_rz_m"]
        if chord is not None and len(chord) >= 3:
            chord = np.asarray(chord, dtype=np.float64)
            axis.plot(
                np.append(chord[:, 0], chord[0, 0]),
                np.append(chord[:, 1], chord[0, 1]),
                color="royalblue",
                linewidth=1.4,
                zorder=3,
            )
        if exact is not None and len(exact) >= 3:
            exact = np.asarray(exact, dtype=np.float64)
            axis.plot(
                np.append(exact[:, 0], exact[0, 0]),
                np.append(exact[:, 1], exact[0, 1]),
                color="seagreen",
                linewidth=1.2,
                zorder=3,
            )
    # analytic core-side region
    region = plot["analytic_region_loops"][cell]
    if region is not None and len(region) >= 3:
        region = np.asarray(region, dtype=np.float64)
        axis.plot(
            np.append(region[:, 0], region[0, 0]),
            np.append(region[:, 1], region[0, 1]),
            color="black",
            linewidth=1.8,
            zorder=5,
        )
    x_point = plot["x_point_rz_m"]
    poloidal.draw_nulls(
        axis,
        magnetic_axis=plot["analytic_axis_rz_m"],
        x_points=(
            np.asarray(x_point).reshape(1, 2)
            if x_point is not None and np.all(np.isfinite(x_point))
            else None
        ),
        style=DEFAULT_INK.variant(xpoint_marker="x", xpoint_color="firebrick"),
    )
    crossing = int(plot["analytic_crossing_count"][cell])
    axis.set_title(
        f"cell {int(cell)}: analytic separatrix crossing count {crossing}\n"
        f"grey hex: atomic cell; blue: chord support; green: exact chain; "
        f"black: analytic core-side region; red: analytic separatrix",
        fontsize=7,
    )


def _draw_figure_two(axis: Any, row: dict[str, Any]) -> None:
    plot = row["plot_data"]
    centres = np.asarray(plot["cell_centres_rz_m"], dtype=np.float64)
    error = np.asarray(plot["chord_current_a"], dtype=np.float64) - np.asarray(
        plot["analytic_current_a"], dtype=np.float64
    )
    wall = np.asarray(plot["wall_rz_m"], dtype=np.float64)
    axis = poloidal_axes(axis)
    limits = np.vstack((centres, wall))
    radial = np.linspace(float(np.min(limits[:, 0])), float(np.max(limits[:, 0])), 181)
    height = np.linspace(float(np.min(limits[:, 1])), float(np.max(limits[:, 1])), 181)
    radius_grid, height_grid = np.meshgrid(radial, height)
    render = LinearNDInterpolator(centres, error, fill_value=np.nan)(
        radius_grid, height_grid
    )
    if np.any(np.isfinite(render)):
        values = np.abs(render[np.isfinite(render)])
        lower = float(np.percentile(values, 5.0)) if values.size else 1.0
        upper = float(np.max(values)) if values.size else 1.0
        lower = max(lower, 1e-300)
        if upper > lower:
            levels = np.geomspace(lower, upper, 9)
            axis.contour(
                radial,
                height,
                np.maximum(np.abs(render), lower),
                levels=levels,
                colors="firebrick",
                linewidths=0.8,
            )
    cut_mask = plot["cut_mask"]
    polygons = [
        np.asarray(loop, dtype=np.float64) for loop in plot["cell_polygons_rz_m"]
    ]
    for cell in np.flatnonzero(cut_mask):
        polygon = polygons[int(cell)]
        axis.plot(polygon[:, 0], polygon[:, 1], color="0.3", linewidth=0.5)
    exact = np.asarray(plot["exact_state_wb"], dtype=np.float64)
    node = np.asarray(plot["node_rz_m"], dtype=np.float64)
    flux_raster = LinearNDInterpolator(node, exact, fill_value=np.nan)(
        radius_grid, height_grid
    )
    if np.any(np.isfinite(flux_raster)):
        levels = np.linspace(float(np.min(exact)), float(np.max(exact)), 12)[1:-1]
        poloidal.draw_flux_contours(
            axis, radial, height, flux_raster, levels, style=DEFAULT_INK, color="0.6"
        )
    poloidal.draw_wall(axis, wall[:, 0], wall[:, 1])
    poloidal.draw_nulls(
        axis,
        magnetic_axis=plot["analytic_axis_rz_m"],
        x_points=(
            np.asarray(plot["x_point_rz_m"]).reshape(1, 2)
            if plot["x_point_rz_m"] is not None
            and np.all(np.isfinite(plot["x_point_rz_m"]))
            else None
        ),
        contain=wall,
    )
    dominated = row["dominated_cells"]
    text = (
        "no cut cells"
        if not dominated
        else ", ".join(
            f"{item['cell']}({item['crossing_count']})" for item in dominated[:4]
        )
    )
    axis.set_title(
        f"{row['case']} {row['requested_cells']} cells: chord minus analytic\n"
        f"per-cell current magnitude; cut cells outlined; dominated {text}",
        fontsize=7,
    )


def _draw_figure_three(axis: Any, row: dict[str, Any]) -> None:
    plot = row["plot_data"]
    cut = np.flatnonzero(plot["cut_mask"])
    if len(cut) == 0:
        axis.set_title("no cut cells")
        return
    analytic = np.asarray(plot["analytic_current_a"])[cut]
    chord = np.asarray(plot["chord_current_a"])[cut]
    exact = np.asarray(plot["exact_current_a"])[cut]
    order = np.argsort(np.abs(chord - analytic))[::-1]
    ordered = np.asarray(order)
    labels = [str(int(cut[index])) for index in ordered]
    positions = np.arange(len(ordered))
    width = 0.38
    axis.bar(
        positions - width / 2,
        np.abs(chord[ordered] - analytic[ordered]),
        width,
        label="chord abs error",
        color="royalblue",
    )
    axis.bar(
        positions + width / 2,
        np.abs(exact[ordered] - analytic[ordered]),
        width,
        label="exact abs error",
        color="seagreen",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_ylabel("|attributed - analytic| current [A]")
    axis.set_xlabel("cut cell index (ranked by chord abs error)")
    axis.legend(fontsize=6)
    axis.set_title(
        f"{row['case']} {row['requested_cells']} cells: absolute per-cell "
        "current attribution error",
        fontsize=8,
    )


def _figure_record(row: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "project_src": "/nova/figures/uniform-cell-clip-and-coupling/xpoint-cell-rca/"
        + f"{_case_key(row['case'], row['requested_cells'])}/{name}",
    }


def _render_row_figures(row: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    plot = row["plot_data"]
    focus_list = (
        [int(plot["x_point_cell"])]
        if plot["x_point_cell"] is not None
        else (
            [int(cell) for cell in plot["focus_cells"]] if plot["focus_cells"] else []
        )
    )
    neighbours = plot["edge_sharing_neighbours"]
    row_root = root / _case_key(row["case"], row["requested_cells"])
    row_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    if focus_list:
        figure, axes = plt.subplots(
            1, len(focus_list), figsize=(5.5 * len(focus_list), 5.5)
        )
        if len(focus_list) == 1:
            axes = [axes]
        for index, cell in enumerate(focus_list):
            _draw_figure_one(axes[index], row, cell, neighbours=neighbours)
        figure.suptitle(
            f"{row['case']} {row['requested_cells']} cells: clip allocation on "
            "the focus cell(s)",
            fontsize=9,
        )
        figure.tight_layout()
        path = row_root / "figure1-focus-cell.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(_figure_record(row, "figure1-focus-cell.png", path))

    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    _draw_figure_two(axis, row)
    figure.tight_layout()
    path = row_root / "figure2-error-contours.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(_figure_record(row, "figure2-error-contours.png", path))

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    _draw_figure_three(axis, row)
    figure.tight_layout()
    path = row_root / "figure3-ranked-bars.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(_figure_record(row, "figure3-ranked-bars.png", path))
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _probe() -> dict[str, Any]:
    """Exercise the operator wiring on the weak -110 row without the full run."""
    configure_dtypes()
    carrier_case, source_case, exact_case = certificate._case(
        "weak-rotation-reactor-static"
    )
    machine = certificate._case_machine(
        "weak-rotation-reactor-static", carrier_case, exact_case, -110
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact_state = certificate._exact_state(
        "weak-rotation-reactor-static", exact_case, coordinates
    )
    operator = oracle_fixture.forward_operator(source_case, machine)
    chord_values, chord_participation, chord_support = _mode_production_integrals(
        operator, exact_state, "chord"
    )
    exact_values, exact_participation, exact_support = _mode_production_integrals(
        operator, exact_state, "exact"
    )
    read = _production_read(operator, exact_state)
    table = _candidate_table(operator, exact_state)
    probes = {
        "machine_nodes": int(len(machine.node)),
        "cells": int(len(chord_values)),
        "chord_participating": int(np.count_nonzero(chord_participation)),
        "exact_participating": int(np.count_nonzero(exact_participation)),
        "chord_zeroth_sum_a": float(np.sum(chord_values[:, 0])),
        "exact_zeroth_sum_a": float(np.sum(exact_values[:, 0])),
        "chord_support_capacity": int(chord_support.support_vertices.shape[1]),
        "exact_support_capacity": int(exact_support.support_vertices.shape[1]),
        "read_status": read["read_status"],
        "read_axis_rz_m": read["axis_rz_m"],
        "o_candidate_count": table["o_candidate_count"],
        "x_candidate_count": table["x_candidate_count"],
        "finite_moments": bool(
            np.all(np.isfinite(np.vstack((chord_values, exact_values))))
        ),
    }
    print("XPOINT_PROBE " + json.dumps(probes, sort_keys=True), flush=True)
    return probes


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=tuple(sorted({name for name, _c in CASE_REQUESTS}))
    )
    parser.add_argument("--requested-cells", type=int, choices=(-110, -300))
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def _aggregate(output: Path = RECEIPT) -> dict[str, Any]:
    rows = [
        json.loads(_part_path(case_name, cells).read_text(encoding="utf-8"))
        for case_name, cells in CASE_REQUESTS
    ]
    revisions = {row["source_revision"] for row in rows}
    if len(revisions) != 1:
        raise RuntimeError(f"rows do not share one source revision: {revisions}")
    job_ids = set()
    for row in rows:
        lane = row["lane"]
        if (
            lane["execution"] != "slurm"
            or lane["partition"] != "all_debug"
            or lane["jax_platform"] != "cpu"
            or not lane["jax_enable_x64"]
        ):
            raise RuntimeError("every row must be an all_debug CPU-x64 measurement")
        job_ids.add(lane["job_id"])
    if len(job_ids) != 1:
        raise RuntimeError("the rows must share one scheduler allocation")
    figures = []
    for row in rows:
        figures.extend(_render_row_figures(row, OUTPUT_ROOT))
    compact_rows = []
    for row in rows:
        compact = dict(row)
        compact.pop("plot_data")
        compact_rows.append(compact)

    headline = {}
    for row in compact_rows:
        key = _case_key(row["case"], row["requested_cells"])
        headline[key] = {
            "boundary_cut_cells": row["cell_census"]["boundary_cut"],
            "x_point_cell": row["cell_census"]["x_point_cell"],
            "max_separatrix_crossing": row["cell_census"]["max_separatrix_crossing"],
            "multi_crossing_cells": row["cell_census"]["multi_crossing_cells"],
            "flux_error_max_abs_wb": row["flux_error"]["max_abs_wb"],
            "x_candidate_count": row["candidate_table"]["x_candidate_count"],
            "dominated": row["dominated_cells"],
        }
    receipt = {
        "schema": "nova.xpoint-cell-allocation-rca",
        "completed": True,
        "measurement_driver_revision": revisions.pop(),
        "contract": {
            "rows": [
                {"case": case_name, "requested_cells": cells}
                for case_name, cells in CASE_REQUESTS
            ],
            "one_row_per_fresh_process": True,
            "one_scheduler_job": True,
            "lane": "all_debug CPU float64",
            "adaptive_relative_tolerance": ADAPTIVE_RELATIVE_TOLERANCE,
            "fixed_interior_points_per_axis": FIXED_INTERIOR_POINTS,
            "selftest": _analytic_self_test(),
        },
        "rows": compact_rows,
        "figures": figures,
        "headline": headline,
    }
    _write_json(output, receipt)
    for case_name, cells in CASE_REQUESTS:
        _part_path(case_name, cells).unlink(missing_ok=True)
    parts = OUTPUT_ROOT / "parts"
    try:
        parts.rmdir()
    except OSError:
        pass
    return receipt


def main() -> None:
    configure_dtypes()
    arguments = _parse()
    if arguments.probe:
        _probe()
        return
    if arguments.selftest:
        result = _analytic_self_test()
        _write_json(OUTPUT_ROOT / "parts" / "selftest.json", result)
        print("SELFTEST " + json.dumps(result["all_passed"]), flush=True)
        raise SystemExit(0 if result["all_passed"] else 1)
    if arguments.aggregate:
        receipt = _aggregate()
        print(json.dumps(receipt["headline"], sort_keys=True), flush=True)
        print("XPOINT_AGGREGATE_EXIT=0", flush=True)
        return
    if arguments.case is None or arguments.requested_cells is None:
        raise SystemExit("--selftest or one --case/--requested-cells pair is required")
    if (arguments.case, arguments.requested_cells) not in CASE_REQUESTS:
        raise SystemExit("the requested case-resolution pair is outside the contract")
    part = _part_path(arguments.case, arguments.requested_cells)
    row = measure_row(arguments.case, arguments.requested_cells)
    _write_json(part, row)
    print(f"XPOINT_ROW_EXIT=0 part={part}", flush=True)


if __name__ == "__main__":
    main()
