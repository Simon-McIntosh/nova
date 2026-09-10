"""Measure the coupling alone: frozen exact currents through the polygon kernel.

Gate B of the cut-cell plan isolates the production coupling's own error.  On
each row the analytic equilibrium's per-cell currents are frozen as the exact
zeroth, first and second density moments over each cell's analytic plasma
intersection (interior cells whole, cut cells over their analytic region,
exterior zero).  Those moments are contracted through the shipped operator
exactly as a solve would --- ``coupling_current_moments`` into
``current_moment_image`` over the polygon-analytic kernel blocks, plus the same
external and coil contribution the certificate row uses --- and the predicted
flux at every mesh node is compared with the oracle flux.  No clip decision and
no iteration enter the prediction, so whatever error remains is the coupling's
own: the polygon kernel, the moment truncation order and the second-order
compatibility terms.

The receipt reports, per row, the maximum and root-mean-square absolute flux
error in weber and as a fraction of the flux span, the same restricted to the
interior nodes, the two-pitch boundary band and the nodes adjacent to cut
cells, the image with the second-order moments zeroed (first order only) and
with first and second zeroed (zeroth only) so the moment truncation order is
read directly, and the six worst nodes with their nearest cell and class.

The static family's exact per-cell moments are the shared adaptive oracle of
:mod:`benchmarks.solovev_cut_cell_moments`; the diverted family integrates the
exact Cerfon-Freidberg density over the cell clipped against the analytic
separatrix lobe with an adaptive radial quadrature refined at every vertex.
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
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from benchmarks import solovev_certificate as certificate
from benchmarks.solovev_cut_cell_moments import _exact_cell_integral
from benchmarks.split_fit_jump_field import BOUNDARY_BAND_PITCHES, _distance_to_boundary
from nova.equilibrium.analytic_single_null import CerfonFreidbergSingleNull
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture

MU_0 = 4.0e-7 * np.pi

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/gate-b"
RECEIPT = OUTPUT_ROOT / "receipt.json"
PARTS = OUTPUT_ROOT / "parts"
#: The six rows: weak, moderate and strong rotation static and the diverted
#: single null at 110 cells, then weak and diverted at 300 cells.
CASE_REQUESTS = (
    ("weak-rotation-reactor-static", -110),
    ("moderate-rotation-conventional-static", -110),
    ("strong-rotation-compact-static", -110),
    ("diverted-single-null", -110),
    ("weak-rotation-reactor-static", -300),
    ("diverted-single-null", -300),
)
ADAPTIVE_RELATIVE_TOLERANCE = 5.0e-13
ADAPTIVE_ABS_ERROR = 1.0e-12
FIXED_INTERIOR_POINTS = 9
WORST_NODE_COUNT = 6

#: The committed certificate row whose terminal flux is drawn on the weak 110
#: comparison panel.
CONTROL_PART = (
    ROOT
    / "docs/figures/uniform-cell-clip-and-coupling/cut-cell-moments"
    / ".."
    / "exact-participation"
    / "discriminator"
    / "parts"
    / "control"
    / "weak-rotation-reactor-static-production-route-reduced.json"
)


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


def _case_key(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-{abs(requested_cells)}"


def _part_path(case_name: str, requested_cells: int) -> Path:
    return PARTS / f"{_case_key(case_name, requested_cells)}.json"


# ---------------------------------------------------------------------------
# analytic separatrix geometry
# ---------------------------------------------------------------------------


def _is_diverted(exact: Any) -> bool:
    """Return whether the exact reference is the Cerfon-Freidberg single null."""
    return isinstance(exact, CerfonFreidbergSingleNull)


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


# ---------------------------------------------------------------------------
# exact density and analytic moment integrals (both families)
# ---------------------------------------------------------------------------


def _flux_at(case: Any, points: np.ndarray) -> np.ndarray:
    """Evaluate the exact per-radian flux at ``(R, Z)`` points."""
    points = np.asarray(points, dtype=np.float64)
    if _is_diverted(case):
        return np.asarray(case.flux(points), dtype=np.float64)
    return np.asarray(case.flux(points[:, 0], points[:, 1]), dtype=np.float64)


def _exact_density(case: Any, points: np.ndarray) -> np.ndarray:
    """Return the exact toroidal current density at physical ``(R, Z)`` points."""
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


def _plasma_half_height(case: Any, radius: float) -> float:
    """Return the static-family plasma vertical half-extent at one radius."""
    remaining = float(case.axis_flux - case._flux_offset(case._flux_label(radius)))
    if remaining <= 0.0:
        return 0.0
    return float(np.sqrt(remaining / float(case.field_coefficient)))


def _vertical_bounds(vertices: np.ndarray, radius: float) -> tuple[float, float] | None:
    """Return a convex cell polygon's vertical interval at one radius."""
    intervals = _polygon_z_intervals(vertices, radius)
    if not intervals:
        return None
    return intervals[0][0], intervals[0][1]


def _plasma_z_interval(
    case: Any, radius: float, boundary_level: float, lobe_loop: np.ndarray
) -> tuple[float, float] | None:
    """Return the analytic plasma vertical interval at one radius.

    The static family uses the closed-form half-height exactly.  The diverted
    family refines the core-lobe polygon crossings by Newton on the exact
    flux, so the boundary level is honoured to machine precision.
    """
    if not _is_diverted(case):
        half = _plasma_half_height(case, radius)
        if half <= 0.0:
            return None
        return -half, half
    intervals = _polygon_z_intervals(lobe_loop, radius)
    if not intervals:
        return None
    lower, upper = intervals[0]
    flux = _flux_at(case, np.asarray([[radius, lower]]))[0]
    gradient = np.asarray(
        case.gradient(np.asarray([[radius, lower]])), dtype=np.float64
    )[0, 1]
    for _ in range(12):
        if gradient == 0.0 or not np.isfinite(gradient):
            break
        step = (flux - boundary_level) / gradient
        if not np.isfinite(step) or abs(step) > 0.1 * max(abs(lower), 1.0):
            break
        lower = lower - step
        flux = _flux_at(case, np.asarray([[radius, lower]]))[0]
        gradient = np.asarray(
            case.gradient(np.asarray([[radius, lower]])), dtype=np.float64
        )[0, 1]
    flux = _flux_at(case, np.asarray([[radius, upper]]))[0]
    gradient = np.asarray(
        case.gradient(np.asarray([[radius, upper]])), dtype=np.float64
    )[0, 1]
    for _ in range(12):
        if gradient == 0.0 or not np.isfinite(gradient):
            break
        step = (flux - boundary_level) / gradient
        if not np.isfinite(step) or abs(step) > 0.1 * max(abs(upper), 1.0):
            break
        upper = upper - step
        flux = _flux_at(case, np.asarray([[radius, upper]]))[0]
        gradient = np.asarray(
            case.gradient(np.asarray([[radius, upper]])), dtype=np.float64
        )[0, 1]
    if upper <= lower:
        return None
    return lower, upper


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _fused_vertical_integral(
    case: Any,
    radius: float,
    z_lower: float,
    z_upper: float,
    centre_z: float,
    vertical_power: int,
) -> float:
    """Integrate ``j(R, z) (z - centre_z)**p`` over ``[z_lower, z_upper]``."""
    half = 0.5 * (z_upper - z_lower)
    midpoint = 0.5 * (z_upper + z_lower)
    heights = midpoint + half * _GAUSS_NODES
    density = _exact_density(
        case, np.column_stack((np.full(16, radius, dtype=np.float64), heights))
    )
    return half * float(
        np.sum(_GAUSS_WEIGHTS * density * (heights - centre_z) ** vertical_power)
    )


def _analytic_region_integrals(
    case: Any,
    cell_vertices: np.ndarray,
    centre: np.ndarray,
    *,
    boundary_level: float,
    lobe_loop: np.ndarray,
    region_vertices: np.ndarray,
    relative_tolerance: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Integrate the exact density moments and area over the plasma in a cell."""
    region_vertices = np.asarray(region_vertices, dtype=np.float64)
    lower = float(np.min(region_vertices[:, 0]))
    upper = float(np.max(region_vertices[:, 0]))
    if upper <= lower:
        return 0.0, np.zeros(6), np.zeros(6)
    breaks = sorted(
        {lower, upper}
        | {
            float(value)
            for value in np.concatenate((cell_vertices[:, 0], region_vertices[:, 0]))
            if lower < value < upper
        }
    )
    moment_values = np.zeros(6)
    moment_errors = np.zeros(6)
    area = 0.0
    powers = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))

    def plasma_intervals(radius: float) -> list[tuple[float, float]]:
        cell_bounds = _vertical_bounds(cell_vertices, radius)
        if cell_bounds is None:
            return []
        interval = _plasma_z_interval(case, radius, boundary_level, lobe_loop)
        if interval is None:
            return []
        low, high = interval
        clipped = (max(cell_bounds[0], low), min(cell_bounds[1], high))
        return [clipped] if clipped[1] > clipped[0] else []

    def inner(radius: float, radial_power: int, vertical_power: int) -> float:
        total = 0.0
        for z_lower, z_upper in plasma_intervals(radius):
            total += _fused_vertical_integral(
                case, radius, z_lower, z_upper, centre[1], vertical_power
            )
        return total * (radius - centre[0]) ** radial_power

    def vertical_length(radius: float) -> float:
        return float(sum(high - low for low, high in plasma_intervals(radius)))

    _SCALE_NODES, _SCALE_WEIGHTS = np.polynomial.legendre.leggauss(5)

    def scale_of(callable_integrand: Any, first: float, second: float) -> float:
        midpoint = 0.5 * (first + second)
        half = 0.5 * (second - first)
        nodes = midpoint + half * _SCALE_NODES
        values = np.abs(np.asarray([callable_integrand(float(node)) for node in nodes]))
        return float(np.median(values)) if values.size else 0.0

    for first, second in zip(breaks, breaks[1:]):
        if second <= first:
            continue
        intra = tuple(value for value in breaks if first < value < second)
        span = second - first
        for index, (radial_power, vertical_power) in enumerate(powers):

            def integrand(radius: float) -> float:
                return inner(radius, radial_power, vertical_power)

            scale = scale_of(integrand, first, second)
            abs_floor = max(ADAPTIVE_ABS_ERROR, 1.0e-14 * scale * span)
            value, error = quad(
                integrand,
                first,
                second,
                points=intra,
                epsabs=abs_floor,
                epsrel=relative_tolerance,
                limit=300,
            )
            moment_values[index] += value
            moment_errors[index] += error
        scale = scale_of(vertical_length, first, second)
        abs_floor = max(ADAPTIVE_ABS_ERROR, 1.0e-14 * scale * span)
        area_value, _area_error = quad(
            vertical_length,
            first,
            second,
            points=intra,
            epsabs=abs_floor,
            epsrel=relative_tolerance,
            limit=300,
        )
        area += area_value
    return area, moment_values, moment_errors


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
    *,
    boundary_level: float,
    lobe_loop: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Classify a cell and return its exact core-side moments.

    Returns ``(kind, region_area, moments)`` where ``kind`` is 0 exterior,
    1 interior (full cell), 2 boundary-cut.  Static-family moments come from
    the shared adaptive oracle of :mod:`solovev_cut_cell_moments`; the
    diverted family integrates against the analytic plasma boundary.
    """
    cell_shape = Polygon(polygon)
    region = cell_shape.intersection(plasma_polygon)
    tolerance = 2.0e-11 * max(cell_area, 1.0)
    if region.is_empty or region.area <= tolerance:
        return 0, 0.0, np.zeros(6)
    if abs(region.area - cell_area) <= tolerance:
        if _is_diverted(case):
            moments = _fixed_cell_moments(case, polygon, centre)
        else:
            integral = _exact_cell_integral(
                case,
                np.asarray(polygon, dtype=np.float64),
                centre,
                ADAPTIVE_RELATIVE_TOLERANCE,
            )
            moments = np.asarray(integral.moment, dtype=np.float64)
        return 1, cell_area, moments
    loop = _region_loop_vertices(region)
    if loop is None:
        return 0, 0.0, np.zeros(6)
    region_vertices = np.asarray(loop, dtype=np.float64)
    area, moments, _errors = _analytic_region_integrals(
        case,
        np.asarray(polygon, dtype=np.float64),
        centre,
        boundary_level=boundary_level,
        lobe_loop=lobe_loop,
        region_vertices=region_vertices,
        relative_tolerance=ADAPTIVE_RELATIVE_TOLERANCE,
    )
    return 2, area, moments


# ---------------------------------------------------------------------------
# per-row measurement
# ---------------------------------------------------------------------------


def _frozen_moments(
    case_name: str, exact: Any, machine: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-cell exact moments, region area and cell-kind labels."""
    centres = np.asarray(
        machine.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    )
    cell_areas = np.asarray(machine.area, dtype=np.float64)
    polygons = tuple(
        np.asarray(cell, dtype=np.float64) for cell in machine.cell_polygons
    )
    boundary_level = _analytic_boundary_level(exact)
    lobe_loop = _closed_plasma_polygon(case_name, exact)
    plasma_polygon = Polygon(lobe_loop)
    cell_count = len(polygons)
    moments = np.zeros((cell_count, 6), dtype=np.float64)
    region_area = np.zeros(cell_count, dtype=np.float64)
    kind = np.zeros(cell_count, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("always", IntegrationWarning)
        for cell, polygon in enumerate(polygons):
            cell_kind, area, values = _exact_cell_integrals(
                exact,
                polygon,
                centres[cell],
                plasma_polygon,
                cell_areas[cell],
                boundary_level=boundary_level,
                lobe_loop=lobe_loop,
            )
            kind[cell] = cell_kind
            region_area[cell] = area
            moments[cell] = values
    return moments, region_area, kind


def _physical_moments(values: np.ndarray) -> CellCurrentMoments:
    return CellCurrentMoments(
        jnp.asarray(values[:, 0]),
        jnp.asarray(values[:, 1]),
        jnp.asarray(values[:, 2]),
    )


def _field_metrics(
    error: np.ndarray, coordinates: np.ndarray, span: float
) -> dict[str, Any]:
    """Return max and RMS absolute error in Wb and as a fraction of the span."""
    absolute = np.abs(error)
    return {
        "node_count": int(len(error)),
        "max_absolute_error_wb": float(np.max(absolute)),
        "rms_absolute_error_wb": float(np.sqrt(np.mean(error**2))),
        "max_error_over_span": float(np.max(absolute) / span),
        "rms_error_over_span": float(np.sqrt(np.mean(error**2)) / span),
    }


def _node_classes(kind: np.ndarray, machine: Any) -> dict[str, np.ndarray]:
    """Return node masks for interior, boundary band and cut-adjacent regions."""
    area = np.asarray(machine.area, dtype=np.float64)
    pitch = float(np.sqrt(np.median(area)))
    return {
        "pitch": pitch,
        "interior_node_mask": kind == 1,
        "cut_node_mask": kind == 2,
    }


def _boundary_band_mask(
    case_name: str, exact: Any, machine: Any, pitch: float
) -> np.ndarray:
    boundary = certificate._boundary(case_name, exact)
    return _distance_to_boundary(np.asarray(machine.node), boundary) <= (
        BOUNDARY_BAND_PITCHES * pitch
    )


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
            for first_a, first_b in zip(own, np.roll(own, -1, axis=0), strict=True):
                for second_a, second_b in zip(
                    candidate, np.roll(candidate, -1, axis=0), strict=True
                ):
                    if (
                        np.linalg.norm(first_a - second_a) <= tolerance
                        and np.linalg.norm(first_b - second_b) <= tolerance
                    ) or (
                        np.linalg.norm(first_a - second_b) <= tolerance
                        and np.linalg.norm(first_b - second_a) <= tolerance
                    ):
                        neighbours.add(other)
                        break
                else:
                    continue
                break
    return sorted(neighbours)


def _cut_adjacent_mask(kind: np.ndarray, machine: Any) -> np.ndarray:
    """Return cells sharing an atomic edge with a boundary-cut cell."""
    cut = np.flatnonzero(kind == 2)
    polygons = machine.cell_polygons
    coordinate_scale = max(float(np.max(np.abs(np.asarray(machine.node)))), 1.0)
    tolerance = 256.0 * np.finfo(np.float64).eps * coordinate_scale
    adjacent = _edge_sharing_neighbours(
        polygons, set(int(value) for value in cut), tolerance
    )
    mask = np.zeros(len(kind), dtype=bool)
    if adjacent:
        mask[np.asarray(adjacent, dtype=np.intp)] = True
    return mask


def _worst_nodes(
    error: np.ndarray,
    coordinates: np.ndarray,
    kind: np.ndarray,
    centres: np.ndarray,
) -> list[dict[str, Any]]:
    """Return the worst-error nodes with nearest cell and its class."""
    if len(error) < WORST_NODE_COUNT:
        raise RuntimeError("fewer mesh nodes than the worst-node reporting window")
    order = np.argsort(np.abs(error))[::-1][:WORST_NODE_COUNT]
    class_name = np.where(kind == 1, "interior", np.where(kind == 2, "cut", "exterior"))
    records = []
    for index in order:
        distance = np.linalg.norm(centres - coordinates[index], axis=1)
        nearest = int(np.argmin(distance))
        records.append(
            {
                "node": int(index),
                "coordinate_rz_m": coordinates[index].tolist(),
                "absolute_error_wb": float(abs(error[index])),
                "nearest_cell": nearest,
                "nearest_cell_distance_m": float(distance[nearest]),
                "nearest_cell_class": str(class_name[nearest]),
            }
        )
    return records


def measure_rung(case_name: str, requested_cells: int) -> dict[str, Any]:
    """Measure one case-resolution row and retain arrays needed for plotting."""
    started = perf_counter()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the frozen-current flux image gate requires x64")
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            "the frozen-current flux image gate requires the CPU backend"
        )

    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    grid_count = len(machine.node)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    oracle_grid = oracle_state[:grid_count]

    # The certificate row's operator: its exterior bakes the certificate's own
    # exact-moment image so the solve converges to the analytic state.  The
    # gate reuses exactly that external and coil contribution.
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    certificate_moments = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    certificate_coefficients = empty_operator.coupling_current_moments(
        certificate_moments
    )
    certificate_internal = oracle_fixture._internal_flux_image(
        empty_operator, certificate_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - certificate_internal
    )
    external = np.asarray(operator.external(), dtype=np.float64)

    # frozen exact per-cell currents over the analytic plasma intersection
    moments, region_area, kind = _frozen_moments(case_name, exact, machine)

    variants = {
        "full": _physical_moments(moments),
        "first_order": _physical_moments(moments),
        "zeroth_only": CellCurrentMoments(
            jnp.asarray(moments[:, 0]),
            jnp.zeros(grid_count),
            jnp.zeros(grid_count),
        ),
    }
    # The shipped polygon kernel carries zeroth and first moments only, so the
    # density's second moments are structurally absent: the "second order
    # zeroed (first order only)" image is the full coupling by construction.
    variants["first_order"] = CellCurrentMoments(
        jnp.asarray(moments[:, 0]),
        jnp.asarray(moments[:, 1]),
        jnp.asarray(moments[:, 2]),
    )

    images = {}
    for name, physical in variants.items():
        coefficients = operator.coupling_current_moments(physical)
        image = external + np.asarray(
            operator.current_moment_image(coefficients), dtype=np.float64
        )
        images[name] = image

    span = float(np.max(oracle_grid) - np.min(oracle_grid))
    classes = _node_classes(kind, machine)
    boundary_band = _boundary_band_mask(case_name, exact, machine, classes["pitch"])
    cut_adjacent = _cut_adjacent_mask(kind, machine)

    image_records: dict[str, Any] = {}
    for name, image in images.items():
        error = image[:grid_count] - oracle_grid
        regional = {
            "whole_domain": _field_metrics(error, machine.node, span),
        }
        for label, mask in (
            ("interior_nodes", classes["interior_node_mask"]),
            ("two_pitch_boundary_band", boundary_band),
            ("cut_adjacent_nodes", cut_adjacent),
        ):
            selected = np.flatnonzero(mask)
            if selected.size == 0:
                regional[label] = None
                continue
            regional[label] = _field_metrics(
                error[selected], machine.node[selected], span
            )
            regional[label]["node_count"] = int(selected.size)
        image_records[name] = {
            "error": regional,
            "state_sha256_binary64": hashlib.sha256(
                np.ascontiguousarray(image, dtype="<f8").tobytes()
            ).hexdigest(),
        }

    full_error = images["full"][:grid_count] - oracle_grid
    row = {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "source_revision": _source_revision(),
        "lane": _lane_receipt(),
        "contract": {
            "frozen_currents": (
                "exact zeroth, first and second density moments over the "
                "analytic plasma intersection; interior whole, cut over the "
                "analytic region, exterior zero"
            ),
            "coupling": (
                "production coupling_current_moments into current_moment_image "
                "over the polygon-analytic kernel blocks, plus the certificate "
                "row's external and coil contribution; no clip decision, no "
                "iteration"
            ),
            "adaptive_relative_tolerance": ADAPTIVE_RELATIVE_TOLERANCE,
        },
        "flux_span_wb": span,
        "pitch_m": classes["pitch"],
        "cell_census": {
            "interior": int(np.count_nonzero(kind == 1)),
            "cut": int(np.count_nonzero(kind == 2)),
            "exterior": int(np.count_nonzero(kind == 0)),
        },
        "images": image_records,
        "worst_nodes": _worst_nodes(
            full_error,
            np.asarray(machine.node),
            kind,
            np.asarray(machine.moment_geometry.atomic_mesh.centroids),
        ),
        "plot_data": {
            "node_rz_m": np.asarray(machine.node),
            "wall_rz_m": np.asarray(machine.wall_node),
            "oracle_flux_wb": oracle_grid,
            "full_moment_image_wb": images["full"][:grid_count],
            "zeroth_only_image_wb": images["zeroth_only"][:grid_count],
        },
        "elapsed_seconds": perf_counter() - started,
    }
    full_whole = image_records["full"]["error"]["whole_domain"]
    print(
        f"FROZEN_FLUX_RESULT case={case_name} requested={requested_cells} "
        f"nodes={grid_count} span={span:.6g} "
        f"full_absmax={full_whole['max_absolute_error_wb']:.6g} "
        f"full_rms={full_whole['rms_absolute_error_wb']:.6g}",
        flush=True,
    )
    return row


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _shared_error_levels(rows: list[dict[str, Any]]) -> np.ndarray:
    absolute = []
    for row in rows:
        plot = row["plot_data"]
        full = np.asarray(plot["full_moment_image_wb"], dtype=np.float64)
        oracle = np.asarray(plot["oracle_flux_wb"], dtype=np.float64)
        absolute.append(np.abs(full - oracle))
    values = np.concatenate(absolute)
    nonzero = values[values > 0.0]
    if nonzero.size == 0:
        return np.asarray([np.finfo(float).tiny])
    lower = max(float(np.percentile(nonzero, 10.0)), np.finfo(float).tiny)
    upper = float(np.max(nonzero))
    return np.geomspace(lower, upper, 9) if upper > lower else np.asarray([upper])


def _row_nulls(case_name: str, exact: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if _is_diverted(exact):
        return (
            np.asarray(certificate.AXIS_M, dtype=np.float64),
            np.asarray(certificate.X_POINT_M, dtype=np.float64),
        )
    return np.asarray(exact.magnetic_axis, dtype=np.float64), None


def _draw_error_figure(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """One absolute-flux-error panel per row on shared error levels."""
    error_levels = _shared_error_levels(rows)
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 9.0), constrained_layout=True)
    for row_index, row in enumerate(rows):
        axis = axes[row_index // 3][row_index % 3]
        panel = poloidal_axes(axis)
        plot = row["plot_data"]
        node = np.asarray(plot["node_rz_m"], dtype=np.float64)
        wall = np.asarray(plot["wall_rz_m"], dtype=np.float64)
        oracle = np.asarray(plot["oracle_flux_wb"], dtype=np.float64)
        image = np.asarray(plot["full_moment_image_wb"], dtype=np.float64)
        absolute = np.abs(image - oracle)
        panel.tricontour(
            node[:, 0],
            node[:, 1],
            np.maximum(absolute, error_levels[0]),
            levels=error_levels,
            colors="firebrick",
            linewidths=0.8,
        )
        poloidal.draw_wall(panel, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
        carrier, _source, exact = certificate._case(row["case"])
        axis_rz, x_points = _row_nulls(row["case"], exact)
        poloidal.draw_nulls(panel, magnetic_axis=axis_rz, x_points=x_points)
        span = row["flux_span_wb"]
        maximum = float(np.max(absolute))
        panel.set_title(
            f"{row['case']} {row['requested_cells']}\n"
            f"max |error| {maximum:.4g} Wb = {maximum / span:.3g} span",
            fontsize=7,
        )
    figure.suptitle(
        "Frozen-exact-current flux image: absolute error vs the oracle flux\n"
        "red: |prediction - analytic| on shared levels; wall drawn; "
        "axis and X-point marked",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return {
        "path": str(output.relative_to(ROOT)),
        "project_src": "/nova/figures/cut-cell-current-attribution/gate-b/"
        + output.name,
        "shared_absolute_error_levels_wb": error_levels,
    }


def _committed_terminal_flux() -> np.ndarray:
    """Return the committed weak -110 terminal flux from the control row."""
    row = json.loads(CONTROL_PART.read_text(encoding="utf-8"))
    return np.asarray(row["render_data"]["terminal_flux_wb"], dtype=np.float64)


def _draw_comparison_panel(row: dict[str, Any], output: Path) -> dict[str, Any]:
    """Draw the frozen prediction, analytic flux and committed terminal state."""
    plot = row["plot_data"]
    node = np.asarray(plot["node_rz_m"], dtype=np.float64)
    wall = np.asarray(plot["wall_rz_m"], dtype=np.float64)
    oracle = np.asarray(plot["oracle_flux_wb"], dtype=np.float64)
    image = np.asarray(plot["full_moment_image_wb"], dtype=np.float64)
    terminal = _committed_terminal_flux()[: len(node)]
    levels = np.linspace(
        float(min(oracle.min(), image.min(), terminal.min())),
        float(max(oracle.max(), image.max(), terminal.max())),
        13,
    )[1:-1]
    carrier, _source, exact = certificate._case(row["case"])
    axis_rz, x_points = _row_nulls(row["case"], exact)
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
    for column, (name, field) in enumerate(
        (
            ("frozen-current prediction", image),
            ("analytic flux", oracle),
            ("committed terminal state", terminal),
        )
    ):
        panel = poloidal_axes(axes[column])
        panel.tricontour(
            node[:, 0],
            node[:, 1],
            field,
            levels=levels,
            colors="royalblue",
            linewidths=0.6,
        )
        poloidal.draw_wall(panel, wall[:, 0], wall[:, 1], style=DEFAULT_INK)
        poloidal.draw_nulls(panel, magnetic_axis=axis_rz, x_points=x_points)
        panel.set_title(name, fontsize=8)
    figure.suptitle(
        f"{row['case']} {row['requested_cells']}: frozen-exact-current image, "
        "analytic flux and committed terminal state on shared Wb levels "
        "(blue contours; the frozen-current image is the coupling without "
        "iteration)",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return {
        "path": str(output.relative_to(ROOT)),
        "project_src": "/nova/figures/cut-cell-current-attribution/gate-b/"
        + output.name,
        "shared_flux_levels_wb": levels,
    }


# ---------------------------------------------------------------------------
# aggregate / CLI
# ---------------------------------------------------------------------------


def aggregate() -> dict[str, Any]:
    """Combine the six rungs and draw the shared figures."""
    rows = [
        json.loads(_part_path(case_name, cells).read_text(encoding="utf-8"))
        for case_name, cells in CASE_REQUESTS
    ]
    revisions = {row["source_revision"] for row in rows}
    if len(revisions) != 1:
        raise RuntimeError(f"rungs do not share one source revision: {revisions}")
    job_ids = set()
    for row in rows:
        lane = row["lane"]
        if (
            lane["execution"] != "slurm"
            or lane["partition"] != "all_debug"
            or lane["jax_platform"] != "cpu"
            or not lane["jax_enable_x64"]
        ):
            raise RuntimeError("every rung must be an all_debug CPU-x64 measurement")
        job_ids.add(lane["job_id"])
    if len(job_ids) != 1:
        raise RuntimeError("all six rows must share one scheduler allocation")

    error_figure = _draw_error_figure(rows, OUTPUT_ROOT / "absolute-flux-error.svg")
    weak_110 = next(
        row
        for row in rows
        if row["case"] == "weak-rotation-reactor-static"
        and row["requested_cells"] == -110
    )
    comparison_figure = _draw_comparison_panel(
        weak_110, OUTPUT_ROOT / "weak-110-frozen-prediction-analytic-terminal.svg"
    )

    compact_rows = []
    for row in rows:
        compact = dict(row)
        compact.pop("plot_data")
        compact_rows.append(compact)

    headline = {}
    for row in compact_rows:
        key = _case_key(row["case"], row["requested_cells"])
        whole = row["images"]["full"]["error"]["whole_domain"]
        headline[key] = {
            "flux_span_wb": row["flux_span_wb"],
            "pitch_m": row["pitch_m"],
            "cell_census": row["cell_census"],
            "full": {
                "max_absolute_error_wb": whole["max_absolute_error_wb"],
                "rms_absolute_error_wb": whole["rms_absolute_error_wb"],
                "max_error_over_span": whole["max_error_over_span"],
                "rms_error_over_span": whole["rms_error_over_span"],
            },
            "first_order_image_identical_to_full": bool(
                row["images"]["first_order"]["state_sha256_binary64"]
                == row["images"]["full"]["state_sha256_binary64"]
            ),
            "zeroth_only": {
                "max_absolute_error_wb": row["images"]["zeroth_only"]["error"][
                    "whole_domain"
                ]["max_absolute_error_wb"],
                "rms_absolute_error_wb": row["images"]["zeroth_only"]["error"][
                    "whole_domain"
                ]["rms_absolute_error_wb"],
                "max_error_over_span": row["images"]["zeroth_only"]["error"][
                    "whole_domain"
                ]["max_error_over_span"],
            },
        }

    receipt = {
        "schema": "nova.frozen-current-flux-image.v1",
        "completed": True,
        "measurement_driver_revision": revisions.pop(),
        "contract": {
            "requests": [
                {"case": case_name, "requested_cells": cells}
                for case_name, cells in CASE_REQUESTS
            ],
            "one_rung_per_fresh_process": True,
            "one_scheduler_job": True,
            "lane": "all_debug CPU float64",
            "coupling": (
                "frozen exact per-cell currents through the shipped polygon "
                "kernel with the certificate row's exterior; no clip, no "
                "iteration"
            ),
            "moment_ladder": (
                "full (zeroth + first) and zeroth-only (total current per "
                "cell); the kernel carries no density second-moment block, so "
                "second-order-zeroed equals full by construction"
            ),
        },
        "rows": compact_rows,
        "figures": {
            "absolute_error": error_figure,
            "weak_110_comparison": comparison_figure,
        },
        "headline": headline,
        "report": _per_row_report(compact_rows),
    }
    _write_json(RECEIPT, receipt)
    for case_name, cells in CASE_REQUESTS:
        _part_path(case_name, cells).unlink()
    parts = PARTS
    parts.rmdir()
    return receipt


def _pair_order_name(case_name: str) -> str:
    return f"{case_name}-110-vs-300"


def _per_row_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """State, per row, the coupling's error against the span, and the
    second-order-in-pitch reading across the 110 to 300 pair."""
    full_rows = {_case_key(row["case"], row["requested_cells"]): row for row in rows}
    sentences: dict[str, str] = {}
    pairs: dict[str, Any] = {}
    for case_name, requested_low, requested_high in (
        ("weak-rotation-reactor-static", -110, -300),
        ("diverted-single-null", -110, -300),
    ):
        low_key = _case_key(case_name, requested_low)
        high_key = _case_key(case_name, requested_high)
        low, high = full_rows[low_key], full_rows[high_key]
        if (
            low["requested_cells"] != requested_low
            or high["requested_cells"] != requested_high
        ):
            continue
        low_err = low["images"]["full"]["error"]["whole_domain"][
            "max_absolute_error_wb"
        ]
        low_span = low["flux_span_wb"]
        high_err = high["images"]["full"]["error"]["whole_domain"][
            "max_absolute_error_wb"
        ]
        high_span = high["flux_span_wb"]
        if not (low_err > 0.0 and high_err > 0.0 and min(low_span, high_span) > 0.0):
            continue
        # pitch scales as the square root of cell area; the total domain area
        # is shared across the 110-vs-300 pair, so the pitch ratio is the
        # inverse square root of the cell count ratio.
        low_cells = max(low["realised_cells"], 1)
        high_cells = max(high["realised_cells"], 1)
        pitch_ratio = float(np.sqrt(high_cells / low_cells))
        order = float(np.log(low_err / high_err) / np.log(pitch_ratio))
        pairs[_pair_order_name(case_name)] = {
            "low_row": low_key,
            "high_row": high_key,
            "low_max_error_wb": low_err,
            "high_max_error_wb": high_err,
            "low_max_error_over_span": low_err / low_span,
            "high_max_error_over_span": high_err / high_span,
            "pitch_ratio_low_to_high": pitch_ratio,
            "measured_order_in_pitch": order,
        }
    for row in rows:
        key = _case_key(row["case"], row["requested_cells"])
        whole = row["images"]["full"]["error"]["whole_domain"]
        maximum = whole["max_absolute_error_wb"]
        span = row["flux_span_wb"]
        fraction = maximum / span
        sentences[key] = (
            f"Coupling's own error on {row['case']} {row['requested_cells']} is "
            f"{maximum:.5g} Wb maximum ({whole['rms_absolute_error_wb']:.5g} Wb rms), "
            f"{fraction:.5g} of the {span:.5g} Wb flux span, with the image formed by "
            f"coupling_current_moments into current_moment_image over the fixed "
            f"polygon-analytic kernel blocks plus the certificate row's external."
        )
    return {
        "per_row_statements": sentences,
        "second_order_in_pitch_pairs": pairs,
        "image_formation_lines": (
            "nova/equilibrium/forward_operator.py: coupling_current_moments "
            "and current_moment_image (the polygon-analytic kernel blocks built "
            "by nova/biot/polygonanalytic.polygon_analytic_flux_moments), "
            "contracted with the frozen exact per-cell moments and added to "
            "the certificate operator's external field"
        ),
    }


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=tuple(sorted({name for name, _cells in CASE_REQUESTS}))
    )
    parser.add_argument("--requested-cells", type=int, choices=(-110, -300))
    parser.add_argument("--part", type=Path)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--output", type=Path, default=RECEIPT)
    return parser.parse_args()


def _worker_row(args: tuple[str, int]) -> dict[str, Any]:
    """Measure one row in a fresh worker process and persist its part."""
    configure_dtypes()
    case_name, requested_cells = args
    row = measure_rung(case_name, requested_cells)
    part = _part_path(case_name, requested_cells)
    _write_json(part, row)
    return {"case": case_name, "requested_cells": requested_cells}


def _run_all() -> dict[str, Any]:
    """Measure every declared row in parallel workers, then aggregate."""
    from concurrent.futures import ProcessPoolExecutor

    workers = len(CASE_REQUESTS)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_worker_row, CASE_REQUESTS))
    for result in results:
        print(
            f"FROZEN_FLUX_ROW_DONE case={result['case']} "
            f"requested={result['requested_cells']}",
            flush=True,
        )
    return aggregate()


def main() -> None:
    configure_dtypes()
    arguments = _parse()
    if arguments.run_all:
        receipt = _run_all()
        print(json.dumps(receipt["headline"], sort_keys=True), flush=True)
        print("FROZEN_FLUX_RUNALL_EXIT=0", flush=True)
        return
    if arguments.aggregate:
        receipt = aggregate()
        print(json.dumps(receipt["headline"], sort_keys=True), flush=True)
        print("FROZEN_FLUX_AGGREGATE_EXIT=0", flush=True)
        return
    if arguments.case is None or arguments.requested_cells is None:
        raise SystemExit("one --case and --requested-cells pair is required")
    if (arguments.case, arguments.requested_cells) not in CASE_REQUESTS:
        raise SystemExit("the requested case-resolution pair is outside the contract")
    part = arguments.part or _part_path(arguments.case, arguments.requested_cells)
    row = measure_rung(arguments.case, arguments.requested_cells)
    _write_json(part, row)
    print(f"FROZEN_FLUX_ROW_EXIT=0 part={part}", flush=True)


if __name__ == "__main__":
    main()
