"""Measure shared-node cell-average quadrature on an analytic equilibrium.

The benchmark derives one equal vertex weight and one centroid weight from the
constant and isotropic second moments of a cell.  A cell evaluation is then one
connectivity gather followed by one dot with that fixed table.  Node values are
shared by neighbouring cells; the table is built once for each cell family and
indexed without change on every equilibrium iteration.

Only cells whose complete polygon lies inside the analytic separatrix enter the
accuracy series.  Boundary clipping and coupling kernels are deliberately absent.
"""

from __future__ import annotations

from math import comb, factorial
import json
from typing import Any

import numpy as np
from scipy import integrate, stats

from benchmarks.efit_constant_current_attribution import (
    ANALYTIC_CASE,
    GRID_SEQUENCE,
    _current_antiderivative,
    _hex_mesh,
)
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import RotatingEquilibrium, reference_cases

EXACTNESS_TOLERANCE = 1.0e-14
REFERENCE_POINT_COUNT = 32


def _rectangle_vertices(width: float, height: float) -> np.ndarray:
    """Return counter-clockwise rectangular offsets from the centroid."""

    return np.asarray(
        [
            (-0.5 * width, -0.5 * height),
            (0.5 * width, -0.5 * height),
            (0.5 * width, 0.5 * height),
            (-0.5 * width, 0.5 * height),
        ]
    )


def _hexagon_vertices(pitch: float) -> np.ndarray:
    """Return the equal-area regular hexagon of the half-offset mesh."""

    circumradius = pitch / np.sqrt(3.0)
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return circumradius * np.column_stack((np.cos(angle), np.sin(angle)))


def _polygon_monomial_average(
    vertices: np.ndarray, radial: int, vertical: int
) -> float:
    """Integrate one Cartesian monomial exactly over a centred convex polygon."""

    total = 0.0
    area = 0.0
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        determinant = first[0] * second[1] - first[1] * second[0]
        triangle_area_factor = abs(determinant)
        area += 0.5 * triangle_area_factor
        for radial_first in range(radial + 1):
            radial_coefficient = (
                comb(radial, radial_first)
                * first[0] ** radial_first
                * second[0] ** (radial - radial_first)
            )
            for vertical_first in range(vertical + 1):
                vertical_coefficient = (
                    comb(vertical, vertical_first)
                    * first[1] ** vertical_first
                    * second[1] ** (vertical - vertical_first)
                )
                first_power = radial_first + vertical_first
                second_power = radial + vertical - first_power
                simplex_moment = (
                    factorial(first_power)
                    * factorial(second_power)
                    / factorial(first_power + second_power + 2)
                )
                total += (
                    triangle_area_factor
                    * radial_coefficient
                    * vertical_coefficient
                    * simplex_moment
                )
    return total / area


def _derive_centroid_vertex_rule(vertices: np.ndarray) -> np.ndarray:
    """Derive equal vertex and centroid weights from area moments."""

    radial_second = _polygon_monomial_average(vertices, 2, 0)
    vertical_second = _polygon_monomial_average(vertices, 0, 2)
    vertex_weight = radial_second / float(np.sum(vertices[:, 0] ** 2))
    vertical_weight = vertical_second / float(np.sum(vertices[:, 1] ** 2))
    if not np.isclose(vertex_weight, vertical_weight, rtol=0.0, atol=2.0e-15):
        raise AssertionError("one symmetric vertex weight does not match both moments")
    centroid_weight = 1.0 - len(vertices) * vertex_weight
    return np.r_[centroid_weight, np.full(len(vertices), vertex_weight)]


def _verify_polynomial_exactness(
    vertices: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    """Verify all monomials through the rule's exact degree."""

    points = np.vstack((np.zeros((1, 2)), vertices))
    rows: list[dict[str, float | int]] = []
    max_relative = 0.0
    max_zero_absolute = 0.0
    for total_degree in range(4):
        for radial_degree in range(total_degree + 1):
            vertical_degree = total_degree - radial_degree
            exact = _polygon_monomial_average(vertices, radial_degree, vertical_degree)
            sampled = float(
                weights
                @ (points[:, 0] ** radial_degree * points[:, 1] ** vertical_degree)
            )
            absolute = abs(sampled - exact)
            if abs(exact) <= EXACTNESS_TOLERANCE:
                relative = 0.0
                max_zero_absolute = max(max_zero_absolute, absolute)
            else:
                relative = absolute / abs(exact)
                max_relative = max(max_relative, relative)
            rows.append(
                {
                    "radial_degree": radial_degree,
                    "vertical_degree": vertical_degree,
                    "exact_average": exact,
                    "rule_average": sampled,
                    "relative_error": relative,
                    "absolute_error": absolute,
                }
            )
    if max(max_relative, max_zero_absolute) > EXACTNESS_TOLERANCE:
        raise AssertionError("centroid-vertex rule is not cubic-exact")

    fourth_exact = _polygon_monomial_average(vertices, 4, 0)
    fourth_sampled = float(weights @ points[:, 0] ** 4)
    if np.isclose(fourth_sampled, fourth_exact, rtol=1.0e-12, atol=1.0e-15):
        raise AssertionError("selected fourth-degree monomial unexpectedly integrates")
    return {
        "polynomial_exactness_degree": 3,
        "verification_tolerance": EXACTNESS_TOLERANCE,
        "maximum_relative_error_for_nonzero_monomials": max_relative,
        "maximum_absolute_error_for_zero_monomials": max_zero_absolute,
        "monomials": rows,
        "fourth_degree_counterexample": {
            "monomial": "radial_coordinate**4",
            "exact_average": fourth_exact,
            "rule_average": fourth_sampled,
            "relative_error": abs(fourth_sampled - fourth_exact) / abs(fourth_exact),
        },
    }


def _gauss_integral(function, lower: float, upper: float, point_count: int) -> float:
    """Integrate a scalar function with a fixed Gauss-Legendre rule."""

    node, weight = np.polynomial.legendre.leggauss(point_count)
    coordinate = 0.5 * (upper - lower) * node + 0.5 * (upper + lower)
    return float(0.5 * (upper - lower) * (weight @ function(coordinate)))


def _rectangle_exact_average(
    case: RotatingEquilibrium, radius: float, width: float
) -> float:
    """Return the closed-form full-rectangle current-density average."""

    lower = radius - 0.5 * width
    upper = radius + 0.5 * width
    return (
        _current_antiderivative(case, upper) - _current_antiderivative(case, lower)
    ) / width


def _rectangle_reference_average(
    case: RotatingEquilibrium, radius: float, width: float
) -> float:
    """Return the fixed thirty-two-point rectangular reference average."""

    lower = radius - 0.5 * width
    upper = radius + 0.5 * width
    return (
        _gauss_integral(
            lambda coordinate: case.toroidal_current_density(coordinate, 0.0),
            lower,
            upper,
            REFERENCE_POINT_COUNT,
        )
        / width
    )


def _hexagonal_height(offset: np.ndarray | float, pitch: float) -> np.ndarray:
    """Return the vertical chord of the point-up regular hexagon."""

    return 2.0 * (pitch - np.abs(offset)) / np.sqrt(3.0)


def _hexagon_exact_average(
    case: RotatingEquilibrium, radius: float, pitch: float
) -> tuple[float, float]:
    """Return an adaptively integrated full-hexagon average and error estimate."""

    half_width = 0.5 * pitch
    area = np.sqrt(3.0) * pitch**2 / 2.0

    def integrand(offset: float) -> float:
        return float(case.toroidal_current_density(radius + offset, 0.0)) * float(
            _hexagonal_height(offset, pitch)
        )

    left, left_error = integrate.quad(
        integrand,
        -half_width,
        0.0,
        epsabs=1.0e-8,
        epsrel=2.0e-13,
        limit=100,
    )
    right, right_error = integrate.quad(
        integrand,
        0.0,
        half_width,
        epsabs=1.0e-8,
        epsrel=2.0e-13,
        limit=100,
    )
    return (left + right) / area, (left_error + right_error) / area


def _hexagon_reference_average(
    case: RotatingEquilibrium, radius: float, pitch: float
) -> float:
    """Return a thirty-two-point rule split across the hexagon's two chords."""

    half_width = 0.5 * pitch
    area = np.sqrt(3.0) * pitch**2 / 2.0

    def integrand(offset: np.ndarray) -> np.ndarray:
        return case.toroidal_current_density(
            radius + offset, np.zeros_like(offset)
        ) * _hexagonal_height(offset, pitch)

    half_points = REFERENCE_POINT_COUNT // 2
    return (
        _gauss_integral(integrand, -half_width, 0.0, half_points)
        + _gauss_integral(integrand, 0.0, half_width, half_points)
    ) / area


def _errors(estimated: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Return absolute and peak-normalised cell-average density errors."""

    difference = estimated - exact
    peak = float(np.max(np.abs(exact)))
    sup = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference**2)))
    return {
        "sup_error_a_per_m2": sup,
        "rms_error_a_per_m2": rms,
        "sup_error_fraction_of_exact_peak": sup / peak,
        "rms_error_fraction_of_exact_peak": rms / peak,
    }


def _fit_order(cell_size: np.ndarray, error: np.ndarray) -> dict[str, float | str]:
    """Fit a power law, preserving a numerical-floor qualification."""

    positive = error > 0.0
    if np.count_nonzero(positive) < 2:
        return {"reading": "numerical floor; fewer than two positive errors"}
    fit = stats.linregress(np.log(cell_size[positive]), np.log(error[positive]))
    numerical_floor = fit.slope <= 0.0 or error[-1] >= error[0]
    return {
        "observed_order": float(fit.slope),
        "order_standard_error": float(fit.stderr),
        "r_squared": float(fit.rvalue**2),
        "reading": (
            "numerical-floor regression; slope is not a convergence claim"
            if numerical_floor
            else (
                "power-law fit"
                if np.all(positive)
                else "qualified fit over positive errors above exact cancellation"
            )
        ),
    }


def _unique_hex_vertices(radial_count: int, vertical_count: int) -> int:
    """Count shared vertices in the finite half-offset hexagonal patch."""

    radial_index, vertical_index = np.indices((radial_count, vertical_count))
    centre = np.column_stack(
        (
            (radial_index + 0.5 * vertical_index).ravel(),
            (np.sqrt(3.0) / 2.0 * vertical_index).ravel(),
        )
    )
    vertices = centre[:, None, :] + _hexagon_vertices(1.0)[None, :, :]
    return len(np.unique(np.round(vertices.reshape(-1, 2), 12), axis=0))


def _evaluation_economy(
    family: str, radial_count: int, vertical_count: int
) -> dict[str, float | int | str]:
    """Return finite-grid and amortised current-density evaluation counts."""

    cell_count = radial_count * vertical_count
    if family == "rectangle":
        node_count = (radial_count + 1) * (vertical_count + 1)
        sharing = "four vertices shared by up to four cells"
    else:
        node_count = _unique_hex_vertices(radial_count, vertical_count)
        sharing = "six vertices shared by up to three cells"
    stencil_total = node_count + cell_count
    return {
        "cell_count": cell_count,
        "shared_mesh_node_evaluations": node_count,
        "centroid_evaluations": cell_count,
        "stencil_total_evaluations": stencil_total,
        "stencil_amortised_evaluations_per_cell": stencil_total / cell_count,
        "centroid_only_total_evaluations": cell_count,
        "centroid_only_evaluations_per_cell": 1.0,
        "reference_total_evaluations": REFERENCE_POINT_COUNT * cell_count,
        "reference_evaluations_per_cell": REFERENCE_POINT_COUNT,
        "sharing": sharing,
    }


def _measure_family(
    family: str, case: RotatingEquilibrium, weights: np.ndarray
) -> dict[str, Any]:
    """Measure one cell family on every analytic lattice resolution."""

    rows: list[dict[str, Any]] = []
    for radial_count, vertical_count in GRID_SEQUENCE:
        half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
        mesh, width, height = _hex_mesh(
            radial_count, vertical_count, case.major_radius, half_height
        )
        if family == "rectangle":
            vertices = _rectangle_vertices(width, height)
            area = width * height
        else:
            vertices = _hexagon_vertices(width)
            area = np.sqrt(3.0) * width**2 / 2.0
        centres = np.asarray(mesh.coordinate)
        corners = centres[:, None, :] + vertices[None, :, :]
        inside = np.all(case.contains(corners[..., 0], corners[..., 1]), axis=1)
        selected_centres = centres[inside]
        selected_corners = corners[inside]
        centroid_density = case.toroidal_current_density(
            selected_centres[:, 0], selected_centres[:, 1]
        )
        corner_density = case.toroidal_current_density(
            selected_corners[..., 0], selected_corners[..., 1]
        )
        gathered = np.concatenate((centroid_density[:, None], corner_density), axis=1)
        stencil_density = gathered @ weights

        if family == "rectangle":
            exact_rows = [
                (
                    _rectangle_exact_average(case, float(radius), width),
                    0.0,
                    _rectangle_reference_average(case, float(radius), width),
                )
                for radius in selected_centres[:, 0]
            ]
        else:
            exact_rows = []
            for radius in selected_centres[:, 0]:
                exact, error_estimate = _hexagon_exact_average(
                    case, float(radius), width
                )
                exact_rows.append(
                    (
                        exact,
                        error_estimate,
                        _hexagon_reference_average(case, float(radius), width),
                    )
                )
        exact_density = np.asarray([row[0] for row in exact_rows])
        reference_density = np.asarray([row[2] for row in exact_rows])
        rows.append(
            {
                "radial_count": radial_count,
                "vertical_count": vertical_count,
                "interior_cell_count": int(np.count_nonzero(inside)),
                "characteristic_cell_size_m": float(np.sqrt(area)),
                "maximum_exact_integration_error_estimate_a_per_m2": float(
                    max(row[1] for row in exact_rows)
                ),
                "centroid_only": _errors(centroid_density, exact_density),
                "vertex_centroid_stencil": _errors(stencil_density, exact_density),
                "reference_32_point": _errors(reference_density, exact_density),
                "evaluation_economy": _evaluation_economy(
                    family, radial_count, vertical_count
                ),
            }
        )

    cell_size = np.asarray([row["characteristic_cell_size_m"] for row in rows])
    convergence: dict[str, Any] = {}
    for variant in (
        "centroid_only",
        "vertex_centroid_stencil",
        "reference_32_point",
    ):
        convergence[variant] = {
            metric: _fit_order(
                cell_size,
                np.asarray([row[variant][metric] for row in rows]),
            )
            for metric in ("sup_error_a_per_m2", "rms_error_a_per_m2")
        }
    return {"resolutions": rows, "convergence": convergence}


def measure() -> dict[str, Any]:
    """Derive the rules and return their exactness, accuracy, and cost receipts."""

    configure_dtypes()
    case = reference_cases()[ANALYTIC_CASE]
    rectangle_unit = _rectangle_vertices(2.0, 2.0)
    hexagon_unit = _hexagon_vertices(np.sqrt(3.0))
    rectangle_weights = _derive_centroid_vertex_rule(rectangle_unit)
    hexagon_weights = _derive_centroid_vertex_rule(hexagon_unit)
    rules = {
        "rectangle": {
            "point_order": "centroid followed by four counter-clockwise corners",
            "weights": rectangle_weights.tolist(),
            "closed_form_weights": {"centroid": "2/3", "each_vertex": "1/12"},
            "centroid_weight": float(rectangle_weights[0]),
            "weight_per_vertex": float(rectangle_weights[1]),
            "edge_midpoints_required_for_degree_three": False,
            "exactness": _verify_polynomial_exactness(
                rectangle_unit, rectangle_weights
            ),
        },
        "regular_hexagon": {
            "point_order": "centroid followed by six angle-ordered vertices",
            "weights": hexagon_weights.tolist(),
            "closed_form_weights": {"centroid": "7/12", "each_vertex": "5/72"},
            "centroid_weight": float(hexagon_weights[0]),
            "weight_per_vertex": float(hexagon_weights[1]),
            "edge_midpoints_required_for_degree_three": False,
            "exactness": _verify_polynomial_exactness(hexagon_unit, hexagon_weights),
        },
    }
    return {
        "analytic_source": ANALYTIC_CASE,
        "interior_cells_only": True,
        "rule_application": {
            "per_cell": "one connectivity gather plus one fixed-weight dot",
            "per_iteration": (
                "evaluate current density once at every shared mesh node and once "
                "at every cell centroid"
            ),
            "precomputation": (
                "build each geometry's weight table once, then index it per cell"
            ),
        },
        "reference_rule": {
            "evaluations_per_cell": REFERENCE_POINT_COUNT,
            "definition": (
                "fixed Gauss-Legendre chord integration of the analytic source; "
                "the exact equilibrium current density is independent of height"
            ),
            "rectangle": "thirty-two radial nodes over the constant-height chord",
            "regular_hexagon": (
                "sixteen radial nodes on each side of the centre, weighted by the "
                "linear hexagonal chord height"
            ),
        },
        "rules": rules,
        "measurements": {
            "rectangle": _measure_family("rectangle", case, rectangle_weights),
            "regular_hexagon": _measure_family(
                "regular_hexagon", case, hexagon_weights
            ),
        },
    }


def main() -> None:
    """Print the stable JSON evidence record."""

    print(json.dumps(measure(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
