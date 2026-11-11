"""Anhysteretic axisymmetric iron, represented by magnetised polygon cells.

Only :func:`solve_magnetisation` and its result are public. Geometry, the B-H
interpolant and the sheet-tile response stay inside this module. Fields and
magnetisations use component order (radial, vertical), in tesla and A/m.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
from scipy.interpolate import PchipInterpolator
import shapely

from nova.biot.tiledassembly import TilePlan, cell_interaction
from nova.jax.config import configure_dtypes

__all__ = ["MagnetisationSolution", "solve_magnetisation"]

_MU0 = 4e-7 * np.pi
_PLAN = TilePlan(64, 256, 256, 16, 48)
_SHEET_LENGTH_SCALE = 1024.0


@dataclass(frozen=True)
class _Mesh:
    polygons: tuple[np.ndarray, ...]
    centroids: np.ndarray
    areas: np.ndarray


def _mesh(region, size: float, sliver: float = 0.15) -> _Mesh:
    """Clip a shared-edge grid and merge cells with insufficient clearance.

    A rectangular strip narrower than h/2 has centroid clearance below h/4;
    merging on that bound removes strips even when their area exceeds the
    sliver floor. The area target is h**2/2. Prefer the smallest union already
    meeting both targets, with longest-edge merges for pieces needing further
    agglomeration. A physically thin
    component can miss the targets, but must admit an interior point at 0.01 h.
    The cell union preserves the authored material to floating-point roundoff.
    A hole inside one grid square is triangulated before merging.
    """
    if not np.isfinite(size) or size <= 0:
        raise ValueError("cell_size must be finite and positive")
    if (
        not isinstance(region, shapely.Polygon | shapely.MultiPolygon)
        or region.is_empty
        or not region.is_valid
        or region.bounds[0] < 0
    ):
        raise ValueError("iron must be a valid nonempty polygon at nonnegative radius")
    r0, z0, r1, z1 = region.bounds
    # Adjacent squares must use identical endpoints to share exact edges.
    radial = np.r_[np.arange(r0, r1, size), r1]
    vertical = np.r_[np.arange(z0, z1, size), z1]
    pieces = []
    for r, right in zip(radial[:-1], radial[1:], strict=True):
        for z, top in zip(vertical[:-1], vertical[1:], strict=True):
            clipped = shapely.box(r, z, right, top).intersection(region)
            for piece in shapely.get_parts(clipped):
                if isinstance(piece, shapely.Polygon) and piece.area > 0:
                    if piece.interiors:
                        pieces.extend(
                            shapely.get_parts(
                                shapely.constrained_delaunay_triangles(piece)
                            )
                        )
                    else:
                        pieces.append(piece)
    floor = max(sliver, 0.5) * size**2
    clearance = 0.25 * size
    while True:
        changed = False
        pending = sorted(
            (k for k, piece in enumerate(pieces) if piece is not None),
            key=lambda k: pieces[k].area,
        )
        for k in pending:
            piece = pieces[k]
            if piece is None or (
                piece.area >= floor * (1 - 1e-12)
                and piece.contains(piece.centroid)
                and piece.centroid.distance(piece.boundary) >= clearance * (1 - 1e-12)
            ):
                continue
            neighbours = [
                (piece.boundary.intersection(other.boundary).length, j)
                for j, other in enumerate(pieces)
                if j != k and other is not None
            ]
            candidates = []
            for shared, j in neighbours:
                if shared <= 0:
                    continue
                merged = piece.union(pieces[j])
                if isinstance(merged, shapely.Polygon) and not merged.interiors:
                    centre = merged.centroid
                    qualified = (
                        merged.area >= floor * (1 - 1e-12)
                        and merged.contains(centre)
                        and centre.distance(merged.boundary) >= clearance * (1 - 1e-12)
                    )
                    # The smallest qualified union preserves resolution across
                    # thin material; an inward full cell need not be consumed.
                    candidates.append(
                        (
                            not qualified,
                            merged.area if qualified else -shared,
                            -shared,
                            j,
                            merged,
                        )
                    )
            if candidates:
                *_, j, merged = min(candidates, key=lambda candidate: candidate[:4])
                pieces[j], pieces[k] = merged, None
                changed = True
        if not changed:
            break
    polygons, centres, areas = [], [], []
    duplicate_tolerance = 1024 * np.finfo(float).eps * max(1.0, *np.abs(region.bounds))
    for piece in pieces:
        if piece is None:
            continue
        # Clipping can leave consecutive vertices separated only by roundoff.
        # Such duplicates are not physical edges and collapse in the sheet tile.
        piece = shapely.remove_repeated_points(piece, tolerance=duplicate_tolerance)
        centre = piece.centroid
        if not piece.contains(centre) or centre.distance(piece.boundary) < clearance:
            # An irreducible concave or thin component needs an interior
            # collocation point; its area centroid can lie outside the iron.
            circle = shapely.maximum_inscribed_circle(piece, tolerance=1e-5 * size)
            interior = shapely.Point(circle.coords[0])
            if interior.distance(piece.boundary) > centre.distance(
                piece.boundary
            ) or not piece.contains(centre):
                centre = interior
        if not piece.contains(centre) or centre.distance(piece.boundary) < 0.01 * size:
            raise ValueError("unresolved iron sliver; reduce cell_size")
        polygons.append(np.asarray(piece.exterior.coords)[:-1])
        centres.append(centre.coords[0])
        areas.append(piece.area)
    if not polygons:
        raise ValueError("iron mesh has no cells")
    return _Mesh(tuple(polygons), np.asarray(centres), np.asarray(areas))


def _interaction(cells, points, plan):
    # B/M is invariant under a common dilation of source and target coordinates.
    # A power of two preserves input bits while moving the tile's nanometre edge
    # identity grid below a picometre in the authored geometry.
    scale = _SHEET_LENGTH_SCALE
    return cell_interaction(
        tuple(p * scale for p in cells.polygons),
        points[:, 0] * scale,
        points[:, 1] * scale,
        plan=plan,
    )


def _fit_law(b_field, permeability_relative):
    """Return piecewise polynomial H(B) with strictly positive, continuous dH/dB.

    Interpolate H, not permeability: a rising permeability is compatible with
    positive differential permeability only while H still rises. Restrict each
    interior Hermite slope to (0, 2*adjacent secant]. The origin slope matches
    its secant permeability; the final slope is 1/mu0 for saturation. An
    interval whose end slope exceeds twice its secant uses positive linear
    derivative ramps joined by a positive constant derivative, integrated as
    two quadratics and a line. Their areas reproduce the interval's H increment exactly.
    """
    b = np.asarray(b_field, dtype=np.float64)
    mu = np.asarray(permeability_relative, dtype=np.float64)
    if (
        b.ndim != 1
        or b.size < 2
        or b.shape != mu.shape
        or not np.all(np.isfinite(b))
        or not np.all(np.isfinite(mu))
        or b[0] != 0
        or np.any(np.diff(b) <= 0)
        or np.any(mu < 1)
    ):
        raise ValueError("B-H table needs increasing B from zero and finite mu_r >= 1")
    h = b / (_MU0 * mu)
    secant = np.diff(h) / np.diff(b)
    if np.any(secant <= 0):
        raise ValueError(
            "B-H table must have strictly positive differential permeability"
        )
    slopes = PchipInterpolator(b, h).derivative()(b)
    bound = np.r_[secant[0], np.minimum(secant[:-1], secant[1:]), secant[-1]]
    slopes = np.clip(slopes, bound * 1e-6, 2 * bound)
    slopes[0] = 1 / (_MU0 * mu[0])
    slopes[-1] = 1 / _MU0
    width = np.diff(b)
    cubic = (slopes[:-1] + slopes[1:] - 2 * secant) / width**2
    quadratic = (3 * secant - 2 * slopes[:-1] - slopes[1:]) / width
    coefficients = np.stack([cubic, quadratic, slopes[:-1], h[:-1]])
    knots, pieces = [b[0]], []
    for i, average in enumerate(secant):
        left, right = slopes[i : i + 2]
        if max(left, right) <= 2 * average:
            pieces.append(coefficients[:, i])
        else:
            # Each ramp occupies fraction f of the interval. Its endpoint
            # contribution is at most average/4, leaving a positive plateau.
            fraction = min(0.25, average / (2 * (left + right)))
            edge = fraction * width[i]
            start, stop = b[i] + edge, b[i + 1] - edge
            left_edge, right_edge = start - b[i], b[i + 1] - stop
            # Use the represented knot distances so even narrow ramps join
            # at the tabulated value and tangent after coordinate rounding.
            middle = (h[i + 1] - h[i] - (left_edge * left + right_edge * right) / 2) / (
                width[i] - (left_edge + right_edge) / 2
            )
            first = h[i] + left_edge * (left + middle) / 2
            last = h[i + 1] - right_edge * (middle + right) / 2
            pieces.extend(
                [
                    [0, (middle - left) / (2 * left_edge), left, h[i]],
                    [0, 0, middle, first],
                    [0, (right - middle) / (2 * right_edge), middle, last],
                ]
            )
            knots.extend([start, stop])
        knots.append(b[i + 1])
    return np.asarray(knots), np.asarray(pieces).T, 1 / (_MU0 * mu[[0, -1]])


def _coefficient(magnitude, knots, coefficients, end_reluctivity):
    """Return c(B) = M(B)/B with constant magnetisation above the table."""
    index = jnp.clip(
        jnp.searchsorted(knots, magnitude, side="right") - 1, 0, knots.size - 2
    )
    offset = magnitude - knots[index]
    a, b, c, d = coefficients[:, index]
    h = ((a * offset + b) * offset + c) * offset + d
    safe = jnp.where(magnitude > 0, magnitude, 1.0)
    reluctivity = jnp.where(magnitude > 0, h / safe, end_reluctivity[0])
    saturated = knots[-1] * (1 / _MU0 - end_reluctivity[1]) / safe
    return jnp.where(magnitude >= knots[-1], saturated, 1 / _MU0 - reluctivity)


def _residual(state, operands):
    gram, external, knots, coefficients, ends = operands
    total = (external + gram @ state).reshape(2, -1)
    # A safe norm keeps the derivative finite at a demagnetised cell. The
    # coefficient there is exactly the origin value, independently of direction.
    squared = jnp.sum(total**2, axis=0)
    magnitude = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
    magnitude = jnp.where(squared > 0, magnitude, 0.0)
    coefficient = _coefficient(magnitude, knots, coefficients, ends)
    return state - (coefficient * total).reshape(-1)


@jax.jit
def _factor(state, operands):
    """Form and factor the differential material Jacobian at a runtime state."""
    jacobian = jax.jacfwd(_residual, argnums=0)(state, operands)
    return jla.lu_factor(jacobian)


@jax.jit
def _update(state, operands, factor, scale):
    """Apply one Newton correction with runtime state, operands and LU factor."""
    residual = _residual(state, operands)
    direction = jla.lu_solve(factor, -residual)
    candidate = state + scale * direction
    return candidate, _residual(candidate, operands)


@dataclass(frozen=True)
class MagnetisationSolution:
    """Solved cell magnetisation and its convergence receipt.

    ``magnetisation`` is (cells, 2) [A/m], ``permeability_relative`` is (cells,).
    ``residual_history`` is the relative material residual, including the seed;
    ``iterations`` counts accepted Newton corrections. ``field`` evaluates the
    iron contribution alone, using the same sheet-tile program at new targets.
    """

    magnetisation: np.ndarray
    permeability_relative: np.ndarray
    residual_history: tuple[float, ...]
    step_scales: tuple[float, ...]
    converged: bool
    termination: str
    _mesh: _Mesh = field(repr=False)
    _plan: TilePlan = field(repr=False)

    @property
    def iterations(self) -> int:
        return len(self.step_scales)

    @property
    def cell_count(self) -> int:
        return self.magnetisation.shape[0]

    def field(self, points: np.ndarray) -> np.ndarray:
        """Return the iron's (B_R, B_Z) [T] at (R, Z) points [m]."""
        points = _points(points)
        if not len(points):
            return np.empty((0, 2))
        response = _interaction(self._mesh, points, self._plan)
        return (response @ self.magnetisation.T.reshape(-1)).reshape(2, -1).T


def _points(points):
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.all(np.isfinite(points))
        or np.any(points[:, 0] < 0)
    ):
        raise ValueError("points must be finite (R, Z) pairs at nonnegative radius")
    return points


def solve_magnetisation(
    region,
    external_field: Callable[[np.ndarray], np.ndarray],
    b_field,
    permeability_relative,
    *,
    cell_size: float,
    tolerance: float = 1e-8,
    max_iterations: int = 60,
    tile_plan: TilePlan = _PLAN,
) -> MagnetisationSolution:
    """Mesh iron and solve M = c(|B|) B for B = B_external + G M.

    ``region`` is a shapely poloidal Polygon or MultiPolygon [m]. The callback
    receives (cells, 2) centroid coordinates and returns (B_R, B_Z) [T] in the
    same shape. The table contains B [T] starting at zero and secant relative
    permeability. Its implied H [A/m] must increase strictly. The law is
    single-valued and has no hysteresis or remanence. A result that exhausts
    the budget or refuses every backtracking grade has ``converged=False``.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64, "the material solve requires float64"
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    law = _fit_law(b_field, permeability_relative)
    cells = _mesh(region, cell_size)
    external = np.asarray(external_field(cells.centroids.copy()), dtype=np.float64)
    if external.shape != cells.centroids.shape or not np.all(np.isfinite(external)):
        raise ValueError(
            "external_field must return finite (cells, 2) field components"
        )
    gram = _interaction(cells, cells.centroids, tile_plan)
    if not np.all(np.isfinite(gram)):
        raise ValueError("the sheet interaction contains nonfinite entries")
    operands = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (gram, external.T.reshape(-1), *law)
    )
    state = jnp.zeros(external.size, dtype=jnp.float64)
    history, scales = [], []
    reference = max(np.linalg.norm(external) / _MU0, 1e-30)
    termination = "iteration-budget"
    for iteration in range(max_iterations + 1):
        residual = _residual(state, operands)
        norm = float(jnp.linalg.norm(residual))
        relative = norm / max(float(jnp.linalg.norm(state)), reference)
        history.append(relative)
        if relative <= tolerance:
            termination = "converged"
            break
        if iteration == max_iterations:
            break
        factor = _factor(state, operands)
        accepted = False
        for grade in range(15):
            scale = 0.5**grade
            candidate, trial = _update(state, operands, factor, jnp.asarray(scale))
            if float(jnp.linalg.norm(trial)) < norm:
                state = candidate
                scales.append(scale)
                accepted = True
                break
        if not accepted:
            termination = "backtracking-refused"
            break
    total = (operands[1] + operands[0] @ state).reshape(2, -1)
    magnitude = jnp.linalg.norm(total, axis=0)
    coefficient = _coefficient(magnitude, *operands[2:])
    mu = 1 / (1 - _MU0 * coefficient)
    return MagnetisationSolution(
        np.asarray(state).reshape(2, -1).T,
        np.asarray(mu),
        tuple(history),
        tuple(scales),
        termination == "converged",
        termination,
        cells,
        tile_plan,
    )
