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
    """Clip a square grid and merge small pieces along their longest boundary.

    The cell union preserves the authored material to floating-point roundoff.
    A hole inside one grid square is triangulated before merging. A disconnected
    sliver that cannot acquire a full edge neighbour is refused rather than
    passed to a near-singular collocation calculation.
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
    floor = sliver * size**2
    for k in sorted(range(len(pieces)), key=lambda k: pieces[k].area):
        piece = pieces[k]
        if piece is None or piece.area >= floor:
            continue
        neighbours = [
            (piece.boundary.intersection(other.boundary).length, j)
            for j, other in enumerate(pieces)
            if j != k and other is not None
        ]
        for shared, j in sorted(neighbours, reverse=True):
            if shared <= 0:
                break
            merged = piece.union(pieces[j])
            if isinstance(merged, shapely.Polygon) and not merged.interiors:
                pieces[j], pieces[k] = merged, None
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
        if (
            piece.area < floor * (1 - 1e-12)
            or not piece.contains(centre)
            or centre.distance(piece.boundary) < 0.01 * size
        ):
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


def _smoothness(cells, bodies):
    """Assemble the natural-boundary Dirichlet energy on each body separately.

    A face contributes length / centroid distance times |M_i - M_j|**2.
    This is the face-normal finite-volume approximation to integral |grad M|**2
    in the poloidal section. No outer face contributes. Cells spanning an
    authored body boundary are excluded because a constant cell value cannot
    represent the two independent traces there.
    """
    polygons = [shapely.Polygon(p) for p in cells.polygons]
    labels = np.full(len(polygons), -1)
    for i, polygon in enumerate(polygons):
        owners = [
            k
            for k, body in enumerate(bodies)
            if polygon.difference(body).area <= 1e-10 * polygon.area
        ]
        if len(owners) == 1:
            labels[i] = owners[0]
    laplacian = np.zeros((len(polygons), len(polygons)))
    tree = shapely.STRtree(polygons)
    for i, polygon in enumerate(polygons):
        for j in tree.query(polygon, predicate="intersects"):
            if j <= i or labels[i] < 0 or labels[i] != labels[j]:
                continue
            shared = polygon.boundary.intersection(polygons[j].boundary).length
            if shared == 0:
                continue
            distance = np.linalg.norm(cells.centroids[i] - cells.centroids[j])
            weight = shared / distance
            laplacian[i, i] += weight
            laplacian[j, j] += weight
            laplacian[i, j] -= weight
            laplacian[j, i] -= weight
    return laplacian


def _smooth_residual(state, operands, penalty):
    residual = _residual(state, operands)
    if jnp.ndim(penalty) == 2:
        residual += (state.reshape(2, -1) @ penalty.T).reshape(-1)
    return residual


@jax.jit
def _factor(state, operands, regularisation=0.0, penalty=0.0):
    """Factor the differential Jacobian with spatial and diagonal penalties."""
    jacobian = jax.jacfwd(_smooth_residual, argnums=0)(state, operands, penalty)
    jacobian += regularisation * jnp.eye(state.size, dtype=state.dtype)
    return jla.lu_factor(jacobian)


@jax.jit
def _update(state, operands, factor, scale, penalty=0.0):
    """Apply one Newton correction with runtime state, operands and LU factor."""
    residual = _smooth_residual(state, operands, penalty)
    direction = jla.lu_solve(factor, -residual)
    candidate = state + scale * direction
    return candidate, _smooth_residual(candidate, operands, penalty)


@dataclass(frozen=True)
class MagnetisationSolution:
    """Solved cell magnetisation and its convergence receipt.

    ``magnetisation`` is (cells, 2) [A/m], ``permeability_relative`` is (cells,).
    ``residual_history`` is the relative material residual, including the seed;
    ``iterations`` counts accepted corrections. ``regularisation_history``,
    ``reduction_ratios`` and ``step_types`` describe each accepted step beside
    ``step_scales``. The ratio measures reduction of the regularised local
    equation, so a regularised step can increase the unregularised residual.
    ``field`` evaluates the iron contribution alone through the sheet tile.
    """

    magnetisation: np.ndarray
    permeability_relative: np.ndarray
    residual_history: tuple[float, ...]
    step_scales: tuple[float, ...]
    converged: bool
    termination: str
    _mesh: _Mesh = field(repr=False)
    _plan: TilePlan = field(repr=False)
    regularisation_history: tuple[float, ...] = ()
    reduction_ratios: tuple[float, ...] = ()
    step_types: tuple[str, ...] = ()
    smoothness_lambda: float = 0.0
    smoothness_history: tuple[float, ...] = ()
    penalised_residual_history: tuple[float, ...] = ()

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
    initial_magnetisation: np.ndarray | None = None,
    smoothness_strength: float = 0.0,
    smoothness_bodies=None,
) -> MagnetisationSolution:
    """Mesh iron and solve M = c(|B|) B for B = B_external + G M.

    ``region`` is a shapely poloidal Polygon or MultiPolygon [m]. The callback
    receives (cells, 2) centroid coordinates and returns (B_R, B_Z) [T] in the
    same shape. The table contains B [T] starting at zero and secant relative
    permeability. Its implied H [A/m] must increase strictly. The law is
    single-valued and has no hysteresis or remanence. An optional initial
    magnetisation has shape (cells, 2) [A/m] on this mesh.

    Try exact Newton first, then adapt a diagonal regularisation lambda.
    The shifted equation F(M + p) + lambda*p = 0 is an implicit pseudo-time
    step. Its linear model predicts zero residual because
    (F'(M) + lambda*I)*p = -F(M). Accept when the actual reduction divided by
    that prediction exceeds one quarter; good agreement reduces lambda.
    This permits passage through a nonzero minimum of ||F|| while bounding
    the nonlinear defect of each regularised step. Trying lambda=0 first
    restores ordinary quadratic Newton convergence near a nonsingular root.
    A budget exhaustion or regularisation refusal has ``converged=False``.

    Optional ``smoothness_strength`` multiplies the initial Jacobian infinity
    norm to set a frozen spatial-penalty scale, exposed as ``smoothness_lambda``.
    The penalty is lambda/2 times the within-body Dirichlet energy. Supply
    ``smoothness_bodies`` as the distinct authored Polygon/MultiPolygon outlines
    when ``region`` is their union: a union loses touching segment boundaries.
    Without it, each connected part of ``region`` is one body. Boundary-spanning
    cells are excluded. The homotopy solves at full, tenth and hundredth scale,
    then at zero, so convergence always means the unpenalised material root.
    ``residual_history`` always records the original material residual;
    ``penalised_residual_history`` and ``smoothness_history`` identify the
    equation and lambda used for each attempted correction.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64, "the material solve requires float64"
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if not np.isfinite(smoothness_strength) or smoothness_strength < 0:
        raise ValueError("smoothness_strength must be finite and nonnegative")
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
    if initial_magnetisation is not None:
        initial = np.asarray(initial_magnetisation, dtype=np.float64)
        if initial.shape != external.shape or not np.all(np.isfinite(initial)):
            raise ValueError("initial_magnetisation must be finite (cells, 2)")
        state = jnp.asarray(initial.T.reshape(-1))
    smoothness_lambda, stage = 0.0, 0
    penalty = jnp.asarray(0.0)
    if smoothness_strength > 0:
        bodies = tuple(
            shapely.get_parts(region)
            if smoothness_bodies is None
            else smoothness_bodies
        )
        if not bodies or any(
            not isinstance(body, shapely.Polygon | shapely.MultiPolygon)
            or body.is_empty
            or not body.is_valid
            for body in bodies
        ):
            raise ValueError("smoothness_bodies must contain valid iron outlines")
        laplacian = jnp.asarray(_smoothness(cells, bodies))
        jacobian = jax.jacfwd(_residual)(state, operands)
        smoothness_lambda = smoothness_strength * float(
            jnp.max(jnp.sum(jnp.abs(jacobian), axis=1))
        )
        penalty = smoothness_lambda * laplacian
    smoothness_history, penalised_history = [], []
    history, scales = [], []
    regularisations, ratios, step_types = [], [], []
    regularisation = 1e-6
    reference = max(np.linalg.norm(external) / _MU0, 1e-30)
    termination = "iteration-budget"
    for iteration in range(max_iterations + 1):
        original = _residual(state, operands)
        denominator = max(float(jnp.linalg.norm(state)), reference)
        relative = float(jnp.linalg.norm(original)) / denominator
        residual = _smooth_residual(state, operands, penalty)
        norm = float(jnp.linalg.norm(residual))
        history.append(relative)
        while (
            smoothness_lambda > 0
            and stage < 3
            and norm / denominator <= max(tolerance, 1e-7)
        ):
            stage += 1
            penalty = smoothness_lambda * (10.0**-stage if stage < 3 else 0) * laplacian
            residual = _smooth_residual(state, operands, penalty)
            norm = float(jnp.linalg.norm(residual))
            regularisation = 1e-6
        active_lambda = smoothness_lambda * (10.0**-stage if stage < 3 else 0)
        smoothness_history.append(active_lambda)
        penalised_history.append(norm / denominator)
        if relative <= tolerance and (smoothness_lambda == 0 or stage >= 3):
            termination = "converged"
            break
        if iteration == max_iterations:
            break
        accepted = False
        for attempt in range(35):
            damping = 0.0 if attempt == 0 else regularisation
            factor = _factor(state, operands, jnp.asarray(damping), penalty)
            candidate, trial = _update(
                state, operands, factor, jnp.asarray(1.0), penalty
            )
            defect = trial + damping * (candidate - state)
            ratio = 1 - (float(jnp.linalg.norm(defect)) / norm) ** 2
            if ratio > 0.25 and bool(jnp.any(candidate != state)):
                state = candidate
                scales.append(1.0)
                regularisations.append(damping)
                ratios.append(ratio)
                step_types.append("newton" if damping == 0 else "regularised")
                if damping > 0:
                    if ratio > 0.9:
                        regularisation *= 0.3
                    elif ratio < 0.5:
                        regularisation *= 2
                accepted = True
                break
            if attempt > 0:
                regularisation *= 10
        if not accepted:
            termination = "regularisation-refused"
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
        tuple(regularisations),
        tuple(ratios),
        tuple(step_types),
        smoothness_lambda,
        tuple(smoothness_history),
        tuple(penalised_history),
    )
