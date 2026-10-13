"""Warm axis and saddle solves with a bounded, explicitly maintained Bell cache.

``maintain_rows`` builds or slides a row window outside the compiled solve.
``solve`` contracts only the iterate triangle's rows and returns NEEDS_ROWS
without building anything when its trajectory exits that window. All numerical
mesh, row and moment operands remain runtime arguments to the device program.
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from math import factorial
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygonanalytic import packed_analytic_moments
from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy

__all__ = ["solve", "maintain_rows"]

POWERS = tuple(
    (degree - vertical, vertical)
    for degree in range(6)
    for vertical in range(degree + 1)
)
DERIVATIVES = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


def _polynomial_row(point, derivative=(0, 0)):
    """Evaluate a derivative of the complete quintic monomial basis."""
    dx, dy = derivative
    return np.asarray(
        [
            factorial(a)
            / factorial(a - dx)
            * factorial(b)
            / factorial(b - dy)
            * point[0] ** (a - dx)
            * point[1] ** (b - dy)
            if a >= dx and b >= dy
            else 0.0
            for a, b in POWERS
        ]
    )


def _triangle_transform(vertices):
    """Map 18 scaled Cartesian vertex jets into 21 polynomial coefficients.

    The three homogeneous constraints remove the quartic term of each edge's
    normal derivative. Isotropic scaling preserves physical edge normals.
    """
    rows = [
        _polynomial_row(vertex, derivative)
        for vertex in vertices
        for derivative in DERIVATIVES
    ]
    for left, right in ((0, 1), (1, 2), (2, 0)):
        tangent = vertices[right] - vertices[left]
        tangent /= np.linalg.norm(tangent)
        normal = np.array((-tangent[1], tangent[0]))
        constraint = normal[0] * _polynomial_row(tangent, (1, 0)) + normal[
            1
        ] * _polynomial_row(tangent, (0, 1))
        constraint[:15] = 0.0
        rows.append(constraint)
    return np.linalg.solve(np.asarray(rows), np.eye(21)[:, :18])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BellMesh:
    """Four triangles per hex, fanned from its first existing sample vertex."""

    triangles: jax.Array
    origins: jax.Array
    scales: jax.Array
    transforms: jax.Array
    inverse_edges: jax.Array
    generators: jax.Array
    local_vertices: jax.Array


def _build_mesh(nodes, sample_index, geometry):
    nodes, indices = np.asarray(nodes), np.asarray(sample_index)
    triangles = np.stack(
        [indices[:, [1, edge, edge + 1]] for edge in range(2, 6)], axis=1
    )
    vertices = nodes[triangles]
    origins = vertices[:, :, 0]
    scales = np.broadcast_to(np.asarray(geometry.pitch)[:, None], triangles.shape[:2])
    local = (vertices - origins[:, :, None]) / scales[:, :, None, None]
    transforms = np.asarray(
        [[_triangle_transform(triangle) for triangle in cell] for cell in local]
    )
    edges = np.stack((local[:, :, 1], local[:, :, 2]), axis=-1)
    return BellMesh(
        *(
            jnp.asarray(value)
            for value in (
                triangles,
                origins,
                scales,
                transforms,
                np.linalg.inv(edges),
                nodes[indices[:, 1:]].mean(axis=1),
                local,
            )
        )
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _BellField:
    """Piecewise Bell polynomial; shared jets make value and gradient continuous."""

    coefficients: jax.Array
    mesh: BellMesh
    jets: jax.Array
    sample_index: jax.Array

    def local_value(self, cell, triangle, point):
        local = (point - self.mesh.origins[cell, triangle]) / self.mesh.scales[
            cell, triangle
        ]
        # Integer powers remain finite through high derivatives at a vertex.
        x = jnp.stack(tuple(local[0] ** degree for degree in range(6)))
        y = jnp.stack(tuple(local[1] ** degree for degree in range(6)))
        basis = jnp.stack(tuple(x[a] * y[b] for a, b in POWERS))
        return jnp.dot(self.coefficients[cell, triangle], basis)

    def value(self, point):
        cell = jnp.argmin(jnp.sum((self.mesh.generators - point) ** 2, axis=-1))
        return self.cell_value(cell, point)

    def cell_value(self, cell, point):
        """Select a fan triangle within a known hex cell."""
        local = (point - self.mesh.origins[cell]) / self.mesh.scales[cell, :, None]
        bary = jnp.einsum("tij,tj->ti", self.mesh.inverse_edges[cell], local)
        bary = jnp.concatenate(
            (bary, 1 - jnp.sum(bary, axis=-1, keepdims=True)), axis=-1
        )
        triangle = jnp.argmax(jnp.min(bary, axis=-1))
        return self.local_value(cell, triangle, point)

    def evaluate(self, point):
        return topology.FieldJet(
            self.value(point),
            jax.grad(self.value)(point),
            jax.hessian(self.value)(point),
        )


def _field_from_jets(jets, sample_index, mesh):
    scale = mesh.scales
    factors = jnp.stack(
        (jnp.ones_like(scale), scale, scale, scale**2, scale**2, scale**2), axis=-1
    )
    scaled = jets[mesh.triangles] * factors[:, :, None, :]
    anchor = scaled[:, :, 0]
    x, y = mesh.local_vertices[..., 0], mesh.local_vertices[..., 1]
    f, gx, gy, hxx, hxy, hyy = (anchor[..., k, None] for k in range(6))
    taylor = jnp.stack(
        (
            f + gx * x + gy * y + 0.5 * hxx * x * x + hxy * x * y + 0.5 * hyy * y * y,
            gx + hxx * x + hxy * y,
            gy + hxy * x + hyy * y,
            jnp.broadcast_to(hxx, x.shape),
            jnp.broadcast_to(hxy, x.shape),
            jnp.broadcast_to(hyy, x.shape),
        ),
        axis=-1,
    )
    # Removing a complete quadratic preserves the Bell space while avoiding
    # cancellation between the flux offset and the higher polynomial terms.
    residual = (scaled - taylor).reshape((*scale.shape, 18))
    coefficients = jnp.einsum("ctij,ctj->cti", mesh.transforms, residual)
    coefficients = coefficients.at[..., :6].add(
        anchor * jnp.asarray((1.0, 1.0, 1.0, 0.5, 1.0, 0.5))
    )
    return _BellField(coefficients, mesh, jets, sample_index)


def _source_jets(point, coupling):
    """Value, gradient and symmetric Hessian for each booked moment basis."""

    def value_gradient(target):
        rows = packed_analytic_moments(
            jnp,
            jnp.broadcast_to(target[0], coupling.norm.shape),
            jnp.broadcast_to(target[1], coupling.norm.shape),
            coupling.edge,
            coupling.weight,
            coupling.norm,
            coupling.centre.T,
            coupling.centre.T,
            coupling.reflection_axis,
            coupling.reflection_partner,
        )
        value = jnp.stack(rows[:3], axis=-1)
        gradient = (
            2
            * jnp.pi
            * target[0]
            * jnp.stack(
                (jnp.stack(rows[6:9], axis=-1), -jnp.stack(rows[3:6], axis=-1)),
                axis=-1,
            )
        )
        return value, gradient

    value, gradient = value_gradient(point)
    hessian = jax.jacfwd(lambda target: value_gradient(target)[1])(point)
    coefficients = jnp.concatenate(
        (
            value[..., None],
            gradient,
            hessian[..., 0, 0, None],
            hessian[..., 0, 1, None],
            hessian[..., 1, 1, None],
        ),
        axis=-1,
    )
    inverse = coupling.inverse_second_moment
    transform = jnp.stack(
        (
            jnp.stack(
                (jnp.ones(len(value)), jnp.zeros(len(value)), jnp.zeros(len(value))),
                axis=-1,
            ),
            jnp.stack(
                (jnp.zeros(len(value)), inverse[:, 0, 0], inverse[:, 0, 1]), axis=-1
            ),
            jnp.stack(
                (jnp.zeros(len(value)), inverse[:, 1, 0], inverse[:, 1, 1]), axis=-1
            ),
        ),
        axis=1,
    )
    return jnp.einsum("cbq,cbm->cmq", coefficients, transform)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _CachedField:
    """Contract only the selected triangle's rows; refuse an uncached cell.

    The bounded cache holds coupling rows, never the current state's dense
    nodal jets. Its lookup maps the mesh's node ids to cache slots.
    """

    mesh: BellMesh
    lookup: jax.Array
    rows: jax.Array
    moments: jax.Array
    exterior: jax.Array

    def _location(self, point):
        cell = jnp.argmin(jnp.sum((self.mesh.generators - point) ** 2, axis=-1))
        local = (point - self.mesh.origins[cell]) / self.mesh.scales[cell, :, None]
        bary = jnp.einsum("tij,tj->ti", self.mesh.inverse_edges[cell], local)
        bary = jnp.concatenate((bary, 1 - jnp.sum(bary, axis=-1, keepdims=True)), -1)
        triangle = jnp.argmax(jnp.min(bary, axis=-1))
        slots = self.lookup[self.mesh.triangles[cell, triangle]]
        return cell, triangle, slots

    def contains(self, point):
        return jnp.all(self._location(point)[2] >= 0)

    def evaluate(self, point):
        cell, triangle, slots = self._location(point)
        jets = self.exterior[slots] + jnp.einsum(
            "ncmq,cm->nq", self.rows[slots], self.moments
        )
        jets = jnp.where(jnp.all(slots >= 0), jets, jnp.nan)
        # Reuse the Bell coefficient construction on this one triangle only.
        local_mesh = BellMesh(
            jnp.arange(3)[None, None, :],
            self.mesh.origins[cell, triangle][None, None, :],
            self.mesh.scales[cell, triangle][None, None],
            self.mesh.transforms[cell, triangle][None, None, :, :],
            self.mesh.inverse_edges[cell, triangle][None, None, :, :],
            self.mesh.generators[cell][None, :],
            self.mesh.local_vertices[cell, triangle][None, None, :, :],
        )
        field = _field_from_jets(jets, jnp.zeros((1, 1), dtype=jnp.int32), local_mesh)
        # Evaluate all six polynomial derivatives in one contraction. Integer
        # power tables avoid both repeated autodiff traces and negative powers.
        scale = self.mesh.scales[cell, triangle]
        local = (point - self.mesh.origins[cell, triangle]) / scale
        x = jnp.stack(tuple(local[0] ** degree for degree in range(6)))
        y = jnp.stack(tuple(local[1] ** degree for degree in range(6)))
        powers = np.asarray(POWERS)
        derivative = np.asarray(DERIVATIVES)
        exponents = powers[None] - derivative[:, None]
        factors = np.asarray(
            [
                [
                    factorial(a) / factorial(a - dx) * factorial(b) / factorial(b - dy)
                    if a >= dx and b >= dy
                    else 0.0
                    for a, b in POWERS
                ]
                for dx, dy in DERIVATIVES
            ]
        )
        basis = (
            x[jnp.asarray(np.maximum(exponents[..., 0], 0))]
            * y[jnp.asarray(np.maximum(exponents[..., 1], 0))]
            * jnp.asarray(factors)
        )
        jet = (basis @ field.coefficients[0, 0]) / jnp.asarray(
            (1.0, scale, scale, scale**2, scale**2, scale**2)
        )
        return topology.FieldJet(
            jet[0], jet[1:3], jnp.asarray(((jet[3], jet[4]), (jet[4], jet[5])))
        )


class FixedPointStatus(IntEnum):
    """Per-point termination, including a recoverable row-window miss."""

    CONVERGED = 0
    NEEDS_ROWS = 1
    WRONG_ROOT = 2
    NOT_CONVERGED = 3


class FixedPointResult(NamedTuple):
    """Device result; only valid points may replace a caller's warm starts."""

    position: jax.Array
    jet: topology.FieldJet
    valid: jax.Array
    status: jax.Array
    iterations: jax.Array
    retried: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RowCache:
    """Immutable bounded rows tied to one mesh, coupling and exterior supply.

    Reuse is valid only for the same geometry and coupling. Changing either
    requires passing ``cache=None`` to maintenance. Exterior jets are refreshed
    on every maintenance call, so a prescribed exterior may move independently.
    ``rows`` has shape (capacity, source cells, three moments, six jet entries).
    """

    mesh: BellMesh
    nodes: jax.Array
    lookup: jax.Array
    rows: jax.Array
    exterior: jax.Array


@dataclass(frozen=True)
class RowMaintenance:
    """Synchronized host maintenance receipt, excluded from solve latency."""

    seconds: float
    nodes_added: int
    nodes_retained: int
    active_nodes: int
    cache_bytes: int


_row_program = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))


def _stack_jet(jet: topology.FieldJet) -> jax.Array:
    return jnp.stack(
        (
            jet.value,
            jet.gradient[..., 0],
            jet.gradient[..., 1],
            jet.hessian[..., 0, 0],
            jet.hessian[..., 0, 1],
            jet.hessian[..., 1, 1],
        ),
        axis=-1,
    )


_exterior_program = jax.jit(
    lambda exterior, points: _stack_jet(jax.vmap(exterior.evaluate)(points))
)


def maintain_rows(
    geometry: topology.TopologyGeometry,
    coupling: topology.BiotMomentCoupling,
    exterior: topology.ExteriorField,
    previous: jax.Array,
    motion_margin: float,
    *,
    cache: RowCache | None = None,
    capacity: int = 512,
) -> tuple[RowCache, RowMaintenance]:
    """Prebuild or slide rows around previous fixed points before calling solve.

    ``motion_margin`` is a nonnegative distance in metres covering anticipated
    motion and the Newton trajectory. The cell circumradius is added so every
    intersected cell contributes all its triangle nodes. Exceeding ``capacity``
    raises before building any rows. No implicit maintenance occurs in solve.
    Coupling and geometry must remain unchanged when retaining a cache.
    """
    start = perf_counter()
    if not np.isfinite(motion_margin) or motion_margin < 0:
        raise ValueError("motion_margin must be finite and nonnegative")
    points = np.asarray(previous)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("previous must contain finite planar fixed points")
    if cache is None:
        samples = np.asarray(geometry.sample_points)
        nodes, inverse = np.unique(
            np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
        )
        mesh = _build_mesh(nodes, inverse.reshape(samples.shape[:-1]), geometry)
        old_lookup = np.full(len(nodes), -1, dtype=np.int32)
    else:
        mesh, nodes = cache.mesh, np.asarray(cache.nodes)
        old_lookup = np.asarray(cache.lookup)
        if capacity != cache.rows.shape[0]:
            raise ValueError("capacity must match the retained row cache")
    distance = np.min(
        np.linalg.norm(np.asarray(mesh.generators)[:, None] - points[None], axis=-1),
        axis=1,
    )
    radius = np.max(
        np.linalg.norm(
            nodes[np.asarray(mesh.triangles)]
            - np.asarray(mesh.generators)[:, None, None],
            axis=-1,
        ),
        axis=(1, 2),
    )
    chosen = np.unique(np.asarray(mesh.triangles)[distance <= motion_margin + radius])
    if not len(chosen) or len(chosen) > capacity:
        raise ValueError(
            f"row window needs {len(chosen)} nodes; capacity is {capacity}"
        )
    retained = old_lookup[chosen] >= 0
    rows = np.zeros((capacity, len(coupling.centre), 3, 6), dtype=np.float64)
    if cache is not None:
        rows[: len(chosen)][retained] = np.asarray(cache.rows)[
            old_lookup[chosen[retained]]
        ]
    added = chosen[~retained]
    slots = np.flatnonzero(~retained)
    for first in range(0, len(added), 3):
        chunk = nodes[added[first : first + 3]]
        padded = np.concatenate((chunk, np.repeat(chunk[-1:], 3 - len(chunk), axis=0)))
        rows[slots[first : first + 3]] = np.asarray(
            _row_program(jnp.asarray(padded), coupling)
        )[: len(chunk)]
    ext = np.zeros((capacity, 6), dtype=np.float64)
    ext[: len(chosen)] = np.asarray(
        _exterior_program(exterior, jnp.asarray(nodes[chosen]))
    )
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[chosen] = np.arange(len(chosen))
    updated = RowCache(
        mesh,
        jnp.asarray(nodes),
        jnp.asarray(lookup),
        jnp.asarray(rows),
        jnp.asarray(ext),
    )
    jax.block_until_ready(updated)
    return updated, RowMaintenance(
        perf_counter() - start,
        len(added),
        int(retained.sum()),
        len(chosen),
        rows.nbytes,
    )


def _root_type(hessian: jax.Array, axis_sign: jax.Array) -> jax.Array:
    """The first point is the signed definite axis; remaining points are saddles."""
    determinant = hessian[:, 0, 0] * hessian[:, 1, 1] - hessian[:, 0, 1] ** 2
    axis = (determinant[0] > 0) & (axis_sign * hessian[0, 0, 0] > 0)
    return (determinant < 0).at[0].set(axis)


def _adaptive(field, seeds, pitch, iterations):
    policy = TopologyPolicy(polish_iterations=iterations)

    def evaluate(position):
        jet = jax.vmap(field.evaluate)(position)
        conditioned, safe = jax.vmap(topology._conditioned_hessian, in_axes=(0, None))(
            jet.hessian, policy.hessian_tolerance
        )
        scale = jnp.max(jnp.abs(safe), axis=(-2, -1))
        matrix = safe / scale[:, None, None]
        gradient = jet.gradient / scale[:, None]
        determinant = (
            matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
        )
        correction = (
            jnp.stack(
                (
                    matrix[:, 1, 1] * gradient[:, 0] - matrix[:, 0, 1] * gradient[:, 1],
                    matrix[:, 0, 0] * gradient[:, 1] - matrix[:, 1, 0] * gradient[:, 0],
                ),
                -1,
            )
            / determinant[:, None]
        )
        length = jnp.linalg.norm(correction, axis=-1)
        usable = (
            conditioned
            & jnp.isfinite(jet.value)
            & jnp.all(jnp.isfinite(correction), axis=-1)
        )
        done = usable & (length <= policy.position_tolerance * pitch)
        correction *= jnp.minimum(1.0, pitch / jnp.where(length > 0, length, 1.0))[
            :, None
        ]
        return jet, correction, usable, done

    jet, correction, usable, done = evaluate(seeds)
    initial = (
        jnp.asarray(0),
        seeds,
        jet,
        correction,
        usable,
        done,
        jnp.zeros(seeds.shape[0], jnp.int32),
    )

    def condition(carry):
        return (carry[0] < iterations) & jnp.any(carry[4] & ~carry[5])

    def update(carry):
        index, position, _, correction, usable, done, counts = carry
        active = usable & ~done
        position -= jnp.where(active[:, None], correction, 0.0)
        jet, correction, usable, done = evaluate(position)
        return index + 1, position, jet, correction, usable, done, counts + active

    _, position, jet, _, _, done, counts = jax.lax.while_loop(
        condition, update, initial
    )
    return position, jet, done, counts


@partial(jax.jit, static_argnames=("iterations",))
def solve(
    cache: RowCache,
    moments: jax.Array,
    seeds: jax.Array,
    previous: jax.Array,
    pitch: jax.Array,
    axis_sign: jax.Array = -1.0,
    *,
    iterations: int = 12,
) -> FixedPointResult:
    """Solve axis then X-points in one adaptive batched device program.

    ``moments`` has shape (source cells, 3); ``seeds`` and ``previous`` have
    shape (points, 2), with the axis first. ``axis_sign`` is -1 for a maximum
    and +1 for a minimum. A wrong root is retried from the last accepted
    ``previous`` point using topology.stationary_read, and rejected if it still
    has the wrong Hessian type. A missing row returns NEEDS_ROWS; the caller
    must run maintenance outside its timed solve before retrying. Refused
    positions describe the attempted trajectory, never an accepted warm start.
    """
    field = _CachedField(cache.mesh, cache.lookup, cache.rows, moments, cache.exterior)
    position, jet, converged, counts = _adaptive(field, seeds, pitch, iterations)
    wrong = ~_root_type(jet.hessian, axis_sign) & jnp.isfinite(jet.value)

    def retry(_):
        policy = TopologyPolicy(polish_iterations=iterations)
        result = jax.vmap(
            lambda point: topology.stationary_read(field, point, pitch, policy)
        )(previous)
        return (
            jnp.where(wrong[:, None], result.position, position),
            jax.tree.map(
                lambda replacement, original: jnp.where(
                    wrong.reshape((len(wrong),) + (1,) * (original.ndim - 1)),
                    replacement,
                    original,
                ),
                result.jet,
                jet,
            ),
            jnp.where(wrong, result.valid, converged),
        )

    position, jet, converged = jax.lax.cond(
        jnp.any(wrong), retry, lambda _: (position, jet, converged), None
    )
    kind = _root_type(jet.hessian, axis_sign)
    needs_rows = ~jax.vmap(field.contains)(position)
    valid = converged & kind & ~needs_rows
    status = jnp.where(
        needs_rows,
        int(FixedPointStatus.NEEDS_ROWS),
        jnp.where(
            ~kind,
            int(FixedPointStatus.WRONG_ROOT),
            jnp.where(
                valid,
                int(FixedPointStatus.CONVERGED),
                int(FixedPointStatus.NOT_CONVERGED),
            ),
        ),
    )
    return FixedPointResult(position, jet, valid, status, counts, wrong)
