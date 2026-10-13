"""Cell-batched booking identity against the sequential reference."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import topology
from nova.equilibrium.clip_quadrature import (
    ClippedCurrentMoments,
    clipped_support_current_moments,
)
from nova.equilibrium.current import (
    Booking,
    FragmentSupport,
    Quadrature,
    book,
)
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial

assert jax.config.jax_enable_x64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Density:
    slope: object = 0.0

    def current_density(self, radius, normalized):
        return jnp.ones_like(radius) + self.slope * normalized


def _cells(number, *, slope=0.2, level=0.0):
    """Tile one quadratic-split cell ``number`` times along radial columns."""
    vertices = jnp.asarray([[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]])
    centre = jnp.asarray([2.0, 0.0])
    points = jnp.asarray(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]], dtype=jnp.float64
    )
    x, y = points.T
    design = jnp.stack([jnp.ones(6), x, y, x * x, x * y, y * y], axis=1)
    coefficients = jnp.asarray([-0.04, 0.0, 0.0, 0.0, 0.0, 1.0])
    fragment = topology.quadratic_cell_fragments(
        vertices - centre, jnp.asarray(4), coefficients.at[0].add(-level)
    )
    selected = jnp.asarray([[False, True]])
    membership = jnp.sum(jnp.where(selected, fragment.area[None], 0), axis=1) / 4
    edge = jnp.pad(
        fragment.edge_interval[None], ((0, 0), (0, 0), (0, 0), (0, 10), (0, 0))
    )
    read = SimpleNamespace(
        field_coefficients=jnp.tile(coefficients[None], (number, 1)),
        axis_flux=jnp.asarray(1.0),
        boundary_flux=jnp.asarray(level),
        fragment_selected=jnp.tile(selected, (number, 1)),
        membership=jnp.tile(membership, (number,)),
        normal_form_cells=jnp.zeros(number, dtype=bool),
        saddle_form=topology.saddle_normal_form(
            centre,
            jnp.asarray([[1.0, 0.0], [0.0, -1.0]]),
            jnp.zeros((2, 2, 2)),
            1e-12,
            jnp.zeros((2, 2, 2, 2)),
        ),
        edge_interval=jnp.tile(edge, (number, 1, 1, 1, 1)),
        valid=jnp.asarray(True),
        qualified=jnp.asarray(True),
    )
    offset = (jnp.arange(number) % 8)[:, None] * jnp.asarray([6.0, 0.0])
    geometry = SimpleNamespace(
        vertices=jnp.tile(vertices, (number, 1, 1)) + offset[:, None, :],
        vertex_count=jnp.full(number, 4),
        centre=jnp.tile(centre[None], (number, 1)) + offset,
        pitch=jnp.ones(number),
        sample_points=jnp.tile((points + centre), (number, 1, 1)) + offset[:, None, :],
        fit_inverse=jnp.tile(jnp.linalg.inv(design), (number, 1, 1)),
    )
    psi = jnp.tile((design @ coefficients), (number, 1))
    return psi, read, geometry, Density(slope=slope)


def _reference_conic_rings(
    vertices, count, centre, pitch, coefficient, selected, segments
):
    """Trace the same cosine-spaced strips as support.geometry.read_polygons."""
    local = (vertices - centre) / pitch
    fragments = topology.quadratic_cell_fragments(
        local, count, coefficient, selected.shape[0]
    )
    left, right = fragments.slice_breaks[:-1], fragments.slice_breaks[1:]
    parameter = 0.5 * (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, segments + 1)))
    x = left[:, None] + (right - left)[:, None] * parameter
    bottom, top, _ = topology._positive_vertical_intervals(local, count, coefficient, x)
    x = jnp.broadcast_to(x[..., None], bottom.shape)
    lower = jnp.stack((x, bottom), axis=-1)
    upper = jnp.stack((x[:, ::-1], top[:, ::-1]), axis=-1)
    rings = jnp.transpose(jnp.concatenate((lower, upper), axis=1), (0, 2, 1, 3))
    labels = fragments.slice_labels
    live = (labels >= 0) & selected[jnp.maximum(labels, 0)]
    live &= (right > left)[:, None]
    rings = centre + pitch * rings.reshape(-1, 2 * (segments + 1), 2)
    live = live.reshape(-1)
    rings = jnp.where(live[:, None, None], rings, centre)
    return rings, jnp.where(live, rings.shape[1], 0), fragments.valid


def _reference_normal_rings(
    vertices, count, centre, form, selected, intervals, segments
):
    """Carry selected sector edges and cubic rays as signed fan rings.

    Straight intervals are owned by the read. Ray intersections use the same
    cubic geometry as its normal-form sectors; no flux level root is involved.
    Opposing fan edges cancel, so holes and disjoint components need no union.
    """
    width = vertices.shape[0]
    slot = jnp.arange(width)
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    edges = vertices[following] - vertices
    live_edge = slot < count
    chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
    sector_live = jnp.zeros(4, dtype=bool).at[chosen].set(selected[:2])
    lower, upper = intervals[..., 0], intervals[..., 1]
    live = selected[:, None, None] & (upper > lower) & live_edge[None, :, None]
    first = vertices[None, :, None, :] + lower[..., None] * edges[None, :, None, :]
    last = vertices[None, :, None, :] + upper[..., None] * edges[None, :, None, :]
    first = first.reshape(-1, 2)
    last = last.reshape(-1, 2)
    straight = jnp.stack((jnp.broadcast_to(centre, first.shape), first, last), axis=1)
    straight_count = jnp.where(live.reshape(-1), 3, 0)

    relative = vertices - form.position
    cross = topology._cross_plane
    roots = topology._cubic_parameters(
        cross(form.cubic[:, None, :], edges[None]),
        cross(form.curvature[:, None, :], edges[None]),
        cross(form.direction[:, None, :], edges[None]),
        -cross(relative[None], edges[None]),
    )
    safe = jnp.where(jnp.isfinite(roots), roots, 0.0)
    points = (
        safe[..., None] * form.direction[:, None, None]
        + safe[..., None] ** 2 * form.curvature[:, None, None]
        + safe[..., None] ** 3 * form.cubic[:, None, None]
    )
    square = jnp.sum(edges * edges, axis=-1)
    fraction = (
        jnp.sum((points - relative[None, :, None]) * edges[None, :, None], axis=-1)
        / jnp.where(square > 0, square, 1.0)[None, :, None]
    )
    hit = (
        live_edge[None, :, None]
        & jnp.isfinite(roots)
        & (roots >= -1e-12)
        & (fraction >= -1e-12)
        & (fraction <= 1 + 1e-12)
    )
    bound = 2 * jnp.max(jnp.linalg.norm(relative, axis=-1))
    parameters = jnp.sort(
        jnp.clip(jnp.where(hit, roots, jnp.inf).reshape(4, -1), 0.0, bound), axis=-1
    )
    parameters = jnp.concatenate(
        (jnp.zeros((4, 1)), parameters, jnp.full((4, 1), bound)), axis=-1
    )
    begin, end = parameters[:, :-1], parameters[:, 1:]
    middle = (begin + end) / 2
    curve = (
        middle[..., None] * form.direction[:, None]
        + middle[..., None] ** 2 * form.curvature[:, None]
        + middle[..., None] ** 3 * form.cubic[:, None]
    )
    inside = jnp.all(
        jnp.where(
            live_edge[None, None],
            cross(edges[None, None], curve[:, :, None] - relative[None, None])
            >= -1e-12,
            True,
        ),
        axis=-1,
    )
    sign = sector_live.astype(vertices.dtype) - jnp.roll(sector_live, 1).astype(
        vertices.dtype
    )
    active = inside & (end > begin) & (sign[:, None] != 0)
    parameter = jnp.linspace(0.0, 1.0, segments + 1)
    t = begin[..., None] + (end - begin)[..., None] * parameter
    arcs = (
        form.position
        + t[..., None] * form.direction[:, None, None]
        + t[..., None] ** 2 * form.curvature[:, None, None]
        + t[..., None] ** 3 * form.cubic[:, None, None]
    )
    arcs = jnp.where((sign < 0)[:, None, None, None], arcs[:, :, ::-1], arcs)
    arcs = arcs.reshape(-1, segments + 1, 2)
    arcs = jnp.concatenate((jnp.broadcast_to(centre, (len(arcs), 1, 2)), arcs), axis=1)
    arcs = jnp.where(active.reshape(-1, 1, 1), arcs, centre)
    straight = jnp.pad(straight, ((0, 0), (0, segments - 1), (0, 0)))
    return (
        jnp.concatenate((straight, arcs)),
        jnp.concatenate(
            (straight_count, jnp.where(active.reshape(-1), segments + 2, 0))
        ),
    )


def _reference_book(psi, read, geometry, profile, quadrature=Quadrature()):
    """Integrate every selected fragment and normalize its current once.

    ``psi`` contains physical flux at the geometry's per-cell sample points.
    The supplied read must be its topology read; using a stale read is a caller
    error. Invalid reads and zero/nonfinite normalization denominators return
    an invalid receipt with nonfinite moments, including under jit and vmap.
    """
    if quadrature.segments < 2:
        raise ValueError("quadrature segments must be at least two")
    psi = jnp.asarray(psi)
    if psi.shape != geometry.sample_points.shape[:-1]:
        raise ValueError("psi must carry flux at every per-cell sample point")
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    span = read.boundary_flux - read.axis_flux
    safe_span = jnp.where(span != 0, span, 1.0)
    normalized = coefficients.at[:, 0].add(-read.axis_flux) / safe_span
    signed = quadrature.sigma * read.field_coefficients
    signed = signed.at[:, 0].add(-quadrature.sigma * read.boundary_flux)

    def cell(_carry, index):
        centre = geometry.centre[index]
        vertices, count = geometry.vertices[index], geometry.vertex_count[index]
        field = FluxFieldPolynomial(
            normalized[index][None],
            centre[None],
            jnp.full((1, 2), geometry.pitch[index]),
            jnp.ones(1, dtype=bool),
        )

        def integrate(rings, counts):
            support = FragmentSupport(
                rings[None],
                counts[None],
                centre[None],
                jnp.ones(1, dtype=bool),
                jnp.ones(1, dtype=bool),
            )
            return clipped_support_current_moments(
                support,
                support.included,
                field,
                profile,
                cut_cell_capacity=1,
                boundary_reduction=False,
            )

        def occupied(_):
            def partial(_):
                def conic(_):
                    rings, counts, valid = _reference_conic_rings(
                        vertices,
                        count,
                        centre,
                        geometry.pitch[index],
                        signed[index],
                        read.fragment_selected[index],
                        quadrature.segments,
                    )
                    moments = integrate(rings, counts)
                    return jax.tree.map(
                        lambda value: jnp.where(valid, value, jnp.nan), moments
                    )

                def normal(_):
                    rings, counts = _reference_normal_rings(
                        vertices,
                        count,
                        centre,
                        read.saddle_form,
                        read.fragment_selected[index],
                        read.edge_interval[index],
                        quadrature.segments,
                    )
                    return integrate(rings, counts)

                return jax.lax.cond(read.normal_form_cells[index], normal, conic, None)

            return jax.lax.cond(
                read.membership[index] >= 1 - 1e-14,
                lambda _: integrate(vertices[None], count[None]),
                partial,
                None,
            )

        moments = jax.lax.cond(
            read.membership[index] > 0,
            occupied,
            lambda _: ClippedCurrentMoments(*(jnp.zeros(1) for _ in range(3))),
            None,
        )
        return None, jax.tree.map(lambda value: value[0], moments)

    _, moments = jax.lax.scan(cell, None, jnp.arange(len(geometry.centre)), unroll=1)
    raw = jnp.sum(moments.cell_current)
    valid = read.valid & read.qualified & (span != 0) & jnp.isfinite(raw)
    if quadrature.target_current is None:
        normalization = jnp.ones_like(raw)
    else:
        valid &= (raw != 0) & jnp.isfinite(quadrature.target_current)
        normalization = quadrature.target_current / jnp.where(raw != 0, raw, 1.0)
    return Booking(
        *(jnp.where(valid, value * normalization, jnp.nan) for value in moments),
        raw,
        normalization,
        valid,
    )


def _compare(cells):
    """Return the batched, reference bookings and each program's warm wall."""
    psi, read, geometry, profile = _cells(cells)
    quadrature = Quadrature(segments=64)
    batched_program = jax.jit(lambda: book(psi, read, geometry, profile, quadrature))
    reference_program = jax.jit(
        lambda: _reference_book(psi, read, geometry, profile, quadrature)
    )
    batched_program(), reference_program()
    start = perf_counter()
    batched = batched_program()
    jax.block_until_ready(batched)
    batched_seconds = perf_counter() - start
    start = perf_counter()
    reference = reference_program()
    jax.block_until_ready(reference)
    reference_seconds = perf_counter() - start
    maximum = 0.0
    relative = 0.0
    for batched_value, reference_value in zip(batched[:3], reference[:3], strict=True):
        batched_value = np.asarray(batched_value)
        reference_value = np.asarray(reference_value)
        maximum = max(maximum, float(np.max(np.abs(batched_value - reference_value))))
        relative = max(
            relative,
            float(
                np.max(
                    np.abs(batched_value - reference_value)
                    / np.maximum(1e-30, np.abs(reference_value))
                )
            ),
        )
    print(
        f"IDENTITY cells={cells} max_abs={maximum:.3e} max_rel={relative:.3e} "
        f"batched_seconds={batched_seconds:.6g} scan_seconds={reference_seconds:.6g}",
        flush=True,
    )
    np.testing.assert_array_equal(
        np.asarray(batched.valid), np.asarray(reference.valid)
    )
    np.testing.assert_allclose(batched[:3], reference[:3], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(
        np.asarray(batched.normalization), np.asarray(reference.normalization)
    )
    assert maximum <= 1e-11
    return maximum, relative


@pytest.mark.parametrize("cells", [1, 64, 200])
def test_vectorised_booking_matches_sequential_scan(cells):
    _compare(cells)


@pytest.mark.slow
def test_vectorised_booking_matches_sequential_scan_on_a_large_lattice():
    _compare(2000)
