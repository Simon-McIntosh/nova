"""Vectorized Bell cut polygons with two connected components when needed."""

import jax
import jax.numpy as jnp

from nova.equilibrium import current
from scripts.prototypes.sub_millisecond.measure_bell_cut import (
    bell_polygon as single_polygon,
)
from scripts.prototypes.sub_millisecond.measure_wall_march import polygon_area


def _split_polygons(geometry, reading, sigma):
    """Resolve four edge crossings as two ordered positive Bell lobes."""
    capacity = geometry.vertices.shape[1]

    def one_cell(vertices, count, centre, pitch, coefficient, level):
        def signed(points):
            local = (points - centre) / pitch
            x, y = local[..., 0], local[..., 1]
            return sigma * (
                coefficient[0]
                + coefficient[1] * x
                + coefficient[2] * y
                + coefficient[3] * x * x
                + coefficient[4] * x * y
                + coefficient[5] * y * y
                - level
            )

        slot = jnp.arange(capacity)
        following = (slot + 1) % jnp.maximum(count, 1)
        next_vertices = vertices[following]
        start = signed(vertices)
        end = signed(next_vertices)
        middle = signed((vertices + next_vertices) / 2)
        quadratic = 2 * (start + end - 2 * middle)
        linear = end - start - quadratic
        discriminant = jnp.maximum(linear * linear - 4 * quadratic * start, 0)
        denominator = jnp.where(jnp.abs(quadratic) > 1e-12, quadratic, 1)
        roots = jnp.stack(
            (
                (-linear - jnp.sqrt(discriminant)) / (2 * denominator),
                (-linear + jnp.sqrt(discriminant)) / (2 * denominator),
            ),
            axis=-1,
        )
        root = jnp.where(
            (roots[:, 0] >= 0) & (roots[:, 0] <= 1), roots[:, 0], roots[:, 1]
        )
        linear_root = -start / jnp.where(jnp.abs(linear) > 1e-12, linear, 1)
        root = jnp.clip(jnp.where(jnp.abs(quadratic) > 1e-12, root, linear_root), 0, 1)
        intersection = vertices + root[:, None] * (next_vertices - vertices)
        live = slot < count
        entering = (start < 0) & (end > 0) & live
        exiting = (start > 0) & (end < 0) & live
        entrances = jnp.nonzero(entering, size=2, fill_value=0)[0]

        def component(entrance, next_entrance):
            steps = jnp.arange(1, capacity + 1)
            edge = (entrance + steps) % jnp.maximum(count, 1)
            possible_exit = exiting[edge]
            exit_step = 1 + jnp.argmax(possible_exit)
            exit_edge = edge[exit_step - 1]
            first = intersection[entrance]
            last = intersection[exit_edge]
            chord_vector = intersection[next_entrance] - last
            normal = jnp.stack((-chord_vector[1], chord_vector[0]))
            fraction = jnp.linspace(0, 1, 6)[1:-1]
            chord = last[None] + fraction[:, None] * chord_vector[None]
            zero = signed(chord)
            positive = signed(chord + normal)
            negative = signed(chord - normal)
            curvature = (positive + negative - 2 * zero) / 2
            slope = (positive - negative) / 2
            radical = slope * slope - 4 * curvature * zero
            safe = jnp.where(jnp.abs(curvature) > 1e-12, curvature, 1)
            first_root = (-slope - jnp.sqrt(jnp.maximum(radical, 0))) / (2 * safe)
            second_root = (-slope + jnp.sqrt(jnp.maximum(radical, 0))) / (2 * safe)
            arc_root = jnp.where(
                jnp.abs(first_root) < jnp.abs(second_root), first_root, second_root
            )
            line_root = -zero / jnp.where(jnp.abs(slope) > 1e-12, slope, 1)
            arc_root = jnp.where(jnp.abs(curvature) > 1e-12, arc_root, line_root)
            arc = chord + arc_root[:, None] * normal[None]
            raw = jnp.concatenate(
                (first[None], vertices[edge], last[None], arc), axis=0
            )
            keep = jnp.concatenate(
                (
                    jnp.ones((1,), bool),
                    steps <= exit_step,
                    jnp.ones((5,), bool),
                )
            )
            valid = jnp.any(possible_exit) & jnp.all(radical >= -1e-9)
            return raw, keep, valid

        centre_inside = signed(centre) > 0
        targets = jnp.where(centre_inside, entrances[::-1], entrances)
        points, keep, validity = jax.vmap(component)(entrances, targets)
        width = 2 * (capacity + 6)

        def compact(raw, mask):
            order = jnp.argsort(jnp.where(mask, jnp.arange(width), width))
            number = jnp.sum(mask, dtype=jnp.int32)
            polygon = jnp.where(jnp.arange(width)[:, None] < number, raw[order], centre)
            return polygon, number

        connected, connected_count = compact(
            points.reshape(width, 2), keep.reshape(width)
        )
        separate, separate_count = jax.vmap(
            lambda raw, mask: compact(
                jnp.pad(raw, ((0, capacity + 6), (0, 0))),
                jnp.pad(mask, ((0, capacity + 6),)),
            )
        )(points, keep)
        polygons = jnp.where(
            centre_inside,
            jnp.stack((connected, jnp.broadcast_to(centre, connected.shape))),
            separate,
        )
        numbers = jnp.where(
            centre_inside,
            jnp.stack((connected_count, jnp.asarray(0, jnp.int32))),
            separate_count,
        )
        four = (jnp.sum(entering) == 2) & (jnp.sum(exiting) == 2)
        return (
            polygons,
            numbers,
            jnp.all(validity) & four & (jnp.sum(numbers) >= 3),
            four,
        )

    return jax.vmap(one_cell)(
        geometry.vertices,
        geometry.vertex_count,
        geometry.centre,
        geometry.pitch,
        reading.field_coefficients,
        jnp.broadcast_to(reading.boundary_flux, (len(geometry.centre),)),
    )


def bell_fraction(geometry, reading, sigma):
    polygon, count, valid = single_polygon(geometry, reading, sigma, 4)
    default = polygon_area(polygon, count)
    split, split_count, split_valid, four = _split_polygons(geometry, reading, sigma)
    split_area = jnp.sum(
        jax.vmap(polygon_area, in_axes=(1, 1), out_axes=1)(split, split_count), axis=1
    )
    area = jnp.where(four, split_area, default)
    return jnp.clip(area / geometry.full_area, 0.0, 1.0), jnp.where(
        four, split_valid, valid
    )


def confined_component(reading):
    """Return the read's closed normal-form component as ``(loop, live)``.

    A read that carries no closed component books the split unrestrained, so
    the absence of the fields is represented as ``None`` rather than as an
    empty loop, which would erase the ordinary cut instead.
    """
    loop = getattr(reading, "closed_loop", None)
    live = getattr(reading, "closed_live", None)
    if loop is None or live is None:
        return None
    return loop, live


MOMENT_SLOTS = 3

BAND_CAPACITY = 160
"""Cells one trace of the composite books, before the booking refuses.

The composite allocates a quadrature over every ring a cell carries, so a trace
per mesh cell costs the whole mesh rather than the boundary band the research
record estimates. ``booking_band`` gathers the cells the composite books to this
capacity, and a band wider than it refuses by name instead of truncating: the
refusal is returned as an invalid booking, the way the saddle booking refuses
past its capacity of 16.
"""


def dense_cells(geometry, reading):
    """Return the whole cells the ordinary bell route covers.

    These are the cells whose quadratic cut is complete, so they book no cut
    support and are excluded from the band the composite applies to.
    """
    return (
        (reading.geometric_membership >= 1 - 1e-14)
        & (geometry.vertex_count == 6)
        & ~reading.booking_cells
    )


def booking_band(geometry, reading):
    """Return the cells the composite books, as this read's cut set."""
    return (reading.membership > 0) & ~dense_cells(geometry, reading)


def composite_cut_moments(
    geometry, reading, profile, sigma, segments=8, capacity=BAND_CAPACITY
):
    """Book every cut cell over (C∩D∩N) ∪ (C∩Dᶜ∩Q) through the shared owner.

    Every cut cell books the same one support: the selected normal-form portion
    inside the effective disk about the saddle and the ordinary quadratic cut
    outside it, both restricted to the closed normal-form component the read
    admits. Neither a scalar blend nor a whole-cell replacement chosen by
    centre distance, crossing count or ordinary validity is used here; those
    remain qualification checks on the read.

    Only the cells the composite books are traced, gathered to ``capacity`` by
    ``booking_band``, and a band wider than that capacity refuses rather than
    truncating, so one trace costs the cut band instead of the whole mesh.
    """
    centre, pitch = geometry.centre, geometry.pitch
    form = reading.saddle_form
    coefficient = reading.field_coefficients
    confined = confined_component(reading)
    signed = sigma * coefficient
    signed = signed.at[:, 0].add(-sigma * reading.boundary_flux)

    def one(index):
        moments, _ = current.composite_cut_moments(
            geometry.vertices[index],
            geometry.vertex_count[index],
            centre[index],
            pitch[index],
            form,
            coefficient[index],
            signed[index],
            profile,
            reading.fragment_selected[index],
            reading.edge_interval[index],
            reading.fragment_selected[index],
            reading.axis_flux,
            reading.boundary_flux,
            reading.disk_position,
            reading.disk_radius,
            segments,
            confined=confined,
        )
        return moments

    band = booking_band(geometry, reading)
    count = jnp.sum(band, dtype=jnp.int32)
    filled = count <= capacity
    indices = jnp.nonzero(band, size=capacity, fill_value=0)[0]
    packed = jax.vmap(one)(indices)
    live = jnp.arange(capacity) < jnp.where(filled, count, 0)
    moments = jnp.zeros((len(centre), MOMENT_SLOTS), dtype=packed.dtype)
    moments = moments.at[indices].add(jnp.where(live[:, None], packed, 0.0))
    return moments, filled


def bell_moments(geometry, reading, profile, sigma, capacity=BAND_CAPACITY):
    """Book the band cells and report whether the band fitted its capacity."""
    moments, filled = composite_cut_moments(
        geometry, reading, profile, sigma, capacity=capacity
    )
    return moments, filled, reading.booking_cells
