"""Vectorized Bell cut polygons with two connected components when needed."""

import jax
import jax.numpy as jnp

from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments
from scripts.prototypes.sub_millisecond.measure_bell_cut import (
    bell_moments as single_moments,
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


def bell_moments(geometry, reading, profile, sigma):
    ordinary, ordinary_valid = single_moments(geometry, reading, profile, sigma, 4)
    split, counts, split_valid, four = _split_polygons(geometry, reading, sigma)
    centre, pitch = geometry.centre, geometry.pitch
    polynomial = reading.field_coefficients
    span = reading.boundary_flux - reading.axis_flux

    def density(x, y):
        flux = (
            polynomial[:, 0]
            + polynomial[:, 1] * x
            + polynomial[:, 2] * y
            + polynomial[:, 3] * x * x
            + polynomial[:, 4] * x * y
            + polynomial[:, 5] * y * y
        )
        return profile.current_density(
            centre[:, 0] + pitch * x, (flux - reading.axis_flux) / span
        )

    origin = density(0, 0)
    radial_positive, radial_negative = density(1, 0), density(-1, 0)
    vertical_positive, vertical_negative = density(0, 1), density(0, -1)
    radial = (radial_positive - radial_negative) / 2
    vertical = (vertical_positive - vertical_negative) / 2
    coefficients = jnp.stack(
        (
            origin,
            radial,
            vertical,
            (radial_positive + radial_negative) / 2 - origin,
            density(1, 1) - radial_positive - vertical_positive + origin,
            (vertical_positive + vertical_negative) / 2 - origin,
        ),
        axis=1,
    )
    cells = len(centre)
    flattened = (cells * 2,)
    values, first = padded_polynomial_current_moments(
        split.reshape(flattened + split.shape[2:]),
        counts.reshape(flattened),
        jnp.broadcast_to(centre[:, None], (cells, 2, 2)).reshape(flattened + (2,)),
        jnp.broadcast_to(pitch[:, None, None], (cells, 2, 2)).reshape(flattened + (2,)),
        jnp.broadcast_to(coefficients[:, None], (cells, 2, 6)).reshape(
            flattened + (6,)
        ),
    )
    split_moments = jnp.stack(
        (
            values.reshape(cells, 2).sum(axis=1),
            first.reshape(cells, 2, 2).sum(axis=1)[:, 0],
            first.reshape(cells, 2, 2).sum(axis=1)[:, 1],
        ),
        axis=1,
    )
    return jnp.where(four[:, None], split_moments, ordinary), jnp.where(
        four, split_valid, ordinary_valid
    )
