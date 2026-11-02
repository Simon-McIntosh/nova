"""Print a clipped cell's fixed and moving booking derivative terms."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

import nova  # noqa: E402
from nova.equilibrium import current, support_read, topology  # noqa: E402
from nova.equilibrium.clip_quadrature import (  # noqa: E402
    _UNIT_NODE,
    _UNIT_WEIGHT,
    fixed_polygon_current_moment_tangent,
    moving_polygon_current_moment_tangent,
)
from tests.test_support_read import UnitDensity, _case  # noqa: E402


def main():
    print("NOVA_FILE", nova.__file__)
    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    reading = support.reading
    geometry = operands.geometry
    cell = 10
    cells, samples = operands.sample_index.shape
    channel = 9
    moments = tuple(jnp.zeros(cells) for _ in range(3))
    psi = support_read.nodal_jets(operands, moments)[operands.sample_index, 0]
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=8)
    field_basis = jnp.eye(6, dtype=psi.dtype)
    indices = jnp.arange(samples + 8)
    field_direction = jnp.broadcast_to(
        field_basis[jnp.clip(indices - samples, 0, 5), None]
        * ((indices >= samples) & (indices < samples + 6))[:, None, None],
        (samples + 8, cells, 6),
    )
    boundary_direction = (indices == samples + 7).astype(psi.dtype)
    signed = quadrature.sigma * reading.field_coefficients
    signed = signed.at[:, 0].add(-quadrature.sigma * reading.boundary_flux)
    signed_direction = quadrature.sigma * field_direction
    signed_direction = signed_direction.at[:, :, 0].add(
        -quadrature.sigma * boundary_direction[:, None]
    )
    centre = geometry.centre[cell]
    vertices = geometry.vertices[cell]
    count = geometry.vertex_count[cell]
    width = geometry.pitch[cell]
    selected = reading.fragment_selected[cell]
    print("CASE", "cell", cell, "channel", channel, "samples", samples)
    print(
        "CLASS",
        "membership",
        float(reading.membership[cell]),
        "normal",
        bool(reading.normal_form_cells[cell]),
        "count",
        int(count),
    )
    print("SELECTED", np.asarray(selected).astype(int).tolist())
    print("SIGNED_DIRECTION", np.asarray(signed_direction[channel, cell]).tolist())

    sample_basis = jnp.eye(samples, dtype=psi.dtype)
    psi_direction = jnp.broadcast_to(
        sample_basis[jnp.minimum(indices, samples - 1), None]
        * (indices < samples)[:, None, None],
        (samples + 8, cells, samples),
    )
    axis_direction = (indices == samples + 6).astype(psi.dtype)

    def density_gradient(radius, flux):
        return jnp.ones_like(radius)

    def serial_reference(index):
        sample = jnp.broadcast_to(
            jax.nn.one_hot(jnp.minimum(index, samples - 1), samples)
            * (index < samples),
            psi.shape,
        )
        field = jnp.broadcast_to(
            jax.nn.one_hot(jnp.clip(index - samples, 0, 5), 6)
            * ((index >= samples) & (index < samples + 6)),
            (cells, 6),
        )
        return current.book_tangent(
            psi,
            sample,
            reading,
            field,
            (index == samples + 6).astype(psi.dtype),
            (index == samples + 7).astype(psi.dtype),
            geometry,
            UnitDensity(),
            density_gradient,
            quadrature,
            booked,
        )

    serial_all = jax.jit(jax.vmap(serial_reference))(indices)
    vector_all = jax.jit(
        lambda: current.book_tangent_channels(
            psi,
            psi_direction,
            reading,
            field_direction,
            axis_direction,
            boundary_direction,
            geometry,
            UnitDensity(),
            density_gradient,
            quadrature,
            booked,
            jnp.concatenate(
                (jnp.arange(samples, samples + 6), jnp.asarray([samples + 7]))
            ),
        )
    )()
    print(
        "FULL_RESPONSE serial vector",
        float(serial_all.vertical_moment[channel, cell]),
        float(vector_all[cell, 2, channel]),
        "normalization",
        float(booked.normalization),
    )
    support_vector = jax.jit(
        lambda: support_read._local_booking_sensitivities(
            operands,
            moments,
            support,
            UnitDensity(),
            density_gradient,
            quadrature,
            booked,
        )
    )()
    print(
        "SUPPORT_RESPONSE",
        float(support_vector[cell, 2, channel]),
        "DIRECT_DIFFERENCE",
        float(jnp.max(jnp.abs(support_vector - vector_all))),
    )
    serial_stack = jnp.stack(tuple(serial_all), axis=-1)
    expected = np.moveaxis(np.asarray(serial_stack), 0, -1)
    observed = np.asarray(support_vector)
    discrepancy = np.abs(expected - observed)
    worst = np.unravel_index(np.argmax(discrepancy), discrepancy.shape)
    print(
        "GLOBAL_WORST",
        worst,
        "serial",
        expected[worst],
        "vector",
        observed[worst],
        "relative",
        np.linalg.norm(expected - observed) / max(np.linalg.norm(expected), 1.0),
    )
    cut = (reading.membership > 0) & (reading.membership < 1 - 1e-14)
    cut_index = jnp.nonzero(cut, size=cells, fill_value=0)[0]
    print("CUT_INDICES", np.asarray(cut_index[: int(jnp.sum(cut))]).tolist())

    cell = int(worst[0])
    centre = geometry.centre[cell]
    vertices = geometry.vertices[cell]
    count = geometry.vertex_count[cell]
    width = geometry.pitch[cell]
    selected = reading.fragment_selected[cell]
    print(
        "DETAIL_TARGET",
        cell,
        "membership",
        float(reading.membership[cell]),
        "normal",
        bool(reading.normal_form_cells[cell]),
        "direction",
        np.asarray(signed_direction[channel, cell]).tolist(),
    )

    if bool(reading.normal_form_cells[cell]):
        rings, counts = current._normal_rings(
            vertices,
            count,
            centre,
            reading.saddle_form,
            selected,
            reading.edge_interval[cell],
            quadrature.segments,
        )
        print("NORMAL_INTERVALS", np.asarray(reading.edge_interval[cell]).tolist())
        return

    local = (vertices - centre) / width
    fragments = topology.quadratic_cell_fragments(
        local, count, signed[cell], len(selected)
    )
    breaks, serial_speed = topology.quadratic_slice_break_tangent(
        local, count, signed[cell], signed_direction[channel, cell]
    )
    _, all_speeds = jax.vmap(
        lambda direction: topology.quadratic_slice_break_tangent(
            local, count, signed[cell], direction
        )
    )(signed_direction[:, cell])
    print(
        "BREAK_SPEED_MAX_DIFF",
        float(jnp.max(jnp.abs(serial_speed - all_speeds[channel]))),
    )
    rings, counts, valid = current._conic_rings(
        vertices, count, centre, width, signed[cell], selected, quadrature.segments
    )

    def cell_moving(index):
        ring, numbers, _ = current._conic_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            geometry.pitch[index],
            signed[index],
            reading.fragment_selected[index],
            quadrature.segments,
        )
        velocity = current._conic_ring_tangents(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            geometry.pitch[index],
            signed[index],
            signed_direction[channel, index],
            reading.fragment_selected[index],
            quadrature.segments,
        )
        response = jax.vmap(
            lambda one_ring, speed, number: moving_polygon_current_moment_tangent(
                one_ring,
                speed,
                number,
                geometry.centre[index],
                geometry.pitch[index],
                jnp.zeros(6),
                UnitDensity(),
            )
        )(ring, velocity, numbers)
        return response.vertical_moment

    serial_cell = jax.jit(cell_moving)(cell)
    position = int(np.nonzero(np.asarray(cut_index) == cell)[0][0])
    cut_batch = cut_index[(position // 8) * 8 : (position // 8 + 1) * 8]
    vector_cells = jax.jit(jax.vmap(cell_moving))(cut_batch)
    batch_slot = int(jnp.argmax(cut_batch == cell))
    print(
        "CELL_BATCH_MOVING serial vector",
        float(jnp.sum(serial_cell)),
        float(jnp.sum(vector_cells[batch_slot])),
    )
    for ring_index, pair in enumerate(
        zip(np.asarray(serial_cell), np.asarray(vector_cells[batch_slot]))
    ):
        if any(abs(v) > 1e-13 for v in pair):
            print("CELL_BATCH_FRAGMENT", ring_index, pair)

    boundary_channels = jnp.concatenate(
        (jnp.arange(samples, samples + 6), jnp.asarray([samples + 7]))
    )

    def cell_all_moving(index):
        ring, numbers, _ = current._conic_rings(
            geometry.vertices[index],
            geometry.vertex_count[index],
            geometry.centre[index],
            geometry.pitch[index],
            signed[index],
            reading.fragment_selected[index],
            quadrature.segments,
        )

        def channel_terms(direction):
            velocity = current._conic_ring_tangents(
                geometry.vertices[index],
                geometry.vertex_count[index],
                geometry.centre[index],
                geometry.pitch[index],
                signed[index],
                direction,
                reading.fragment_selected[index],
                quadrature.segments,
            )
            terms = jax.vmap(
                lambda one_ring, speed, number: moving_polygon_current_moment_tangent(
                    one_ring,
                    speed,
                    number,
                    geometry.centre[index],
                    geometry.pitch[index],
                    jnp.zeros(6),
                    UnitDensity(),
                )
            )(ring, velocity, numbers)
            return terms.vertical_moment

        return jax.vmap(channel_terms)(signed_direction[boundary_channels, index])

    nested = jax.jit(jax.vmap(cell_all_moving))(cut_batch)[batch_slot, 2]
    print(
        "NESTED_MOVING",
        float(jnp.sum(nested)),
        "SINGLE_MOVING",
        float(jnp.sum(serial_cell)),
    )
    for ring_index, pair in enumerate(zip(np.asarray(serial_cell), np.asarray(nested))):
        if abs(pair[0] - pair[1]) > 1e-12:
            print("FIRST_DIFFERING_FRAGMENT", ring_index, pair)
            break

    serial_batch = cut_index[:32]
    serial_slot = int(jnp.argmax(serial_batch == cell))

    def channel_outer(index):
        field = jax.nn.one_hot(jnp.clip(index - samples, 0, 5), 6)
        field *= (index >= samples) & (index < samples + 6)
        direction = quadrature.sigma * field
        direction = direction.at[0].add(
            -quadrature.sigma * (index == samples + 7).astype(psi.dtype)
        )

        def one_cell(cell_index):
            ring, numbers, _ = current._conic_rings(
                geometry.vertices[cell_index],
                geometry.vertex_count[cell_index],
                geometry.centre[cell_index],
                geometry.pitch[cell_index],
                signed[cell_index],
                reading.fragment_selected[cell_index],
                quadrature.segments,
            )
            velocity = current._conic_ring_tangents(
                geometry.vertices[cell_index],
                geometry.vertex_count[cell_index],
                geometry.centre[cell_index],
                geometry.pitch[cell_index],
                signed[cell_index],
                direction,
                reading.fragment_selected[cell_index],
                quadrature.segments,
            )
            moving = jax.vmap(
                lambda one_ring, speed, number: moving_polygon_current_moment_tangent(
                    one_ring,
                    speed,
                    number,
                    geometry.centre[cell_index],
                    geometry.pitch[cell_index],
                    jnp.zeros(6),
                    UnitDensity(),
                )
            )(ring, velocity, numbers)
            return velocity, moving.vertical_moment

        return jax.vmap(one_cell)(serial_batch)

    outer_velocity, outer_moving = jax.jit(jax.vmap(channel_outer))(indices)
    outer_velocity = np.asarray(outer_velocity[channel, serial_slot])
    outer_moving = np.asarray(outer_moving[channel, serial_slot])
    local_velocity = np.asarray(
        current._conic_ring_tangents(
            vertices,
            count,
            centre,
            width,
            signed[cell],
            signed_direction[channel, cell],
            selected,
            quadrature.segments,
        )
    )
    print(
        "CHANNEL_OUTER_MOVING",
        float(np.sum(outer_moving)),
        "VECTOR_MOVING",
        float(np.sum(np.asarray(nested))),
    )
    for ring_index, (old, new) in enumerate(zip(outer_moving, np.asarray(nested))):
        if abs(old - new) > 1e-12:
            print("CHANNEL_OUTER_FIRST_FRAGMENT", ring_index, old, new)
            vertex_difference = np.max(
                np.abs(outer_velocity[ring_index] - local_velocity[ring_index]),
                axis=1,
            )
            vertex = int(np.argmax(vertex_difference))
            print(
                "CHANNEL_OUTER_FIRST_VERTEX",
                ring_index,
                vertex,
                "serial",
                outer_velocity[ring_index, vertex].tolist(),
                "vector",
                local_velocity[ring_index, vertex].tolist(),
            )
            break

    padded = jnp.pad(cut_index, (0, (-cells) % 32))
    first_batch = padded[:32]
    active = jnp.arange(32) < jnp.sum(cut)

    def conditional_channel(index):
        field = jax.nn.one_hot(jnp.clip(index - samples, 0, 5), 6)
        field *= (index >= samples) & (index < samples + 6)
        direction = quadrature.sigma * field
        direction = direction.at[0].add(
            -quadrature.sigma * (index == samples + 7).astype(psi.dtype)
        )

        def one_cell(cell_index):
            def conic(_):
                ring, numbers, valid = current._conic_rings(
                    geometry.vertices[cell_index],
                    geometry.vertex_count[cell_index],
                    geometry.centre[cell_index],
                    geometry.pitch[cell_index],
                    signed[cell_index],
                    reading.fragment_selected[cell_index],
                    quadrature.segments,
                )
                velocity = current._conic_ring_tangents(
                    geometry.vertices[cell_index],
                    geometry.vertex_count[cell_index],
                    geometry.centre[cell_index],
                    geometry.pitch[cell_index],
                    signed[cell_index],
                    direction,
                    reading.fragment_selected[cell_index],
                    quadrature.segments,
                )
                response = jax.vmap(
                    lambda one_ring, speed, number: (
                        moving_polygon_current_moment_tangent(
                            one_ring,
                            speed,
                            number,
                            geometry.centre[cell_index],
                            geometry.pitch[cell_index],
                            jnp.zeros(6),
                            UnitDensity(),
                        )
                    )
                )(ring, velocity, numbers)
                return jnp.where(valid, jnp.sum(response.vertical_moment), jnp.nan)

            return jax.lax.cond(
                reading.normal_form_cells[cell_index],
                lambda _: jnp.zeros((), psi.dtype),
                conic,
                None,
            )

        values = jax.vmap(one_cell)(first_batch)
        scattered = (
            jnp.zeros((cells,), psi.dtype)
            .at[first_batch]
            .add(jnp.where(active, values, 0.0))
        )
        return values, scattered

    conditional_values, conditional_scatter = jax.jit(jax.vmap(conditional_channel))(
        indices
    )
    print(
        "CONDITIONAL_VALUE",
        float(conditional_values[channel, serial_slot]),
        "SCATTERED_VALUE",
        float(conditional_scatter[channel, cell]),
        "REFERENCE_VALUE",
        float(serial_all.vertical_moment[channel, cell]),
    )

    def no_fixed(index):
        field = jnp.broadcast_to(
            jax.nn.one_hot(jnp.clip(index - samples, 0, 5), 6)
            * ((index >= samples) & (index < samples + 6)),
            (cells, 6),
        )
        return current.book_tangent(
            psi,
            jnp.zeros_like(psi),
            reading,
            field,
            jnp.zeros((), psi.dtype),
            (index == samples + 7).astype(psi.dtype),
            geometry,
            UnitDensity(),
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            booked,
        ).vertical_moment

    isolated = jax.jit(jax.vmap(no_fixed))(indices)
    scalar = jax.jit(serial_reference)(jnp.asarray(channel)).vertical_moment
    print(
        "SERIAL_NO_FIXED",
        float(isolated[channel, cell]),
        "SERIAL_SCALAR",
        float(scalar[cell]),
        "SERIAL_ALL",
        float(serial_all.vertical_moment[channel, cell]),
    )

    target_centre = np.asarray(centre)
    original_tangents = current._conic_ring_tangents

    def report_velocity(
        label,
        one_centre,
        one_direction,
        lower_velocity,
        upper_velocity,
        x_end,
        x_speed,
        bottom_end,
        top_end,
        polygon_lower,
        polygon_upper,
        lower_edge,
        upper_edge,
        lower_edge_speed,
        upper_edge_speed,
        root_lower,
        root_upper,
    ):
        if (
            np.max(np.abs(one_centre - target_centre)) < 1e-12
            and abs(one_direction[2] - 1.0) < 1e-12
        ):

            def source(y):
                if abs(y - polygon_lower) < 1e-8:
                    return "lower edge"
                if abs(y - polygon_upper) < 1e-8:
                    return "upper edge"
                return "root"

            print(
                "CLASSIFICATION",
                label,
                "ring",
                10,
                "slot",
                1,
                "x",
                float(x_end),
                "x_rate",
                float(x_speed),
                "polygon",
                (float(polygon_lower), float(polygon_upper)),
                "interval",
                (float(bottom_end), float(top_end)),
            )
            print(
                "CLASSIFICATION_ENDPOINT",
                label,
                "lower",
                source(bottom_end),
                "velocity",
                lower_velocity.tolist(),
                "edge",
                int(lower_edge),
                "edge_rate",
                float(lower_edge_speed),
                "root_rate",
                float(root_lower),
            )
            print(
                "CLASSIFICATION_ENDPOINT",
                label,
                "upper",
                source(top_end),
                "velocity",
                upper_velocity.tolist(),
                "edge",
                int(upper_edge),
                "edge_rate",
                float(upper_edge_speed),
                "root_rate",
                float(root_upper),
            )

    def traced_tangents(label):
        def trace(
            vertices,
            count,
            one_centre,
            pitch,
            coefficient,
            direction,
            selection,
            segments,
        ):
            velocity = original_tangents(
                vertices,
                count,
                one_centre,
                pitch,
                coefficient,
                direction,
                selection,
                segments,
            )
            one_local = (vertices - one_centre) / pitch
            one_fragments = topology.quadratic_cell_fragments(
                one_local, count, coefficient, selection.shape[0]
            )
            left = one_fragments.slice_breaks[:-1]
            right = one_fragments.slice_breaks[1:]
            parameter = 0.5 * (1.0 - jnp.cos(jnp.linspace(0.0, jnp.pi, segments + 1)))
            x = left[:, None] + (right - left)[:, None] * parameter
            _, break_rate = topology.quadratic_slice_break_tangent(
                one_local, count, coefficient, direction
            )
            x_rate = (
                break_rate[:-1, None]
                + (break_rate[1:, None] - break_rate[:-1, None]) * parameter
            )
            bottom, top, _ = topology._positive_vertical_intervals(
                one_local, count, coefficient, x
            )
            polygon_lower, polygon_upper = topology._vertical_polygon_interval(
                one_local, count, x
            )
            point_x = x[3, 8]
            point_rate = x_rate[3, 8]
            slot = jnp.arange(one_local.shape[0])
            following = jnp.where(slot + 1 < count, slot + 1, 0)
            edge = one_local[following] - one_local
            safe_dx = jnp.where(edge[:, 0] != 0, edge[:, 0], 1.0)
            fraction = (point_x - one_local[:, 0]) / safe_dx
            live = (
                (slot < count)
                & (edge[:, 0] != 0)
                & (fraction >= -1e-12)
                & (fraction <= 1 + 1e-12)
            )
            edge_y = one_local[:, 1] + fraction * edge[:, 1]
            slope = edge[:, 1] / safe_dx
            probe_fraction = (
                point_x + 1e-9 * jnp.sign(point_rate) - one_local[:, 0]
            ) / safe_dx
            entered = (
                (slot < count)
                & (edge[:, 0] != 0)
                & (probe_fraction >= -1e-12)
                & (probe_fraction <= 1 + 1e-12)
            )

            def owner(bound):
                near = live & (jnp.abs(edge_y - bound) < 1e-8)
                moving = near & entered
                candidates = jnp.where(jnp.any(moving), moving, near)
                return jnp.argmin(jnp.where(candidates, jnp.abs(slope), jnp.inf))

            low_edge = owner(jnp.min(jnp.where(live, edge_y, jnp.inf)))
            high_edge = owner(jnp.max(jnp.where(live, edge_y, -jnp.inf)))

            def root_rate(y):
                field = (
                    direction[0]
                    + direction[1] * point_x
                    + direction[2] * y
                    + direction[3] * point_x**2
                    + direction[4] * point_x * y
                    + direction[5] * y**2
                )
                gx = coefficient[1] + 2 * coefficient[3] * point_x + coefficient[4] * y
                gy = coefficient[2] + coefficient[4] * point_x + 2 * coefficient[5] * y
                return -(field + gx * point_rate) / jnp.where(gy != 0, gy, 1)

            jax.debug.callback(
                lambda *args: report_velocity(label, *args),
                one_centre,
                direction,
                velocity[10, 8],
                velocity[10, 9],
                point_x,
                point_rate,
                bottom[3, 8, 1],
                top[3, 8, 1],
                polygon_lower[3, 8],
                polygon_upper[3, 8],
                low_edge,
                high_edge,
                slope[low_edge] * point_rate,
                slope[high_edge] * point_rate,
                root_rate(bottom[3, 8, 1]),
                root_rate(top[3, 8, 1]),
            )
            return velocity

        return trace

    current._conic_ring_tangents = traced_tangents("serial")
    try:

        def debug_reference(index):
            return serial_reference(index)

        debug_result = jax.jit(jax.vmap(debug_reference))(indices)
        print("DEBUG_SERIAL_VALUE", float(debug_result.vertical_moment[channel, cell]))
    finally:
        current._conic_ring_tangents = original_tangents

    current._conic_ring_tangents = traced_tangents("vector")
    try:
        debug_vector = jax.jit(
            lambda: current.book_tangent_channels(
                psi,
                psi_direction,
                reading,
                field_direction,
                axis_direction,
                boundary_direction,
                geometry,
                UnitDensity(),
                density_gradient,
                quadrature,
                booked,
                jnp.concatenate(
                    (jnp.arange(samples, samples + 6), jnp.asarray([samples + 7]))
                ),
            )
        )()
        print("DEBUG_VECTOR_VALUE", float(debug_vector[cell, 2, channel]))
    finally:
        current._conic_ring_tangents = original_tangents

    def terms(direction):
        velocity = current._conic_ring_tangents(
            vertices,
            count,
            centre,
            width,
            signed[cell],
            direction,
            selected,
            quadrature.segments,
        )
        fixed = jax.vmap(
            lambda ring, number: fixed_polygon_current_moment_tangent(
                ring,
                number,
                centre,
                width,
                jnp.zeros(6),
                jnp.zeros(6),
                lambda radius, flux: jnp.ones_like(radius),
            )
        )(rings, counts)
        moving = jax.vmap(
            lambda ring, speed, number: moving_polygon_current_moment_tangent(
                ring, speed, number, centre, width, jnp.zeros(6), UnitDensity()
            )
        )(rings, velocity, counts)
        return (
            velocity,
            jnp.stack(tuple(fixed), axis=-1),
            jnp.stack(tuple(moving), axis=-1),
        )

    serial_velocity, serial_fixed, serial_moving = jax.jit(terms)(
        signed_direction[channel, cell]
    )
    all_velocity, all_fixed, all_moving = jax.jit(jax.vmap(terms))(
        signed_direction[:, cell]
    )
    vector_velocity = all_velocity[channel]
    print("RINGS", len(counts), "valid", bool(valid), "live", int(jnp.sum(counts >= 3)))
    print(
        "VELOCITY_MAX_DIFF", float(jnp.max(jnp.abs(serial_velocity - vector_velocity)))
    )
    print(
        "FIXED_SUM serial vector",
        float(jnp.sum(serial_fixed[:, 2])),
        float(jnp.sum(all_fixed[channel, :, 2])),
    )
    print(
        "MOVING_SUM serial vector",
        float(jnp.sum(serial_moving[:, 2])),
        float(jnp.sum(all_moving[channel, :, 2])),
    )

    labels = np.asarray(fragments.slice_labels)
    left = np.asarray(fragments.slice_breaks[:-1])
    right = np.asarray(fragments.slice_breaks[1:])
    speeds = np.asarray(serial_speed)
    vector_speeds = np.asarray(all_speeds[channel])
    for index, number in enumerate(np.asarray(counts)):
        if number < 3:
            continue
        strip, fragment = divmod(index, labels.shape[1])
        print(
            "FRAGMENT",
            index,
            "strip",
            strip,
            "label",
            int(labels[strip, fragment]),
            "x",
            left[strip],
            right[strip],
            "dx",
            speeds[strip],
            speeds[strip + 1],
            "dx_vector",
            vector_speeds[strip],
            vector_speeds[strip + 1],
            "fixed",
            float(serial_fixed[index, 2]),
            float(all_fixed[channel, index, 2]),
            "moving",
            float(serial_moving[index, 2]),
            float(all_moving[channel, index, 2]),
        )
        ring = np.asarray(rings[index])
        sv = np.asarray(serial_velocity[index])
        vv = np.asarray(vector_velocity[index])
        for edge_index in range(int(number)):
            end = (edge_index + 1) % int(number)
            delta = ring[end] - ring[edge_index]
            node = _UNIT_NODE[:, None]
            speed_s = sv[edge_index] + node * (sv[end] - sv[edge_index])
            speed_v = vv[edge_index] + node * (vv[end] - vv[edge_index])
            y = ring[edge_index, 1] + _UNIT_NODE * delta[1] - float(centre[1])
            edge_s = np.sum(
                _UNIT_WEIGHT * (speed_s[:, 0] * delta[1] - speed_s[:, 1] * delta[0]) * y
            )
            edge_v = np.sum(
                _UNIT_WEIGHT * (speed_v[:, 0] * delta[1] - speed_v[:, 1] * delta[0]) * y
            )
            print(
                "EDGE",
                index,
                edge_index,
                "start",
                ring[edge_index].tolist(),
                "end",
                ring[end].tolist(),
                "velocity_start",
                sv[edge_index].tolist(),
                "velocity_end",
                sv[end].tolist(),
                "vector_velocity_start",
                vv[edge_index].tolist(),
                "vector_velocity_end",
                vv[end].tolist(),
                "vertical_term",
                edge_s,
                edge_v,
            )


if __name__ == "__main__":
    main()
