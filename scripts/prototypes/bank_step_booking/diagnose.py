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

    serial_all = jax.jit(
        lambda: jax.vmap(
            lambda index: current.book_tangent(
                psi,
                psi_direction[index],
                reading,
                field_direction[index],
                axis_direction[index],
                boundary_direction[index],
                geometry,
                UnitDensity(),
                density_gradient,
                quadrature,
                booked,
            )
        )(indices)
    )()
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
    cut = (reading.membership > 0) & (reading.membership < 1 - 1e-14)
    cut_index = jnp.nonzero(cut, size=cells, fill_value=0)[0]
    print("CUT_INDICES", np.asarray(cut_index[: int(jnp.sum(cut))]).tolist())

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
    cut_batch = cut_index[:8]
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
