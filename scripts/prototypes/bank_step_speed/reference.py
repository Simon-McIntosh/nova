"""Retained directional differentiation control for the geometry response."""

from dataclasses import replace
import jax
import jax.numpy as jnp
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.support_read import nodal_jets


def _normal_geometry_booking_tangent(
    operands, moments, jet_tangent, support, profile, quadrature, booked
):
    """Move admitted saddle rays and their edge intervals with the Bell field.

    The only differentiated inputs are the jets at the saddle's Bell triangle.
    Its stationary position follows the implicit Hessian equation. The small
    normal-form and edge-intersection expressions then supply ring velocities;
    moving-polygon moments integrate those velocities on the selected rings.
    """
    geometry = operands.geometry
    reading = support.reading
    cells = len(geometry.centre)
    zero = current.ClippedCurrentMoments(
        *(jnp.zeros(cells, dtype=jet_tangent.dtype) for _ in range(3))
    )
    active = (
        (reading.membership > 0)
        & (reading.membership < 1 - 1e-14)
        & reading.normal_form_cells
    )

    def evaluate(_):
        total_jets = nodal_jets(operands, moments)
        field = fixed_points.field_from_jets(
            total_jets, operands.sample_index, operands.mesh
        )
        direction_field = fixed_points.field_from_jets(
            jet_tangent, operands.sample_index, operands.mesh
        )
        point = reading.saddle_form.position
        pitch = jnp.median(geometry.pitch)
        jet = field.evaluate(point)
        point_tangent = -jnp.linalg.solve(
            jet.hessian, direction_field.evaluate(point).gradient
        )

        def varied_field(amount):
            return replace(
                field,
                coefficients=field.coefficients + amount * direction_field.coefficients,
                jets=field.jets + amount * direction_field.jets,
            )

        def form_inputs(amount, position):
            varied = varied_field(amount)
            derivatives = topology._curvature_derivatives(varied, position, pitch)
            return varied.evaluate(position).hessian, *derivatives

        (hessian, third, fourth), (hessian_tangent, third_tangent, fourth_tangent) = (
            jax.jvp(
                form_inputs,
                (jnp.zeros((), point.dtype), point),
                (jnp.ones((), point.dtype), point_tangent),
            )
        )
        sigma = quadrature.sigma
        form = reading.saddle_form
        _, form_tangent = jax.jvp(
            lambda position, h, d3, d4: topology.saddle_normal_form(
                position,
                sigma * h,
                sigma * d3,
                topology.TopologyPolicy().hessian_tolerance,
                sigma * d4,
            ),
            (point, hessian, third, fourth),
            (point_tangent, hessian_tangent, third_tangent, fourth_tangent),
        )
        psi = total_jets[operands.sample_index, 0]
        coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
        span = reading.boundary_flux - reading.axis_flux
        normalized = coefficients.at[:, 0].add(-reading.axis_flux) / span
        chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
        indices = jnp.nonzero(active, size=cells, fill_value=0)[0]
        count = jnp.sum(active, dtype=jnp.int32)

        def add(index, carried):
            cell = indices[index]
            vertices = geometry.vertices[cell]
            vertex_count = geometry.vertex_count[cell]
            centre = geometry.centre[cell]
            intervals = reading.edge_interval[cell]
            selected = reading.fragment_selected[cell]

            def edges(position, direction, curvature, cubic):
                changed = form._replace(
                    position=position,
                    direction=direction,
                    curvature=curvature,
                    cubic=cubic,
                )
                return topology._normal_form_cell_fragments(
                    vertices,
                    vertex_count,
                    changed,
                    topology.TopologyPolicy().edge_tolerance,
                ).edge_interval[chosen]

            inputs = (form.position, form.direction, form.curvature, form.cubic)
            changes = (
                form_tangent.position,
                form_tangent.direction,
                form_tangent.curvature,
                form_tangent.cubic,
            )
            _, interval_tangent = jax.jvp(edges, inputs, changes)
            interval_tangent = jnp.pad(
                interval_tangent,
                ((0, intervals.shape[0] - 2), (0, 0), (0, 0), (0, 0)),
            )

            def rings(position, direction, curvature, cubic, interval):
                changed = form._replace(
                    position=position,
                    direction=direction,
                    curvature=curvature,
                    cubic=cubic,
                )
                return current._normal_rings(
                    vertices,
                    vertex_count,
                    centre,
                    changed,
                    selected,
                    interval,
                    quadrature.segments,
                )[0]

            rings_now, ring_count = current._normal_rings(
                vertices,
                vertex_count,
                centre,
                form,
                selected,
                intervals,
                quadrature.segments,
            )
            _, velocity = jax.jvp(
                rings,
                (*inputs, intervals),
                (*changes, interval_tangent),
            )

            def moving(ring, speed, number):
                return current.moving_polygon_current_moment_tangent(
                    ring,
                    speed,
                    number,
                    centre,
                    geometry.pitch[cell],
                    normalized[cell],
                    profile,
                )

            moved = jax.vmap(moving)(rings_now, velocity, ring_count)
            return jax.tree.map(
                lambda prior, change: prior.at[cell].add(jnp.sum(change)),
                carried,
                moved,
            )

        raw = jax.lax.fori_loop(0, count, add, zero)
        total = jnp.sum(raw.cell_current)
        return current.ClippedCurrentMoments(
            *(
                booked.normalization * change
                - (
                    0.0
                    if quadrature.target_current is None
                    else value * total / booked.raw_current
                )
                for change, value in zip(raw, booked[:3], strict=True)
            )
        )

    return jax.lax.cond(jnp.any(active), evaluate, lambda _: zero, None)
