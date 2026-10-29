"""Localise disagreement between the algebraic and directional geometry rates."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp

import nova
from nova.equilibrium import current, fixed_points, support_read, topology
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_step_speed.reference import (
    _normal_geometry_booking_tangent as reference_tangent,
)


def emit(kind, **values):
    print(kind, json.dumps(values, sort_keys=True, default=float), flush=True)


def difference(actual, reference):
    return float(
        jnp.linalg.norm(actual - reference)
        / jnp.maximum(jnp.linalg.norm(reference), 1e-30)
    )


def main():
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, state, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = jax.jit(
        lambda m: support_read.read(bank, cache, m, seeds, roots, convention, policy)
    )(state)
    assert bool(support.valid)
    index = jnp.arange(state[0].size)
    shape = jnp.sin(1.7 * index) + jnp.cos(2.3 * index)
    direction = (shape, 0.5 * jnp.roll(shape, 17), -0.2 * jnp.roll(shape, 31))

    def stages(m, s):
        jets = support_read.nodal_jets(bank, m)
        field = fixed_points.field_from_jets(jets, bank.sample_index, bank.mesh)
        direction_jets, direction_field = support_read._directional_field(
            bank, direction
        )
        point = s.reading.saddle_form.position
        pitch = jnp.median(bank.geometry.pitch)
        point_rate = -jnp.linalg.solve(
            field.evaluate(point).hessian, direction_field.evaluate(point).gradient
        )

        def varied(amount):
            return replace(
                field,
                coefficients=field.coefficients + amount * direction_field.coefficients,
                jets=field.jets + amount * direction_field.jets,
            )

        def inputs(amount, position):
            changed = varied(amount)
            third, fourth = topology._curvature_derivatives(changed, position, pitch)
            return changed.evaluate(position).hessian, third, fourth

        (h, d3, d4), (hr, d3r, d4r) = jax.jvp(
            inputs,
            (jnp.zeros((), point.dtype), point),
            (jnp.ones((), point.dtype), point_rate),
        )
        ah, ahr, ad3, ad4, ad3r, ad4r = support_read._curvature_rate(
            field, direction_field, point, point_rate, pitch
        )
        form = s.reading.saddle_form
        sigma = quadrature.sigma
        _, form_rate = jax.jvp(
            lambda p, hh, t, f: topology.saddle_normal_form(
                p,
                sigma * hh,
                sigma * t,
                topology.TopologyPolicy().hessian_tolerance,
                sigma * f,
            ),
            (point, h, d3, d4),
            (point_rate, hr, d3r, d4r),
        )
        algebraic_rate = support_read._normal_form_rate(
            form,
            sigma * ah,
            sigma * ahr,
            sigma * ad3,
            sigma * ad3r,
            sigma * ad4,
            sigma * ad4r,
            point_rate,
        )
        psi = jets[bank.sample_index, 0]
        booked = current.book(psi, s.reading, bank.geometry, profile, quadrature)
        new = support_read._normal_geometry_booking_tangent(
            bank, m, direction_jets, s, profile, quadrature, booked
        )
        old = reference_tangent(bank, m, direction_jets, s, profile, quadrature, booked)
        active = (
            (s.reading.membership > 0)
            & (s.reading.membership < 1 - 1e-14)
            & s.reading.normal_form_cells
        )
        cell = jnp.argmax(active)
        vertices = bank.geometry.vertices[cell]
        count = bank.geometry.vertex_count[cell]
        centre = bank.geometry.centre[cell]
        selected = s.reading.fragment_selected[cell]
        intervals = s.reading.edge_interval[cell]
        chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]
        form_inputs = (form.position, form.direction, form.curvature, form.cubic)
        form_changes = (
            form_rate.position,
            form_rate.direction,
            form_rate.curvature,
            form_rate.cubic,
        )

        def edges(position, direction, curvature, cubic):
            changed = form._replace(
                position=position, direction=direction, curvature=curvature, cubic=cubic
            )
            return topology._normal_form_cell_fragments(
                vertices, count, changed, topology.TopologyPolicy().edge_tolerance
            ).edge_interval[chosen]

        _, interval_rate = jax.jvp(edges, form_inputs, form_changes)
        interval_rate = jnp.pad(
            interval_rate, ((0, intervals.shape[0] - 2), (0, 0), (0, 0), (0, 0))
        )
        rings, ring_count = current._normal_rings(
            vertices, count, centre, form, selected, intervals, quadrature.segments
        )

        def rings_fn(position, direction, curvature, cubic, interval):
            changed = form._replace(
                position=position, direction=direction, curvature=curvature, cubic=cubic
            )
            return current._normal_rings(
                vertices,
                count,
                centre,
                changed,
                selected,
                interval,
                quadrature.segments,
            )[0]

        _, velocity = jax.jvp(
            rings_fn, (*form_inputs, intervals), (*form_changes, interval_rate)
        )
        new_velocity = support_read._normal_ring_velocity(
            vertices,
            count,
            centre,
            form,
            selected,
            intervals,
            quadrature.segments,
            algebraic_rate,
            ring_count,
        )
        return (
            (h, ah),
            (hr, ahr),
            (d3, ad3),
            (d3r, ad3r),
            (d4, ad4),
            (d4r, ad4r),
            (form_rate.direction, algebraic_rate[1]),
            (form_rate.curvature, algebraic_rate[2]),
            (form_rate.cubic, algebraic_rate[3]),
            (velocity, new_velocity),
            (jnp.concatenate(tuple(old)), jnp.concatenate(tuple(new))),
            cell,
            jnp.sum(active),
            ring_count,
        )

    *pairs, cell, active, ring_count = jax.jit(stages)(state, support)
    for label, pair in zip(
        (
            "h",
            "h_rate",
            "third",
            "third_rate",
            "fourth",
            "fourth_rate",
            "direction_rate",
            "curvature_rate",
            "cubic_rate",
            "ring_velocity",
            "booking_rate",
        ),
        pairs,
        strict=True,
    ):
        emit(
            "DIFFERENCE",
            name=label,
            relative=difference(pair[1], pair[0]),
            reference_norm=float(jnp.linalg.norm(pair[0])),
            actual_norm=float(jnp.linalg.norm(pair[1])),
        )
    emit(
        "CELL", index=int(cell), active=int(active), rings=int(jnp.sum(ring_count > 0))
    )


if __name__ == "__main__":
    main()
