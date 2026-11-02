"""Locate a bank channel's first moving geometry derivative mismatch."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import current, fixed_points, support_read, topology
from scripts.prototypes.bank_newton_step.measure import machine


def report(name, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    difference = np.abs(actual - expected)
    index = np.unravel_index(np.argmax(difference), difference.shape)
    print(
        name,
        json.dumps(
            {
                "relative": float(
                    np.linalg.norm(actual - expected)
                    / max(np.linalg.norm(expected), 1e-20)
                ),
                "max_absolute": float(difference[index]),
                "index": tuple(map(int, index)),
                "actual": float(actual[index]),
                "expected": float(expected[index]),
            }
        ),
        flush=True,
    )


def main():
    root = Path(__file__).resolve().parents[3]
    print(
        "HEADER",
        json.dumps(
            {
                "revision": subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "tree": str(root),
                "cwd": str(Path.cwd().resolve()),
                "module": support_read.__file__,
                "nova": nova.__file__,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
            }
        ),
        flush=True,
    )
    bank, cache, roots, convention, policy, profile, quadrature, trial, _ = machine(
        "diverted"
    )
    cells = trial[0].size
    channel = int(os.environ.get("BANK_CHANNEL", "7"))
    direction = tuple(
        (jnp.arange(cells) == channel % cells).astype(trial[0].dtype)
        * (channel // cells == kind)
        for kind in range(3)
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = jax.jit(support_read.read)(
        bank, cache, trial, seeds, roots, convention, policy
    )
    jets = support_read.nodal_jets(bank, trial)
    field = fixed_points.field_from_jets(jets, bank.sample_index, bank.mesh)
    jet_rate = support_read._directional_field(bank, direction)[0]
    direction_field = fixed_points.field_from_jets(
        jet_rate, bank.sample_index, bank.mesh
    )
    point = support.reading.saddle_form.position
    point_rate = -jnp.linalg.solve(
        field.evaluate(point).hessian, direction_field.evaluate(point).gradient
    )
    pitch = jnp.median(bank.geometry.pitch)

    def form_inputs(amount, position):
        varied = replace(
            field,
            coefficients=field.coefficients + amount * direction_field.coefficients,
            jets=field.jets + amount * direction_field.jets,
        )
        return varied.evaluate(position).hessian, *topology._curvature_derivatives(
            varied, position, pitch
        )

    primal, tangent = jax.jvp(
        form_inputs, (jnp.zeros(()), point), (jnp.ones(()), point_rate)
    )
    actual = support_read._curvature_rates(
        field, direction_field, point, point_rate, pitch
    )
    for label, observed, expected in zip(
        ("hessian", "hessian_rate", "third", "third_rate", "fourth", "fourth_rate"),
        actual,
        (primal[0], tangent[0], primal[1], tangent[1], primal[2], tangent[2]),
        strict=True,
    ):
        report(label, observed, expected)
    sigma = quadrature.sigma
    form = support.reading.saddle_form
    _, form_tangent = jax.jvp(
        lambda p, h, t, f: topology.saddle_normal_form(
            p,
            sigma * h,
            sigma * t,
            topology.TopologyPolicy().hessian_tolerance,
            sigma * f,
        ),
        (point, *primal),
        (point_rate, *tangent),
    )
    form_rates = support_read._normal_form_rate(
        form,
        sigma * actual[0],
        sigma * actual[1],
        sigma * actual[2],
        sigma * actual[3],
        sigma * actual[4],
        sigma * actual[5],
    )
    for label, observed, expected in zip(
        ("direction_rate", "curvature_rate", "cubic_rate"),
        form_rates,
        form_tangent[1:4],
        strict=True,
    ):
        report(label, observed, expected)
    changes = (point_rate, *form_rates)
    expected_changes = form_tangent[:4]
    active = np.nonzero(
        np.asarray(
            (support.reading.membership > 0)
            & (support.reading.membership < 1 - 1e-14)
            & support.reading.normal_form_cells
        )
    )[0]
    print("ACTIVE", active.tolist(), flush=True)
    for cell in active:
        vertices = bank.geometry.vertices[cell]
        count = bank.geometry.vertex_count[cell]
        centre = bank.geometry.centre[cell]
        intervals = support.reading.edge_interval[cell]
        selected = support.reading.fragment_selected[cell]
        chosen = jnp.nonzero(form.positive, size=2, fill_value=0)[0]

        def edges(p, d, c, b):
            changed = form._replace(position=p, direction=d, curvature=c, cubic=b)
            return topology._normal_form_cell_fragments(
                vertices, count, changed, topology.TopologyPolicy().edge_tolerance
            ).edge_interval[chosen]

        primals = form[:4]
        _, expected_interval = jax.jvp(edges, primals, expected_changes)
        actual_interval = support_read._edge_interval_rate(
            vertices, count, form, changes, topology.TopologyPolicy().edge_tolerance
        )[chosen]
        report(f"cell_{cell}_interval", actual_interval, expected_interval)

        def padded(value):
            return jnp.pad(value, ((0, intervals.shape[0] - 2), (0, 0), (0, 0), (0, 0)))

        def rings(p, d, c, b, interval):
            changed = form._replace(position=p, direction=d, curvature=c, cubic=b)
            return current._normal_rings(
                vertices, count, centre, changed, selected, interval, 8
            )[0]

        _, expected_velocity = jax.jvp(
            rings, (*primals, intervals), (*expected_changes, padded(expected_interval))
        )
        actual_velocity = support_read._normal_ring_rates(
            vertices,
            count,
            centre,
            form,
            selected,
            intervals,
            padded(actual_interval),
            changes,
            8,
        )
        report(f"cell_{cell}_ring_velocity", actual_velocity, expected_velocity)


if __name__ == "__main__":
    main()
