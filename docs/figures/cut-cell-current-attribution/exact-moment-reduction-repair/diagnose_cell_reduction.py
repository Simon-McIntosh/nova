"""Separate density sampling from polynomial contraction on one recorded cell."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from nova.jax.config import configure_dtypes


configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmarks import exact_cut_cell_moment_stages as stages  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium.clip_quadrature import (  # noqa: E402
    _quadratic_support,
    clipped_support_current_moments,
    _density_coefficients,
    _density_sample_field,
    _integrate_current_points,
    _quadrature_from_arrays,
    _sampled_arc_polynomial_moments,
)
from nova.equilibrium.forward_operator import set_support_clip_mode  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
).strip()
print(f"revision={revision} tree={ROOT} command=diagnose-cell-reduction", flush=True)

case_name = "weak-rotation-reactor-static"
requested_cells = 110
cell = 108
label = f"{case_name}-cells-{requested_cells}"
carrier, source, exact = certificate._case(case_name)
machine = certificate._case_machine(case_name, carrier, exact, -requested_cells)
operator = fixture.forward_operator(source, machine)
with np.load(
    ROOT / "docs/figures/plasma-cell-read-fidelity/map-fidelity" / f"{label}-exact.npz"
) as bank:
    state = bank["analytic"]
set_support_clip_mode("exact")
jax.clear_caches()
incoming, effective, production, field, _topology, selected, profile = (
    stages._evaluate_path(operator, state)
)
index = jnp.asarray([cell], dtype=jnp.int32)
vertices = jnp.asarray(effective.support_vertices)[index]
count = jnp.asarray(effective.vertex_count)[index]
centres = jnp.asarray(effective.centroids)[index]
points, psi_norm, _radial, _vertical, polynomial_centre, coordinate_scale = (
    _density_sample_field(field, index)
)
density = profile.confined.current_density(points[..., 0], psi_norm)
coefficients = _density_coefficients(density)
polynomial = _sampled_arc_polynomial_moments(
    vertices,
    count,
    polynomial_centre,
    coordinate_scale,
    coefficients,
    centres,
)
fan_points, fan_weights = _quadrature_from_arrays(
    vertices, count, centres, jnp.ones(1, dtype=bool)
)
fan = _integrate_current_points(
    fan_points, fan_weights, field, index, centres, profile.confined
)
reference_row = json.loads((stages.OUTPUT / f"{label}.json").read_text())
reference = next(
    item["production_signed_winding_order_16_moments"]
    for item in reference_row["simple_separatrix_cut"]["cells"]
    if item["cell"] == cell
)


def singleton_row(value):
    array = jnp.asarray(value)
    if array.ndim and array.shape[0] == len(selected):
        return array[index]
    return array


singleton_support = jax.tree.map(singleton_row, incoming)
singleton_field = jax.tree.map(singleton_row, field)
singleton_coefficient = -jnp.asarray(singleton_field.coefficient)
singleton_coefficient = singleton_coefficient.at[:, 0].add(1.0)
singleton_effective = _quadratic_support(
    singleton_support.support_vertices,
    singleton_support.vertex_count,
    singleton_support.centroids,
    singleton_coefficient,
    singleton_field.centre,
    singleton_field.scale,
    jnp.ones(1, dtype=bool),
)
singleton_direct = _sampled_arc_polynomial_moments(
    singleton_effective.support_vertices,
    singleton_effective.vertex_count,
    singleton_field.centre,
    singleton_field.scale,
    coefficients,
    singleton_support.centroids,
)
singleton = clipped_support_current_moments(
    singleton_support,
    jnp.ones(1, dtype=bool),
    singleton_field,
    profile,
    cut_cell_capacity=1,
)
print(
    json.dumps(
        {
            "cell": cell,
            "vertex_count": int(np.asarray(count)[0]),
            "incoming_boundary": bool(np.asarray(incoming.boundary)[cell]),
            "incoming_included": bool(np.asarray(incoming.included)[cell]),
            "selected": bool(np.asarray(selected)[cell]),
            "density_sample_min": float(np.asarray(density).min()),
            "density_sample_max": float(np.asarray(density).max()),
            "density_coefficient_linf": float(np.abs(np.asarray(coefficients)).max()),
            "production_moments": np.asarray(production)[cell].tolist(),
            "singleton_bank_moments": [
                float(np.asarray(value)[0]) for value in singleton
            ],
            "singleton_effective_vertex_count": int(
                np.asarray(singleton_effective.vertex_count)[0]
            ),
            "singleton_effective_boundary": bool(
                np.asarray(singleton_effective.boundary)[0]
            ),
            "singleton_effective_included": bool(
                np.asarray(singleton_effective.included)[0]
            ),
            "singleton_direct_moments": [
                float(np.asarray(value)[0]) for value in singleton_direct
            ],
            "polynomial_moments": [float(np.asarray(value)[0]) for value in polynomial],
            "fan_moments": [float(np.asarray(value)[0]) for value in fan],
            "order_16_reference_moments": reference,
        },
        sort_keys=True,
    ),
    flush=True,
)
