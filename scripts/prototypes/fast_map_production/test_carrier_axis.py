"""The certificate field's declared axis is retained on its own carrier."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import shlex
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.fast_map_production.kernel import read
from scripts.prototypes.support.certificate_fixtures import cached_carrier
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


class SampledField:
    def __init__(self, analytic):
        self.analytic = analytic

    def value(self, point):
        return self.analytic.value(point)

    def evaluate(self, point):
        return self.analytic.evaluate(point)

    def sampled_coefficient(self, geometry):
        samples = jax.vmap(self.value)(geometry.sample_points.reshape(-1, 2))
        return jnp.einsum(
            "nks,ns->nk",
            geometry.fit_inverse,
            samples.reshape(geometry.sample_points.shape[:-1]),
        )


def test_certified_carrier_retains_analytic_axis():
    assert jax.config.jax_enable_x64 is True
    case, _, _ = certificate._case("weak-rotation-reactor-static", clip_mode="exact")
    machine = cached_carrier(case, -500, None)
    assert machine.cache["hit"], "the certified carrier must already be published"
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic, _, _, _ = _analytic_inputs("limited")
    field = SampledField(certificate_field(analytic, "limited"))
    expected_axis = np.asarray((float(analytic.radius), float(analytic.shift[1])))
    position = jnp.asarray(expected_axis)[None]
    np.testing.assert_allclose(
        np.asarray(field.evaluate(position[0]).gradient), 0.0, atol=1e-12
    )
    roots = fixed_points.FixedPointResult(
        position,
        jax.vmap(field.evaluate)(position),
        jnp.ones(1, bool),
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1, bool),
    )
    assert bool(jnp.any(topology._inside_cells(geometry, position[0])))
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    eigenvalues = np.linalg.eigvalsh(
        np.asarray(convention.sigma * roots.jet.hessian[0])
    )
    assert np.all(eigenvalues < 0)
    reading = read(field, roots, geometry, convention, TopologyPolicy())
    assert bool(reading.valid) and bool(reading.qualified)
    np.testing.assert_allclose(reading.axis, expected_axis, atol=1e-12)

    reversed_convention = topology.TopologyConvention.from_cocos(17, -1.0)
    refused = read(field, roots, geometry, reversed_convention, TopologyPolicy())
    assert not bool(refused.valid)
    assert bool(jnp.all(jnp.isnan(refused.axis)))


def test_solver_loads_the_certified_carrier():
    from scripts.prototypes.fast_map_production.solver import certified_machine

    case, _, exact = certificate._case(
        "weak-rotation-reactor-static", clip_mode="exact"
    )
    machine = certified_machine(case, "limited", 550, exact)
    assert machine.cache["hit"]
    assert len(machine.cell_polygons) == 553


@pytest.mark.parametrize("job", ("deep-job.sh", "two_merit_job.sh"))
def test_solver_job_arguments_match_parser(job, monkeypatch):
    from scripts.prototypes.fast_map_production import solver

    lines = (Path(__file__).parent / job).read_text().splitlines()
    commands = []
    for index, line in enumerate(lines):
        if '"$python" -m scripts.prototypes.fast_map_production.solver' not in line:
            continue
        if "if " not in line:
            continue
        command = line.strip()
        while command.endswith("\\"):
            index += 1
            command = command[:-1] + " " + lines[index].strip()
        tokens = shlex.split(command.split("; then")[0].replace("$size", "500"))
        commands.append(
            tokens[tokens.index("scripts.prototypes.fast_map_production.solver") + 1 :]
        )
    assert len(commands) == 1
    parsed = []
    monkeypatch.setattr(solver, "run", parsed.append)
    monkeypatch.setattr(sys, "argv", ["solver", *commands[0]])
    solver.main()
    assert len(parsed) == 1
    assert parsed[0].kind == "diverted"
