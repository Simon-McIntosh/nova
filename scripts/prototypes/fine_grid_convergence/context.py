"""Construct the fixed-shape certificate map for local derivative probes."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_map_production import program
from scripts.prototypes.fast_map_production.kernel import read
from scripts.prototypes.fine_grid_convergence.path_weight import install
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs

install()


@dataclass(frozen=True)
class Context:
    geometry: topology.TopologyGeometry
    reference: jax.Array
    scale: jax.Array
    operands: tuple
    window: fixed_points.RowCache
    values: object
    jets: object
    fit: jax.Array
    rule: object
    profile: object
    seeds: jax.Array
    convention: topology.TopologyConvention
    policy: TopologyPolicy
    axis_sign: float

    def map(self, moments):
        return program.map_moments(moments, *self.operands)

    def reading(self, moments):
        cached = fixed_points._CachedField(
            self.window.mesh,
            self.window.lookup,
            self.window.rows,
            moments,
            self.window.exterior,
        )
        found = fixed_points.solve(
            self.window,
            moments,
            self.seeds,
            self.seeds,
            jnp.median(self.geometry.pitch),
            self.axis_sign,
        )
        gauss_flux = self.values.apply(moments).reshape(self.rule.points.shape[:-1])
        gauss_coefficient = jnp.einsum("nkq,nq->nk", self.fit, gauss_flux)
        slots = self.window.lookup[self.jets.sample_index]
        complete = jnp.all(slots >= 0, axis=1)
        safe_slots = jnp.maximum(slots, 0)
        band_values = self.window.exterior[safe_slots, 0] + jnp.einsum(
            "nscm,cm->ns", self.window.rows[safe_slots, ..., 0], moments
        )
        band_coefficient = jnp.einsum(
            "nks,ns->nk", self.geometry.fit_inverse, band_values
        )
        coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
        field = program.FusedField(cached, coefficient)
        return read(field, found, self.geometry, self.convention, self.policy)


def build(cells, cache):
    assert jax.config.jax_enable_x64 is True
    carrier, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs("diverted")
    field = certificate_field(analytic_field, "diverted")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda psi: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(samples)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, _ = build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, _ = build_jets(coupling, geometry, field, reference)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    window = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(len(jets.nodes), dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )
    seeds = jnp.asarray(roots)
    axis_sign = float(jnp.sign(field.evaluate(seeds[0]).hessian[0, 0]))
    operands = (
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        profile,
        seeds,
        convention,
        policy,
        booked.raw_current,
        axis_sign,
    )
    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )
    return Context(
        geometry,
        reference,
        scale,
        operands,
        window,
        values,
        jets,
        fit,
        rule,
        profile,
        seeds,
        convention,
        policy,
        axis_sign,
    )
