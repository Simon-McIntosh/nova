"""Measure traced connected-fragment booking on cached certificate carriers."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from copy import copy
import json
import os
from pathlib import Path
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.support.geometry import (
    analytic_moments,
    analytic_polygons,
    certificate_field,
    pack,
    read_polygons,
)
from scripts.prototypes.support.measure import (
    exterior_from_cache,
    machine_from_cache,
    override,
    partition,
    stage,
)
from tests.equilibrium.test_topology_read import _analytic_inputs

assert jax.config.jax_enable_x64


def measure(kind, cells, directory, cache):
    """Keep the map immutable and replace its private booking on a copy."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("certificate rows require the GPU measurement lane")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        "diverted-single-null" if kind == "diverted" else "weak-rotation-reactor-static"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    wall = certificate._diverted_wall(exact) if kind == "diverted" else None
    requested = -500 if cells == 550 else -cells
    machine, _ = stage(
        "cached-machine", lambda: machine_from_cache(carrier, requested, wall, cache)
    )
    assert machine.cache["hit"] and machine.cache["readonly"]
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    empty = fixture.forward_operator(source, machine)
    physical, exterior, exterior_cache = exterior_from_cache(
        source, exact, machine, empty, analytic, cache
    )
    assert exterior_cache["hit"] and exterior_cache["readonly"]
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(
        "exact"
    )
    branch = (
        topology.TopologyClass.DIVERTED
        if kind == "diverted"
        else topology.TopologyClass.LIMITED
    )
    target, _, _ = certificate._closed_form_current_target(
        name, source, operator, physical
    )
    fixed, _ = stage(
        "legacy-partition",
        lambda: jax.jit(lambda psi: operator._support_partition(psi, branch))(analytic),
    )
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading, _ = stage(
        "topology-read",
        lambda: jax.jit(topology.read)(field, geometry, convention, TopologyPolicy()),
    )
    assert bool(reading.valid) and bool(reading.qualified)
    psi = jax.vmap(jax.vmap(field.value))(geometry.sample_points)
    reference = analytic_polygons(exact, machine.cell_polygons)
    refined = analytic_polygons(
        exact,
        machine.cell_polygons,
        points=16385,
        subdivisions=1 if kind == "diverted" else 0,
    )
    uncertainty = float(
        np.max(
            np.abs(
                np.asarray([p.area for p in reference])
                - np.asarray([p.area for p in refined])
            )
            / np.asarray(geometry.full_area)
        )
    )
    reference = refined
    truth = analytic_moments(source, reference, np.asarray(geometry.centre))
    total = jnp.sum(truth.cell_current)
    polygons = read_polygons(geometry, reading, convention.sigma)
    polygon_truth = analytic_moments(source, polygons, np.asarray(geometry.centre))
    rule = current.Quadrature(sigma=convention.sigma, target_current=total)
    evaluate = jax.jit(
        lambda samples, read, mesh: current.book(
            samples, read, mesh, operator.source.core, rule
        )
    )
    booked, wall_seconds = stage(
        "traced-book", lambda: evaluate(psi, reading, geometry)
    )
    assert bool(booked.valid)
    moment = CellCurrentMoments(*booked[:3])
    np.savez_compressed(
        directory / f"{kind}-{cells}-moments.npz",
        **{
            label: np.asarray(value)
            for label, value in zip(
                ["cell_current", "radial_moment", "vertical_moment"],
                moment,
                strict=True,
            )
        },
        expected=np.asarray(truth),
        polygon_truth=np.asarray(polygon_truth),
        membership=np.asarray(reading.membership),
        pitch=np.asarray(geometry.pitch),
        centre=np.asarray(geometry.centre),
        full_area=np.asarray(geometry.full_area),
        normal_form_cells=np.asarray(reading.normal_form_cells),
    )
    exact_support = pack(reference, fixed[3])
    exact_partition = partition(
        operator,
        analytic,
        jnp.asarray(fixture._analytic_axis_flux(exact)),
        jnp.asarray(0.0),
        fixture._analytic_saddle(exact),
        exact_support,
    )
    exact_operator = override(operator, exact_partition)
    exact_raw, _ = stage(
        "exact-book",
        lambda: jax.jit(
            lambda state: exact_operator.cell_current_moments(state, branch)
        )(analytic),
    )
    exact_book = operator.scaled_current_moments(
        exact_raw, total / jnp.sum(exact_raw.cell_current)
    )
    truth_image = operator.current_moment_image(
        operator.coupling_current_moments(truth)
    )
    span = abs(float(fixture._analytic_axis_flux(exact)))
    shadow = np.asarray(operator.residual_shadow_mask(analytic, branch))[
        : len(machine.node)
    ]

    def error(moments):
        delta = np.asarray(operator.current_moment_image(moments) - truth_image)[
            : len(machine.node)
        ]
        return float(np.max(np.abs(np.where(shadow, 0, delta))) / span)

    active = copy(operator)

    def private_booking(self, state, requested_class=None):
        del state, requested_class
        return self.coupling_current_moments(
            CellCurrentMoments(*evaluate(psi, reading, geometry)[:3])
        )

    active.cell_current_moments = MethodType(private_booking, active)
    read_coupled, _ = stage(
        "private-booking-override",
        lambda: active.cell_current_moments(analytic, branch),
    )
    legacy, _ = stage(
        "legacy-book",
        lambda: jax.jit(lambda state: operator.cell_current_moments(state, branch))(
            analytic
        ),
    )
    legacy = operator.scaled_current_moments(
        legacy, target / jnp.sum(legacy.cell_current)
    )
    legacy_delta = np.asarray(
        exterior + operator.current_moment_image(legacy) - analytic
    )[: len(machine.node)]
    legacy_map = float(np.max(np.abs(np.where(shadow, 0, legacy_delta))) / span)
    if kind == "diverted" and cells == 550:
        assert abs(legacy_map / 0.197327 - 1) < 5e-6, legacy_map
    density_scale = np.maximum(
        np.abs(np.asarray(truth.cell_current)),
        abs(float(total))
        * np.asarray(geometry.full_area)
        / float(jnp.sum(geometry.full_area)),
    )
    errors = np.abs(np.asarray(moment) - np.asarray(truth)) / density_scale
    errors[1:] /= np.asarray(geometry.pitch)
    clip = (
        np.abs(
            np.asarray(moment) / float(booked.normalization) - np.asarray(polygon_truth)
        )
        / density_scale
    )
    clip[1:] /= np.asarray(geometry.pitch)
    radius = float(exact.major_radius)
    eta = np.asarray(geometry.pitch) / radius
    geometric_coefficient = 0.274714 if kind == "limited" else 2.1805365680219544
    max_density = float(
        np.max(
            np.abs(
                source.toroidal_current_density(
                    np.asarray(geometry.vertices)[..., 0],
                    np.asarray(geometry.vertices)[..., 1],
                )
            )
        )
    )
    current_coefficient = (
        geometric_coefficient
        * max_density
        * float(jnp.sum(geometry.full_area))
        / abs(float(total))
    )
    budget = current_coefficient * eta**2 + 2e-6
    private = np.asarray(reading.membership) == 0
    private_total = float(np.abs(np.asarray(booked.cell_current)[private]).sum())
    region = np.asarray(reading.normal_form_cells)
    row = dict(
        case=kind,
        requested_cells=cells,
        realised_cells=len(machine.node),
        device=str(jax.devices()[0]),
        job_id=os.environ.get("SLURM_JOB_ID"),
        book_seconds=wall_seconds,
        analytic_current=float(total),
        booked_current=float(jnp.sum(booked.cell_current)),
        net_current_relative_error=float(abs(jnp.sum(booked.cell_current) / total - 1)),
        private_cell_count=int(private.sum()),
        private_current=private_total,
        read_image_error=error(read_coupled),
        exact_image_error=error(exact_book),
        legacy_image_error=error(legacy),
        legacy_map_error=legacy_map,
        oracle_area_fraction_uncertainty=uncertainty,
        clip_current_error=float(clip[0].max()),
        clip_first_moment_error=float(clip[1:].max()),
        current_budget_coefficient=current_coefficient,
        smooth_current_budget_ratio=float(np.max(errors[0, ~region] / budget[~region])),
        smooth_moment_budget_ratio=float(np.max(errors[1:, ~region] / budget[~region])),
        saddle_current_budget_ratio=float(np.max(errors[0, region] / budget[region]))
        if region.any()
        else 0.0,
        saddle_moment_budget_ratio=float(np.max(errors[1:, region] / budget[region]))
        if region.any()
        else 0.0,
        completed=True,
    )
    path = directory / f"{kind}-{cells}.json"
    path.write_text(json.dumps(row, indent=2) + "\n")
    print("BOOKING_ROW " + json.dumps(row, sort_keys=True), flush=True)
    return row
