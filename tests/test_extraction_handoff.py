"""Exact free-boundary flux handoff to the structured extraction service."""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import (
    FluxLattice,
    GreenSourceRepresentation,
    evaluate_forward_equilibrium,
    extract_flux_surface_geometry,
)
from nova.equilibrium.flux_surface_extraction import _axis_connected_core
from tests import test_equilibrium_forward_solve as forward_fixture

pytest_plugins = ("tests.test_equilibrium_forward_solve",)

NODE_FLUX_RELATIVE_TOLERANCE = 5.0e-12
ARC_INVALID_COUNT_TOLERANCE = 0
EXTRACTION_CORE_FLOOR = 200


def _rectangle(radius: float, height: float, size: float = 0.05) -> np.ndarray:
    """Return the section used by the shared free-boundary fixture."""
    half = 0.5 * size
    return np.asarray(
        (
            (radius - half, height - half),
            (radius + half, height - half),
            (radius + half, height + half),
            (radius - half, height + half),
        )
    )


def _fixture_sources(profile) -> GreenSourceRepresentation:
    """Retain the exact section description used to build fixture Green blocks."""
    angle = (
        2.0
        * np.pi
        * np.arange(forward_fixture.CONDUCTORS)
        / (forward_fixture.CONDUCTORS)
    )
    conductor = np.c_[1.0 + 0.62 * np.cos(angle), 0.62 * np.sin(angle)]
    return GreenSourceRepresentation(
        external_sections=tuple(
            _rectangle(radius, height) for radius, height in conductor
        ),
        external_current=np.asarray(profile.operator.external_current),
        plasma_sections=tuple(
            _rectangle(radius, height) for radius, height in profile.lattice.coordinate
        ),
        external_kernel="hybrid_rectangle",
        plasma_kernel="hybrid_rectangle",
    )


@dataclass(frozen=True)
class DenseExtraction:
    """One dense exact evaluation and the service record it drives."""

    lattice: FluxLattice
    flux: jnp.ndarray
    inside_limiter: jnp.ndarray
    record: dict[str, jnp.ndarray]
    wall_seconds: float
    core_count: int


@pytest.fixture(scope="module")
def dense_extraction(machine, converged) -> DenseExtraction:
    """Evaluate the shared equilibrium once on a service-sized lattice."""
    profile, _seed, _vacuum = machine
    dense = FluxLattice(
        np.linspace(profile.lattice.radius[0], profile.lattice.radius[-1], 33),
        np.linspace(profile.lattice.height[0], profile.lattice.height[-1], 33),
    )
    start = time.perf_counter()
    flux = evaluate_forward_equilibrium(
        converged, dense, _fixture_sources(profile)
    ).block_until_ready()
    wall_seconds = time.perf_counter() - start

    mesh_r, mesh_z = np.meshgrid(dense.radius, dense.height, indexing="xy")
    _wall, wall_flux = forward_fixture._wall_loop()
    inside_limiter = jnp.asarray(forward_fixture._solovev(mesh_r, mesh_z) >= wall_flux)
    span = converged.topology.boundary_flux - converged.topology.axis_flux
    psi_n = (flux - converged.topology.axis_flux) / span
    core_count = int(np.asarray(_axis_connected_core(psi_n, inside_limiter)).sum())

    profile_grid = jnp.linspace(0.0, 1.0, 101, dtype=flux.dtype)
    field_function = jnp.sqrt(
        profile.source.core.field_function_squared(
            profile_grid,
            profile.source.boundary_field_function,
            converged.topology.flux_span,
        )
    )
    major_radius = converged.topology.axis[0]
    record = extract_flux_surface_geometry(
        flux,
        jnp.asarray(dense.radius),
        jnp.asarray(dense.height),
        inside_limiter,
        axis_psi=converged.topology.axis_flux,
        boundary_psi=converged.topology.boundary_flux,
        profile_coefficients=jnp.zeros(2, dtype=flux.dtype),
        coefficient_scale=jnp.ones(2, dtype=flux.dtype),
        ip_amperes=converged.moments.plasma_current,
        major_radius=major_radius,
        boundary_toroidal_field=(profile.source.boundary_field_function / major_radius),
        field_function_psi_n=profile_grid,
        field_function=field_function,
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=12,
        n_surface_bins=14,
    )
    jnp.asarray(record["rho_face"]).block_until_ready()
    print(
        "dense exact evaluation: "
        f"shape={flux.shape}, core_cells={core_count}, "
        f"valid={bool(record['valid'])}, "
        f"invalid_arcs={int(record['surface_arc_invalid_count'])}, "
        f"wall_seconds={wall_seconds:.6f}"
    )
    return DenseExtraction(
        lattice=dense,
        flux=flux,
        inside_limiter=inside_limiter,
        record=record,
        wall_seconds=wall_seconds,
        core_count=core_count,
    )


def test_exact_evaluator_reproduces_the_solve_nodes(machine, converged):
    """The Green re-evaluation agrees with the converged 25 by 25 map."""
    profile, _seed, _vacuum = machine
    evaluated = np.asarray(
        evaluate_forward_equilibrium(
            converged, profile.lattice, _fixture_sources(profile)
        )
    )
    solved = (
        np.asarray(converged.flux[: profile.lattice.node_count])
        .reshape(profile.lattice.shape)
        .T
    )
    scale = max(float(np.max(np.abs(solved))), 1.0e-30)
    absolute = float(np.max(np.abs(evaluated - solved)))
    relative = absolute / scale
    print(
        "solve-node exact evaluation: "
        f"max_abs_wb={absolute:.6e}, relative={relative:.6e}"
    )
    assert evaluated.shape == (25, 25)
    assert relative < NODE_FLUX_RELATIVE_TOLERANCE


def test_dense_exact_map_clears_the_service_core_floor(converged, dense_extraction):
    """Resolution, not a lowered validity floor, makes the record valid."""
    coarse_core_count = int(np.asarray(converged.domains.core).sum())
    assert coarse_core_count < EXTRACTION_CORE_FLOOR
    assert dense_extraction.core_count >= EXTRACTION_CORE_FLOOR
    assert bool(dense_extraction.record["valid"])
    assert dense_extraction.wall_seconds > 0.0


def test_dense_exact_map_meets_the_arc_validity_contract(dense_extraction):
    """The exact handoff feeds every surface through the service arc tolerance."""
    record = dense_extraction.record
    assert bool(record["surface_arc_valid"])
    assert int(record["surface_arc_invalid_count"]) == ARC_INVALID_COUNT_TOLERANCE
    assert np.all(np.isfinite(np.asarray(record["rho_face"])))
