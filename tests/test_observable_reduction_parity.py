"""Transformation parity of derived moment and conservation observations."""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.conservation import conservation_ledger
    from nova.equilibrium.domain import DomainMasks, PlasmaDomain
    from nova.equilibrium.observation import (
        ClippedIntegralMeasure,
        observe_moments,
        summation_error_bound,
    )
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.stencil_mesh import StencilMesh
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes


def _mesh(shape: tuple[int, int] = (9, 10)) -> StencilMesh:
    """Return one regular half-offset mesh with complete seven-cell rings."""

    column, row = np.indices(shape)
    radius = 1.1 + 0.03 * (column + 0.5 * row)
    height = 0.03 * np.sqrt(3.0) * row / 2.0
    coordinate = np.c_[radius.ravel(), height.ravel()]
    area = np.full(len(coordinate), np.sqrt(3.0) * 0.03**2 / 2.0)
    return StencilMesh(coordinate, hex_stencil(shape), area)


def _source() -> ForwardSource:
    """Return a smooth absolute closure that exercises fitted gradients."""

    return ForwardSource(
        core=DomainProfile(
            p_prime=lambda psi_norm: -2.0e4 * (1.0 - 0.2 * psi_norm),
            ff_prime=lambda psi_norm: -0.15 * (1.0 - 0.1 * psi_norm),
        ),
        boundary_pressure=800.0,
        boundary_field_function=2.5,
    )


def test_scalar_and_jitted_vmap_share_one_observable_evaluation():
    """The repaired reductions retain their association under a leading batch."""

    configure_dtypes()
    mesh = _mesh()
    source = _source()
    radius = jnp.asarray(mesh.node_radius)
    base_area = jnp.asarray(mesh.cell_area)
    label = jnp.full(mesh.node_count, int(PlasmaDomain.CORE), dtype=jnp.int8)
    coordinate = jnp.asarray(mesh.coordinate)
    seed_flux = (
        0.31 * (coordinate[:, 0] - 1.2) ** 2
        + 0.17 * coordinate[:, 1] ** 2
        + 0.03 * coordinate[:, 0] * coordinate[:, 1]
    )

    def observe(flux):
        psi_norm = 0.15 + 0.2 * flux
        masks = DomainMasks(label=label, psi_norm=psi_norm)
        area = base_area * (1.0 + 0.01 * psi_norm)
        volume_elements = 2.0 * jnp.pi * radius * area
        measure = ClippedIntegralMeasure(
            area=area,
            volume=volume_elements,
            radial_volume=radius * volume_elements,
            cell_current=1.0e5 * area,
            pressure_volume=(2.0e3 + psi_norm) * volume_elements,
            field_volume=(0.2 + psi_norm**2) * volume_elements,
            masks=masks,
        )
        moments = observe_moments(measure, jnp.asarray(0.7))
        conservation = conservation_ledger(mesh, flux, source, masks, 0.7)
        return jnp.stack(
            (moments.volume, moments.major_radius, conservation.divergence_j)
        )

    scalar = observe(seed_flux)
    transformed = jax.jit(jax.vmap(observe))(seed_flux[jnp.newaxis, ...])[0]
    jax.block_until_ready((scalar, transformed))

    np.testing.assert_array_equal(np.asarray(scalar), np.asarray(transformed))

    volume_elements = (
        2.0 * jnp.pi * radius * base_area * (1.0 + 0.01 * (0.15 + 0.2 * seed_flux))
    )
    volume_bound = float(summation_error_bound(volume_elements))
    radial_bound = float(summation_error_bound(radius * volume_elements))
    major_radius_bound = radial_bound / float(jnp.sum(volume_elements))
    major_radius_bound += (
        float(jnp.sum(radius * volume_elements))
        * volume_bound
        / float(jnp.sum(volume_elements)) ** 2
    )
    difference = np.abs(np.asarray(transformed) - np.asarray(scalar))
    assert difference[0] <= volume_bound
    assert difference[1] <= major_radius_bound
