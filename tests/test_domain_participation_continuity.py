"""Continuity pin for source participation at a domain-label hand-off."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygonanalytic import (
    polygon_analytic_field_moments,
    polygon_analytic_flux_moments,
)
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.separatrix_clip import (
    AtomicCellMesh,
    padded_polynomial_current_moments,
)
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import CellCurrentMoments


def _hexagon(centre, pitch):
    angle = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    return centre + pitch / np.sqrt(3.0) * np.c_[np.cos(angle), np.sin(angle)]


def _participation_sweep():
    pitch = 0.18
    centre = np.asarray([1.5, 0.0])
    polygon = _hexagon(centre, pitch)
    mesh = AtomicCellMesh.from_cells([polygon], centroids=np.asarray([centre]))
    polyline_radius = float(np.max(polygon[:, 0]) + 0.1 * pitch)
    signed_flux = polyline_radius - mesh.node_coordinates[:, 0]
    core_support = mesh.traced_clip(jnp.asarray(signed_flux))
    common_support = mesh.traced_clip(jnp.asarray(-signed_flux))

    source = ForwardSource(
        core=DomainProfile(
            p_prime=lambda psi: -(1.0 + 0.2 * (psi - 1.0) + 0.15 * (psi - 1.0) ** 2),
            ff_prime=lambda psi: jnp.zeros_like(psi),
        )
    )

    gradient_scale = jnp.asarray([[0.7 / pitch, -0.4 / pitch]])

    def density_gradient(density):
        return density[:, None] * gradient_scale

    def support_moments(centroid_density, _shared_density, support):
        current, first = support.linear_current_moments(
            centroid_density, density_gradient(centroid_density)
        )
        return CellCurrentMoments(current, first[:, 0], first[:, 1])

    target_r = np.asarray([1.22, 1.79, 2.55, 3.8])
    target_z = np.asarray([0.19, -0.27, 0.41, -0.63])
    flux = polygon_analytic_flux_moments(
        target_r, target_z, polygon, expansion_point=centre
    )
    radial, vertical = polygon_analytic_field_moments(
        target_r, target_z, polygon, expansion_point=centre
    )
    blocks = np.stack([np.stack(flux), np.stack(radial), np.stack(vertical)])

    states = []
    labels = []
    for displacement in np.asarray([-4, -2, -1, 1, 2, 4]) * 1.0e-4:
        psi_norm = 1.0 - displacement
        label = PlasmaDomain.CORE if psi_norm <= 1.0 else PlasmaDomain.COMMON_SOL
        labels.append(label.name)
        masks = DomainMasks(
            label=jnp.asarray([int(label)], dtype=jnp.int8),
            psi_norm=jnp.asarray([psi_norm]),
        )
        shared_masks = DomainMasks(
            label=jnp.full(len(mesh.node_coordinates), int(label), dtype=jnp.int8),
            psi_norm=jnp.full(len(mesh.node_coordinates), psi_norm),
        )
        moments = source.current_moments(
            jnp.asarray([centre[0]]),
            masks,
            jnp.asarray(mesh.node_coordinates[:, 0]),
            shared_masks,
            lambda centroid, shared: support_moments(centroid, shared, core_support),
            support_moments,
            core_support,
            common_support,
        )
        vector = np.asarray(jnp.stack(moments))[:, 0]
        states.append(
            {
                "current": vector[:1],
                "moments": vector,
                "fields": np.einsum("m,qmt->qt", vector, blocks),
            }
        )

    ratios = []
    for name in ("current", "moments", "fields"):
        fine = np.abs(states[3][name] - states[2][name])
        coarse = np.abs(states[4][name] - states[1][name])
        ratios.extend(np.ravel(coarse / fine))
    return core_support, labels, np.asarray(ratios)


def test_non_crossing_cell_participation_is_continuous_through_label_flip():
    """A cell outside the crossing set must not switch its whole source on."""
    support, labels, ratios = _participation_sweep()

    assert bool(support.boundary[0]) is False
    assert labels[2:4] == ["COMMON_SOL", "CORE"]

    message = (
        "a non-crossing centroid-label flip changed clip-selected participation; "
        "continuity requires every current, moment and composed-field "
        f"epsilon-doubling ratio near 2, measured "
        f"[{ratios.min():.6f}, {ratios.max():.6f}]"
    )
    np.testing.assert_allclose(ratios, 2.0, rtol=2.0e-2, atol=0.0, err_msg=message)


def test_clipped_evaluation_converges_to_full_stencil_limit():
    """Every moment and assembled field shares the full-cell path limit."""
    centre = np.asarray([1.5, 0.0])
    width = 0.2
    polygon = centre + np.asarray(
        [
            [-0.5 * width, -0.5 * width],
            [0.5 * width, -0.5 * width],
            [0.5 * width, 0.5 * width],
            [-0.5 * width, 0.5 * width],
        ]
    )
    mesh = AtomicCellMesh.from_cells([polygon], centroids=np.asarray([centre]))
    source = ForwardSource(
        core=DomainProfile(
            p_prime=lambda psi: -jnp.ones_like(psi),
            ff_prime=lambda psi: jnp.zeros_like(psi),
        )
    )
    radius = jnp.asarray([centre[0]])
    shared_radius = jnp.asarray(mesh.node_coordinates[:, 0])
    masks = DomainMasks(
        label=jnp.asarray([int(PlasmaDomain.CORE)], dtype=jnp.int8),
        psi_norm=jnp.asarray([0.5]),
    )
    shared_masks = DomainMasks(
        label=jnp.full(
            len(mesh.node_coordinates), int(PlasmaDomain.CORE), dtype=jnp.int8
        ),
        psi_norm=jnp.full(len(mesh.node_coordinates), 0.5),
    )

    coefficients = jnp.asarray(
        [[100.0, 300.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    scale = jnp.asarray([[width / 2.0, width / 2.0]])
    complete_support = mesh.traced_clip(jnp.ones(len(mesh.node_coordinates)))
    full_current, full_first = padded_polynomial_current_moments(
        complete_support.support_vertices,
        complete_support.vertex_count,
        complete_support.centroids,
        scale,
        coefficients,
    )
    full_values = jnp.asarray([full_current[0], full_first[0, 0], full_first[0, 1]])

    def support_moments(_centroid_density, _shared_density, support):
        current, first = padded_polynomial_current_moments(
            support.support_vertices,
            support.vertex_count,
            support.centroids,
            scale,
            coefficients,
        )
        return CellCurrentMoments(current, first[:, 0], first[:, 1])

    target_r = np.asarray([1.22, 1.79, 2.55, 3.8])
    target_z = np.asarray([0.19, -0.27, 0.41, -0.63])
    flux = polygon_analytic_flux_moments(
        target_r, target_z, polygon, expansion_point=centre
    )
    radial, vertical = polygon_analytic_field_moments(
        target_r, target_z, polygon, expansion_point=centre
    )
    blocks = np.stack([np.stack(flux), np.stack(radial), np.stack(vertical)])
    full_fields = np.einsum("m,qmt->qt", np.asarray(full_values), blocks)

    observed_missing = []
    moment_ratios = []
    field_ratios = []
    for missing_fraction in (2.8e-4, 2.0e-4, 1.0e-4, 3.0e-5):
        cut = polygon[:, 0].max() - width * missing_fraction
        signed_flux = cut - mesh.node_coordinates[:, 0]
        core_support = mesh.traced_clip(jnp.asarray(signed_flux))
        common_support = mesh.traced_clip(jnp.asarray(-signed_flux))
        moments = source.current_moments(
            radius,
            masks,
            shared_radius,
            shared_masks,
            lambda centroid, shared: support_moments(
                centroid, shared, complete_support
            ),
            support_moments,
            core_support,
            common_support,
        )
        vector = np.asarray(jnp.stack(moments))[:, 0]
        actual_missing = float(1.0 - core_support.area[0] / core_support.full_area[0])
        fields = np.einsum("m,qmt->qt", vector, blocks)
        observed_missing.append(actual_missing)
        moment_ratios.append(
            np.max(np.abs(vector - np.asarray(full_values)) / np.abs(full_values))
        )
        field_ratios.append(
            np.max(np.abs(fields - full_fields), axis=1)
            / np.max(np.abs(full_fields), axis=1)
        )

    observed_missing = np.asarray(observed_missing)
    moment_ratios = np.asarray(moment_ratios)
    field_ratios = np.asarray(field_ratios)
    np.testing.assert_allclose(observed_missing, [2.8e-4, 2.0e-4, 1.0e-4, 3.0e-5])
    assert np.all(moment_ratios <= 4.1 * observed_missing)
    assert np.all(field_ratios <= 4.1 * observed_missing[:, None])


def test_cubic_support_is_c1_at_full_cell_transition_without_smoothing():
    """Edge-vanishing cubic moments and their assembled fields cross fill C1."""
    centre = np.asarray([1.5, 0.0])
    half_width = np.asarray([0.1, 0.08])
    polygon = centre + half_width * np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    mesh = AtomicCellMesh.from_cells([polygon], centroids=centre[None, :])
    node_u = jnp.asarray((mesh.node_coordinates[:, 0] - centre[0]) / half_width[0])
    target_r = np.asarray([1.22, 1.79, 2.55, 3.8])
    target_z = np.asarray([0.19, -0.27, 0.41, -0.63])
    flux = polygon_analytic_flux_moments(
        target_r, target_z, polygon, expansion_point=centre
    )
    radial, vertical = polygon_analytic_field_moments(
        target_r, target_z, polygon, expansion_point=centre
    )
    blocks = jnp.asarray(
        np.stack([np.stack(flux), np.stack(radial), np.stack(vertical)])
    )

    def composed(cut):
        support = mesh.traced_clip(cut - node_u)
        coefficients = jnp.asarray(
            [[cut, -1.0, 0.2 * cut, 0.1 * cut, -0.2, 0.0, -0.1, 0.0, 0.0, 0.0]]
        )
        current, first = padded_polynomial_current_moments(
            support.support_vertices,
            support.vertex_count,
            support.centroids,
            jnp.asarray(half_width)[None, :],
            coefficients,
        )
        moments = jnp.asarray([current[0], first[0, 0], first[0, 1]])
        fields = jnp.einsum("m,qmt->qt", moments, blocks).ravel()
        return jnp.concatenate([moments, fields])

    displacement = 2.0e-7
    left_value, left_derivative = jax.jvp(
        composed, (jnp.asarray(1.0 - displacement),), (jnp.asarray(1.0),)
    )
    right_value, right_derivative = jax.jvp(
        composed, (jnp.asarray(1.0 + displacement),), (jnp.asarray(1.0),)
    )
    surface = composed(jnp.asarray(1.0))
    np.testing.assert_allclose(left_value, surface, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(right_value, surface, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        left_derivative, right_derivative, rtol=2.0e-6, atol=2.0e-9
    )
