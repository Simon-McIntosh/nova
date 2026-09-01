"""Current attribution remains defined on partial-ring plasma supports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay

from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.equilibrium.stencil_mesh import (
    StencilMesh,
    fixed_profile_current_moments,
)
from nova.jax.config import configure_dtypes


FIXTURE_ROOT = Path(__file__).parents[1] / "scripts"
REFERENCE_INPUTS = (
    FIXTURE_ROOT / "ring_quadrature" / "inputs" / "coarse-fixture-reference-inputs.npz"
)
DIRECT_TARGETS = (
    FIXTURE_ROOT / "ring_attribution" / "inputs" / "direct-target-matrices.npz"
)
CURRENT_LOCALISATION = (
    FIXTURE_ROOT / "ring_quadrature" / "inputs" / "source-shift-localization.npz"
)


@dataclass(frozen=True)
class AffineProfile:
    """A profile that makes an affine test flux an affine current density."""

    def current_density(self, radius, psi_norm):
        return 2.0 + 3.0 * radius - 4.0 * psi_norm


def _complete_ring_sensitivity_mask(reference, direct):
    """Reconstruct the six-neighbour, all-full-hex restriction locally."""
    centres = reference["consistent_centres"]
    canonical_vertices = centres[:, None, :] + direct["cell_hex_offsets"]
    regular_hex = np.zeros(len(centres), dtype=bool)
    tolerance = 512.0 * np.finfo(np.float64).eps * np.max(np.abs(canonical_vertices))
    for cell, count in enumerate(reference["legacy_vertex_count"]):
        if count != 6:
            continue
        vertices = reference["legacy_vertices"][cell, :6]
        separation = np.linalg.norm(
            vertices[:, None, :] - canonical_vertices[cell][None, :, :], axis=2
        )
        regular_hex[cell] = (
            max(
                float(np.max(np.min(separation, axis=0))),
                float(np.max(np.min(separation, axis=1))),
            )
            <= tolerance
        )

    triangulation = Delaunay(centres)
    offsets, flat_neighbours = triangulation.vertex_neighbor_vertices
    complete_ring = np.zeros(len(centres), dtype=bool)
    for cell in range(len(centres)):
        neighbours = flat_neighbours[offsets[cell] : offsets[cell + 1]]
        ring = np.r_[cell, neighbours]
        complete_ring[cell] = len(neighbours) == 6 and bool(np.all(regular_hex[ring]))
    return complete_ring


def _fixture_stencil():
    configure_dtypes()
    with np.load(REFERENCE_INPUTS) as stored:
        reference = {name: stored[name] for name in stored.files}
    with np.load(DIRECT_TARGETS) as stored:
        direct = {name: stored[name] for name in stored.files}

    centres = direct["centre_coordinates"]
    sample_index = direct["cell_sample_index"]
    sample_coordinates = direct["unique_vertex_coordinates"]
    np.testing.assert_array_equal(direct["target_coordinates"][: len(centres)], centres)
    np.testing.assert_array_equal(
        direct["target_coordinates"][len(centres) :], sample_coordinates
    )

    central = int(np.argmin(np.linalg.norm(centres - centres.mean(axis=0), axis=1)))
    distance = np.linalg.norm(centres - centres[central], axis=1)
    neighbours = np.argsort(distance)[1:7]
    mesh = StencilMesh(
        coordinate=centres,
        stencil=np.asarray([[central, *neighbours]], dtype=np.intp),
        area=np.ones(len(centres)),
    )
    support_centre = np.arange(len(centres), dtype=np.intp)
    stencil = mesh.current_moment_stencil(
        support_centre=support_centre,
        sampling_node_coordinate=sample_coordinates,
        sampling_cell_node=sample_index[:, 1:] - len(centres),
    )
    support = SimpleNamespace(
        support_vertices=reference["support_vertices"],
        vertex_count=reference["support_vertex_count"],
        centroids=reference["consistent_centres"],
    )
    centroid_flux = 0.1 + 0.2 * centres[:, 0] - 0.3 * centres[:, 1]
    sample_flux = 0.1 + 0.2 * sample_coordinates[:, 0] - 0.3 * sample_coordinates[:, 1]
    return (
        reference,
        stencil,
        support,
        centroid_flux,
        sample_flux,
        _complete_ring_sensitivity_mask(reference, direct),
    )


def test_every_nonempty_fixture_support_receives_its_exact_current():
    """Own-node scatter conserves current on complete and partial-ring cells."""
    reference, stencil, support, centroid_flux, sample_flux, _complete_ring = (
        _fixture_stencil()
    )
    moments = stencil.support_flux_moments(
        AffineProfile(), centroid_flux, sample_flux, support
    )

    value_pool = jnp.concatenate([jnp.asarray(centroid_flux), jnp.asarray(sample_flux)])
    gathered = value_pool[stencil.ring_gather_index]
    coefficient = jnp.einsum(
        "rps,rs->rp", jnp.asarray(stencil.ring_flux_weight), gathered
    )
    exact_current, _exact_first = fixed_profile_current_moments(
        AffineProfile(),
        support.support_vertices,
        support.vertex_count,
        support.centroids,
        stencil.ring_sampling_centre,
        stencil.ring_coordinate_scale,
        coefficient,
    )
    exact = np.asarray(exact_current)
    scattered = np.asarray(moments.cell_current)
    nonempty = reference["support_vertex_count"] >= 3

    assert int(np.count_nonzero(nonempty)) == 447
    assert int(np.count_nonzero(nonempty & (exact != 0.0) & (scattered == 0.0))) == 0
    np.testing.assert_array_equal(scattered, exact)
    np.testing.assert_array_equal(scattered.sum(), exact.sum())

    with np.load(CURRENT_LOCALISATION) as stored:
        fixture_exact = stored["analytic_m0"]
    current_scatter = np.zeros_like(fixture_exact)
    current_scatter[stencil.ring_centre] = fixture_exact[stencil.ring_centre]
    assert (
        int(
            np.count_nonzero(
                nonempty & (fixture_exact != 0.0) & (current_scatter == 0.0)
            )
        )
        == 0
    )
    np.testing.assert_array_equal(current_scatter, fixture_exact)


def test_complete_ring_sensitivity_control_reproduces_rim_omission():
    """Restricting current to complete rings reproduces all 96 lost supports."""
    reference, stencil, _support, _centroid_flux, _sample_flux, complete_ring = (
        _fixture_stencil()
    )
    nonempty = reference["support_vertex_count"] >= 3
    omitted = nonempty & ~complete_ring
    with np.load(CURRENT_LOCALISATION) as stored:
        exact_current = stored["analytic_m0"]
    restricted = np.zeros_like(exact_current)
    restricted[complete_ring] = exact_current[complete_ring]
    lost = omitted & (restricted == 0.0)
    missing_current = float(np.sum(exact_current[omitted]))
    missing_fraction = abs(missing_current / float(np.sum(exact_current)))

    assert int(np.count_nonzero(complete_ring)) == 356
    assert int(np.count_nonzero(nonempty & complete_ring)) == 351
    assert int(np.count_nonzero(omitted)) == 96
    assert int(np.count_nonzero(lost)) == 96
    np.testing.assert_allclose(
        missing_current, -689_027.2099774083, rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        missing_fraction, 0.045918550584580275, rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(stencil.ring_centre, np.arange(len(nonempty)))


def test_complementary_clips_recombine_to_full_cell_current():
    """Two complementary material partitions conserve the uncut current."""
    configure_dtypes()
    centre = np.asarray([[6.2, 0.1]])
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    polygon = centre[0] + 0.24 * np.column_stack([np.cos(angle), np.sin(angle)])
    mesh = AtomicCellMesh.from_cells([polygon], centroids=centre)
    signed = jnp.asarray(mesh.node_coordinates[:, 0] - centre[0, 0])
    full = mesh.traced_clip(jnp.ones(len(mesh.node_coordinates)))
    first = mesh.traced_clip(signed)
    second = mesh.traced_clip(-signed)
    scale = np.asarray([[0.24, 0.24]])
    coefficient = np.asarray(
        [
            [
                0.1 + 0.2 * centre[0, 0] - 0.3 * centre[0, 1],
                0.2 * 0.24,
                -0.3 * 0.24,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )

    def current(support):
        value, _first = fixed_profile_current_moments(
            AffineProfile(),
            support.support_vertices,
            support.vertex_count,
            support.centroids,
            centre,
            scale,
            coefficient,
        )
        return float(value[0])

    full_current = current(full)
    recombined = current(first) + current(second)
    scaled_error = abs(recombined - full_current) / abs(full_current)
    assert scaled_error <= 2.0e-15


def test_uncut_interior_scatter_is_bitwise_identical_to_direct_contraction():
    """Writing an own-node result leaves an ordinary interior value unchanged."""
    reference, stencil, _support, centroid_flux, sample_flux, complete_ring = (
        _fixture_stencil()
    )
    full_support = SimpleNamespace(
        support_vertices=reference["legacy_vertices"],
        vertex_count=reference["legacy_vertex_count"],
        centroids=reference["consistent_centres"],
    )
    scattered = stencil.support_flux_moments(
        AffineProfile(), centroid_flux, sample_flux, full_support
    )
    value_pool = jnp.concatenate([jnp.asarray(centroid_flux), jnp.asarray(sample_flux)])
    gathered = value_pool[stencil.ring_gather_index]
    coefficient = jnp.einsum(
        "rps,rs->rp", jnp.asarray(stencil.ring_flux_weight), gathered
    )
    direct_current, direct_first = fixed_profile_current_moments(
        AffineProfile(),
        full_support.support_vertices,
        full_support.vertex_count,
        full_support.centroids,
        stencil.ring_sampling_centre,
        stencil.ring_coordinate_scale,
        coefficient,
    )
    interior = int(np.flatnonzero(complete_ring)[0])

    np.testing.assert_array_equal(
        scattered.cell_current[interior], direct_current[interior]
    )
    np.testing.assert_array_equal(
        scattered.radial_moment[interior], direct_first[interior, 0]
    )
    np.testing.assert_array_equal(
        scattered.vertical_moment[interior], direct_first[interior, 1]
    )
