"""The cold seed of the weak certificate row books its label region at unit
amplitude under the reinstated clip.

The certificate solve normalises every map evaluation to the declared plasma
current, so the cold seed's support must start within reach of the target:
at the seed the read boundary of a compact current-disc field encloses far
less than the profile-owned label region, and the clip alone dropped about
half of it, which doubled the booked profile before the first trip.  The
production chord clip now completes a profile-participating cell the level
would drop at its full atom rather than excluding it, so the seed's booked
current at unit amplitude stays within two percent of the declared target
(measured 1.2% on this tree) and no labelled cell is excluded.

These tests pin that weak 110-cell seed directly: the booked-current
amplitude, the exclusion invariant, and the regression that the support does
not book roughly half of the label region as it did when the clip inherited
the seed's undersized boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.jax.config import configure_dtypes

CASE = "weak-rotation-reactor-static"
CELLS = -110


def _weak_row():
    """Return the operator, seed, target and requested class of the weak -110 row."""
    configure_dtypes()
    import benchmarks.solovev_certificate as certificate
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.stencil_mesh import StencilMesh
    from scripts.analytic_oracle_fixtures import measure as oracle_fixture

    carrier_case, source_case, exact = certificate._case(CASE)
    machine = certificate._case_machine(CASE, carrier_case, exact, CELLS)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(CASE, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    mesh = StencilMesh(machine.node, machine.stencil, machine.area)
    profile = ForwardProfile(
        operator, mesh, newton_steps=certificate.recovery.NEWTON_STEPS
    )
    target, centroid, current_receipt = certificate._closed_form_current_target(
        CASE, source_case, operator, exact_physical
    )
    seed, _branch, _receipt = certificate._production_seed(
        profile, CASE, target, centroid, current_receipt
    )
    from nova.equilibrium.topology import TopologyClass

    return operator, profile, np.asarray(seed), float(target), TopologyClass.LIMITED


def _label_region_current(operator, source, masks, sample_psi_norm):
    """Return the whole-atom current of the labelled confined cells."""
    import jax.numpy as jnp

    atomic = operator.moment_geometry.atomic_mesh
    labels = np.asarray(masks.profile_participation, dtype=bool)
    full = atomic.traced_clip(
        jnp.ones(len(np.asarray(atomic.node_coordinates)))
    ).qualify(labels)
    moment_masks = operator._moment_support_masks(masks, full)
    moments = source.current_moments(
        moment_masks,
        operator.support_current_moments,
        full,
        sample_flux=sample_psi_norm,
    )
    return float(
        np.sum(
            np.asarray(operator.coupling_current_moments(moments, None).cell_current)
        )
    )


@pytest.mark.slow
def test_weak_110_cold_seed_books_the_label_region_at_unit_amplitude():
    """The seed's booked current stays within two percent of the target."""
    import jax.numpy as jnp

    operate, _profile, seed, target, requested = _weak_row()
    seed_moments = operate.cell_current_moments(jnp.asarray(seed), requested)
    booked = float(np.sum(np.asarray(seed_moments.cell_current)))
    amplitude = target / booked
    # Before the completion rule the support inherited the seed's undersized
    # boundary and booked 0.49 of the target (amplitude 2.05): the doubling.
    # The completed support books the label region (measured 1.012 here), the
    # closest every candidate support reaches short of the analytic state
    # itself (0.992).
    assert abs(amplitude - 1.0) <= 0.02
    assert abs(booked / target - 1.0) <= 0.02


@pytest.mark.slow
def test_weak_110_cold_seed_excludes_no_labelled_cell():
    """The seed's support contains every profile-participating cell."""
    import jax.numpy as jnp

    operate, _profile, seed, _target, requested = _weak_row()
    physical = jnp.asarray(seed)[: operate.physical_node_number]
    masks, topology, _c, _a = operate._fixed_design_read(physical, requested)
    sample_flux = operate.sample_node_flux(jnp.asarray(seed))
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    support = operate._profile_support(masks, topology, physical, sample_psi_norm)
    labels = np.asarray(masks.profile_participation, dtype=bool)
    included = np.asarray(support.included, dtype=bool)
    dropped = labels & ~included
    assert not np.any(dropped)


@pytest.mark.slow
def test_weak_110_cold_seed_books_the_label_region_not_half():
    """The support books the labelled region, never roughly half of it."""
    import jax.numpy as jnp

    operate, profile, seed, target, requested = _weak_row()
    physical = jnp.asarray(seed)[: operate.physical_node_number]
    masks, topology, _c, _a = operate._fixed_design_read(physical, requested)
    sample_flux = operate.sample_node_flux(jnp.asarray(seed))
    sample_psi_norm = (sample_flux - topology.axis_flux) / topology.flux_span
    label_current = _label_region_current(
        operate, profile.source, masks, sample_psi_norm
    )
    seed_moments = operate.cell_current_moments(jnp.asarray(seed), requested)
    booked = float(np.sum(np.asarray(seed_moments.cell_current)))
    # The clip-compatible support books the label region; the pre-fix support
    # booked only about half of it.
    assert booked >= 0.8 * label_current
    assert booked >= 0.8 * target
