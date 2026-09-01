"""Evaluation-time open-field-line support selection."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.continuation import SeparatrixContinuation
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.observation import current_ledger, declared_pressure
from nova.equilibrium.sol_closure import EichSolClosure, EichWidth, SolDecayVariant
from nova.equilibrium.source import (
    ContinuationForm,
    DomainProfile,
    ForwardSource,
    SeparatrixContinuity,
)
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes


def _source() -> ForwardSource:
    """Return a source whose open continuation differs away from the edge."""

    core = DomainProfile(
        p_prime=lambda psi: 2.0 + jnp.zeros_like(psi),
        ff_prime=lambda psi: jnp.zeros_like(psi),
    )
    continuation = SeparatrixContinuation(
        form=ContinuationForm.HERMITE_POLYNOMIAL,
        continuity=SeparatrixContinuity.VALUE_AND_GRADIENT,
        support=0.5,
    ).extend_open_field_line(core)
    return ForwardSource(core=core, common_sol=continuation)


def _masks(labels, psi_norm) -> DomainMasks:
    return DomainMasks(
        label=jnp.asarray(labels, dtype=jnp.int8),
        psi_norm=jnp.asarray(psi_norm, dtype=jnp.float64),
    )


def test_live_open_support_uses_flux_material_and_private_mask_eager_and_jit():
    """Flux chooses SOL support while topology vetoes private and material cells."""

    configure_dtypes()
    labels = jnp.asarray(
        [
            PlasmaDomain.CORE,
            PlasmaDomain.CORE,
            PlasmaDomain.PRIVATE_FLUX,
            PlasmaDomain.EXCLUDED_MATERIAL,
        ],
        dtype=jnp.int8,
    )

    def evaluate(psi_norm):
        return DomainMasks(label=labels, psi_norm=psi_norm).open_field_line

    psi_norm = jnp.asarray([0.9, 1.1, 1.1, 1.1])
    expected = np.asarray([False, True, False, False])
    np.testing.assert_array_equal(evaluate(psi_norm), expected)
    np.testing.assert_array_equal(jax.jit(evaluate)(psi_norm), expected)


def test_consistent_labels_reproduce_label_selected_current_bit_exactly():
    """A mesh with no boundary-straddling cell is numerically unchanged."""

    configure_dtypes()
    source = _source()
    masks = _masks(
        [
            PlasmaDomain.CORE,
            PlasmaDomain.COMMON_SOL,
            PlasmaDomain.PRIVATE_FLUX,
            PlasmaDomain.EXCLUDED_MATERIAL,
        ],
        [0.6, 1.2, 0.8, 1.2],
    )
    radius = jnp.ones(4)
    old = jnp.where(
        masks.core,
        source.core.current_density(radius, jnp.where(masks.core, masks.psi_norm, 0.0)),
        0.0,
    ) + jnp.where(
        masks.common_sol,
        source.common_sol.current_density(
            radius, jnp.where(masks.common_sol, masks.psi_norm, 1.0)
        ),
        0.0,
    )

    np.testing.assert_array_equal(source.current_density(radius, masks), old)


def test_stale_label_cannot_delay_open_field_line_current_selection():
    """A moved flux value selects SOL current before a component relabel."""

    configure_dtypes()
    source = _source()
    masks = _masks([PlasmaDomain.CORE, PlasmaDomain.CORE], [0.8, 1.2])
    radius = jnp.ones(2)
    observed = source.current_density(radius, masks)
    frozen_label = source.core.current_density(radius, masks.psi_norm)
    expected_open = source.common_sol.current_density(radius[1:], masks.psi_norm[1:])

    assert float(observed[0]) == float(frozen_label[0])
    assert float(observed[1]) == float(expected_open[0])
    assert float(observed[1]) != float(frozen_label[1])
    relative_delta = abs(float(observed[1] / frozen_label[1] - 1.0))
    assert relative_delta > 0.1


def test_straddling_cell_moment_selects_profile_at_each_quadrature_flux():
    """One cell integrates both sides instead of switching its whole moment."""

    configure_dtypes()
    source = _source()
    masks = _masks([PlasmaDomain.CORE], [0.99])
    evaluation_flux = jnp.asarray([0.95, 1.05])
    radius = jnp.ones(2)

    def support_moments(profile, _centroid_flux, _sample_flux, _support):
        values = profile.current_density(radius, evaluation_flux)
        total = jnp.sum(values)[None]
        return CellCurrentMoments(total, 2.0 * total, 3.0 * total)

    moments = source.current_moments(
        masks,
        support_moments,
        object(),
        sample_flux=jnp.zeros(0),
    )
    confined_only = jnp.sum(source.core.current_density(radius, evaluation_flux))
    expected = source.core.current_density(radius[:1], evaluation_flux[:1])[0]
    expected += source.common_sol.current_density(radius[1:], evaluation_flux[1:])[0]

    np.testing.assert_allclose(moments.cell_current, [expected])
    assert float(moments.cell_current[0]) != float(confined_only)
    assert abs(float(moments.cell_current[0] / confined_only - 1.0)) > 0.01


def test_observations_derive_common_support_instead_of_reading_the_label():
    """Primitive and ledger reporting follow the current evaluation support."""

    configure_dtypes()
    source = _source()
    masks = _masks([PlasmaDomain.CORE, PlasmaDomain.COMMON_SOL], [1.2, 0.8])
    radius = jnp.ones(2)
    pressure = declared_pressure(source, masks, radius, jnp.asarray(1.0))
    expected = jnp.asarray(
        [
            source.common_sol.pressure(radius[:1], masks.psi_norm[:1], 0.0, 1.0)[0],
            source.core.pressure(radius[1:], masks.psi_norm[1:], 0.0, 1.0)[0],
        ]
    )
    np.testing.assert_array_equal(pressure, expected)

    ledger = current_ledger(jnp.asarray([3.0, 5.0]), masks)
    assert float(ledger.common_sol) == 3.0
    assert float(ledger.core) == 5.0


def test_measured_sol_closure_uses_the_same_live_support():
    """The measured SOL policy shares the evaluation-time support contract."""

    configure_dtypes()
    source = _source()
    closure = EichSolClosure(
        width=EichWidth(
            outboard_midplane_poloidal_field_t=0.5,
            heat_flux_width_m=0.01,
            normalized_flux_width=0.2,
            outboard_midplane_radius_m=1.0,
            flux_span_wb=1.0,
        ),
        spreading_length_m=0.02,
        spreading_fraction=0.5,
    )
    masks = _masks([PlasmaDomain.CORE, PlasmaDomain.COMMON_SOL], [1.1, 0.9])
    current = closure.current_density(
        source.core, jnp.ones(2), masks, SolDecayVariant.SINGLE_LENGTH
    )

    assert float(current[0]) != 0.0
    assert float(current[1]) == 0.0


def test_solve_paths_do_not_route_on_the_common_sol_label():
    """Production evaluation seams contain no common component-label branch."""

    implementations = (
        inspect.getsource(ForwardSource.current_density),
        inspect.getsource(ForwardSource.current_moments),
        inspect.getsource(ForwardSource.declared_support),
        inspect.getsource(ForwardFluxOperator._support_partition),
    )
    assert all("COMMON_SOL" not in implementation for implementation in implementations)
    assert not hasattr(ForwardFluxOperator, "shared_domain_masks")
