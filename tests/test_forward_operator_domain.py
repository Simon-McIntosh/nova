"""Profile-owned residual-domain and current-participation contracts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.continuation import SeparatrixContinuation
from nova.equilibrium.domain import (
    DomainMasks,
    PlasmaDomain,
    profile_domain_change,
)
from nova.equilibrium.fixed_point import picard
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import (
    ContinuationForm,
    DomainProfile,
    ForwardSource,
    SeparatrixContinuity,
)
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.jax.config import configure_dtypes


def _core_profile() -> DomainProfile:
    """Return a core profile with a nonzero continuation anchor."""

    return DomainProfile(
        p_prime=lambda psi: -(2.0 - psi),
        ff_prime=lambda psi: jnp.zeros_like(psi),
    )


def _continued_source() -> ForwardSource:
    """Return one source with a declared common-SOL continuation."""

    core = _core_profile()
    policy = SeparatrixContinuation(
        form=ContinuationForm.HERMITE_POLYNOMIAL,
        continuity=SeparatrixContinuity.VALUE_AND_GRADIENT,
        support=0.5,
    )
    return ForwardSource(
        core=core,
        common_sol=policy.extend(core, PlasmaDomain.COMMON_SOL),
    )


def test_residual_support_has_no_boundary_flux_partition() -> None:
    """Residual participation comes from achieved saddle-aware domain labels."""

    implementation = inspect.getsource(ForwardFluxOperator._support_partition)

    assert "boundary_flux" not in implementation
    assert "profile_participation" in implementation
    assert "_fixed_design_read" in implementation
    assert "traced_clip(-" not in implementation


def test_profile_owned_moments_count_common_sol_current_once() -> None:
    """A declared common-SOL cell participates once without a boundary clip."""

    configure_dtypes()
    source = _continued_source()
    masks = DomainMasks(
        label=jnp.asarray(
            [
                PlasmaDomain.CORE,
                PlasmaDomain.COMMON_SOL,
                PlasmaDomain.PRIVATE_FLUX,
                PlasmaDomain.EXCLUDED_MATERIAL,
            ],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.asarray([0.5, 1.25, 1.25, 1.25]),
    )
    radius = jnp.ones(4)

    def support_moments(profile, centroid_flux, _sample_flux, _support):
        density = profile.current_density(radius, centroid_flux)
        return CellCurrentMoments(density, 2.0 * density, 3.0 * density)

    moments = source.current_moments(
        masks,
        support_moments,
        object(),
        sample_flux=masks.psi_norm,
    )
    core_density = source.core.current_density(radius[:1], masks.psi_norm[:1])[0]
    common_density = source.common_sol.current_density(
        radius[1:2], masks.psi_norm[1:2]
    )[0]
    expected = np.asarray([core_density, common_density, 0.0, 0.0])

    np.testing.assert_allclose(moments.cell_current, expected)
    np.testing.assert_allclose(moments.radial_moment, 2.0 * expected)
    np.testing.assert_allclose(moments.vertical_moment, 3.0 * expected)
    assert float(moments.cell_current[1]) != float(core_density + common_density)


def _point_cell_operator(masks: DomainMasks) -> ForwardFluxOperator:
    """Return a fixed-label point-cell operator with one owner per stencil."""

    operator = object.__new__(ForwardFluxOperator)
    operator.grid = SimpleNamespace(
        coordinate=jnp.c_[jnp.ones(4), jnp.arange(4, dtype=jnp.float64)],
        node_number=4,
    )
    operator.wall = SimpleNamespace(node_number=0)
    operator.sample = None
    operator.source = ForwardSource(core=_core_profile())
    operator.area = jnp.ones(4)
    operator.cell_average_stencil = jnp.zeros((4, 5), dtype=jnp.int32)
    operator.cell_average_weight = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
    operator.use_linear_moments = False
    operator._fixed_design_read = lambda _psi, _requested=None: (
        masks,
        None,
        masks.core,
        jnp.asarray(True),
    )
    return operator


def test_point_cell_arm_uses_profile_owned_support_eager_and_jit() -> None:
    """Point-cell averaging admits common SOL and excludes both shadow classes."""

    configure_dtypes()
    masks = DomainMasks(
        label=jnp.asarray(
            [
                PlasmaDomain.CORE,
                PlasmaDomain.COMMON_SOL,
                PlasmaDomain.PRIVATE_FLUX,
                PlasmaDomain.EXCLUDED_MATERIAL,
            ],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.asarray([0.5, 1.25, 0.5, 1.25]),
    )
    operator = _point_cell_operator(masks)

    for evaluate in (
        operator.cell_current_moments,
        jax.jit(operator.cell_current_moments),
    ):
        moments = evaluate(jnp.zeros(4))
        owner_current = float(moments.cell_current[0])
        assert owner_current != 0.0
        np.testing.assert_allclose(
            moments.cell_current,
            [owner_current, owner_current, 0.0, 0.0],
        )

    implementation = inspect.getsource(ForwardFluxOperator.cell_current_moments)
    assert "profile_participation" in implementation
    assert "declared_support" not in implementation
    assert "boundary_flux" not in implementation


def test_shadow_cell_has_zero_residual_sensitivity_eager_and_jit() -> None:
    """A shadowed trial component contributes neither residual nor tangent."""

    masks = DomainMasks(
        label=jnp.asarray(
            [
                PlasmaDomain.CORE,
                PlasmaDomain.COMMON_SOL,
                PlasmaDomain.PRIVATE_FLUX,
                PlasmaDomain.EXCLUDED_MATERIAL,
            ],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.zeros(4),
    )
    operator = _point_cell_operator(masks)
    operator.external = lambda _current=None: jnp.zeros(4)
    operator.internal = lambda psi, *_args: 2.0 * psi
    operator.residual_shadow_mask = lambda _psi, _requested=None: masks.private_flux
    state = jnp.arange(4, dtype=jnp.float64)
    shadow_direction = jnp.asarray([0.0, 0.0, 1.0, 0.0])

    for residual in (operator.residual, jax.jit(operator.residual)):
        value, sensitivity = jax.jvp(
            residual,
            (state,),
            (shadow_direction,),
        )
        assert float(value[2]) == 0.0
        np.testing.assert_array_equal(sensitivity, jnp.zeros(4))


def test_profile_domain_change_attributes_shadow_transitions_eager_and_jit() -> None:
    """Domain receipts distinguish entered, left, and static-material cells."""

    previous = DomainMasks(
        label=jnp.asarray(
            [PlasmaDomain.CORE, PlasmaDomain.PRIVATE_FLUX, PlasmaDomain.CORE],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.zeros(3),
    )
    current = DomainMasks(
        label=jnp.asarray(
            [
                PlasmaDomain.PRIVATE_FLUX,
                PlasmaDomain.CORE,
                PlasmaDomain.EXCLUDED_MATERIAL,
            ],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.zeros(3),
    )

    for observe in (profile_domain_change, jax.jit(profile_domain_change)):
        change = observe(previous, current)
        assert int(change.shadow_entered) == 1
        assert int(change.shadow_left) == 1
        assert int(change.shadow_changed) == 2
        assert int(change.material_changed) == 1


def test_fixed_point_receipt_records_each_shadow_mask_change_eager_and_jit() -> None:
    """Every fixed-trip outer evaluation publishes its changed-cell count."""

    def solve(initial):
        return picard(
            lambda state: state + 1.0,
            initial,
            evaluations=3,
            relaxation=1.0,
            shadow_mask_fn=lambda state: jnp.asarray(
                [state[0] >= 1.0, state[0] >= 2.0]
            ),
        ).shadow_mask_changes

    for observe in (solve, jax.jit(solve)):
        np.testing.assert_array_equal(observe(jnp.zeros(1)), [1, 1, 0])
