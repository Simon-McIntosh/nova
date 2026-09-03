"""Public contracts for declared-current amplitude elimination."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.source import (
        CurrentNormalisationError,
        DomainProfile,
        ForwardSource,
        NormalisationPolicy,
    )
    from nova.equilibrium.stencil_mesh import CellCurrentMoments
    from tests import test_equilibrium_forward_solve as forward_fixture


def _moments() -> CellCurrentMoments:
    return CellCurrentMoments(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([3.0, 4.0]),
        jnp.asarray([5.0, 6.0]),
    )


def _operator() -> ForwardFluxOperator:
    """Return the smallest operator surface the constrained map evaluates."""
    operator = object.__new__(ForwardFluxOperator)
    operator.cell_current_moments = lambda _state, _requested=None: _moments()
    operator.external = lambda _current=None, _prescribed=None: jnp.zeros(2)
    operator.current_moment_image = lambda moments: jnp.asarray(
        [jnp.sum(moments.cell_current), jnp.sum(moments.radial_moment)]
    )
    operator._exclude_shadow_residual = lambda _psi, image, _requested=None: image
    return operator


def test_public_map_enforces_the_exact_declared_current() -> None:
    operator = _operator()
    profile = object.__new__(ForwardProfile)
    profile.operator = operator
    target = 12.0

    image = jax.jit(profile.flux_map(target_current=target))(jnp.zeros(2))
    scaled, amplitude = operator.normalised_current_moments(jnp.zeros(2), target)

    assert float(jnp.sum(scaled.cell_current)) == target
    assert float(image[0]) == target
    assert float(amplitude) == pytest.approx(4.0)


def test_guard_fails_loudly_without_clipping() -> None:
    with pytest.raises(CurrentNormalisationError, match="outside"):
        ForwardFluxOperator.current_normalisation_amplitude(1.0, 1.0e-9)


@pytest.mark.parametrize("unscaled", [0.0, -1.0, np.nan])
def test_guard_checks_admissibility_before_division(unscaled: float) -> None:
    with pytest.raises(CurrentNormalisationError, match="outside"):
        ForwardFluxOperator.current_normalisation_amplitude(1.0, unscaled)


def test_common_amplitude_scales_every_current_moment() -> None:
    scaled = ForwardFluxOperator.scaled_current_moments(_moments(), 2.5)

    np.testing.assert_allclose(scaled.cell_current, [2.5, 5.0])
    np.testing.assert_allclose(scaled.radial_moment, [7.5, 10.0])
    np.testing.assert_allclose(scaled.vertical_moment, [12.5, 15.0])


def test_receipt_declares_the_recovered_source_amplitude() -> None:
    profile = DomainProfile(p_prime=lambda value: value, ff_prime=lambda value: value)
    source = ForwardSource(core=profile)

    absolute = source.normalisation_record()
    constrained = source.normalisation_record(amplitude=jnp.asarray(1.25))

    assert int(absolute.policy) == int(NormalisationPolicy.ABSOLUTE)
    assert float(absolute.amplitude) == 1.0
    assert not bool(absolute.rescaled)
    assert int(constrained.policy) == int(NormalisationPolicy.DECLARED_SCALAR_CURRENT)
    assert float(constrained.amplitude) == pytest.approx(1.25)
    assert bool(constrained.rescaled)


def test_public_observation_receipt_carries_the_recovered_amplitude() -> None:
    profile, seed, _vacuum = forward_fixture.machine.__wrapped__()
    unscaled = jnp.sum(profile.operator.cell_current_moments(seed).cell_current)
    target = 1.2 * unscaled

    result = profile.observe(seed, target_current=target)

    np.testing.assert_allclose(result.moments.plasma_current, target, rtol=1.0e-12)
    np.testing.assert_allclose(result.normalisation.amplitude, 1.2, rtol=1.0e-12)
    assert result.normalisation.policy_name == "declared_scalar_current"


def test_absent_target_delegates_to_the_ordinary_operator_path() -> None:
    def sentinel(state):
        return state + 3.0

    def flux_map(current, requested, target, prescribed):
        return sentinel

    operator = SimpleNamespace(flux_map=flux_map)
    profile = object.__new__(ForwardProfile)
    profile.operator = operator

    mapped = profile.flux_map(current=jnp.asarray([2.0]))

    assert mapped is sentinel
