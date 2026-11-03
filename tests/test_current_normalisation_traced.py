"""Admissibility refusal for the declared-current amplitude under tracing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import CurrentNormalisationError
from nova.jax.config import configure_dtypes

configure_dtypes()

TARGET_CURRENT = 1.0e6


def amplitude_under_linearize(unscaled_current: float) -> jax.Array:
    """Return the amplitude a linearized evaluation reports for one input."""

    amplitude, _tangent = jax.linearize(
        lambda value: ForwardFluxOperator.current_normalisation_amplitude(
            TARGET_CURRENT, value
        ),
        jnp.asarray(unscaled_current),
    )
    return amplitude


@pytest.mark.parametrize("unscaled_current", [-2.0, 1.0e-9])
def test_traced_inadmissible_operand_reports_a_non_finite_amplitude(
    unscaled_current: float,
) -> None:
    amplitude = amplitude_under_linearize(unscaled_current)
    assert not np.isfinite(np.asarray(amplitude))


def test_traced_admissible_operand_reports_the_scaled_amplitude() -> None:
    amplitude = amplitude_under_linearize(2.0)
    assert np.isfinite(np.asarray(amplitude))
    assert float(amplitude) == pytest.approx(TARGET_CURRENT / 2.0)


def test_concrete_inadmissible_operand_raises_the_intended_refusal() -> None:
    with pytest.raises(CurrentNormalisationError):
        ForwardFluxOperator.current_normalisation_amplitude(
            TARGET_CURRENT, jnp.asarray(-2.0)
        )