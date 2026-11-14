"""Remove the local iron-field gain and run the in-vessel reference check."""

import jax.numpy as jnp
import pytest

from nova.biot import magnetisation as iron

production = iron._residual


def no_cross_field_gain(state, operands):
    if len(operands) == 7:
        operands = (*operands[:6], jnp.zeros_like(operands[6]))
    return production(state, operands)


iron._residual = no_cross_field_gain
raise SystemExit(
    pytest.main(
        [
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_magnetisation.py",
            "-k",
            "in_vessel_fill_matches_six_limb_reference",
        ]
    )
)
