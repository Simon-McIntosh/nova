"""Remove the local iron-field gain and run a constitutive reference check."""

import argparse

import jax.numpy as jnp
import pytest

from nova.biot import magnetisation as iron

production = iron._residual


def no_cross_field_gain(state, operands):
    if len(operands) == 7:
        operands = (*operands[:6], jnp.zeros_like(operands[6]))
    return production(state, operands)


iron._residual = no_cross_field_gain
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fast", action="store_true")
args = parser.parse_args()
raise SystemExit(
    pytest.main(
        [
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_magnetisation.py",
            "-k",
            "cross_field_gain_keeps_linear_reference"
            if args.fast
            else "in_vessel_fill_matches_six_limb_reference",
        ]
    )
)
