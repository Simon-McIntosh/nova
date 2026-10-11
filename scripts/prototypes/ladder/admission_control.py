"""Make the membership refusal test fail when its numerical guard is disabled."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from nova.equilibrium import topology  # noqa: E402


def permissive_bound(_area, _geometry):
    return jnp.asarray(1e-7)


topology._membership_roundoff_bound = permissive_bound
raise SystemExit(
    pytest.main(
        [
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_read_ten_thousand_cells.py::test_large_membership_overfill_is_refused",
        ]
    )
)
