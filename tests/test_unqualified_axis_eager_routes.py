"""Eager forward-profile reads refuse states without an admitted axis."""

from __future__ import annotations

from dataclasses import replace

import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

from nova.equilibrium.topology import NoQualifiedAxisError
from tests.test_equilibrium_forward_solve import _with_direct_samples
from tests.test_fused_same_state_read import (
    _first_unqualified_map_state,
)

pytest_plugins = ("tests.test_equilibrium_forward_solve",)


@pytest.fixture(scope="module")
def sampled(machine):
    """Return a profile with direct samples and one refused map state."""
    profile, seed, _vacuum = machine
    operator, sampled_seed = _with_direct_samples(profile, seed)
    profile = replace(profile, operator=operator)
    return profile, sampled_seed, _first_unqualified_map_state(profile, sampled_seed)


def test_eager_profile_reads_serve_the_qualified_state(sampled):
    """Both linear integral branches and moment reads serve an admitted state."""
    profile, seed, _unqualified = sampled

    profile.operator.cell_current_moments(seed)
    profile._integral_state(seed)
    profile._integral_state(seed, target_current=1.0)


@pytest.mark.parametrize(
    "read",
    (
        lambda profile, state: profile.operator.cell_current_moments(state),
        lambda profile, state: profile._integral_state(state),
        lambda profile, state: profile._integral_state(state, target_current=1.0),
    ),
    ids=("cell-current-moments", "linear-integrals", "normalised-linear-integrals"),
)
def test_eager_profile_reads_name_the_unqualified_axis_refusal(sampled, read):
    """Every eager read rejects the same unqualified state instead of serving it."""
    profile, _seed, unqualified = sampled

    with pytest.raises(NoQualifiedAxisError, match="qualified magnetic-axis"):
        read(profile, jnp.asarray(unqualified))
