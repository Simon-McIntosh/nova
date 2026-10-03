"""Exact-clip cold seeds choose a boundary level that books unit current."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


class _BoundaryLevelOperator:
    use_linear_moments = True
    grid = SimpleNamespace(node_number=2)
    physical_node_number = 4

    def __init__(self, slope: float = 25.0, clip_mode: str = "exact"):
        self.slope = float(slope)
        self.clip_mode = clip_mode
        self.read_count = 0

    def __getattr__(self, name: str):
        # ``_clip_consistent_seed`` reads the operator's active clip mode under a
        # name ending in ``clip_mode``; the fake exposes its own explicitly.
        if name.endswith("clip_mode"):
            return self.clip_mode
        raise AttributeError(name)

    def read(self, _state, _requested_class):
        self.read_count += 1
        return None, SimpleNamespace(flux_span=-2.0)

    def cell_current_moments(self, state, requested_class=None):
        assert requested_class is TopologyClass.LIMITED
        wall_level = jnp.mean(jnp.asarray(state)[2:4])
        booked = 50.0 - self.slope * wall_level
        zero = jnp.zeros(1, dtype=jnp.float64)
        return CellCurrentMoments(jnp.atleast_1d(booked), zero, zero)


def _profile(operator: _BoundaryLevelOperator) -> ForwardProfile:
    profile = object.__new__(ForwardProfile)
    profile.operator = operator
    return profile


def test_exact_clip_seed_solves_boundary_level_for_unit_amplitude() -> None:
    configure_dtypes()
    operator = _BoundaryLevelOperator(clip_mode="exact")
    seed = jnp.asarray([4.0, 3.0, 0.0, 0.0, 8.0], dtype=jnp.float64)

    adjusted = _profile(operator)._clip_consistent_seed(
        seed, 100.0, TopologyClass.LIMITED
    )
    moments = operator.cell_current_moments(adjusted, TopologyClass.LIMITED)

    np.testing.assert_array_equal(np.asarray(adjusted)[[0, 1, 4]], [4.0, 3.0, 8.0])
    np.testing.assert_array_equal(np.asarray(adjusted)[2:4], [-2.0, -2.0])
    assert float(jnp.sum(moments.cell_current)) == pytest.approx(100.0)
    assert operator.read_count == 1


def test_nonexact_seed_is_bit_identical_and_does_not_read_operator() -> None:
    configure_dtypes()
    operator = _BoundaryLevelOperator(clip_mode="chord")
    seed = jnp.asarray([4.0, 3.0, 0.0, 0.0, 8.0], dtype=jnp.float64)

    adjusted = _profile(operator)._clip_consistent_seed(
        seed, 100.0, TopologyClass.LIMITED
    )

    np.testing.assert_array_equal(adjusted, seed)
    assert operator.read_count == 0


def test_exact_clip_seed_refuses_an_unbracketed_current() -> None:
    configure_dtypes()
    operator = _BoundaryLevelOperator(slope=0.0, clip_mode="exact")
    seed = jnp.asarray([4.0, 3.0, 0.0, 0.0, 8.0], dtype=jnp.float64)

    with pytest.raises(
        ValueError,
        match=r"does not bracket unit amplitude: a\(0\)=2, a\(1\)=2",
    ):
        _profile(operator)._clip_consistent_seed(seed, 100.0, TopologyClass.LIMITED)
