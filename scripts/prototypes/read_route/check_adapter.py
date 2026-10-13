"""Both polynomial fields preserve their nodal sampling contracts."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from scripts.prototypes.read_route.bell import read_bell
from scripts.prototypes.read_route.jets import read_with_jets


def sampled_read(field, geometry, convention, policy):
    return _sampled_flux(field, geometry)


def _sampled_flux(field, geometry):
    raise AssertionError("adapter must supply the nodal sampler")


class LocalCentreField:
    jets = jnp.asarray([[2.0], [3.0], [5.0]])
    sample_index = jnp.asarray([[0, 1, 2]])

    def cell_value(self, cell, point):
        return jnp.asarray(7.0)


def geometry():
    return SimpleNamespace(
        fit_inverse=jnp.eye(3)[None], centre=jnp.zeros((1, 2)), pitch=jnp.ones(1)
    )


def test_bell_sampler_uses_local_centre(monkeypatch):
    monkeypatch.setattr(topology, "read", sampled_read)
    coefficient, centre = read_bell(LocalCentreField(), geometry(), None, None)
    np.testing.assert_array_equal(coefficient, [[7.0, 3.0, 5.0]])
    np.testing.assert_array_equal(centre, [7.0])


def test_hermite_sampler_uses_stored_centre(monkeypatch):
    monkeypatch.setattr(topology, "read", sampled_read)
    coefficient, centre = read_with_jets(
        jnp.zeros((3, 1, 3, 6)),
        (jnp.zeros(1),) * 3,
        jnp.asarray([[2.0, 0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0, 0], [5.0, 0, 0, 0, 0, 0]]),
        jnp.zeros((3, 2)),
        jnp.asarray([[0, 1, 2]]),
        geometry(),
        None,
        None,
    )
    np.testing.assert_array_equal(coefficient, [[2.0, 3.0, 5.0]])
    np.testing.assert_array_equal(centre, [2.0])
