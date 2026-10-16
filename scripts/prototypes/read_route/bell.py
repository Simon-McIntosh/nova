"""Read-route Bell-mesh adapter over the production fixed-point mesh owner.

The mesh, field, polynomial-row and triangle-transform machinery lives in
``nova.equilibrium.fixed_points``; re-exporting it here keeps the read-route
prototype importers resolving against a single owner.
"""

import jax.numpy as jnp

from nova.equilibrium.fixed_points import (
    BellField,
    BellMesh,
    build_mesh,
    field_from_jets,
)
from scripts.prototypes.read_route.jets import read_field

__all__ = [
    "BellField",
    "BellMesh",
    "build_mesh",
    "field_from_jets",
    "read_bell",
    "read_cached_bell",
]


def read_bell(field, geometry, convention, policy):
    """Use exact nodal samples and the Bell field in the existing topology body."""

    return read_field(field, geometry, convention, policy)


def read_cached_bell(payload, geometry, convention, policy):
    """Contract runtime source moments and form the Bell field inside the read."""
    matrix, moments, exterior, sample_index, mesh = payload
    jets = exterior + jnp.einsum("ncmq,cm->nq", matrix, jnp.stack(tuple(moments), -1))
    return read_bell(
        field_from_jets(jets, sample_index, mesh), geometry, convention, policy
    )
