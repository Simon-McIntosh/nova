"""Consumption-side IMAS IO: the standard-name-keyed zarr signal store.

The transcoders here move data between true IMAS IDSs and a tensorized,
standard-name-keyed zarr store: :mod:`~nova.io.ingest` extracts a mapped IDS
into the store, :mod:`~nova.io.egress` materialises a valid IDS back out, and
:mod:`~nova.io.signalstore` persists the store keyed by source identity. Name
semantics come from the managed imas-standard-names / imas-standard-names-catalog
packages via :mod:`~nova.io.standardname`; nova owns no standard-name schema.
The legacy G-EQDSK reader/writer (:mod:`~nova.io.geqdsk`) also lives here.
"""

from nova.io.egress import IdsEgress
from nova.io.ingest import IdsIngest, SignalSpec, PROVISIONAL_NAMESPACE
from nova.io.signalstore import SignalStore
from nova.io.standardname import (
    NameSource,
    Resolution,
    StandardNameResolver,
    UnknownStandardName,
)
from nova.io.streaming import SignalStream

__all__ = [
    "IdsEgress",
    "IdsIngest",
    "NameSource",
    "PROVISIONAL_NAMESPACE",
    "Resolution",
    "SignalSpec",
    "SignalStore",
    "SignalStream",
    "StandardNameResolver",
    "UnknownStandardName",
]
