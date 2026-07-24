"""Assemble a true IMAS IDS from the standard-name signal store.

Egress is the boundary materialiser: the spine computes on the dense zarr
store, and at the end of a run this builder reconstructs a real IDS from the
stored arrays via :class:`~nova.imas.ids_entry.IdsEntry` (which wraps
``IDSFactory``), validates it, and writes it out with a single ``put()`` as
HDF5 or IMAS-netCDF. The round-trip ``zarr -> IDS -> validate() -> zarr`` is
what keeps the store from drifting into a private schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray

from nova.utilities.importmanager import check_import
from nova.imas.ids_entry import IdsEntry
from nova.io.ingest import PROVISIONAL_NAMESPACE, SignalSpec

with check_import("imas"):
    import imas


@dataclass
class IdsEgress:
    """Reconstruct and write an IDS from a stored signal dataset."""

    def _variable(self, dataset: xarray.Dataset, name: str) -> str:
        """Return the dataset variable holding ``name`` (plain or provisional)."""
        if name in dataset:
            return name
        provisional = f"{PROVISIONAL_NAMESPACE}/{name}"
        if provisional in dataset:
            return provisional
        raise KeyError(f"standard name {name!r} not present in the signal store")

    def assemble(
        self,
        dataset: xarray.Dataset,
        specs: list[SignalSpec],
        *,
        ids_name: str,
        dd_version: str,
    ) -> IdsEntry:
        """Return an :class:`IdsEntry` rebuilt from the stored signals.

        The dataset's shared ``time`` coordinate becomes the IDS time base
        (homogeneous_time). Each spec writes its array back to the node it was
        extracted from, so the assembled IDS mirrors the ingested structure.
        """
        entry = IdsEntry(name=ids_name, dd_version=str(dd_version))
        time = np.asarray(dataset["time"].values, dtype=float)
        entry.ids.time = time
        entry.ids.ids_properties.homogeneous_time = 1
        if any(spec.node == "time_slice" for spec in specs):
            entry.ids.time_slice.resize(len(time))
        for spec in specs:
            values = np.asarray(dataset[self._variable(dataset, spec.standard_name)])
            parent, _, leaf = spec.path.rpartition(".")
            branch = f"{parent}.*" if parent else "*"
            with entry.node(f"{spec.node}:{branch}"):
                entry[leaf, :] = values
        return entry

    def write(self, entry: IdsEntry, uri: str, *, dd_version: str) -> str:
        """Validate the assembled IDS and write it with a single put().

        Returns the URI written. Raises if validation fails so a malformed
        store can never masquerade as a valid pulse.
        """
        entry.ids.validate()
        database = imas.DBEntry(uri, "w", dd_version=str(dd_version))
        try:
            database.put(entry.ids)
        finally:
            database.close()
        return uri

    def egress(
        self,
        dataset: xarray.Dataset,
        specs: list[SignalSpec],
        uri: str,
        *,
        ids_name: str | None = None,
        dd_version: str | None = None,
    ) -> str:
        """Assemble from ``dataset`` and write to ``uri`` in one call."""
        ids_name = ids_name or dataset.attrs["ids_name"]
        dd_version = dd_version or dataset.attrs["dd_version"]
        entry = self.assemble(dataset, specs, ids_name=ids_name, dd_version=dd_version)
        return self.write(entry, uri, dd_version=dd_version)
