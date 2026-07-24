"""Transcode an IMAS IDS into the standard-name-keyed signal store.

Ingest performs a whole-IDS extraction: given an in-memory IDS (the caller's
single ``get()``) and a signal map -- the facility-signal to standard-name
mapping that codex discovers in production -- it pulls each mapped node into a
dense array via the Access Layer tensorizer semantics
(:class:`~nova.imas.ids_index.IdsIndex`) and assembles an
:class:`xarray.Dataset` keyed by standard name over the shared time base.

nova invents no names here: every key is resolved against ISN/ISNC through
:class:`~nova.io.standardname.StandardNameResolver`. A signal whose name is
grammar-valid but absent from the catalog is kept under a clearly-marked
provisional namespace and flagged for a catalog-fork contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray

from nova.imas.ids_index import IdsIndex
from nova.io.signalstore import SignalStore
from nova.io.standardname import StandardNameResolver

#: Variable-name prefix marking a signal whose standard name is grammar-valid
#: but not (yet) in the installed catalog -- a candidate fork contribution.
PROVISIONAL_NAMESPACE = "provisional"


@dataclass(frozen=True)
class SignalSpec:
    """One entry in a facility-signal to standard-name map.

    Parameters
    ----------
    standard_name:
        The IMAS standard name the extracted signal is keyed by.
    node:
        The IDS extraction node passed to :class:`IdsIndex` (e.g.
        ``"time_slice"`` for equilibrium global quantities, ``"coil"`` for
        pf_active).
    path:
        The attribute path within the node (e.g. ``"global_quantities.ip"``).
    """

    standard_name: str
    node: str
    path: str


@dataclass
class IdsIngest:
    """Extract mapped IDS signals into a standard-name-keyed dataset.

    Parameters
    ----------
    resolver:
        Standard-name resolver used to key and annotate each signal.
    cocos:
        COCOS convention of the source data, carried as a store attribute.
    """

    resolver: StandardNameResolver = field(default_factory=StandardNameResolver)
    cocos: int = 11

    def _variable_name(self, name: str, provisional: bool) -> str:
        """Return the store variable name, namespacing provisional signals."""
        if provisional:
            return f"{PROVISIONAL_NAMESPACE}/{name}"
        return name

    def tensorize(
        self,
        ids,
        specs: list[SignalSpec],
        *,
        uri: str,
        dd_version: str,
        provenance: str = "",
    ) -> xarray.Dataset:
        """Return a standard-name-keyed dataset extracted from ``ids``.

        The IDS is assumed homogeneous in time (the shared time base lives at
        the IDS level); each signal is stored over the ``time`` coordinate with
        its unit, source, and originating IDS path as variable attributes.
        """
        time = np.asarray(ids.time, dtype=float)
        data_vars: dict[str, xarray.DataArray] = {}
        provisional: list[str] = []
        for spec in specs:
            resolution = self.resolver.resolve(spec.standard_name)
            array = np.asarray(IdsIndex(ids, spec.node).array(spec.path))
            dims = self._dims(spec.standard_name, array.shape, len(time))
            attrs = {
                "standard_name": spec.standard_name,
                "ids_node": spec.node,
                "ids_path": spec.path,
                "source": resolution.source.value,
                "status": resolution.status,
            }
            if resolution.unit is not None:
                attrs["units"] = resolution.unit
            if resolution.kind is not None:
                attrs["kind"] = resolution.kind
            variable = self._variable_name(spec.standard_name, resolution.provisional)
            if resolution.provisional:
                provisional.append(spec.standard_name)
            data_vars[variable] = xarray.DataArray(array, dims=dims, attrs=attrs)
        dataset = xarray.Dataset(data_vars, coords={"time": time})
        dataset.attrs = {
            "uri": uri,
            "dd_version": str(dd_version),
            "ids_name": ids.metadata.name,
            "cocos": self.cocos,
            "provenance": provenance,
            "provisional_names": provisional,
        }
        return dataset

    @staticmethod
    def _dims(name: str, shape: tuple[int, ...], ntime: int) -> tuple[str, ...]:
        """Return dimension names, using ``time`` for the leading time axis."""
        if shape and shape[0] == ntime:
            return ("time",) + tuple(f"{name}_dim{i}" for i in range(1, len(shape)))
        return tuple(f"{name}_dim{i}" for i in range(len(shape)))

    def ingest(
        self,
        ids,
        specs: list[SignalSpec],
        store: SignalStore,
        *,
        provenance: str = "",
    ) -> xarray.Dataset:
        """Tensorize ``ids`` and persist it into ``store``, returning the data."""
        dataset = self.tensorize(
            ids,
            specs,
            uri=store.uri,
            dd_version=store.dd_version,
            provenance=provenance,
        )
        store.write(dataset)
        return dataset
