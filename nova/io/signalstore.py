"""Standard-name-keyed zarr signal store.

The signal store is the primary data plane for the spine: every signal is a
variable in an :class:`xarray.Dataset` keyed by its IMAS standard name, over a
shared time coordinate, persisted as one group inside a zarr store. The group
name is a content hash of the source identity (pulse URI + DD version) so a
re-ingest of the same pulse is a cache hit.

Persistence reuses :class:`~nova.database.zarrstore.ZarrStore` (the identity
cache tier) unchanged -- this class only adds the signal-oriented construction
and the source-identity cache key. COCOS, unit, and provenance travel as store
attributes so the tensorized store can never drift into a private schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray
import zarr

from nova.database.zarrstore import ZarrStore


@dataclass
class SignalStore(ZarrStore):
    """A grouped zarr store whose group is a source-identity cache key.

    Parameters
    ----------
    uri:
        Source pulse URI (or any stable source identifier).
    dd_version:
        Data Dictionary version the source was read at. Combined with ``uri``
        into the group name so signals from the same pulse at the same DD
        version reuse one group and a different version keys a distinct group.
    """

    uri: str = ""
    dd_version: str = ""

    def __post_init__(self):
        """Derive the group from the source identity when unset."""
        super().__post_init__()
        if self.group is None:
            self.group = self.cache_key

    @property
    def cache_key(self) -> str:
        """Return the content-hash group name for the source identity."""
        return self.hash_attrs({"uri": self.uri, "dd_version": self.dd_version})

    def cached(self) -> bool:
        """Return True when this source identity already has a stored group."""
        if not self.is_store():
            return False
        root = zarr.open_group(store=self._mapper(), mode="r")
        return self.group in root

    def write(self, data: xarray.Dataset):
        """Store the signal dataset, replacing any existing group in place."""
        self.data = data
        return self.store_overwrite()

    def read(self) -> xarray.Dataset:
        """Load and return the stored signal dataset for this identity."""
        self.data = xarray.Dataset()
        self.load()
        return self.data
