"""Standard-name-keyed zarr signal store.

The signal store is the primary data plane for the spine: every signal is a
variable in an :class:`xarray.Dataset` keyed by its IMAS standard name, over a
shared time coordinate, persisted as one group inside a zarr store. The group
name is a content hash of the source identity (pulse URI + DD version) *and* the
COCOS convention of the stored fields, so a re-ingest of the same pulse at the
same convention is a cache hit and a pulse whose convention differs keys a
distinct group. The convention is derived from the DD version the source was
read at, so a group can never be served under a key that does not state it.

Persistence reuses :class:`~nova.database.zarrstore.ZarrStore` (the identity
cache tier) unchanged -- this class only adds the signal-oriented construction
and the source-identity cache key. Unit and provenance travel as store
attributes so the tensorized store can never drift into a private schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray
import zarr

from nova.database.zarrstore import ZarrStore


class StaleConventionError(RuntimeError):
    """Raised when a stored group was written before the key carried COCOS."""


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
    cocos:
        COCOS convention the stored fields are in. Derived from ``dd_version``
        (DD3 -> COCOS 11, DD4 -> COCOS 17) when left unset, and carried in
        :attr:`cache_key` so fields in two conventions can never share a group.
    """

    uri: str = ""
    dd_version: str = ""
    cocos: int = 0

    def __post_init__(self):
        """Derive the convention and group from the source identity when unset."""
        super().__post_init__()
        if not self.cocos and self.dd_version:
            # local import: sourcemap imports this package's ingest module, so a
            # module-level import here would cycle
            from nova.io.sourcemap import dd_equilibrium_convention

            self.cocos = dd_equilibrium_convention(self.dd_version)
        if self.group is None:
            self.group = self.cache_key

    @property
    def cache_key(self) -> str:
        """Return the content-hash group name for the source identity and convention."""
        return self.hash_attrs(
            {"uri": self.uri, "dd_version": self.dd_version, "cocos": self.cocos}
        )

    @property
    def legacy_cache_key(self) -> str:
        """Return the group name a pre-convention key produced, identity only.

        A stored group labelled this way declares no convention, so it cannot be
        matched by a convention-keyed read and is detected rather than served.
        """
        return self.hash_attrs({"uri": self.uri, "dd_version": self.dd_version})

    def legacy_group(self) -> str | None:
        """Return the pre-convention group present in the store, if any.

        A group written before the convention entered the key is keyed on the
        source identity alone, so its presence means the stored fields declare
        no convention.
        """
        if not self.is_store():
            return None
        legacy = self.legacy_cache_key
        if legacy == self.cache_key:
            return None
        root = zarr.open_group(store=self._mapper(), mode="r")
        return legacy if legacy in root else None

    def stale_group(self) -> str | None:
        """Return the pre-convention group when only it is present.

        A group written before the convention entered the key is keyed on the
        source identity alone. When that group exists while the convention-keyed
        one does not, the stored fields' convention is unknown, so the entry is
        reported rather than served.
        """
        legacy = self.legacy_group()
        if legacy is None or self.cached():
            return None
        return legacy

    def reclaim_legacy_group(self) -> str | None:
        """Remove the pre-convention group the convention-keyed write supersedes.

        A legacy group is refused on read while it is the only one present, but
        once the convention-keyed group exists it can no longer be detected as
        stale, so it would survive as an unreachable orphan. Removing it on
        write keeps one group per source; the removed group is named so the
        re-ingest reports what it reclaimed.
        """
        legacy = self.legacy_group()
        if legacy is None or legacy == self.group:
            return None
        root = zarr.open_group(store=self._mapper(), mode="a")
        del root[legacy]
        print(
            f"SignalStore: removed legacy group {legacy!r} for {self.uri!r} at "
            f"DD {self.dd_version}; the convention-keyed group {self.group!r} "
            f"(COCOS {self.cocos}) supersedes it"
        )
        return legacy

    def cached(self) -> bool:
        """Return True when this source identity already has a stored group."""
        if not self.is_store():
            return False
        root = zarr.open_group(store=self._mapper(), mode="r")
        return self.group in root

    def write(self, data: xarray.Dataset):
        """Store the signal dataset, replacing existing groups for the source."""
        self.data = data
        self.reclaim_legacy_group()
        return self.store_overwrite()

    def read(self) -> xarray.Dataset:
        """Load the stored dataset, refusing an entry written under the old key.

        A lone group keyed on the source identity carries no stated convention,
        so it is refused by name instead of being served as though it were the
        convention-keyed entry.
        """
        stale = self.stale_group()
        if stale is not None:
            raise StaleConventionError(
                f"cache group {stale!r} for {self.uri!r} at DD {self.dd_version} "
                "was written before the key carried the stored fields' COCOS "
                f"convention; the convention-keyed group {self.group!r} (COCOS "
                f"{self.cocos}) is absent, so the entry is refused rather than "
                "served and must be re-ingested"
            )
        self.data = xarray.Dataset()
        self.load()
        return self.data
