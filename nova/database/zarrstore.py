"""Facilitate structured access to a zarr-backed identity cache.

The identity cache stores compiled frames and solved operators keyed by a
content hash of the machine identity. This store backs that cache with zarr:
each cache entry is a named group inside a single ``.zarr`` store, and stale
entries are evicted with a native group delete rather than the whole-file
rewrite the netCDF backend is forced into. netCDF remains available through
:class:`~nova.database.netcdf.netCDF` for interchange and export.
"""

from dataclasses import dataclass, field
import gc

import xarray
import zarr

from nova.database.filepath import FilePath


@dataclass
class ZarrStore(FilePath):
    """Provide regulated access to a grouped zarr store."""

    group: str | None = field(default=None, repr=False)
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

    @FilePath.filepath.getter  # type: ignore
    def filepath(self):
        """Extend FilePath.filepath to include the zarr store suffix."""
        return super().filepath.with_suffix(".zarr")

    def subgroup(self, *subgroups: str) -> str | None:
        """Return the compound group path for the given subgroups."""
        subgroup = tuple(
            group for group in (self.group,) + subgroups if group is not None
        )
        if len(subgroup) == 0:
            return None
        return "/".join(subgroup)

    def is_store(self) -> bool:
        """Return True when the zarr store directory exists on the host."""
        return self.fsys.isdir(str(self.filepath))

    def get_mode(self, mode=None) -> str:
        """Return the zarr write mode, appending into an existing store."""
        if mode is not None:
            return mode
        if self.is_store():
            return "a"
        return "w"

    def _mapper(self):
        """Return the store target, an fsspec mapper for remote hosts."""
        if self.host is not None:
            return self.fsys.get_mapper(str(self.filepath))
        return str(self.filepath)

    def store(self, mode=None):
        """Store data as a group within the zarr store."""
        self.data.to_zarr(
            self._mapper(),
            group=self.group,
            mode=self.get_mode(mode),
            consolidated=False,
        )
        self.data.close()
        gc.collect()
        return self

    def delete_group(self):
        """Evict a cache group via a native zarr delete, keeping its siblings."""
        if self.group is None or not self.is_store():
            return self
        root = zarr.open_group(store=self._mapper(), mode="a")
        if self.group in root:
            del root[self.group]
        return self

    def store_overwrite(self):
        """Store data, replacing the group in place if it already exists."""
        self.delete_group()
        return self.store(mode=self.get_mode())

    def load(self):
        """Load a group from the zarr store and merge it into data.

        A missing store or absent group signals a cache miss as
        :class:`FileNotFoundError` so callers can treat every backend
        uniformly; zarr otherwise reports an absent group as a
        :class:`zarr.errors.GroupNotFoundError` (a ``ValueError``).
        """
        try:
            data = xarray.open_zarr(
                self._mapper(), group=self.group, consolidated=False
            )
        except zarr.errors.NodeNotFoundError as error:
            raise FileNotFoundError(str(self.filepath)) from error
        with data:
            data.load()
            self.data = self.data.merge(
                data, combine_attrs="drop_conflicts", compat="override"
            )
        return self
