"""Facilitate structured access to netCDF data."""

from dataclasses import dataclass, field
import gc

import xarray
import xxhash

from nova.database.filepath import FilePath


@dataclass
class netCDF(FilePath):
    """Provide regulated access to netCDF database."""

    group: str | None = field(default=None, repr=False)
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

    @FilePath.filepath.getter  # type: ignore
    def filepath(self):
        """Extend FilePath.filepath to include netCDF suffix."""
        return super().filepath.with_suffix(".nc")

    def subgroup(self, *subgroups: str) -> str | None:
        """Return subgroup."""
        subgroup = tuple(
            group for group in (self.group,) + subgroups if group is not None
        )
        if len(subgroup) == 0:
            return None
        return "/".join(subgroup)

    def hash_attrs(self, attrs: dict) -> str:
        """Return xxh32 hex hash of attrs dict."""
        xxh32 = xxhash.xxh32()
        xxh32.update(str(attrs))
        return xxh32.hexdigest()

    def get_mode(self, mode=None) -> str:
        """Return file access mode."""
        if mode is not None:
            return mode
        if self.is_file():
            return "a"
        return "w"

    def store(self, mode=None):
        """Store data as group within netCDF file."""
        mode = self.get_mode(mode)
        if self.host is not None:  # remote write
            with self.fsys.open(str(self.filepath), mode + "b") as file:
                self.data.to_netcdf(file, mode=mode, group=self.group)
        else:
            self.data.to_netcdf(self.filepath, mode=mode, group=self.group)
        self.data.close()
        gc.collect()
        return self

    def delete_group(self):
        """Delete group from netCDF file if it exists."""
        if not self.is_file() or self.group is None:
            return self
        try:
            import h5py

            with h5py.File(self.filepath, "a") as f:
                if self.group in f:
                    del f[self.group]
        except (ImportError, OSError):
            pass  # h5py not available or file locked
        return self

    def load(self):
        """Load dataset from file."""
        with xarray.open_dataset(self.filepath, group=self.group, cache=True) as data:
            data.load()
            self.data = self.data.merge(
                data, combine_attrs="drop_conflicts", compat="override"
            )
        return self
