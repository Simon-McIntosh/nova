"""Facilitate structured access to netCDF data."""
from dataclasses import dataclass, field
import gc
import os
import sys

import xarray
import xxhash

from nova.database.filepath import FilePath


@dataclass
class netCDF(FilePath):
    """Provide regulated access to netCDF database."""

    group: str | None = None
    data: xarray.Dataset | xarray.DataArray = \
        field(default_factory=xarray.Dataset, repr=False)

    def __post_init__(self):
        """Forward post init for for cooperative inheritance."""
        super().__post_init__()

    @FilePath.filepath.getter  # type: ignore
    def filepath(self):
        """Extend FilePath.filepath to include netCDF suffix."""
        return super().filepath.with_suffix('.nc')

    def subgroup(self, *subgroups: str) -> str | None:
        """Return subgroup."""
        subgroup = tuple(group for group in (self.group,) + subgroups
                         if group is not None)
        if len(subgroup) == 0:
            return None
        return '/'.join(subgroup)

    def _clear(self):
        """Clear datafile at self.filepath."""
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)

    @property
    def clear_cache(self):
        """Clear cached datafile at self.filepath."""
        if os.path.isfile(self.filepath):
            remove = input('Confirm removal of the following cached datafile:'
                           f'\n{self.filepath}\nProceed (Y/n)?')
            if remove == '' or remove.lower() == 'y':
                os.remove(self.filepath)
            return
        sys.stdout.write(f'Cached datafile clear:\n{self.filepath}')

    def hash_attrs(self, attrs: dict) -> str:
        """Return xxh32 hex hash of attrs dict."""
        xxh32 = xxhash.xxh32()
        xxh32.update(str(attrs))
        return xxh32.hexdigest()

    def mode(self, mode=None) -> str:
        """Return file access mode."""
        if mode is not None:
            return mode
        if self.is_file():
            return 'a'
        return 'w'

    def store(self, mode=None):
        """Store data as group within netCDF file."""
        mode = self.mode(mode)
        if self.host is not None:  # remote write
            with self.fsys.open(str(self.filepath), mode+'b') as file:
                self.data.to_netcdf(file, mode=mode, group=self.group)
        else:
            self.data.to_netcdf(self.filepath, mode=mode, group=self.group)
        self.data.close()
        gc.collect()
        return self

    def load(self):
        """Load dataset from file."""
        with xarray.open_dataset(
                self.filepath, group=self.group, cache=True) as data:
            data.load()
            self.data = self.data.merge(data, combine_attrs='drop_conflicts')
        return self
