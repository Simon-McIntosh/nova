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
            import netCDF4

            filepath = str(self.filepath)
            with netCDF4.Dataset(filepath, "r+") as nc:
                # Handle both 'group' and '/group' formats
                group_key = self.group.lstrip("/")
                if group_key in nc.groups:
                    # netCDF4 doesn't support deleting groups directly,
                    # so we need to recreate the file without this group
                    pass  # Fall through to file recreation approach
                else:
                    return self  # Group doesn't exist, nothing to delete
        except (ImportError, OSError):
            pass

        # Fallback: recreate file without the target group
        try:
            # Load all other groups, delete file, rewrite without target group
            import netCDF4

            filepath = str(self.filepath)
            group_key = self.group.lstrip("/")

            # Get list of all groups
            with netCDF4.Dataset(filepath, "r") as nc:
                groups = list(nc.groups.keys())

            if group_key not in groups:
                return self  # Group doesn't exist

            # Load data from all OTHER groups
            other_data = {}
            for grp in groups:
                if grp != group_key:
                    with xarray.open_dataset(filepath, group=grp) as ds:
                        other_data[grp] = ds.load()

            # Delete the file
            import os

            os.remove(filepath)

            # Rewrite all other groups
            for grp, ds in other_data.items():
                mode = "w" if not self.is_file() else "a"
                ds.to_netcdf(filepath, mode=mode, group=grp)
                ds.close()

        except (ImportError, OSError, KeyError) as e:
            import warnings

            warnings.warn(f"Failed to delete group: {e}")
        return self

    def store_overwrite(self):
        """Store data, overwriting the group if it exists."""
        self.delete_group()
        # Use append mode if file exists (other groups preserved), else write
        mode = "a" if self.is_file() else "w"
        if self.host is not None:
            with self.fsys.open(str(self.filepath), mode + "b") as file:
                self.data.to_netcdf(file, mode=mode, group=self.group)
        else:
            self.data.to_netcdf(self.filepath, mode=mode, group=self.group)
        self.data.close()
        gc.collect()
        return self

    def load(self):
        """Load dataset from file."""
        with xarray.open_dataset(self.filepath, group=self.group, cache=True) as data:
            data.load()
            self.data = self.data.merge(
                data, combine_attrs="drop_conflicts", compat="override"
            )
        return self
