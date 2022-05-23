"""Manage file data access for frame and biot instances."""
from dataclasses import dataclass, field
import os

import xarray

from nova.definitions import root_dir


@dataclass
class FilePath:
    """Manage to access to data via store and load methods."""

    filename: str = field(default=None, repr=True)
    group: str = field(default=None, repr=True)
    path: str = field(default=None, repr=False)
    data: xarray.Dataset = field(default=None, repr=False)

    def set_path(self, subpath: str):
        """Set default path."""
        self.path = os.path.join(root_dir, subpath)

    def check_path(self, path):
        """Return self.path if path is None."""
        if path is None:
            if self.path is None:
                raise FileNotFoundError('default path not set')
            return self.path
        return path

    def check_dir(self, filename, path):
        """Return full filepath, check and make directory."""
        if not (directory := os.path.dirname(filename)):
            directory = self.check_path(path)
            filename = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return filename

    def file(self, filename=None, path=None, extension='.nc'):
        """Return full netCDF file path."""
        if filename is None:
            filename = self.filename
        if not os.path.splitext(filename)[1]:
            filename += extension
        return self.check_dir(filename, path)

    def netcdf_path(self, *labels, group_prefix=True) -> str:
        """Return path for netcdf group."""
        if group_prefix:
            labels = (self.group,) + labels
        labels = [label for label in labels if label is not None]
        return '/'.join(labels)

    @staticmethod
    def mode(file: str) -> str:
        """Return netcdf file access mode."""
        if os.path.isfile(file):
            return 'a'
        return 'w'

    @property
    def filepath(self):
        """Return full filepath for netCDF (.nc) data using default path."""
        assert self.path is not None
        return self.file(self.filename)

    @property
    def isfile(self):
        """Return status of default netCDF file."""
        return os.path.isfile(self.filepath)

    def store(self, filename=None, path=None, group=None):
        """Store data within hdf file."""
        file = self.file(filename, path)
        self.data.to_netcdf(file, group=group, mode=self.mode(file))
        return self

    def load(self, filename=None, path=None, group=None):
        """Load dataset from file."""
        file = self.file(filename, path)
        with xarray.open_dataset(file, group=group) as data:
            self.data = data
            self.data.load()
        return self
