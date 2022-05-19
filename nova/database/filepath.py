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

    def check_dir(self, file, path):
        """Return full filepath, check and make directory."""
        if not (directory := os.path.dirname(file)):
            directory = self.check_path(path)
            file = os.path.join(directory, file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return file

    def file(self, file, path=None, extension='.nc'):
        """Return full netCDF file path."""
        if not os.path.splitext(file)[1]:
            file += extension
        return self.check_dir(file, path)

    def netcdf_path(self, *labels, group_prefix=True) -> str:
        """Return path for netcdf group."""
        if group_prefix:
            labels = (self.group,) + labels
        labels = [label for label in labels if label is not None]
        return '/'.join(labels)

    @property
    def filepath(self):
        """Return full filepath for netCDF (.nc) data using default path."""
        assert self.path is not None
        return self.file(self.filename)

    @property
    def isfile(self):
        """Return status of default netCDF file."""
        return os.path.isfile(self.filepath)

    def _disable_HDF5_lock(self):
        """Disable HDF5 file locking via sysenv."""
        if 'HDF5_USE_FILE_LOCKING' not in os.environ:
            os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    def store(self, mode='a'):
        """Store data within hdf file."""
        file = self.file(self.filename)
        if not os.path.isfile(file):
            mode = 'w'
        try:
            self.data.to_netcdf(file, group=self.group, mode=mode)
        except OSError:
            self._disable_HDF5_lock()
            self.data.to_netcdf(file, group=self.group, mode=mode)
        return self

    def load(self, lazy=True):
        """Load dataset from file."""
        file = self.file(self.filename)
        with xarray.open_dataset(file, group=self.group) as data:
            self.data = data
            if not lazy:
                self.data.load()
        return self
