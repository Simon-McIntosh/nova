"""Manage file data access for frame and biot instances."""
from abc import abstractmethod
from dataclasses import dataclass, field
import os

from nova.definitions import root_dir


@dataclass
class FilePath:
    """Manage to access to data via store and load methods."""

    path: str = field(default=None)

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

    @property
    def filepath(self):
        """Return full filepath for netCDF data using default path."""
        assert self.path is not None
        return self.file(self.filename)

    @property
    def isfile(self):
        """Return status of default netCDF file."""
        return os.path.isfile(self.filepath)

    @abstractmethod
    def store(self, file: str, path=None):
        """Store frame and subframe as groups within hdf file."""

    @abstractmethod
    def load(self, file: str, path=None):
        """Load frameset from file."""
