"""Manage file data access for frame and biot instances."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os

from nova.definitions import root_dir


@dataclass
class FilePath(ABC):
    """Frame data base class - store/load frame and biot data."""

    path: str = field(default=None)

    def __post_init__(self):
        """Init dataset path."""
        if self.path is None:
            self.path = os.path.join(root_dir, 'data/Nova/coilsets')

    def check_path(self, path):
        """Return self.path if path is None."""
        if path is None:
            return self.path
        return path

    def file(self, file, path=None, extension='.nc'):
        """Return full netCDF file path."""
        if not os.path.splitext(file)[1]:
            file += extension
        return os.path.join(self.check_path(path), file)

    @abstractmethod
    def store(self, file: str, path=None):
        """Store frame and subframe as groups within hdf file."""

    @abstractmethod
    def load(self, file: str, path=None):
        """Load frameset from file."""
