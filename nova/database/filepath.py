"""Manage file data access for frame and biot instances."""
from dataclasses import dataclass, field
import os

import appdirs
try:
    import imas
    IMPORT_IMAS = True
except ImportError:
    IMPORT_IMAS = False
import xarray
import xxhash

import nova
from nova.definitions import root_dir


@dataclass
class FilePath:
    """Manage to access to data via store and load methods."""

    filename: str | None = field(default=None, repr=True)
    group: str | None = field(default=None, repr=True)
    path: str | None = field(default=None, repr=False)
    directory: str = 'user_data'
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)

    def __post_init__(self):
        """Forward post init for for cooperative inheritance."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    @staticmethod
    def get_path(directory: str, subpath: str) -> str:
        """Return full filepath."""
        if hasattr(appdirs, (appattr := f'{directory}_dir')):
            app = getattr(appdirs, appattr)
            if subpath == 'nova':
                return app(nova.__name__, version=nova.__version__)
            if subpath == 'imas' and IMPORT_IMAS:
                name, version = imas.__name__.split('_', 1)
                return app(appname=name, version=version)
            return app(subpath)
        if directory == 'root':
            directory = root_dir
        if not subpath:
            return directory
        return os.path.join(directory, subpath)

    def set_path(self, subpath=None):
        """Set default path."""
        self.path = self.get_path(self.directory, subpath)

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
        """Return full netCDF file path and group."""
        if filename is None:
            filename = self.filename
        if not os.path.splitext(filename)[1]:
            filename += extension
        return self.check_dir(filename, path)

    def netcdf_group(self, group=None):
        """Return netcdf group."""
        if group is None:
            return self.group
        return group

    def netcdf_path(self, *labels, group_prefix=True) -> str:
        """Return path for netcdf group."""
        if group_prefix:
            labels = (self.group,) + labels
        labels = tuple(label for label in labels if label is not None)
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

    def hash_attrs(self, attrs: dict) -> str:
        """Return xxh32 hex hash of attrs dict."""
        xxh32 = xxhash.xxh32()
        xxh32.update(attrs.__str__())
        return xxh32.hexdigest()

    def store(self, filename=None, path=None, group=None):
        """Store data within hdf file."""
        file = self.file(filename, path)
        group = self.netcdf_group(group)
        self.data.to_netcdf(file, group=group, mode=self.mode(file))
        return self

    def load(self, filename=None, path=None, group=None):
        """Load dataset from file."""
        file = self.file(filename, path)
        group = self.netcdf_group(group)
        with xarray.open_dataset(file, group=group) as data:
            self.data = data
            self.data.load()
        return self
