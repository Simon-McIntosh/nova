"""Perform post-processing analysis on Fourier perterbed TFC dataset."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import xarray

from nova.database.filepath import FilePath


@dataclass
class DataAttrs:
    """Manage simulation and group dataset labels."""

    name: str = None
    filename: str = 'vault'
    datapath: str = 'data/Assembly'
    group: str = field(init=False, default=None)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        self.group = f'{self.__class__.__name__.lower()}'
        if self.name is not None:
            self.group += f'/{self.name}'
        self.set_path(self.datapath)


@dataclass
class ModelData(ABC, FilePath, DataAttrs):
    """Perform Fourier analysis on TFC deformations."""

    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    ncoil: ClassVar[int] = 18

    def __post_init__(self):
        """Load / build dataset."""
        super().__post_init__()
        try:
            self.load(lazy=False)
        except (FileNotFoundError, OSError, KeyError):
            self.build()

    @abstractmethod
    def build(self):
        """Build dataset."""


@dataclass
class ModelBase(ModelData):
    """FFT model baseclass."""

    def __post_init__(self):
        """Load finite impulse response filter."""
        super().__post_init__()
        self.filter = {dimension: self._filter(dimension).data
                       for dimension in ['radial', 'tangential']}

    @abstractmethod
    def _filter(self, dimension: str):
        """Return complex filter."""
