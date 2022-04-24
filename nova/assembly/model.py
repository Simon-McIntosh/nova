"""Perform post-processing analysis on Fourier perterbed TFC dataset."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
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

    @staticmethod
    def fft(data, axis=-2):
        """Apply fft to dataset."""
        data.attrs['ncoil'] = data.dims['index']
        data.attrs['nyquist'] = data.ncoil // 2
        data['mode'] = range(data.nyquist + 1)
        data['coefficient'] = ['real', 'imag', 'amplitude', 'phase']
        dimensions = list(data.delta.dims)
        dimensions[axis] = 'mode'
        dimensions = tuple(dimensions) + ('coefficient',)
        data['fft'] = dimensions, \
            np.zeros(tuple(data.dims[dim] for dim in dimensions))

        coefficient = np.fft.rfft(data['delta'].data, axis=axis)
        data.fft[..., 0] = coefficient.real
        data.fft[..., 1] = coefficient.imag
        data.fft[..., 2] = np.abs(coefficient) / data.nyquist
        data.fft[:, 0, :, 2] /= 2
        if data.ncoil % 2 == 0:
            data.fft[:, data.nyquist, :, 2] /= 2
        data.fft[..., 3] = np.angle(coefficient)


@dataclass
class ModelBase(ModelData):
    """FFT model baseclass."""

    def __post_init__(self):
        """Load finite impulse response filter."""
        super().__post_init__()
        self.load_filter()

    def load_filter(self):
        """Extract filter from dataset."""
        self.filter = {dimension: self._filter(dimension).data
                       for dimension in ['radial', 'tangential']}

    @abstractmethod
    def _filter(self, label: str):
        """Return complex filter."""
