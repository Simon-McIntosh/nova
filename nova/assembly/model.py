"""Perform post-processing analysis on Fourier perterbed TFC dataset."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray

from nova.database.netcdf import netCDF


@dataclass
class DataAttrs:
    """Manage simulation and group dataset labels."""

    name: str | None = None
    filename: str = "vault"
    dirname: Path | str = "data/Assembly"
    group: str | None = field(init=False, default=None)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        if self.group is None:
            self.group = f"{self.__class__.__name__.lower()}"
        if self.name is not None:
            self.group += f"/{self.name}"


@dataclass
class Dataset(ABC, netCDF, DataAttrs):
    """
    Manage build, storage, and retrival of an xarray dataset.

    TFC ansys data is stored in an IO shared folder at:
    \\\\io-ws-ccstore1\\ANSYS_Data\\mcintos\\sector_modules
    """

    filename: str = "vault"
    basename: str = "root"
    data: xarray.Dataset = field(init=False, repr=False, default_factory=xarray.Dataset)

    def __post_init__(self):
        """Load / build dataset."""
        super().__post_init__()
        # self.set_path(self.datapath)
        try:
            self.load()
        except (FileNotFoundError, OSError, KeyError):
            self.build()

    @abstractmethod
    def build(self):
        """Build dataset."""


@dataclass
class ModelData(Dataset):
    """Perform Fourier analysis on TFC deformations."""

    ncoil: ClassVar[int] = 18

    @staticmethod
    def fft(data, axis=-2):
        """Apply fft to dataset."""
        data.attrs["ncoil"] = data.sizes["index"]
        data.attrs["nyquist"] = data.ncoil // 2
        data["mode"] = range(data.nyquist + 1)
        data["coefficient"] = ["real", "imag", "amplitude", "phase"]
        dimensions = list(data.delta.dims)
        dimensions[axis] = "mode"
        dimensions = tuple(dimensions) + ("coefficient",)
        data["fft"] = dimensions, np.zeros(tuple(data.sizes[dim] for dim in dimensions))
        coefficient = np.fft.rfft(data["delta"].data, axis=axis)
        data.fft[..., 0] = coefficient.real
        data.fft[..., 1] = coefficient.imag
        data.fft[..., 2] = np.abs(coefficient) / data.nyquist
        data.fft[..., 0, :, 2] /= 2
        if data.ncoil % 2 == 0:
            data.fft[..., data.nyquist, :, 2] /= 2
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
        self.filter = {
            dimension: self._filter(dimension).data
            for dimension in ["radial", "tangential"]
        }

    @abstractmethod
    def _filter(self, label: str):
        """Return complex filter."""
