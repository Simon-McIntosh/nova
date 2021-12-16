"""Biot-Savart calculation base class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Union

import numpy as np
import numpy.typing as npt
import pandas
import xarray

from nova.electromagnetic.biotset import BiotSet
from nova.electromagnetic.dataarray import DataArray


@dataclass
class DaskBiotVector:
    """Store Biot vectors - implement subset of xarray dataset interface."""

    index: pandas.Index
    chunks: Union[int, str, Mapping] = None
    data: xarray.Dataset = field(init=False)

    def __post_init__(self):
        """Create empty dataset."""
        self.data = xarray.Dataset(coords=dict(index=self.index))

    def __setitem__(self, attr, value):
        """Inset index aligned item to dataset."""
        value = xarray.DataArray(value, dims=['index'])
        if self.chunks is not None:  # convert to dask array
            value = value.chunk(self.chunks)
        self.data[attr] = value

    def __getitem__(self, attr):
        """Return dataarray from dataset."""
        return self.data[attr]

    def __delitem__(self, key):
        """Delete variable from dataset."""
        del self.data[key]

    def __iter__(self):
        """Itterate through variables in dataset."""
        return iter(self.data)

    def __len__(self):
        """Return number of variables in dataset."""
        return len(self.data)


class BiotVector(DataArray):
    """Biot Vector data container built from DataFrame extension."""

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.update_columns()

    def update_metadata(self, data, columns, attrs, metadata):
        """Extend FrameAttrs update_metadata."""
        if columns is not None:
            metadata = {'required': columns, 'array': columns,
                        'default': dict.fromkeys(columns, 0.)} | metadata
        super().update_metadata(data, columns, attrs, metadata)


@dataclass
class BiotMatrix(BiotSet):
    """Store Biot matricies."""

    columns: list[str] = field(default_factory=lambda: [])
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Init data."""
        super().__post_init__()
        self.data = xarray.Dataset(
            coords=dict(source=self.get_index('source'),
                        plasma=self.source.index[self.source.plasma].to_list(),
                        target=self.get_index('target')))

    def initialize_dataset(self):
        """Initialize dataarrays."""
        for var in self.columns:
            self.data[var] = xarray.DataArray(
                0., dims=['target', 'source'],
                coords=[self.data.target, self.data.source])
        for var in self.columns:  # unit filaments
            self.data[f'_{var}'] = xarray.DataArray(
                0., dims=['target', 'plasma'],
                coords=[self.data.target, self.data.plasma])

    def get_index(self, frame):
        """Return matrix coordinate, reduce if flag True."""
        biotframe = getattr(self, frame)
        if biotframe.reduce:
            return biotframe.biotreduce.index.to_list()
        return biotframe.index.to_list()


@dataclass
class BiotSolve(ABC, BiotMatrix):
    """Biot-Savart base-class. Define calculaiton interface."""

    mu_o = 4*np.pi*1e-7

    vector: BiotVector = field(init=False, repr=False)

    def __post_init__(self):
        """Init static and unit datasets."""
        super().__post_init__()
        self.vector = BiotVector(index=self.index, columns=self.columns)
        self.calculate_vectors()
        self.store_matrix()

    def calculate_vectors(self):
        """Calculate vector and scalar potential and magnetic field."""
        coeff = self.calculate_coefficients()
        self.calculate_vector_potential(coeff)
        self.calculate_scalar_potential(coeff)
        self.calculate_magnetic_field(coeff)

    @abstractmethod
    def calculate_coefficients(self) -> dict[str, npt.ArrayLike]:
        """Return interaction coefficients."""

    @abstractmethod
    def calculate_vector_potential(self, coeff) -> dict[str, npt.ArrayLike]:
        """
        Calculate target vector potential, Wb/Amp-turn-turn..

        Define in cylindrical (r, phi, z) or cartesian (x, y, z) coordinates.

        """

    def calculate_scalar_potential(self, coeff) -> npt.ArrayLike:
        """Calculate scalar potential, axisymmetric-only."""

    @abstractmethod
    def calculate_magnetic_field(self, coeff):
        """
        Calculate magnetic field, T/Amp-turn-turn.

        Define in cylindrical (r, phi, z) or cartesian (x, y, z) coordinates.

        """

    @property
    def shape(self):
        """Return source-target shape."""
        return (len(self.target), len(self.source))

    def store_matrix(self):
        """
        Store interaction matrices.

        Extract plasma (unit) interaction from full matrix.
        Multiply by source and target turns.
        Apply reduction summations.

        """
        for col in self.vector:
            matrix = self.vector[col].reshape(*self.shape)  # .to_numpy()
            plasma = matrix[:, self.source.plasma]
            if self.source.turns:
                matrix *= self.source('nturn').reshape(*self.shape)
            if self.target.turns:
                matrix *= (turns := self.target('nturn').reshape(*self.shape))
                plasma *= turns[:, self.source.plasma]
            # reduce
            if self.source.reduce and self.source.biotreduce.reduce:
                matrix = np.add.reduceat(
                    matrix, self.source.biotreduce.indices, axis=1)
            if self.target.reduce and self.target.biotreduce.reduce:
                matrix = np.add.reduceat(
                    matrix, self.target.biotreduce.indices, axis=0)
                plasma = np.add.reduceat(
                    plasma, self.target.biotreduce.indices, axis=0)
            # link source
            source_link = self.source.biotreduce.link
            if self.source.reduce and len(source_link) > 0:
                for link in source_link:  # sum linked columns
                    ref, factor = source_link[link]
                    matrix[:, ref] += factor * matrix[:, link]
                matrix = np.delete(matrix, list(source_link), 1)
            # link target
            target_link = self.target.biotreduce.link
            if self.target.reduce and len(target_link) > 0:
                for link in target_link:  # sum linked columns
                    ref, factor = target_link[link]
                    matrix[ref, :] += factor * matrix[link, :]
                    plasma[ref, :] += factor * plasma[link, :]
                matrix = np.delete(matrix, list(target_link), 0)
                plasma = np.delete(plasma, list(target_link), 0)
            # store
            self.data[col] = (['target', 'source'], matrix)
            self.data[f'_{col}'] = (['target', 'plasma'], plasma)
