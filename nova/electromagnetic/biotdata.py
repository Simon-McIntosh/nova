"""Biot-Savart calculation base class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import xarray

from nova.electromagnetic.biotset import BiotSet
from nova.electromagnetic.metaframe import MetaFrame
from nova.electromagnetic.dataarray import DataArray


class BiotVector(DataArray):
    """Store Biot vectors."""

    def __init__(self, data=None, index=None, columns=None, attrs=None,
                 **metadata):
        super().__init__(data, index, columns, attrs, **metadata)
        self.update_columns()

    def extract_attrs(self, data, attrs):
        """Extend FrameAttrs.extract_attrs, lanuch custom metaframe."""
        if not self.hasattrs('metaframe'):
            self.attrs['metaframe'] = MetaFrame(
                self.index,
                required=['Psi', 'Ax', 'Ay', 'Az', 'Bx', 'By', 'Bz'],
                array=['Psi', 'Ax', 'Ay', 'Az', 'Bx', 'By', 'Bz'],
                default=dict().fromkeys(['Psi', 'Ax', 'Ay', 'Az',
                                         'Bx', 'By', 'Bz'], 0.0))
        super().extract_attrs(data, attrs)


@dataclass
class BiotMatrix(BiotSet):
    """Store Biot matricies."""

    data_vars: list[str] = field(init=False, default_factory=lambda: [
        'Psi', 'Ax', 'Ay', 'Az', 'Bx', 'By', 'Bz'])
    static: xarray.Dataset = field(init=False, repr=False)
    unit: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Init static and unit datasets."""
        super().__post_init__()
        self.static = xarray.Dataset(
            coords=dict(source=self.get_coord('source'),
                        target=self.get_coord('target')))
        self.unit = xarray.Dataset(
            coords=dict(source=self.source.index[self.source.plasma],
                        target=self.get_coord('target')))

    def initialize(self):
        """Initialize dataarrays."""
        for var in self.data_vars:
            self.static[var] = xarray.DataArray(0., dims=['target', 'source'],
                                                coords=self.static.coords)
            self.unit[var] = xarray.DataArray(0., dims=['target', 'source'],
                                              coords=self.unit.coords)

    def get_coord(self, frame):
        """Return matrix coordinate, reduce if flag True."""
        biotframe = getattr(self, frame)
        if biotframe.reduce:
            return biotframe.biotreduce.index
        return biotframe.index


@dataclass
class BiotSolve(ABC, BiotMatrix):
    """Biot-Savart base-class. Define calculaiton interface."""

    mu_o = 4*np.pi*1e-7

    vector: BiotVector = field(init=False, repr=False)

    def __post_init__(self):
        """Init static and unit datasets."""
        super().__post_init__()
        self.calculate_vectors()
        self.store_matrix(self.vector)

    def calculate_vectors(self):
        """Calculate vector and scalar potential and magnetic field."""
        self.vector = BiotVector(index=self.index)
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

    def store_matrix(self, vector: BiotVector):
        """
        Store interaction matrices.

        Extract plasma (unit) interaction from full matrix.
        Multiply by source and target turns.
        Apply reduction summations.

        """
        for col in vector:
            static = vector[col].reshape(*self.shape)
            unit = static[:, self.source.plasma]
            if self.source.turns:
                static *= self.source('nturn').reshape(*self.shape)
            if self.target.turns:
                static *= (turns := self.target('nturn').reshape(*self.shape))
                unit *= turns[:, self.source.plasma]
            # reduce
            if self.source.reduce and self.source.biotreduce.reduce:
                static = np.add.reduceat(
                    static, self.source.biotreduce.indices, axis=1)
            if self.target.reduce and self.target.biotreduce.reduce:
                static = np.add.reduceat(
                    static, self.target.biotreduce.indices, axis=0)
                unit = np.add.reduceat(
                    unit, self.target.biotreduce.indices, axis=0)
            # link source
            source_link = self.source.biotreduce.link
            if self.source.reduce and len(source_link) > 0:
                for link in source_link:  # sum linked columns
                    static[:, source_link[link]] += static[:, link]
                static = np.delete(static, list(source_link), 1)
            # link target
            target_link = self.target.biotreduce.link
            if self.target.reduce and len(target_link) > 0:
                for ref in target_link:  # sum linked columns
                    static[ref, :] += static[target_link[ref], :]
                    unit[ref, :] += unit[target_link[ref], :]
                static = np.delete(static, target_link.values(), 0)
                unit = np.delete(unit, target_link.values(), 0)

            self.static[col] = (['target', 'source'], static)
            self.unit[col] = (['target', 'source'], unit)
