"""Biot-Savart calculation base class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import xarray

from nova.electromagnetic.biotset import BiotSet
from nova.electromagnetic.dataframe import DataFrame


@dataclass
class BiotMatrix:

    #source: BiotFrame
    static: npt.ArrayLike = field(init=False, repr=False)
    plasma: npt.ArrayLike = field(init=False, repr=False)

    #def __post_init__(self):
    #    filament = BiotFilament(self.source, self.target)


class BiotVector(DataFrame):
    """Store Biot vectors."""

    def __init__(self, data=None, index=None, columns=None, attrs=None, **metadata):
        metadata = {'required': ['Psi', 'Ax', 'Ay', 'Az', 'Bx', 'By', 'Bz'],
                    'Default': dict().fromkeys(['Psi', 'Ax', 'Ay', 'Az',
                                                'Bx', 'By', 'Bz'], 0.0)} | metadata
        super().__init__(data, index, columns, attrs, **metadata)
        self.update_columns()


@dataclass
class BiotSavart(ABC, BiotSet):
    """Biot calculation base-class, Define calculaiton interface."""

    mu_o = 4*np.pi*1e-7

    vector: BiotVector = field(init=False, repr=False)

    def __post_init__(self):
        """Init BoitSet and calculate interaction."""
        super().__post_init__()
        self.vector = BiotVector(index=self.index)
        self.calculate()

    '''
    def save_matrix(self, vector: npt.ArrayLike):
        """
        Save interaction matrices.

        Split plasma interaction from full matrix.
        Multiply static source and target turns to full matrix only.

        Parameters
        ----------
        M : array-like, shape(target*source,)
            Unit turn source-target interaction matrix.

        Returns
        -------
        None.

        """
        # extract plasma interaction
        _M_ = M.reshape(self.nT, self.nS)[:, self.source.plasma]
        if self.source_turns:
            M *= self.source._nturn_
        if self.target_turns:
            M *= self.target._nturn_
        _M = M.reshape(self.nT, self.nS)  # source-target reshape (matrix)
        # reduce
        if self.reduce_source and len(self.source._reduction_index) < self.nS:
            _M = np.add.reduceat(_M, self.source._reduction_index, axis=1)
        if self.reduce_target and len(self.target._reduction_index) < self.nT:
            _M = np.add.reduceat(_M, self.target._reduction_index, axis=0)
        return _M, _M_  # turn-turn interaction, unit plasma interaction
    '''

    def calculate(self):
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
