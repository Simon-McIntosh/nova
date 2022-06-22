#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:30:46 2022

@author: mcintos
"""


    '''
    def calculate_vector_potential(self):
        """Calculate target vector potential (r, phi, z), Wb/Amp-turn-turn."""
        self.vector['Aphi'] = self.Aphi.compute()

    def calculate_scalar_potential(self):
        """Calculate scalar potential."""
        self.vector['Psi'] = self.Psi.compute()

    def calculate_magnetic_field(self):
        """Calculate magnetic field (r, phi, z), T/Amp-turn-turn."""
        self.vector['Br'] = self.Br.compute()
        self.vector['Bz'] = self.Bz.compute()
    '''



@dataclass
class DaskBiotVector:
    """Store Biot vectors - implement subset of xarray dataset interface."""

    source: list[str]
    target: list[str]
    chunks: Union[int, str, Mapping] = None
    data: xarray.Dataset = field(init=False)

    def __post_init__(self):
        """Create empty dataset."""
        self.data = xarray.Dataset(coords=dict(source=self.source,
                                               target=self.target))

    def __setitem__(self, attr, value):
        """Inset index aligned item to dataset."""
        value = xarray.DataArray(value, dims=['target', 'source'])
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
class BiotBase(ABC, BiotMatrix):
    """Biot-Savart base-class. Define calculaiton interface."""

    #vector: DaskBiotVector = field(init=False, repr=False)


    def __post_init__(self):
        """Build full and unit-turn datasets."""
        super().__post_init__()
        #self.vector = DaskBiotVector(
        #    source=self.get_index('source', reduce=False),
        #    target=self.get_index('target', reduce=False))
        #self.calculate_vectors()


    '''
    def calculate_vectors(self):
        """Calculate vector and scalar potential and magnetic field."""
        self.calculate_coefficients()
        self.calculate_vector_potential()
        self.calculate_scalar_potential()
        self.calculate_magnetic_field()

    @abstractmethod
    def calculate_coefficients(self) -> dict[str, npt.ArrayLike]:
        """Return interaction coefficients."""

    @abstractmethod
    def calculate_vector_potential(self) -> dict[str, npt.ArrayLike]:
        """
        Calculate target vector potential, Wb/Amp-turn-turn..

        Define in cylindrical (r, phi, z) or cartesian (x, y, z) coordinates.

        """

    def calculate_scalar_potential(self) -> npt.ArrayLike:
        """Calculate scalar potential, axisymmetric-only."""

    @abstractmethod
    def calculate_magnetic_field(self):
        """
        Calculate magnetic field, T/Amp-turn-turn.

        Define in cylindrical (r, phi, z) or cartesian (x, y, z) coordinates.

        """

    @property
    def shape(self):
        """Return source-target shape."""
        return (len(self.target), len(self.source))
    '''
