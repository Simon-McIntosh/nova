"""Methods for saving and manipulating AC loss data."""

from dataclasses import dataclass, field
from typing import Union

import xarray
import numpy as np

from nova.biot.biotframe import BiotFrame
from nova.electromagnetic.coilset import CoilSet


class DataProperty:
    """Properties mixin for DataArray and DataSet."""

    @property
    def time(self):
        """Return time array."""
        return self.data.coords['time'].values


@dataclass
class DataArray(DataProperty):
    """
    Manage CoilSet data with an DataArray.

    Parameters
    ----------
    data : Union[tuple[np.array, BiotFrame, str], xarray.DataArray]

        - tuple[time, target, name]
            time : array-like, shape(nt,)
                Array of time values.
            target : BiotFrame
                Target biotframe from forcefield instance.
            name : str
                Data attribute label.

        - xarray.DataArray
            Exsisting data array.
    """

    data: Union[tuple[np.array, BiotFrame, str], xarray.DataArray] = \
        field(repr=False)
    name: str = field(init=False)
    attrs: list = field(init=False, default_factory=list)

    def __post_init__(self):
        """Assemble data strucutre."""
        if not isinstance(self.data, xarray.DataArray):
            self.data = self._initialize(*self.data)
        self.name = self.data.name
        self.attrs = [attr for attr in self.data.attrs]

    def _initialize(self, time: np.array, target: BiotFrame, name: str):
        """Return empty xarray.DataArray instance with simulation metadata.

        time : array-like, shape(nt,)
            Array of time values.
        target : BiotFrame
            Target biotframe from forcefield instance.
        name : str
            Data attribute label.

        Raises
        ------
        IndexError
            Requested data attribute not found in target frame.

        Returns
        -------
        None.

        """
        # initialize data array
        nt = len(time)  # time instance number
        if target.reduce:
            index = target.index[target._reduction_index]
            nT = index.shape[0]
            indices = range(nT)
        else:
            index = target.index
            nT = target.nT
            indices = target._reduction_index  # slices to reduce.
        dims = ('time', 'turn')
        coords = {'time': time, 'turn': index}
        data = np.zeros((nt, nT))
        # extract attributes
        attrs = {'nT': nT, 'indices': np.append(indices, nT)}
        for attr in ['dx', 'dz', 'nturn', 'coil', 'x', 'z']:
            value = target[attr].values
            if attr in ['dx', 'dz', 'nturn', 'coil']:
                value = value[target._reduction_index]
            elif target.reduce:
                value = np.add.reduceat(value, target._reduction_index)
                value /= np.add.reduceat(np.ones(target.nT),
                                         target._reduction_index)
            if value.dtype == 'object':
                value = value.astype('U60')  # convert to character array
            attrs[attr] = value
        return xarray.DataArray(data, dims=dims, coords=coords,
                                name=name, attrs=attrs)


@dataclass
class DataSet(DataProperty):
    """Manage a collection of coilset data with xarray.Dataset."""

    data: Union[tuple[np.array, BiotFrame, list[str]], xarray.Dataset] = \
        field(repr=False)
    names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Load data set."""
        if not isinstance(self.data, xarray.Dataset):
            self.data = self._initialize(*self.data)
        self.names = [name for name in self.data]

    def _initialize(self, time: np.array, target: BiotFrame, names: list[str]):
        """Return initialized DataSet."""
        data = xarray.Dataset()
        for name in names:
            data[name] = DataArray((time, target, name)).data
        return data

    def __getitem__(self, key: str):
        """Return labelled dataset."""
        return self.data[key]

    def __setitem__(self, key: str, array: xarray.DataArray):
        """Assign data to key."""
        self.data[key] = array

    def __delitem__(self, key: str):
        """Delete item."""
        self.data.drop_vars(key)

    def __iter__(self):
        """Return keys."""
        return iter(self.data)

    def __len__(self):
        """Return data array number."""
        return len(self.data)


if __name__ == '__main__':

    cs = CoilSet()
    cs.add_coil(5, 1, 0.5, 0.5, dCoil=0.2)
    cs.add_coil(6, 1, 0.5, 0.5, dCoil=0.1)
    cs.add_coil(5, -1, 0.5, 0.5, dCoil=0.05)

    cs.biot_instances = ['forcefield']
    cs.forcefield.solve_biot()

    da = DataArray((np.linspace(0, 1, 20), cs.forcefield.target, 'B'))

    ds = DataSet((np.linspace(0, 1, 20), cs.forcefield.target, ['Bx', 'Bz']))

    '''

    cda = CoilDataArray(np.linspace(0, 1, 20), cs.forcefield.target, 'B')
    for i, t in enumerate(cda.time()):
        cs.Ic = 1e3*t
        cda.data[i] = cs.forcefield.B

    cda.save('Bdiff', 'diffrent forcefield values', machine='ITER', coilset=None,
             dCoil=cs.dCoil, dPlasma=cs.dPlasma, replace=False)
    '''
