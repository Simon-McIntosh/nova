
import os.path
import intake
from xarray import DataArray
import numpy as np

from nova.definitions import root_dir
from nova.electromagnetic.coilset import CoilSet


class CoilArray:
    """Manage CoilSet data with an xarray.DataArray."""

    _data_attributes = ['dx', 'dz', 'Nt']

    def __init__(self, time, target, name='B', **kwargs):
        self.initialize_directory()
        self.initialize_data(time, target, name, **kwargs)

    def initialize_directory(self):
        self.directory = os.path.join(root_dir, 'data/ACloss')

    def initialize_data(self, time, target, name='B', **kwargs):
        # initialize data array
        nt = len(time)  # time instance number
        nT = target.nT  # turn number
        index = target.index
        dims = ('t', 'turn')
        coords = {'t': time, 'turn': index}
        data = np.zeros((nt, nT))
        # extract attributes
        indices = target._reduction_index  # Paired indices, slices to reduce.
        attrs = {'nC': target._nC, 'indices': indices}
        for attribute in self._data_attributes:
            if attribute in target._dataframe_attributes:
                value = getattr(target, attribute)[indices]
            elif attribute in target:
                value = target[attribute].values[indices]
            else:
                raise IndexError(f'attribute {attribute} not present in '
                                 f'target: {target.columns}')
            attrs[attribute] = value
        attrs.update({'rms': target.rms, 'x': target.x, 'z': target.z})
        self.data = DataArray(data, dims=dims, coords=coords,
                              name=name, attrs=attrs, **kwargs)

    def time(self):
        for t in self.data.coords['t'].values:
            yield t

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        """Save data to netCDF file."""
        self.data.to_netcdf(os.path.join(self.directory, 'tmp.nc'))


class ACloss:

    def __init__(self):
        self.directory = os.path.join(root_dir, 'data/ACloss')


if __name__ == '__main__':

    cs = CoilSet()
    cs.add_coil(5, 1, 0.5, 0.5, dCoil=0.2)
    cs.add_coil(6, 1, 0.5, 0.5, dCoil=0.2)
    cs.add_coil(5, -1, 0.5, 0.5, dCoil=0.2)

    cs.biot_instances = ['forcefield']
    cs.forcefield.solve_biot()

    #ca = CoilArray(np.linspace(0, 1, 10), cs.forcefield.target)

    with CoilArray(np.linspace(0, 1, 10), cs.forcefield.target) as ca:
        for t in ca.time():
            print(t)


    '''
    time = np.linspace(0, 1, 10)
    index = ['f1', 'f2', 'f3']
    var = 'B'

    nt = len(time)
    nT = len(index)
    data = np.zeros((nt, nT))  # initialize data
    dims = ('t', var)
    coords = {'t': time, var: index}
    da = DataArray(data, coords=coords, dims=dims)
    '''
