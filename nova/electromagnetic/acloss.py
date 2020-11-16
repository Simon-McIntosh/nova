
import os.path
import intake
from xarray import DataArray
import numpy as np

from nova.definitions import root_dir


class CoilArray(DataArray):
    """
    CoilArray extends xarray.DataArray to manage CoilSet data.

    Examples
    --------
    >>> nT = 5  # target number
    >>> t = np.linspace(0, 100, 50)  # time array
    >>> ca = CoilArray(t, )

    """

    __slots__ = 'directory'

    def __init__(self, time, turns, variable='B', **kwargs):
        nt = len(time)  # time instance number
        nT = 2#len(turns)  # target number
        print(turns, nT)
        data = np.zeros((nt, nT))  # initialize data
        dims = ('t', variable)
        coords = {'t': time, 'B': [1,2]}
        # , coords=coords, dims=dims, **kwargs
        print('init')
        DataArray.__init__(self, data, dims=dims, coords=coords)
        self.initialize_directory()

    def initialize_directory(self):
        self.directory = os.path.join(root_dir, 'data/ACloss')

    #def time(self):
    #    for t in self.coords['t']:
    #        yield t

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.to_netcdf(os.path.join(self.directory, 'tmp.nc'))
        # save data

class ACloss:

    def __init__(self):
        self.directory = os.path.join(root_dir, 'data/ACloss')


if __name__ == '__main__':

    ca = CoilArray(np.linspace(0, 1, 10), ['f1', 'f2', 'f3'])

    #with CoilArray(np.linspace(0, 1, 10), ['f1', 'f2', 'f3']) as ca:
    #    for t in ca.time():
    #        print(t)


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
