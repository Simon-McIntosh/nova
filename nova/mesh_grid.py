from amigo.geom import grid
from amigo.pyplot import plt
from matplotlib.collections import LineCollection
import numpy as np


class MeshGrid:
    '''
    construct 2d poloidal grid
    '''

    def __init__(self, n, limit, xscale='linear', zscale='linear'):
        '''
        Attributes:
            n (int or [int, int]): mesh dimension n or [nx, nz]
            limit (list): ['xmin', 'xmax', 'zmin', 'zmax']
        '''
        self._n = n
        self._limit = limit
        self.xscale = xscale
        self.zscale = zscale
        self.update()
        
    def update(self):
        '''
        update grid
        '''
        self.x2d, self.z2d, self.x, self.z, self._nx, self._nz = \
            grid(self._n, self._limit, eqdsk=False)
        for var in ['x', 'z']:
            self.scale(var)
        
    def scale(self, xz):
        if getattr(self, f'{xz}scale') == 'log':
            x = getattr(self, xz)
            nx = getattr(self, f'n{xz}')
            nz = int(self.n / nx)
            setattr(self, xz, 10**np.linspace(
                    np.log10(x[0]), np.log10(x[-1]), nx))
            if xz == 'x':
                x2d = np.dot(x.reshape(-1, 1), np.ones((1, nz)))
            else:
                x2d = np.dot(np.ones((nz, 1)), x.reshape(1, -1))
            setattr(self, f'{xz}2d', x2d)
            
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        if n != self._n:
            self._n = n
            self.update()

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, nx):
        if nx != self._nx:
            self._nx = nx
            self.update()

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, nz):
        if nz != self._nz:
            self._nz = nz
            self.update()

    @property
    def limit(self):
        return self._limit

    @limit.setter
    def limit(self, limit):
        if np.array(limit != self._limit).any():
            self._limit = limit
            self.update()

    def update_limit(self, index, value, plot=False):
        limit = np.copy(self._limit)
        limit[index] = value
        self.limit = limit
        if plot:
            self.plot()
            
    @property
    def xz(self):
        return self.x, self.z

    def eqdsk(self):
        '''
        return grid as eqdsk labeled dict
        '''
        return {'x2d': self.x2d, 'z2d': self.z2d,
                'x': self.x, 'z': self.z, 'nx': self.nx, 'nz': self.nz}

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        linewidth = kwargs.pop('linewidth', 0.4)
        zorder = kwargs.pop('zorder', -100)
        for coordinate, step in zip(['x', 'z'], [1, -1]):
            n_ = getattr(self, f'n{coordinate}')
            lines = np.zeros((n_, 2, 2))
            for i in range(2):
                index = tuple([slice(None), -i][::step])
                lines[:, i, 0] = self.x2d[index]
                lines[:, i, 1] = self.z2d[index]
            segments = LineCollection(lines, linewidth=linewidth,
                                      zorder=zorder)
            ax.add_collection(segments)
        if self.xscale == 'linear' and self.zscale == 'linear':
            ax.axis('equal')
        else:
            ax.set_xscale(self.xscale)
            ax.set_yscale(self.zscale)
        if ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1):
            ax.set_xlim(self.limit[:2])
            ax.set_ylim(self.limit[2:])

        
if __name__ == '__main__':

    mg = MeshGrid(1e3, [5, 7.5, 8, 12], xscale='linear')
    mg.plot()
