"""Grid methods."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import shapely.geometry
import numpy as np

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.polygen import polygen
from nova.utilities.pyplot import plt


# pylint:disable=unsubscriptable-object


'''

def grid(n, limit, eqdsk=False):
    if len(np.shape(limit)) > 1:
        limit = np.array(limit).flatten()
    xo, zo = limit[:2], limit[2:]
    try:  # n ([nx, nz])
        nx, nz = n
    except TypeError:  # n (int)
        dxo, dzo = (xo[-1] - xo[0]), (zo[-1] - zo[0])
        ar = dxo / dzo
        nz = np.max([int(np.sqrt(n / ar)), 3])
        nx = np.max([int(n / nz), 3])
    x = np.linspace(xo[0], xo[1], nx)
    z = np.linspace(zo[0], zo[1], nz)
    x2d, z2d = np.meshgrid(x, z, indexing='ij')
    if eqdsk:
        return {'x2d': x2d, 'z2d': z2d, 'x': x, 'z': z, 'nx': nx, 'nz': nz}
    else:
        return x2d, z2d, x, z, nx, nz

'''


@dataclass
class PolyCoil:
    """Manage coil's bounding polygon."""

    limit: InitVar[
        Union[dict[str, list[float]], list[float]], shapely.geometry.Polygon]
    delta: InitVar[float]
    poly: shapely.geometry.Polygon = field(init=False, repr=False)

    def __post_init__(self, limit, delta):
        """Init bounding polygon."""
        self.poly = self.generate(limit)

    def generate(self, limit):
        """
        Generate bounding polygon.

        Parameters
        ----------
        polygon :
            - dict[str, list[float]], polyname: *args
            - list[float], shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        Raises
        ------
        IndexError
            Malformed bounding box, shape is not (4,).
            Malformed bounding loop, shape is not (n, 2).

        Returns
        -------
        polygon : Polygon
            Limit boundary.

        """
        if isinstance(limit, PolyCoil):
            return limit
        if isinstance(limit, shapely.geometry.Polygon):
            return self.orient(limit)
        if isinstance(limit, dict):
            polys = [polygen(section)(*limit[section]) for section in limit]
            poly = shapely.ops.unary_union(polys)
            try:
                return self.orient(poly)
            except AttributeError as nonintersecting:
                raise AttributeError('non-overlapping polygons specified in '
                                     f'{limit}') from nonintersecting
        limit = np.array(limit)  # to numpy array
        if limit.ndim == 1:   # limit bounding box
            if len(limit) == 4:  # [xmin, xmax, zmin, zmax]
                xlim, zlim = limit[:2], limit[2:]
                x_center, z_center = np.mean(xlim), np.mean(zlim)
                width, height = np.diff(xlim)[0], np.diff(zlim)[0]
                poly = polygen('rectangle')(x_center, z_center, width, height)
                return self.orient(poly)
            raise IndexError('malformed bounding box\n'
                             f'limit: {limit}\n'
                             'require [xmin, xmax, zmin, zmax]')
        if limit.shape[1] != 2:
            limit = limit.T
        if limit.ndim == 2 and limit.shape[1] == 2:  # loop
            poly = shapely.geometry.Polygon(limit)
            return self.orient(poly)
        raise IndexError('malformed bounding loop\n'
                         f'shape(limit): {limit.shape}\n'
                         'require (n,2)')

    @staticmethod
    def orient(poly):
        """Return coerced polygon boundary as a positively oriented curve."""
        return shapely.geometry.polygon.orient(poly)

    def plot(self):
        """Plot boundary polygon."""
        plt.plot(*self.poly.exterior.xy)

    @property
    def xlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.poly.bounds[::2]

    @property
    def width(self) -> float:
        """Return polygon bounding box width."""
        return np.diff(self.xlim)[0]

    @property
    def zlim(self) -> tuple[float]:
        """Return polygon bounding box x limit (xmin, xmax)."""
        return self.poly.bounds[1::2]

    @property
    def height(self) -> float:
        """Return polygon bounding box height, [xmin, xmax]."""
        return np.diff(self.zlim)[0]


@dataclass
class CoilGrid:
    """Construct 2d grid."""

    limit: InitVar[
        Union[dict[str, list[float]], list[float]], shapely.geometry.Polygon]
    delta: InitVar[float] = field(default=0)
    section: str = 'hex'
    scale: float = 1.
    offset: bool = True
    snap: bool = False
    frame: Frame = field(
        init=False, repr=False,
        default=Frame(required=['x', 'z'], available=['section']))

    def __post_init__(self, limit, delta):
        """Generate grid."""
        self.polycoil = PolyCoil(limit, delta)

    def __len__(self):
        """Return grid length."""
        return len(self.frame)

    def set_delta(self, delta):
        """
        Update grid delta.

            - delta <= 0: point number
            - delta > 0: grid dimension
        """

    '''
        if delta is None or delta == 0:
            Nf = 1
            delta = np.max([dx, dz])
        elif delta == -1:  # mesh per-turn
            Nf = frame['Nt']
            if 'section' not in mesh:
                mesh['section'] = 'circle'
            if frame['section'] == 'circle':
                delta = (np.pi * ((dx + dz) / 4)**2 / Nf)**0.5
            else:
                delta = (dx * dz / Nf)**0.5
        elif delta < -1:
            Nf = -delta  # set filament number
            if frame['section'] == 'circle':
                delta = (np.pi * (dx / 2)**2 / Nf)**0.5
            else:
                delta = (dx * dz / Nf)**0.5
        elif delta > 0:
            nx = np.max([1, int(np.round(dx / delta))])
            nz = np.max([1, int(np.round(dz / delta))])
            Nf = nx * nz
    '''

    def clear(self):
        """Clear grid."""
        self.frame = self.frame.iloc[0:0]

    def linspace(self, limit, delta):
        """Return ."""
        start, stop = limit
        ndiv = int(np.ceil((stop-start) / delta))
        stop = start + ndiv * delta
        offset = ndiv*delta - np.diff(limit)[0]
        start -= offset/2
        stop -= offset/2
        if self.offset:
            return np.linspace(start, stop, ndiv+1)
        return np.linspace(start + delta/2, stop - delta/2, ndiv-1)

        #if self.snap
        #return np.linspace(xo[0], xo[1], nx)

    def generate(self):
        """Generate grid."""
        delta = 0.03
        delta_x = 2*delta
        delta_z = delta/np.sqrt(3)

        x_space = self.linspace(self.polycoil.xlim, delta_x)
        z_space = self.linspace(self.polycoil.zlim, delta_z)

        x_grid, z_grid = np.meshgrid(x_space, z_space, indexing='ij')
        if self.offset:
            x_grid[:, 1::2] += delta_x/2
        self.frame.insert(x_grid.flatten(), z_grid.flatten(),
                          dl=delta_x, dt=delta_z,
                          section=self.section, scale=self.scale)

        plt.plot(self.frame.x, self.frame.z, 'C3.')
        '''
        dxo, dzo = (xo[-1] - xo[0]), (zo[-1] - zo[0])
        ar = dxo / dzo
        nz = np.max([int(np.sqrt(n / ar)), 3])
        nx = np.max([int(n / nz), 3])
        x = np.linspace(xo[0], xo[1], nx)
        z = np.linspace(zo[0], zo[1], nz)
        x2d, z2d = np.meshgrid(x, z, indexing='ij')
        '''


if __name__ == '__main__':

    #grid = Grid([5, 17.5, 8, 12], 0.5)

    coilgrid = CoilGrid({'r': [6, 3, 0.4, 0.2]}, offset=True,
                        scale=1)

    plt.figure()
    plt.axis('off')
    plt.axis('equal')

    coilgrid.generate()
    coilgrid.frame.polyplot()
    coilgrid.polycoil.plot()





    """
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
    def n2d(self):
        return (self.nx, self.nz)

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
        self._plot(self.x2d, self.z2d, self.limit[:2], self.limit[2:],
                   xscale=self.xscale, zscale=self.zscale, ax=ax, **kwargs)

    @staticmethod
    def _plot(x2d, z2d, xlim, zlim,
              xscale='linear', zscale='linear', ax=None, **kwargs):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        linewidth = kwargs.pop('linewidth', 0.4)
        zorder = kwargs.pop('zorder', -100)
        color = kwargs.pop('color', 'gray')
        alpha = kwargs.pop('alpha', 0.5)
        for n_, step in zip(x2d.shape, [1, -1]):
            lines = np.zeros((n_, 2, 2))
            for i in range(2):
                index = tuple([slice(None), -i][::step])
                lines[:, i, 0] = x2d[index]
                lines[:, i, 1] = z2d[index]
            segments = LineCollection(lines, linewidth=linewidth,
                                      zorder=zorder, color=color, alpha=alpha)
            ax.add_collection(segments)
        if xscale == 'linear' and zscale == 'linear':
            ax.axis('equal')
        else:
            ax.set_xscale(xscale)
            ax.set_yscale(zscale)
        if ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1):
            ax.set_xlim(xlim)
            ax.set_ylim(zlim)




if __name__ == '__main__':

    mg = MeshGrid(1e3, [5, 7.5, 8, 12], xscale='linear')
    mg.plot()

    """
