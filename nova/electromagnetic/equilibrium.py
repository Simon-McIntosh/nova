"""Forward free-boundary equilibrium solver."""
from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt

from nova.database.netcdf import netCDF
from nova.electromagnetic.firstwall import FirstWall
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.plasmagrid import PlasmaGrid
from nova.electromagnetic.polyplot import Axes
from nova.geometry.polygon import Polygon
from nova.utilities.pyplot import plt


@dataclass
class Equilibrium(Axes, netCDF, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'equilibrium'
    grid: PlasmaGrid = field(repr=False, default=None)
    wall: FirstWall = field(repr=False, default=None)
    separatrix: npt.ArrayLike = field(init=False, default=None, repr=False)

    @property
    def separatrix(self):
        """Return input plasma separatrix trimmed to first wall."""
        if self.loop is None:
            self.update_separatrix(self.firstwall)
        return Polygon(self.loop).poly.intersection(self.firstwall.poly)

    def update_separatrix(self, loop):
        """
        Update plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        loop : array-like (n, 2), Polygon, dict[str, list[float]], list[float]
            Bounding loop.

        """
        if self.sloc['plasma'].sum() == 0:
            return
        try:
            inloop = self.pointloop.update(loop)
        except numba.TypingError:
            loop = Polygon(loop).boundary
            inloop = self.pointloop.update(loop)
        self.loop = loop

        plasma = self.aloc['plasma']
        ionize = self.aloc['ionize']
        nturn = self.aloc['nturn']
        area = self.aloc['area']
        ionize[plasma] = inloop
        self._update_nturn(plasma, ionize, nturn, area)
        self.update_version()

    @staticmethod
    @numba.njit
    def _update_nturn(plasma, ionize, nturn, area):
        nturn[plasma] = 0
        ionize_area = area[ionize]
        nturn[ionize] = ionize_area / np.sum(ionize_area)

    def update(self, psi_boundary):
        """Update plasma seperatrix."""
        #s_psi = self.boundary.psi.min()
        s_psi = psi_boundary.min()

        #self.grid.plot(levels=[s_psi], colors='r')
        #self.grid.plot(levels=21)

        plasma = self.aloc['plasma']
        ionize = self.aloc['ionize']
        nturn = self.aloc['nturn']
        area = self.aloc['area']
        ionize[plasma] = self.grid.psi < s_psi
        self._update_nturn(plasma, ionize, nturn, area)

        self.update_version()

        #self.subframe.polyplot('plasma')
        #self.boundary.plot()

    def plot(self)
        if (poly := self.separatrix) is not None and not poly.is_empty:
            self.axes.add_patch(descartes.PolygonPatch(
                poly.__geo_interface__,
                facecolor='C4', alpha=0.75, linewidth=0.5, zorder=-10))

'''
    #loop: npt.ArrayLike = field(init=False, default=None, repr=False)


    def store(self, filename: str, path=None):
        """Extend netCDF.store, store data as netCDF in hdf5 file."""
        self.data = xarray.Dataset()
        self.data['loop_coorinates'] = ['x', 'z']
        self.data['loop'] = ('loop_index', 'loop_coorinates'), self.loop
        super().store(filename, path)

    def load(self, filename: str, path=None):
        """Extend netCDF.load, load data from hdf5."""
        super().load(filename, path)
        self.loop = self.data['loop'].data
        return self
'''
