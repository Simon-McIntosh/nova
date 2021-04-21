
from dataclasses import dataclass, field

import descartes
import numpy as np
import numpy.typing as npt
import pandas
import pygeos

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polygon import Polygon, PolyFrame
from nova.utilities.pyplot import plt


# self._ionize_index = self._plasma[self._mpc_referance]
# self.biot_instances = ['plasmafilament', 'plasmagrid']
# _boundary: shapely.geometry.Polygon = field(init=False, repr=False)

# name=name, plasma=True, active=True, part='plasma'

'''
self.plasmagrid.generate_grid(**kwargs)
grid_factor = self.dPlasma/self.plasmagrid.dx
# self._add_vertical_stabilization_coils()
self.plasmagrid.cluster_factor = 1.5*grid_factor
self.plasmafilament.add_plasma()
'''


@dataclass
class PlasmaGrid(PoloidalGrid):
    """Grid plasma region."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float
    turn: str = 'hexagon'
    tile: bool = field(init=False, default=True)
    default: dict = field(init=False, default_factory=lambda: {
        'nturn': 1, 'part': 'plasma', 'name': 'Plasma', 'plasma': True})

    def set_conditional_attributes(self):
        """Set conditional attrs for plasma grid."""
        self.ifthen('delta', -1, 'turn', 'rectangle')
        self.ifthen('turn', 'rectangle', 'tile', False)

    def insert(self, *required, iloc=None, **additional):
        """
        Extend PoloidalGrid.insert.

        Add plasma to coilset and generate bounding plasma grid.

        Plasma inserted into frame with subframe meshed accoriding
        to delta and trimmed to the plasma's boundary curve.

        """
        return super().insert(*required, iloc=iloc, **additional)


@dataclass
class Plasma(PlasmaGrid):
    """Set plasma separatix, ionize plasma filaments."""

    plasma: PolyFrame = field(init=False, repr=False, default=None)
    bounday: PolyFrame = field(init=False, repr=False, default=None)
    nfilament: int = field(init=False)
    tree: pygeos.STRtree = field(init=False, repr=False, default=None)

    def insert(self, *required, iloc=None, **additional):
        """Store plasma index and plasma boundary and generate STR tree."""
        index = super().insert(*required, iloc=None, **additional)
        self.nfilament = self.subframe.plasma.sum()
        self.boundary = self.frame.at[index[0], 'poly']
        self.tree = self.generate_tree()

    def generate_tree(self):
        """
        Return STR plasma tree, read-only.

        Construct STR tree from plasma filaments to enable fast search in
        free-boundary calculations.
        Pygeos creates tree on first evaluation.

        Parameters
        ----------
        tree : pygeos.STRtree
            STR tree.

        Raises
        ------
        TypeError
            Tree not pygeos.STRtree.

        Returns
        -------
        plasma_tree
            pygeos STRtree.

        """
        coords = self.subframe.loc[self.subframe.plasma, ['x', 'z']].values
        points = pygeos.points(coords)
        return pygeos.STRtree(points)

    @property
    def index(self):
        """Return plasma boolean index, read-only."""
        return self.subframe.plasma

    @property
    def ionize(self):
        """Return plasma ionization index, read-only."""
        return self.subframe.ionize

    @property
    def separatrix(self):
        """
        Manage plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        loop : dict[str, list[float]], array-like or Polygon
            Bounding loop.

        Returns
        -------
        separatrix : Polygon
            Plasma separatrix.

        """
        return self.plasma

    @separatrix.setter
    def separatrix(self, loop):
        poly = Polygon(loop).poly
        self.plasma = poly.intersection(self.boundary)
        index = self.tree.query(pygeos.from_shapely(poly),
                                predicate='contains')

        ionize = np.full(self.nfilament, False)
        ionize[index] = True

        self.subframe.loc[self.plasma, 'ionize'] = ionize

        self.subframe.loc[self.subframe.plasma, 'nturn'] = 0
        self.subframe.loc[self.subframe.ionize, 'nturn'] = 1

    def update_nturn(self, current=None):
        """Update plasma filament turns."""
        #if current is None:
        #    current = self.subframe.

        '''
        if isinstance(loop, pandas.DataFrame):
            loop = loop.values
        if isinstance(loop, shapely.geometry.Polygon):
            polygon = loop
        elif len(loop) == 0:
            return
        else:
            polygon = shapely.geometry.Polygon(loop)

        # intersection of separatrix and plasma_boundary
        #separatrix = polygon.intersection(self.plasma_boundary)
        separatrix = polygon
        self._separatrix = separatrix
        # update coil - polygon and polygon derived attributes
        self.coil.loc['Plasma', 'polygon'] = separatrix
        self.coil.update_polygon(index='Plasma')
        self.coil.Np = 1
        # update subcoil
        self.subcoil.ionize = self.plasma_tree.query(
            pygeos.io.from_shapely(separatrix), predicate='contains')
        self.update_plasma_turns = True
        self.update_plasma_current = True
        '''

    def plot(self):
        """Plot plasma boundary and separatrix."""
        plt.plot(*self.boundary.exterior.xy, '-', color='gray')
        axes = plt.gca()
        axes.add_patch(descartes.PolygonPatch(
            self.plasma, facecolor='C4', alpha=0.75, linewidth=0.5))

    '''
    @property
    def ionize(self):
        """
        Manage plasma ionization_index.

        Set index to True for all intra-spearatrix filaments.

        Parameters
        ----------
        index : array-like, shape(nP,)
            ionization index for plasma filament bundle, shape(nP,).

        Returns
        -------
        _ionize_index : array-like, shape(nC,)
            Ionization index.

        """
        return self._ionize_index

    @ionize.setter
    def ionize(self, index):
        active = np.full(self.nP, False)
        active[index] = True
        self._ionize_index[self.plasma] = active
        self.Np = 1  # initalize turn number
    '''

    @property
    def plasma_index(self):
        """
        Return plasma index on subcoil frame, read-only.

        Returns
        -------
        plasma_index : array-like
            subcoil plasma index

        """
        return self.subcoil.plasma

    @property
    def ionize_index(self):
        """
        Return plasma ionization index, read-only.

        Returns
        -------
        ionize : array-like, shape(n,)
            Subcoil ionization index (nturn>0).

        """
        return self.subcoil.ionize[self.subcoil.plasma]

    @property
    def Np(self):
        r"""
        Plasma filament turn number.

        Parameters
        ----------
        value : float or array-like
            Set turn number of plasma filaments

            Ensure :math:`\sum |Np| = 1`.

        Returns
        -------
        Np : np.array, shape(nP,)
            Plasma filament turn number.

        """
        return self._nturn[self.plasma]

    @Np.setter
    def Np(self, value):
        self._nturn[self.plasma & ~self._ionize_index] = 0
        self._nturn[self.plasma & self._ionize_index] = value
        # normalize plasma turn number
        nturn_sum = np.sum(self._nturn[self.plasma])
        if nturn_sum > 0:
            self._nturn[self.plasma] /= nturn_sum
        self._update_dataframe['nturn'] = True

    @property
    def nP(self):
        """Return number of plasma filaments."""
        return np.sum(self.plasma)

    @property
    def nPlasma(self):
        """Return number of active plasma fillaments."""
        return len(self.Np[self.Np > 0])

    @property
    def Ip(self):
        """
        Return plasma line current [A].

        Returns
        -------
        It : float
            sum(It) (float): plasma line current [A]

        """
        return self._Ic[self._plasma]

    @Ip.setter
    def Ip(self, value):
        self._Ic[self._plasma] = value
        self._update_dataframe['Ic'] = True

    @property
    def Ip_sum(self):
        """Net plasma current."""
        return self.Ip.sum()

    @property
    def Ip_sign(self):
        """Plasma polarity."""
        return np.sign(self.Ip_sum)

