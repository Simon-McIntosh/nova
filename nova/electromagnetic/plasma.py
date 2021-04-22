"""Manage plasma attributes."""
from dataclasses import dataclass, field

import descartes
import numpy as np
import pygeos

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.frameloc import FrameLoc
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polygon import Polygon, PolyFrame
from nova.utilities.pyplot import plt


# self.biot_instances = ['plasmafilament', 'plasmagrid']

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
        'nturn': 1, 'part': 'plasma', 'name': 'Plasma', 'plasma': True,
        'active': True})

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
class Plasma(PlasmaGrid, FrameLoc):
    """Set plasma separatix, ionize plasma filaments."""

    number: int = field(init=False, default=0)
    bounday: PolyFrame = field(init=False, repr=False, default=None)
    tree: pygeos.STRtree = field(init=False, repr=False, default=None)
    separatrix: PolyFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['ionize'], 'array': ['ionize', 'area', 'nturn']}
        self.subframe.update_columns()

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc['ionize', ['x', 'z', 'section', 'area',
                                   'Ic', 'It', 'nturn']].__str__()

    def insert(self, *required, iloc=None, **additional):
        """Store plasma index and plasma boundary and generate STR tree."""
        index = super().insert(*required, iloc=None, **additional)
        self.number = self.subframe.plasma.sum()
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
        points = pygeos.points(self.loc['plasma', 'x'],
                               self.loc['plasma', 'z'])
        return pygeos.STRtree(points)

    def update(self, loop):
        """
        Update plasma separatrix.

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
        poly = Polygon(loop).poly
        self.separatrix = poly.intersection(self.boundary)
        within = self.tree.query(pygeos.from_shapely(self.separatrix),
                                 predicate='contains')
        ionize = np.full(self.number, False)
        ionize[within] = True
        self.loc['plasma', 'ionize'] = ionize
        self.loc['plasma', 'nturn'] = 0
        self.loc['ionize', 'nturn'] = \
            self.loc['ionize', 'area'] / self.loc['ionize', 'area'].sum()
        # self.update_plasma_turns = True
        # self.update_plasma_current = True

    def plot(self):
        """Plot plasma boundary and separatrix."""
        plt.plot(*self.boundary.exterior.xy, '-', color='gray')
        axes = plt.gca()
        axes.add_patch(descartes.PolygonPatch(
            self.separatrix, facecolor='C4', alpha=0.75, linewidth=0.5,
            zorder=-10))
        plt.axis('equal')
        plt.axis('off')

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.sloc['Plasma', 'Ic'])
