"""Manage plasma attributes."""
from dataclasses import dataclass, field

import descartes
import numpy as np
import numpy.typing as npt
import shapely

from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polyplot import Axes
from nova.geometry import inpoly
from nova.geometry.polygon import Polygon
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

    turn: str = 'hexagon'
    tile: bool = field(init=False, default=True)
    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
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
class Plasma(PlasmaGrid, FrameSetLoc, Axes):
    """Set plasma separatix, ionize plasma filaments."""

    number: int = field(init=False, default=0)
    boundary: Polygon = field(init=False, repr=False, default=None)
    separatrix: Polygon = field(init=False, repr=False, default=None)
    tree: shapely.STRtree = field(init=False, repr=False, default=None)
    plasma_index: npt.ArrayLike = field(init=False, repr=False, default=None)
    plasma_points: npt.ArrayLike = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['ionize'], 'array': ['ionize', 'area', 'nturn']}
        self.subframe.update_columns()

    def __len__(self):
        """Return number of plasma filaments."""
        return self.index.sum()

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc['ionize', ['x', 'z', 'section', 'area',
                                   'Ic', 'It', 'nturn']].__str__()

    def insert(self, *args, required=None, iloc=None, **additional):
        """Store plasma index and plasma boundary and generate STR tree."""
        super().insert(*args, required=None, iloc=None, **additional)
        self.normalize()
        self.generate()

    def normalize(self):
        """Normalize plasma turn number for multiframe plasmas."""
        if self.sloc['plasma'].sum() == 1:
            return
        self.linkframe(self.Loc['plasma', :].index.tolist())
        self.Loc['plasma', 'nturn'] = \
            self.Loc['plasma', 'area'] / np.sum(self.Loc['plasma', 'area'])
        self.loc['plasma', 'nturn'] = \
            self.loc['plasma', 'area'] / np.sum(self.loc['plasma', 'area'])

    def generate(self):
        """Generate plasma attributes, build STR tree."""
        self.number = self.loc['plasma'].sum()
        if self.number > 0:
            self.boundary = Polygon(self.Loc['plasma', 'poly'][0])
            self.tree = self.generate_tree()
            self.plasma_index = self.loc['plasma']
            self.plasma_points = self.loc['plasma', ['x', 'z']].to_numpy()

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
        return shapely.STRtree([polyframe.centroid
                                for polyframe in self.loc['plasma', 'poly']])
        '''
        return pygeos.STRtree(
            pygeos.points(self.loc['plasma', ['x', 'z']].to_numpy()))
        '''

    '''
    def plasma_poly(self, loop):
        """Return pygeos polygon built from loop."""
        if isinstance(loop, pygeos.lib.Geometry):
            return loop
        if isinstance(loop, np.ndarray):
            return pygeos.polygons(loop)
        if isinstance(loop, shapely.geometry.Polygon):
            return pygeos.from_shapely(loop)
        return pygeos.from_shapely(Polygon(loop).poly)
    '''

    def update(self, loop):
        """
        Update plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        loop : array-like (n, 2), Polygon, dict[str, list[float]], list[float]
            Bounding loop.

        """

        '''
        try:
            self.separatrix = pygeos.polygons(loop)
        except TypeError:
            loop = Polygon(loop).points[:, ::2]
            self.separatrix = pygeos.polygons(loop)
        within = self.tree.query(self.separatrix, predicate='intersects')
        '''

        '''
        self.separatrix = Polygon(loop)
        within = self.tree.query_items(self.separatrix.poly)
        within = [index for index in within
                  if self.tree.geometries[index].within(self.separatrix.poly)]
        ionize_filament = np.full(self.number, False)
        ionize_filament[within] = True
        '''

        ionize_filament = inpoly.is_inside_sm_parallel(self.plasma_points, loop)
        self.loc[self.plasma_index, 'ionize'] = ionize_filament
        self.loc[self.plasma_index, 'nturn'] = 0
        ionize = self.loc['ionize']
        self.loc[ionize, 'nturn'] = \
            self.loc[ionize, 'area'] / np.sum(self.loc[ionize, 'area'])

        # self.update_plasma_turns = True
        # self.update_plasma_current = True

    def plot(self, axes=None, boundary=True):
        """Plot plasma boundary and separatrix."""
        self.axes = axes
        if self.separatrix is not None:
            self.axes.add_patch(descartes.PolygonPatch(
                self.separatrix.poly.__geo_interface__,
                facecolor='C4', alpha=0.75, linewidth=0.5, zorder=-10))
        if boundary:
            self.boundary.plot_boundary(self.axes, color='gray')
        plt.axis('equal')
        plt.axis('off')

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.sloc['Plasma', 'Ic'])
