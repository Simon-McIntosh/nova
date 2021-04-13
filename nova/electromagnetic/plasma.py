
from dataclasses import dataclass, field

import shapely.geometry
import numpy as np
import pygeos
import pandas

from nova.electromagnetic.frame import Frame
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polygrid import Polygon

# self._ionize_index = self._plasma[self._mpc_referance]


@dataclass
class Plasma(PoloidalGrid):
    """Generate plasma."""

    frame: Frame = field(repr=False)
    subframe: Frame = field(repr=False)
    delta: float
    tile: bool = field(init=False, default=True)
    fill: bool = field(init=False, default=True)
    turn: str = 'hexagon'
    #_boundary: shapely.geometry.Polygon = field(init=False, repr=False)

    def update_conditionals(self):
        """Update conditional attrs."""

    #def insert(self, *required, iloc=None, **additional):
    #    """Extend PoloidalGrid.insert."""
    #
    #    #PolyGeom(polygon)
    #    #if len(required) == isinstance(required[0], (dict, list, np.ndarray)):



    '''
    def insert(self, boundary, iloc=None, **additional):
        """
        Extend Coil.insert.

        Add plasma to coilset and generate plasma grid.

        Plasma inserted into coilframe with subcoils meshed accoriding
        to delta and trimmed to the inital boundary curve.

        Parameters
        ----------
        boundary : array_like or Polygon
            External plasma boundary. Coerced into positively oriented curve.
        name : str, optional
            Plasma coil name.
        delta : float, optional
            Plasma subcoil dimension. If None defaults to self.dPlasma
        **kwargs : dict
            Keyword arguments passed to PlasmaGrid.generate_grid()

        Returns
        -------
        None.

        """
        #self.biot_instances = ['plasmafilament', 'plasmagrid']
        self.boundary = boundary
        # construct plasma coil from polygon
        additional['poly'] = self.boundary
        required = [additional.pop(attr, self.frame.metaframe.default[attr])
                    for attr in self.frame.metaframe.required]
        super().insert(*required, iloc=iloc, **additional)
    '''

    # name=name, plasma=True, active=True, part='plasma'

    '''
        self.plasmagrid.generate_grid(**kwargs)
        grid_factor = self.dPlasma/self.plasmagrid.dx
        # self._add_vertical_stabilization_coils()
        self.plasmagrid.cluster_factor = 1.5*grid_factor
        self.plasmafilament.add_plasma()
    '''

    @property
    def boundary(self):
        """
        Manage plasma limit boundary.

        Parameters
        ----------
        boundary : array-like, shape(4,) or array-like, shape(n, 2) or Polygon
            External plasma boundary (limit).
            Coerced as a positively oriented curve.

            - array-like, shape(4,) bounding box [xmin, xmax, zmin, zmax]
            - array-like, shape(n,2) bounding loop [x, z]

        Raises
        ------
        IndexError
            Malformed bounding box, shape is not (4,).

            Malformed bounding loop, shape is not (n, 2).

        Returns
        -------
        plasma_boundary : Polygon
            Plasma limit boundary.

        """
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if not isinstance(boundary, shapely.geometry.Polygon):
            boundary = np.array(boundary)  # to numpy array
            if boundary.ndim == 1:   # limit bounding box
                if len(boundary) == 0:
                    return
                elif len(boundary) == 4:
                    polygon = shapely.geometry.box(*boundary[::2],
                                                   *boundary[1::2])
                else:
                    raise IndexError('malformed bounding box\n'
                                     f'boundary: {boundary}\n'
                                     'require [xmin, xmax, zmin, zmax]')
            elif boundary.ndim == 2 and (boundary.shape[0] == 2 or
                                         boundary.shape[1] == 2):  # loop
                if boundary.shape[1] != 2:
                    boundary = boundary.T
                polygon = shapely.geometry.Polygon(boundary)
            else:
                raise IndexError('malformed bounding loop\n'
                                 f'shape(boundary): {boundary.shape}\n'
                                 'require (n,2)')
        else:
            polygon = boundary
        # orient polygon
        polygon = shapely.geometry.polygon.orient(polygon)
        self._boundary = polygon
        #if 'plasmagrid' in self.biot_instances:
        #    self.plasmagrid.plasma_boundary = polygon

    @property
    def separatrix(self):
        """
        Manage plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        loop : DataFrame or array-like, shape(n,2) or Polygon
            Bounding loop.

        Returns
        -------
        separatrix : Polygon
            Plasma separatrix.

        """
        return self._separatrix

    @separatrix.setter
    def separatrix(self, loop):
        if isinstance(loop, pandas.DataFrame):
            loop = loop.values
        if isinstance(loop, shapely.geometry.Polygon):
            polygon = loop
        elif len(loop) == 0:
            return
        else:
            polygon = shapely.geometry.Polygon(loop)
        if not polygon.is_valid:
            polygon = pygeos.creation.polygons(loop)
            polygon = pygeos.constructive.make_valid(polygon)
            area = [pygeos.area(pygeos.get_geometry(polygon, i))
                    for i in range(pygeos.get_num_geometries(polygon))]
            polygon = pygeos.get_geometry(polygon, np.argmax(area))
            polygon = shapely.geometry.Polygon(
                pygeos.get_coordinates(polygon))
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

    @property
    def plasma_tree(self):
        """
        Return STR plasma tree, read-only.

        Construct STR tree from plasma filaments to enable fast search in
        free-boundary calculations. Create link to tree on first call.
        pygeos creates tree on first evaluation

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
        if not hasattr(self, '_plasma_tree'):  # link to pygeos STRtree
            self._plasma_tree = pygeos.STRtree(pygeos.points(
                self.subcoil.x[self.plasma_index],
                self.subcoil.z[self.plasma_index]))
        return self._plasma_tree

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
        ionize : array_like, shape(n,)
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

