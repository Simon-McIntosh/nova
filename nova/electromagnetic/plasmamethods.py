"""
Plasma methods mixin.

Methods inserted into CoilSet class. Requies CoilMethods.

"""
import numpy as np
import pandas as pd
import shapely
import pygeos


class PlasmaMethods:
    """Plasma methods mixin."""

    def __init__(self):
        self.biot_instances = ['plasmagrid', 'plasmafilament']
        self.default_attributes = {'plasma_boundary': [],
                                   'separatrix': []}

    @property
    def dPlasma(self):
        """
        Manage plasma filament dimension.

        Parameters
        ----------
        dPlasma : float
            Plasma filament dimension.

        Returns
        -------
        dPlasma : float
            Plasma filament dimension.

        """
        self._check_default('dPlasma')
        return self._dPlasma

    @dPlasma.setter
    def dPlasma(self, dPlasma):
        self._dPlasma = dPlasma
        self._default_attributes['dPlasma'] = dPlasma

    def add_plasma(self, boundary, name='Plasma', dPlasma=None):
        """
        Add plasma coil to coilset and generate plasma grid.

        Plasma coil inserted into coilframe with subcoils meshed accoriding
        to dPlasma and trimmed to the inital boundary curve.

        Parameters
        ----------
        boundary : array_like or Polygon
            External plasma boundary. Coerced as a positively oriented curve.
        name : str, optional
            Plasma coil name.
        dPlasma : float, optional
            Plasma subcoil dimension. If None defaults to self.dPlasma

        Returns
        -------
        None.

        """
        if dPlasma is not None:  # update plasma subcoil dimension
            self.dPlasma = dPlasma
        self.plasma_boundary = boundary
        # construct plasma coil from polygon
        self.add_coil(0, 0, 0, 0, polygon=self.plasma_boundary,
                      cross_section='polygon', turn_section='rectangle',
                      dCoil=self.dPlasma, name=name, plasma=True, power=True,
                      part='plasma')

    @property
    def plasma_boundary(self):
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
        return self._plasma_boundary

    @plasma_boundary.setter
    def plasma_boundary(self, boundary):
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
        self._plasma_boundary = polygon
        self.plasmagrid.plasma_boundary = polygon

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
    def separatrix(self):
        """
        Manage plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update flag to True.

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
        if isinstance(loop, pd.DataFrame):
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
        separatrix = polygon.intersection(self.plasma_boundary)
        self._separatrix = separatrix
        # update coil - polygon and polygon derived attributes
        self.coil.loc['Plasma', 'polygon'] = separatrix
        self.coil.update_polygon(index='Plasma')
        # ubdate subcoil
        self.subcoil.ionize = self.plasma_tree.query(
            pygeos.io.from_shapely(separatrix), predicate='contains')
        self.update_plasma = True

    @property
    def plasma_index(self):
        """
        Return plasma index on subcoil frame, read-only.

        Returns
        -------
        plasma_index : array-like
            subcoil plasma index

        """
        return self.subcoil._plasma_index

    @property
    def ionize_index(self):
        """
        Return plasma ionization index, read-only.

        Returns
        -------
        ionize : array_like, shape(n,)
            Subcoil ionization index (Nt>0).

        """
        return self.subcoil.ionize

    @property
    def nP(self):
        """Return number of plasma filaments."""
        return self.subcoil.nP

    @property
    def Np(self):
        """
        Manage plasma filament **turn** number.

        Property should not be confused with nP (number of plasma filaments)

        Parameters
        ----------
        Np : float or array-like, shape(,)
            Plasma filament turn number.

        Returns
        -------
        Np : array-like, shape(nP,)
            Plasma filament turn number.

        """
        return self.subcoil.Np

    @Np.setter
    def Np(self, Np):
        self.subcoil.Np = Np
        self.coil.Np = self.subcoil.Np.sum()

    @property
    def Ip(self):
        """
        Manage net plasma current.

        Parameters
        ----------
        Ip : float
            Net plasma current.
            Set in coil and subcoil frames

        Returns
        -------
        Ip_sum : float
            Net plasma current.

        """
        return self.coil.Ip_sum

    @Ip.setter
    def Ip(self, Ip):
        self.coil.Ip = Ip
        self.subcoil.Ip = Ip
