"""
Plasma methods mixin.

Methods inserted into CoilSet class. Requies CoilMethods.

"""
import numpy as np
import shapely
import pygeos


class PlasmaMethods:
    """Plasma methods mixin."""

    def __init__(self):
        self.biot_instances = ['plasmagrid', 'plasmafilament']
        self.default_attributes = {'plasma_boundary': None,
                                   'plasma_tree': None}

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
                      cross_section='rectangle',
                      dCoil=self.dPlasma, name=name, plasma=True,
                      part='plasma')
        #self.plasma_tree = pygeos.STRtree(pygeos.points(
        #    self.subcoil.loc[self.plasma_index, ['x', 'z']].values))
        # generate plasma grid
        # self.plasma.generate_grid()

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
        if boundary is not None:
            if not isinstance(boundary, shapely.geometry.Polygon):
                boundary = np.array(boundary)  # to numpy array
                if boundary.ndim == 1:   # limit bounding box
                    if len(boundary) == 4:
                        polygon = shapely.geometry.box(*boundary[::2],
                                                       *boundary[1::2])
                    else:
                        raise IndexError('malformed bounding box\n'
                                         f'boundary: {boundary}\n'
                                         'require [xmin, xmax, zmin, zmax]')
                elif boundary.ndim == 2 and boundary.shape[1] == 2:  # loop
                    polygon = shapely.geometry.Polygon(boundary)
                else:
                    raise IndexError('malformed bounding loop\n'
                                     f'shape(boundary): {boundary.shape}\n'
                                     'require (n,2)')
            else:
                polygon = boundary
            # orient polygon
            polygon = shapely.geometry.polygon.orient(polygon, sign=1.0)
            self._plasma_boundary = polygon

    '''
    @property
    def plasma_tree(self):
        """
        Manage STR plasma tree.

        Construct STR tree from plasma filaments to enable fast search in
        free-boundary calculations.

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
        return self._plasma_tree

    @plasma_tree.setter
    def plasma_tree(self, tree):
        if tree is not None:
            if not isinstance(tree, pygeos.STRtree):
                raise TypeError('requires pygeos.STRtree\n'
                                f'passed {type(tree)}')
            self._plasma_tree = tree
    '''

    @property
    def plasma_index(self):
        """
        Return plasma index on subcoil frame.

        Returns
        -------
        plasma_index : array-like
            subcoil plasma index

        """
        return self.subcoil._plasma_index

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
