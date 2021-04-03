"""
Plasma methods mixin.

Methods inserted into CoilSet class. Requies CoilMethods.

"""
import operator
import itertools
import warnings

import numpy as np
import pandas as pd
import shapely
import pygeos

from nova.electromagnetic.topology import TopologyError
from nova.utilities.pyplot import plt


class PlasmaMethods:
    """Plasma methods mixin."""

    def __init__(self):
        self.default_attributes = {'plasma_boundary': [], 'separatrix': []}

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


    def _add_vertical_stabilization_coils(self, apex=25, dz_factor=1.1,
                                          n_subcoil=25):
        """
        Add vertical stabilization coils.

        Pair of truncated cone shaped coils inserted on high field side of
        plasma. Coil pair linked with a multi-point constraint (mpc) with the
        upper coil constrained in current with a vaule equal to the negated
        lower coil current (Iupper=-Ilower). Coil geometroy chosen to generate
        roughly uniform radial field across the entire plasma rejoin.


        Parameters
        ----------
        apex : float, optional
            Apex angle of coil cone in degrees. The default is 25.
        dz_factor : float, optional
            Multiplicative factor applied to the plasma boundary vertical
            extent. The default is 1.1.
        n_subcoil : int, optional
            Number of subcoils to discritize each coil cone. The default is 25.

        Returns
        -------
        None.

        """
        apex *= np.pi/180
        bounds = self.plasma_boundary.bounds
        zlim = bounds[1::2]
        dz = zlim[1]-zlim[0]
        zo = np.mean(zlim)
        xmax = 1.5*bounds[2]
        xo = bounds[0] - dz_factor * dz/2 * np.tan(apex)
        if xo < 0:
            apex_min = np.arctan(bounds[0] / (dz_factor * dz/2))
            warn_txt = f'Cone apex {xo: 1.2f} < 0\n'
            warn_txt += 'Decreasing apex angle '
            warn_txt += fr'from {apex * 180/np.pi}$^o$ '
            warn_txt += fr'to {apex_min * 180/np.pi}$^o$'
            warnings.warn(warn_txt)
            xo, apex = 0, apex_min
        dL = (xmax-xo) / np.sin(apex)
        dCoil = dL / n_subcoil
        # add conical coils
        for i in range(2):
            self.add_shell([xo, xmax],
                           [zo, zo + (-1)**i * (xmax-xo) / np.tan(apex)],
                           dt=self.dPlasma, dCoil=dCoil, dShell=0,
                           label='Zfb', feedback=True)
        self.add_mpc(self.coil.index[-2:], -1)

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
        if 'plasmagrid' in self.biot_instances:
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

    def update_topology_index(self):
        if self.plasmagrid._update_topology_index:
            Opoint = self.plasmagrid.Opoint
            if self.plasmagrid.nO == 0:
                self.plasmagrid.plot_flux()
                self.plasmagrid.plot_topology(True)
                raise TopologyError('no Opoints found')
            elif self.plasmagrid.nO == 1:
                self._Oindex = 0
            else:
                plasma_centroid = np.array([self.coil.x[self.coil.plasma],
                                            self.coil.z[self.coil.plasma]]).T
                dL = np.linalg.norm(Opoint-plasma_centroid, axis=1)
                self._Oindex = np.argmin(dL)
            if not self.separatrix.contains(
                    shapely.geometry.Point(*Opoint[self._Oindex])):
                raise TopologyError('no Opoints found within separatrix')
            Opsi = self.plasmagrid.Opsi[self._Oindex]
            Xpsi = self.plasmagrid.Xpsi
            self._Xindex = np.argmin(abs(Opsi-Xpsi))
            self._locate_Xpoint()
            self.plasmagrid._update_topology_index = False

    def _locate_Xpoint(self):
        Xz = self.plasmagrid.Xpoint[self._Xindex][1]
        Oz = self.plasmagrid.Opoint[self._Oindex][1]
        if Xz < Oz:
            self._Xloc = 'lower'
        else:
            self._Xloc = 'upper'

    @property
    def Xloc(self):
        """
        Return location descriptor of primary X-point relitive to O-point.

        - lower : X-point below O-point.
        - upper : X-point above O-point.
        - limit : Separatrix limited by plasma_boundary.

        Returns
        -------
        Xloc : str
            Xpoint location descriptor.

        """
        self.update_topology_index()
        return self._Xloc

    @property
    def Oindex(self):
        self.update_topology_index()
        return self._Oindex

    @property
    def Opoint(self):
        return self.plasmagrid.Opoint[self.Oindex]

    @property
    def Opsi(self):
        return self.plasmagrid.Opsi[self.Oindex]

    @property
    def Xindex(self):
        self.update_topology_index()
        return self._Xindex

    @property
    def Xpoint(self):
        return self.plasmagrid.Xpoint[self.Xindex]

    @property
    def Xpsi(self):
        return self.plasmagrid.Xpsi[self.Xindex]

    def _trim_contour(self, contour):
        if self.Xloc in ['lower', 'upper']:
            trimed_contour = []
            compare = operator.ge if self.Xloc == 'lower' else operator.le
            index = compare(contour[:, 1], self.Xpoint[1])
            group = [list(i)[:2][0] for __, i in
                     itertools.groupby(enumerate(index), key=lambda x: x[-1])]
            ngroup = len(group)
            ncontour = len(contour)
            for i in range(ngroup):
                if group[i][1]:  # compare is True
                    start = group[i][0]
                    stop = ncontour if i == ngroup-1 else group[i+1][0]
                    trimed_contour.append(contour[slice(start, stop)])
        else:
            trimed_contour = [contour]
        return trimed_contour

    def _close_contour(self, trimed_contour, alpha, ndx=3):
        max_gap = ndx*self.coil.dx[self.coil.plasma]
        closed_contour = []
        for ct in trimed_contour:
            gap = np.linalg.norm(ct[0]-ct[-1])
            if gap <= max_gap and len(ct) >= 3:  # contour closed
                if gap > 0:  # close gap
                    dX = np.linalg.norm(self.Xpoint-ct[0])
                    ct = np.append(ct, ct[:1], axis=0)
                    '''
                    if dX < max_gap and alpha == 1:
                        Xpoint = self.Xpoint.reshape(-1, 2)
                        #ct = np.append(Xpoint, np.append(ct, Xpoint, axis=0),
                        #               axis=0)
                        ct = np.append(ct, ct[:1], axis=0)
                    else:
                        ct = np.append(ct, ct[:1], axis=0)
                    '''
                closed_contour.append(ct)
        return closed_contour

    def update_separatrix(self, plot=False, **kwargs):
        if 'Psi' in kwargs:
            Psi = kwargs['psi']
            alpha = (Psi-self.Opsi) / (self.Xpsi-self.Opsi)
        else:
            alpha = kwargs.get('alpha', 1-1e-3)
            Psi = alpha * (self.Xpsi-self.Opsi) + self.Opsi
        Opoint = shapely.geometry.Point(self.Opoint)
        contours = self.plasmagrid.contour(Psi)
        closed_contours = []
        for contour in contours:
            trimed_contour = self._trim_contour(contour)
            closed_contours.extend(self._close_contour(trimed_contour, alpha))
        self._separatrix = []  # clear current separatrix
        for cc in closed_contours:
            polygon = shapely.geometry.Polygon(cc)
            if polygon.contains(Opoint):
                self.separatrix = polygon
                break
        if not self._separatrix:  # separatrix not found
            self.plot()
            self.plasmagrid.plot_topology(True)
            self.plasmagrid.plot_flux()
            for i, contour in enumerate(contours):
                label = rf'all $\alpha$={alpha:1.2f}' if i == 0 else None
                plt.plot(*contour.T, 'k-', label=label)
            for i, closed_contour in enumerate(closed_contours):
                label = 'closed' if i == 0 else None
                plt.plot(*closed_contour.T, 'C3-', label=label)
            plt.legend(loc='upper right')
            raise TopologyError('closed separatrix containing Opoint '
                                'not found')
        if plot:
            plt.plot(*self.separatrix.boundary.xy, '-', color=0.4*np.ones(3))

    def plot_null(self, legend=True):
        """
        Plot primary nulls.

        Parameters
        ----------
        legend : bool, optional
            Include legend. The default is True.

        Returns
        -------
        None.

        """
        color = 'C3'
        plt.plot(*self.Opoint, 'o', mfc='none', mec=color, mew=1.25, ms=6,
                 label='X-point')
        plt.plot(*self.Xpoint, 'x', mec=color, mew=1.25, ms=6, label='O-point')
        if legend:
            plt.legend(loc='center right')

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
            Subcoil ionization index (Nt>0).

        """
        return self.subcoil.ionize[self.subcoil.plasma]

    @property
    def nP(self):
        """Return number of plasma filaments."""
        return self.subcoil.nP

    @property
    def Np(self):
        """
        Manage plasma filament **turn** number.

        Property should not be confused with nP (number of plasma filaments)

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
        self.update_plasma_current = True
