import shapely.geometry
import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import skimage.measure
import sklearn.cluster
import nlopt
import pandas as pd

from nova.utilities.pyplot import plt
from nova.utilities import geom


class TopologyError(Exception):
    """Raise topology error."""


class Topology:
    """
    Extract topology from poloidal flux and field maps.

    Abstract base class inherited by Grid.

    """

    _optimizer_instances = {'newton': 'LD_TNEWTON',
                            'mma': 'LD_MMA'}

    _interpolate_attributes = ['Psi', 'B']

    _topology_attributes = ['null_cluster',
                            'coil_center', 'update_coil_center',
                            'Xpoint', 'Opoint', 'Xpsi', 'Opsi',
                            'update_topology', 'update_topology_index',
                            'field_quantile',
                            'cluster_factor', 'unique_factor',
                            'optimizer', 'filter_sigma',
                            'ftol_rel', 'xtol_rel']

    def __init__(self):
        """Extend biot attributes."""
        self._biot_attributes += [
            attribute_pair for attribute in self._interpolate_attributes
            for attribute_pair in [f'_{attribute}_spline',
                                   f'_update_{attribute}_spline']]
        self._biot_attributes += [f'_{attribute}'
                                  for attribute in self._topology_attributes]
        self._default_biot_attributes.update(
            {f'_update_{attribute}_spline': True
             for attribute in self._interpolate_attributes})
        self._default_biot_attributes.update(
            {'_null_cluster': [],
             '_coil_center': [], 'update_coil_center': True,
             '_Xpoint': [], '_Opoint': [],
             '_field_quantile': 0.05,
             '_cluster_factor': 1.5, '_unique_factor': 0.5,
             '_update_topology': True, '_update_topology_index': True,
             '_optimizer': 'mma', '_filter_sigma': 1.5,
             '_ftol_rel': 1e-12, '_xtol_rel': 1e-12})

    def _flag_update(self, status):
        """
        Implement CoilMatrix method hook to set local update flags.

        Parameters
        ----------
        status : bool
            Update status, propogate if True.

        Returns
        -------
        None.

        """
        if status:
            for attribute in self._interpolate_attributes:
                setattr(self, f'_update_{attribute}_spline', True)
            self._update_topology = True

    def solve_topology(self):
        """
        Perform global topology update.

        - Extract global null points.
        - Sort by null type (X or O).
        - Extract poloidal flux at each null point.
        - Sort each null type array by poloidal flux.
        - Merge repeated nulls

        Returns
        -------
        None.

        """
        if self._update_topology:
            self.global_null()

    def global_null(self, plot=False, **kwargs):
        """
        Locate all field nulls. Categorize as X or O points.

        Cluster grid points with an absolute field below factor*Bmin.
        Launch local gradient based optimizers at geometric centers of each
        cluster. Categorize nulls based on local field curvature.

        - Xpoint : saddle
        - Opoint : concave or convex

        Parameters
        ----------
        plot : bool, optional
            Plot flag. The default is False.

        Keyword Arguments
        -----------------
        field_quantile : float, optional
            Lower quantile to search for low field clusters.
        cluster_factor : float, optional
            Factor applied to max grid spacing to produce the maximum
            neighbour seperation used by the DBSCAN algorithum.
            The default is self.cluster_factor.
        unique_factor : float, optional
            Factor applied to max grid spacing to produce the minimum
            neighbour seperation for unique null points.
            The default is self.unique_factor.

        Returns
        -------
        None.

        """
        field_quantile = kwargs.get('field_quantile', self.field_quantile)
        cluster_factor = kwargs.get('cluster_factor', self.cluster_factor)
        unique_factor = kwargs.get('unique_factor', self.unique_factor)
        dx = np.max([self.dx, self.dz])  # maximum grid delta
        eps_cluster = cluster_factor*dx  # max neighbour seperation in DBSCAN
        eps_unique = unique_factor*dx  # min seperation for unique nulls
        # (re)initialize null point arrays
        self._Xpoint, self._Opoint = [], []
        # field null clusters
        self._null_cluster, _field_Opoint = [], []
        Bthreshold = np.quantile(self.B, field_quantile, interpolation='lower')
        index = self.B < Bthreshold  # threshold
        if np.sum(index) > 0:  # protect against uniform zero field
            xt, zt = self.x2d[index], self.z2d[index]  # threshold points
            dbscan = sklearn.cluster.DBSCAN(eps=eps_cluster, min_samples=1)
            cluster_index = dbscan.fit_predict(np.array([xt, zt]).T)
            for i in range(np.max(cluster_index)+1):
                # cluster coordinates
                x_cluster = xt[cluster_index == i]
                z_cluster = zt[cluster_index == i]
                self._null_cluster.append([x_cluster, z_cluster])
        for x_cl, z_cl in self._null_cluster + list(self.plasma_vertex):
            # coordinates of cluster centre
            xc = np.mean(x_cl)
            zc = np.mean(z_cl)
            # resolve local field null
            xn, zn = self.local_null((xc, zc))
            null_type = self.null_type((xn, zn))
            if null_type == 'X':
                self._Xpoint.append([xn, zn])
            elif null_type == 'O':
                _field_Opoint.append([xn, zn])
        for xc, zc in _field_Opoint + list(self.coil_center):
            xn, zn = self._refine_Opoint(xc, zc)
            if self.null_type((xn, zn)) == 'O':
                self._Opoint.append([xn, zn])
        # convert to unique np.arrays and
        self._sort_null(eps=eps_unique)  # sort by poloidal flux
        self._update_topology = False
        self._update_topology_index = True
        if plot:
            self.plot_topology(plot_clusters=True)

    def _refine_Opoint(self, xc, zc):
        p = self._flux_curvature((xc, zc))
        concave = p[np.argmax(np.abs(p))] < 0
        xn, zn = self.local_Opoint((xc, zc), concave)
        return xn, zn

    @property
    def coil_center(self):
        """
        Return geometric centers of coils that fall within grid bounds.

        Returns
        -------
        coil_center : array-like, shape(n, 2)
            Coordinates of coil centers (x, z).

        """
        if self._update_coil_center:
            self.update_coil_center()
            self._update_coil_center = False
        return self._coil_center

    @property
    def plasma_vertex(self):
        """
        Return plasma verticies.

        Returns
        -------
        plasma_vertex : array-like, shape(n, 2)
            Corrdinates of upper and lower plasma verticies.

        """
        if self._update_coil_center:
            self.update_coil_center()
            self._update_coil_center = False
        return self._plasma_vertex

    def update_coil_center(self):
        """
        Extract coil centers from source BiotFrame.

        Method called by Grid instance when (re)generating grid.

        Returns
        -------
        None.

        """
        _coil_center = []
        _plasma_vertex = []
        reduction_index = self.source._reduction_index
        turn_number = np.add.reduceat(np.ones(self.source.nC), reduction_index)
        xc = np.sqrt(np.add.reduceat(self.source.rms**2, reduction_index) /
                     turn_number)
        zc = np.add.reduceat(self.source.z, reduction_index) / turn_number
        zmin = np.minimum.reduceat(self.source.z, reduction_index)
        zmax = np.maximum.reduceat(self.source.z, reduction_index)
        # plasma vertex
        for iloc in self.source._plasma_iloc:
            _plasma_vertex.append([xc[iloc], zmin[iloc]])
            _plasma_vertex.append([xc[iloc], zmax[iloc]])
        grid_polygon = self.grid_polygon
        for i, (x, z), in enumerate(zip(xc, zc)):
            if grid_polygon.contains(shapely.geometry.Point(x, z)):
                _coil_center.append([x, z])
        self._coil_center = np.array(_coil_center)
        self._plasma_vertex = np.array(_plasma_vertex)

    def plot_topology(self, plot_clusters=False, ax=None, color='C3',
                      legend=False):
        """
        Plot topological points (X-points, O-points).

        Parameters
        ----------
        ax : axis, optional
            Plot axis. The default is plt.gca().
        plot_clusters : bool, optional
            Plot threshold null point clusters. The default is False.
        color : str
            Null point marker color. The default is darkgray

        Returns
        -------
        None.

        """
        self.solve_topology()
        if ax is None:
            ax = plt.gca()
        if plot_clusters:
            for cluster in self._null_cluster:
                ax.plot(cluster[0], cluster[1], 'C7.', ms=4)
                ax.plot(*np.mean(cluster, axis=1), 'k.', ms=4)  # centers
            ax.plot(*self._plasma_vertex.T, 'C0.')
            ax.plot(*self._coil_center.T, 'C3.')
        if self.nX > 0:  # X-ponits
            ax.plot(*self.Xpoint.T, 'x', label=f'X-point {self.nX}',
                    ms=6, mew=1, color=color)
        if self.nO > 0:  # O-ponits
            ax.plot(*self.Opoint.T, 'o', label=f'O-point {self.nO}',
                    markerfacecolor='none', mew=1, ms=6, color=color)
        if (self.nX > 0 or self.nO > 0) and (legend or plot_clusters):
            ax.legend(loc='center right')

    @property
    def field_quantile(self):
        """
        Manage field_quantile.

        Lower quantile to search for low field clusters. Field multiplied by
        radial coordinate to balance reduction of field at high radii.
        Setting flags topology update.

        Parameters
        ----------
        field_quantile : float
            DESCRIPTION.

        Returns
        -------
        field_quantile : float

        """
        return self._field_quantile

    @field_quantile.setter
    def field_quantile(self, field_quantile):
        if field_quantile <= 0 or field_quantile > 1:
            raise TopologyError('feild quantile '
                                f'{field_quantile} bound by 0 and 1')
        self._field_quantile = field_quantile
        self._update_topology = True

    @property
    def cluster_factor(self):
        """
        Manage cluster factor.

        Factor applied to max grid spacing to produce the maximum
        neighbour seperation used by the DBSCAN algorithum.
        Setting flags topology update.

        Parameters
        ----------
        cluster_factor : float
            DESCRIPTION.

        Returns
        -------
        cluster_factor : float

        """
        return self._cluster_factor

    @cluster_factor.setter
    def cluster_factor(self, cluster_factor):
        self._cluster_factor = cluster_factor
        self._update_topology = True

    @property
    def unique_factor(self):
        """
        Manage unique factor.

        Factor applied to max grid spacing to produce the minimum
        neighbour seperation for unique null points.
        Setting flags topology update.

        Parameters
        ----------
        unique_factor : float
            DESCRIPTION.

        Returns
        -------
        unique_factor : float

        """
        return self._unique_factor

    @unique_factor.setter
    def unique_factor(self, unique_factor):
        self._unique_factor = unique_factor
        self._update_topology = True

    @property
    def ftol_rel(self):
        """
        Manage relitive termination tolarance of objective function.

        Adjustments to ftol_rel trigger a global topology update.

        Parameters
        ----------
        ftol_rel : float
            relitive minimization tolarance on objective.

        Returns
        -------
        ftol_rel : float

        """
        return self._ftol_rel

    @ftol_rel.setter
    def ftol_rel(self, ftol_rel):
        self._ftol_rel = ftol_rel
        self._update_topology = True

    @property
    def xtol_rel(self):
        """
        Manage relitive termination tolarance of objective function.

        Adjustments to xtol_rel trigger a global topology update.

        Parameters
        ----------
        xtol_rel : float
            relitive minimization tolarance on objective.

        Returns
        -------
        xtol_rel : float

        """
        return self._xtol_rel

    @xtol_rel.setter
    def xtol_rel(self, xtol_rel):
        self._xtol_rel = xtol_rel
        self._update_topology = True

    @property
    def Xpoint(self):
        """
        Return Xpoints sorted by poloidal flux.

        Returns
        -------
        Xpoint : ndarray, shape(nX, 2)
            Xpoints sorted by poloidal flux.

        """
        self.solve_topology()
        return self._Xpoint

    @property
    def Opoint(self):
        """
        Return Opoints sorted by poloidal flux.

        Returns
        -------
        Opoint : ndarray, shape(nO, 2)
            Opoints sorted by poloidal flux.

        """
        self.solve_topology()
        return self._Opoint

    @property
    def Xpsi(self):
        """
        Return sorted poloidal flux at Xpoints.

        Returns
        -------
        Xpsi : ndarray, shape(nX, 2)
            Sorted poloidal flux at Xpoints.

        """
        self.solve_topology()
        return self._Xpsi

    @property
    def Opsi(self):
        """
        Return sorted poloidal flux at Opoints.

        Returns
        -------
        Opsi : ndarray, shape(nO, 2)
            Sorted poloidal flux at Opoints.

        """
        self.solve_topology()
        return self._Opsi

    @property
    def nX(self):
        """
        Return number of Xpoints.

        Returns
        -------
        nX : int
            Number of Xpoints.

        """
        self.solve_topology()
        return len(self._Xpoint)

    @property
    def nO(self):
        """
        Return number of Opoints.

        Returns
        -------
        nO : int
            Number of Opoints.

        """
        self.solve_topology()
        return len(self._Opoint)

    def contour(self, flux, plot=False, ax=None, **kwargs):
        """
        Return flux contours.

        Parameters
        ----------
        flux : float or list[float]
            Contour levels.
        plot : bool, optional
            Plot contours. The default is False.
        ax : axes, optional
            Plot axes. The default is None.

            - None: plots to current axes

        **kwargs : dict
            Keyword arguments passed to plot.

        Returns
        -------
        contours : list[array-like, shape(n, 2)]
            Contour coordinates.

        """
        index = skimage.measure.find_contours(self.Psi, flux)
        contours = [[] for __ in range(len(index))]
        for i, idx in enumerate(index):
            contours[i] = np.array([self.x_index(idx[:, 0]),
                                    self.z_index(idx[:, 1])]).T
        if plot:
            if ax is None:
                ax = plt.gca()
            for contour in contours:
                plt.plot(contour[:, 0], contour[:, 1], **kwargs)
        return contours

    def _sort_null(self, eps=1e-3):
        """
        Sort and merge null points.

        Null points sorted by poloidal flux. Points with a neighbour seperation
        less than eps are merged.

        Parameters
        ----------
        eps : float, optional
            Minimum neighbour seperation for unique null points.
            The default is 1e-3.

        Returns
        -------
        None.

        """
        for null_type in ['X', 'O']:
            Xnull = getattr(self, f'_{null_type}point')
            Xnull = geom.unique2D(Xnull, eps=eps, bound=self.grid_boundary)[1]
            Psi = []
            for i in range(len(Xnull)):
                Psi.append(self.interpolate('Psi').ev(*Xnull[i]))
            Psi = np.array(Psi)
            sort_index = np.argsort(Psi)
            setattr(self, f'_{null_type}psi', Psi[sort_index])
            setattr(self, f'_{null_type}point', Xnull[sort_index])

    def local_null(self, xo):
        """
        Return local field null (minimum absolutle poloidal field).

        Parameters
        ----------
        xo : array-like, shape(2,)
            Seed coordinates (x, y).

        Returns
        -------
        x : array-like, shape(2,)
            Null coordinates (x, z).

        """
        opt = self._get_opt('field', minimize=True)
        return opt.optimize(self._bound_point(*xo))

    def local_Opoint(self, xo, minimize):
        """
        Return local Opoint (minimum or maximum of poloidal flux).

        Parameters
        ----------
        xo : array-like, shape(2,)
            Seed coordinates (x, y).
        minimize : bool
            minimize flag.

            - True : minimize objective
            - False : maximuze objective

        Returns
        -------
        x : array-like, shape(2,)
            Opoint coordinates (x, z).

        """
        opt = self._get_opt('flux', minimize=minimize)
        return opt.optimize(self._bound_point(*xo))

    def interpolate(self, attribute):
        """
        Return RectBivariateSpline for attribute (Lazy evaluation).

        Parameters
        ----------
        attribute : str
            Attriburte label.

        Returns
        -------
        _{attribute}_spline: RectBivariateSpline
            Grid interpolant.

        """
        self._evaluate_spline(attribute)
        return getattr(self, f'_{attribute}_spline')  # interpolant

    @property
    def filter_sigma(self):
        """
        Manage kernal width for gaussian filter.

        Set width to zero to dissable filtering.

        Parameters
        ----------
        sigma : float
            Kernal width of gaussian filter.

        """
        return self._filter_sigma

    @filter_sigma.setter
    def filter_sigma(self, sigma):
        for attribute in self._interpolate_attributes:
            setattr(self,  f'_update_{attribute}_spline', True)
        self._filter_sigma = sigma

    def _evaluate_spline(self, attribute):
        update_flag = f'_update_{attribute}_spline'
        if getattr(self, update_flag):
            # compute interpolant
            z = getattr(self, attribute)
            # filter
            sigma = self.filter_sigma
            if sigma != 0:
                z = scipy.ndimage.gaussian_filter(z, sigma)
            setattr(self, f'_{attribute}_spline',
                    scipy.interpolate.RectBivariateSpline(
                        self.x, self.z, z, bbox=self.grid_boundary))
            setattr(self, update_flag, False)

    @property
    def spline_update_status(self):
        """
        Return update status for spline interpolants.

        Returns
        -------
        spline_update_status: pandas.Series
            Update status for spline interpolants.

        """
        return pd.Series({attribute:
                          getattr(self, f'_update_{attribute}_spline')
                          for attribute in self._spline_attributes})

    def null_type(self, x):
        """
        Return feild null type.

        Parameters
        ----------
        x : array-like, shape(2,)
            Coordinates of null-point (x, z).

        Raises
        ------
        TopologyError
            One or both princaple flux curvatures equal to zero
            (plane or cylindrical surface).

        Returns
        -------
        null_type : str

            - X : X-point
            - O : O-point.

        """
        Pmax, Pmin = self._flux_curvature(x)
        Pratio = np.max(abs(np.array([Pmax, Pmin])) /
                        np.min(abs(np.array([Pmax, Pmin]))))
        if np.isclose(Pmax, 0) or np.isclose(Pmin, 0):
            raise TopologyError('Field null froms cylinder or plane surface')
        elif Pratio > 100:
            return '-'
        elif np.sign(Pmax) == np.sign(Pmin):
            return 'O'
        else:
            return 'X'

    def _flux_curvature(self, x):
        """
        Return principal curvatures in poloidal flux at x.

        Parameters
        ----------
        x : array-like, shape(2,)
            Polidal coordinates for curvature calculation.

        Returns
        -------
        Pmax : float
            Maximum principal curvature.
        Pmin : float
            Minimum principal curvature.

        """
        # flux derivatives
        Px = self.interpolate('Psi').ev(*x, dx=1)
        Pz = self.interpolate('Psi').ev(*x, dy=1)
        Pxx = self.interpolate('Psi').ev(*x, dx=2)
        Pzz = self.interpolate('Psi').ev(*x, dy=2)
        Pxz = self.interpolate('Psi').ev(*x, dx=1, dy=1)
        # mean curvature
        H = (Px**2 + 1)*Pzz - 2*Px*Pz*Pxz + (Pz**2 + 1)*Pxx
        H = -H/(2*(Px**2 + Pz**2 + 1)**(1.5))
        # gaussian curvature
        K = (Pxx*Pzz - Pxz**2) / (1 + Px**2 + Pz**2)**2
        # principal curvature
        Pmax = H + np.sqrt(H**2 - K)
        Pmin = H - np.sqrt(H**2 - K)
        return Pmax, Pmin

    def _opt_name(self, opt_name, minimize=True):
        """
        Return optimization instance name.

        Parameters
        ----------
        opt_name : str
            Optimization name.
        minimize : bool, optional
            Minimize objective function. The default is True.

        Returns
        -------
        opt_name : str
            optimization instance name dependant on minimize flag:
            - True : opt_name = f'_min_{opt_name}'
            - False : opt_name = f'_max_{opt_name}'

        """
        return f'_min_{opt_name}' if minimize else f'_max_{opt_name}'

    @property
    def optimizer(self):
        """
        Manage nlopt optimizer.

        Avalible optimizers listed in self._optimizer_instances.

        Parameters
        ----------
        _optimizer : str
            Set optimiser name.

        Returns
        -------
        _optimizer : str
            Return name of current optimizer.

        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, name):
        if name not in self._optimizer_instances:
            raise IndexError(f'Optimizer {name} not present in '
                             f'{self._optimizer_instances.keys()}')
        self._optimizer = name

    def _get_opt(self, name, minimize=True):
        """
        Set nlopt optimization instance.

        Parameters
        ----------
        opt_name : str
            Optimization name.
        minimize : bool, optional
            Minimize objective function. The default is True.

        Returns
        -------
        None.

        """
        opt_name = self._opt_name(name, minimize)
        if not hasattr(self, opt_name):
            opt_instance = nlopt.opt(
                getattr(nlopt, self._optimizer_instances[self.optimizer]), 2)
            objective = getattr(self, f'_{name}')
            if minimize:
                opt_instance.set_min_objective(objective)
            else:
                opt_instance.set_max_objective(objective)
            opt_instance.set_ftol_rel(self.ftol_rel)
            opt_instance.set_xtol_rel(self.xtol_rel)
            opt_instance.set_lower_bounds(self.grid_boundary[::2])
            opt_instance.set_upper_bounds(self.grid_boundary[1::2])
            setattr(self, opt_name, opt_instance)
        else:
            opt_instance = getattr(self, opt_name)
            if opt_instance.get_ftol_rel != self.ftol_rel:
                opt_instance.set_ftol_rel = self.ftol_rel
            if opt_instance.get_xtol_rel != self.xtol_rel:
                opt_instance.set_xtol_rel = self.xtol_rel
        return opt_instance

    def _field(self, x, grad):
        if grad.size > 0:
            grad[:] = self._gradient(x, 'B')
        return self.interpolate('B').ev(*x).item()

    def _flux(self, x, grad):
        if grad.size > 0:
            grad[:] = self._gradient(x, 'Psi')
        return self.interpolate('Psi').ev(*x).item()

    def _objective(self, x, attribute):
        return self.interpolate(attribute).ev(*x).item()

    def _gradient(self, x, attribute):
        """
        Return gradient of objective function.

        Parameters
        ----------
        x : array-like
            Query coordinates (x, z).
        attribute : str
            Attribute name (interpolate attribute).

        Returns
        -------
        grad : array-like, shape(2,)
            Gradient of objective function.

        """
        return [self.interpolate(attribute).ev(*x, dx=1, dy=0).item(),
                self.interpolate(attribute).ev(*x, dx=0, dy=1).item()]

    def _bound_point(self, x, z):
        # bound x
        if x < self.grid_boundary[0]:
            x = self.grid_boundary[0]
        elif x > self.grid_boundary[1]:
            x = self.grid_boundary[1]
        # bound z
        if z < self.grid_boundary[2]:
            z = self.grid_boundary[2]
        elif z > self.grid_boundary[3]:
            z = self.grid_boundary[3]
        return x, z
