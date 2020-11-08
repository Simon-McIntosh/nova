import itertools

import pygmo as pg
import shapely.geometry
import numpy as np
import scipy.optimize
import scipy.interpolate
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

    _interpolate_attributes = ['Psi', 'B']

    _topology_attributes = ['Opoint', 'Opsi', 'Xpoint', 'Xpsi',
                            'ftol_rel']

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
            {'_update_topology': True, '_Xpoint': [], '_Opoint': [],
             '_ftol_rel': 1e-6})

    @property
    def ftol_rel(self):
        """
        Manage relitive termination tolarance of objective function.

        Adjustments to ftol_rel trigger a global topology update.

        Parameters
        ----------
        ftol_rel : float
            relitive minimization tolarance on objective. The default is 1e-9.

        Returns
        -------
        ftol_rel : float

        """
        return self._ftol_rel

    @ftol_rel.setter
    def ftol_rel(self, ftol_rel):
        update = ~np.isclose(self._ftol_rel, ftol_rel, atol=1e-16)
        self._ftol_rel = ftol_rel
        if update:
            self._update_ftol_rel('field', True)
            self._update_ftol_rel('flux', True)
            self._update_ftol_rel('flux', False)
            self._update_topology = True

    def _update_ftol_rel(self, opt_name, minimize):
        opt_name = self._opt_name(opt_name, minimize)
        if hasattr(self, opt_name):
            getattr(self, opt_name).set_ftol_rel(self._ftol_rel)

    def _flag_update(self, status):
        if status:
            for attribute in self._interpolate_attributes:
                setattr(self, f'_update_{attribute}_spline', True)
            self._update_topology = True

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

    def _evaluate_spline(self, attribute):
        update_flag = f'_update_{attribute}_spline'
        if getattr(self, update_flag):
            # compute interpolant
            setattr(self, f'_{attribute}_spline',
                    scipy.interpolate.RectBivariateSpline(
                        self.x, self.z, getattr(self, attribute)))
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
        if np.isclose(Pmax, 0) or np.isclose(Pmin, 0):
            raise TopologyError('Field null froms cylinder or plane surface')
        elif np.sign(Pmax) == np.sign(Pmin):
            return 'O'
        else:
            return 'X'

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

    def _get_opt(self, opt_name, minimize=True):
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
        objective = getattr(self, f'_{opt_name}')
        opt_name = self._opt_name(opt_name, minimize)
        if not hasattr(self, opt_name):
            opt_instance = nlopt.opt(nlopt.LD_MMA, 2)
            # opt_instance = nlopt.opt(nlopt.LD_LBFGS, 2)
            # opt_instance = nlopt.opt(nlopt.LD_SLSQP, 2)
            if minimize:
                opt_instance.set_min_objective(objective)
            else:
                opt_instance.set_max_objective(objective)
            opt_instance.set_ftol_rel(self.ftol_rel)
            opt_instance.set_lower_bounds(self.grid_boundary[::2])
            opt_instance.set_upper_bounds(self.grid_boundary[1::2])
            setattr(self, opt_name, opt_instance)
        else:
            opt_instance = getattr(self, opt_name)
        return opt_instance

    def _field(self, x, grad):
        if grad.size > 0:
            grad[:] = self._gradient(x, 'B')
        return self.interpolate('B').ev(*x).item()

    def _flux(self, x, grad):
        if grad.size > 0:
            grad[:] = self._gradient(x, 'Psi')
        return self.interpolate('Psi').ev(*x).item()

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
        return [self.interpolate(attribute).ev(*x, dx=1).item(),
                self.interpolate(attribute).ev(*x, dy=1).item()]

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

    def _global_null(self, plot=False):
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

        Returns
        -------
        None.

        """
        self._Xpoint, self._Opoint = [], []  # (re)initialize null point arrays
        Bthreshold = np.quantile(self.B*self.x2d, 0.05)
        index = self.B*self.x2d < Bthreshold  # threshold
        if np.sum(index) > 0:  # protect against uniform zero field
            xt, zt = self.x2d[index], self.z2d[index]  # threshold points
            eps = 1.25 * np.max([self.dx, self.dz])  # max neighbour seperation
            dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=1)
            cluster_index = dbscan.fit_predict(np.array([xt, zt]).T)
            xc, zc = [], []
            for i in range(np.max(cluster_index)+1):
                # cluster coordinates
                x_cluster = xt[cluster_index == i]
                z_cluster = zt[cluster_index == i]
                # coordinates of cluster centre
                xc.append(np.mean(x_cluster))
                zc.append(np.mean(z_cluster))
                # resolve local field null
                xn, zn = self.local_null((xc[-1], zc[-1]))
                if self.null_type((xn, zn)) == 'X':
                    self._Xpoint.append([xn, zn])
                else:  # Opoint - refine using local flux gradient search
                    concave = self._flux_curvature((xn, zn))[0] < 0
                    xn_opt, zn_opt = self.local_Opoint((xn, zn), concave)
                    dx = np.sqrt((xn_opt-xn)**2 + (zn_opt-zn)**2)
                    if dx < np.max(
                            [np.max(x_cluster) - np.min(x_cluster),
                             np.max(z_cluster) - np.min(z_cluster)]):
                        self._Opoint.append([xn_opt, zn_opt])
                    else:
                        self._Opoint.append([xn, zn])
            # convert to unique np.arrays
            self._Xpoint = geom.unique2D(self._Xpoint, eps=1e-3,
                                         bound=self.grid_boundary)[1]
            self._Opoint = geom.unique2D(self._Opoint, eps=1e-3,
                                         bound=self.grid_boundary)[1]
            if plot:
                plt.plot(xt, zt, '.', color='darkgray')  # threshold points
                plt.plot(xc, zc, 'k.')  # cluster centers
                plt.plot(*self._Xpoint.T, 'C0X')  # Xponits
                plt.plot(*self._Opoint.T, 'C1o')  # Opoints

    def _sort_flux(self, null_type):
        """
        Sort null arrays by poloidal flux.

        Parameters
        ----------
        null_type : str
            null type, X or O.

        Raises
        ------
        IndexError
            null_type not in [X, O].

        Returns
        -------
        None.

        """
        if null_type not in ['X', 'O']:
            raise IndexError(f'null_type {null_type} not in [X, O]')
        Psi = []
        Xnull = getattr(self, f'_{null_type}point')
        for i in range(len(Xnull)):
            Psi.append(self.interpolate('Psi').ev(*Xnull[i]))
        Psi = np.array(Psi)
        sort_index = np.argsort(Psi)
        setattr(self, f'_{null_type}psi', Psi[sort_index])
        setattr(self, f'_{null_type}point', Xnull[sort_index])

    def update_topology(self):
        """
        Perform global topology update.

        - Extract global null points.
        - Sort by null type (X or O).
        - poloidal flux at each null point.
        - Sort each null type array by poloidal flux.

        Returns
        -------
        None.

        """
        if self._update_topology:
            self._global_null()
            for null_type in ['X', 'O']:
                self._sort_flux(null_type)
            self._update_topology = False

    @property
    def Xpoint(self):
        """
        Return Xpoints sorted by poloidal flux.

        Returns
        -------
        Xpoint : ndarray, shape(nX, 2)
            Xpoints sorted by poloidal flux.

        """
        self.update_topology()
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
        self.update_topology()
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
        self.update_topology()
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
        self.update_topology()
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
        self.update_topology()
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
        self.update_topology()
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
