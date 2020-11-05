import itertools

import pygmo as pg
import shapely.geometry
import numpy as np
import scipy.optimize
import scipy.interpolate
import skimage.measure
import nlopt
import pandas as pd

from nova.utilities.pyplot import plt
from nova.utilities import geom
from nova.electromagnetic.meshgrid import MeshGrid


class TopologyError(Exception):
    """Raise topology error."""


class Topology:
    """
    Extract topology from poloidal flux and field maps.

    Abstract base class inherited by Grid.

    """

    _interpolate_attributes = ['Psi', 'B']

    _topology_attributes = ['polarity', 'Opoint', 'Opsi', 'Xpoint', 'Xpsi']

    def __init__(self):
        """Extend biot attributes."""
        self._biot_attributes += [
            attribute_pair for attribute in self._interpolate_attributes
            for attribute_pair in [f'_{attribute}_spline',
                                   f'_update_{attribute}_spline']]
        self._biot_attributes += [
            attribute_pair for attribute in self._topology_attributes
            for attribute_pair in [f'_{attribute}', f'_update_{attribute}']]
        self._default_biot_attributes.update(
            {f'_update_{attribute}_spline': True
             for attribute in self._interpolate_attributes})

    def _flag_update(self, status):
        if status:
            for attribute in self._interpolate_attributes:
                setattr(self, f'_update_{attribute}_spline', True)
            for attribute in self._topology_attributes:
                setattr(self, f'_update_{attribute}', True)

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
            contours[i] = np.array([self._x(idx[:, 0]), self._z(idx[:, 1])]).T
        if plot:
            if ax is None:
                ax = plt.gca()
            for contour in contours:
                plt.plot(contour[:, 0], contour[:, 1], **kwargs)
        return contours

    def set_points(self, n=50, plot=False):
        mg = MeshGrid(n, self.grid_boundary)[-4:]
        '''
        x, z, nx, nz
        mg = MeshGrid(self.n, self.grid_boundary)  # set mesh
        self.n2d = [mg.nx, mg.nz]  # shape
        self.x, self.z = mg.x, mg.z  # axes
        # trace index interpolators
        self._x = interp1d(range(self.n2d[0]), self.x)
        self._z = interp1d(range(self.n2d[1]), self.z)
        # grid deltas
        self.dx = np.diff(self.grid_boundary[:2])[0] / (mg.nx - 1)
        self.dz = np.diff(self.grid_boundary[2:])[0] / (mg.nz - 1)
        # 2d coordinates
        self.x2d = mg.x2d
        self.z2d = mg.z2d


        self.point_grid = {'x': x, 'z': z}
        n = nx * nz
        dtype = [('x', float), ('z', float), ('B', float), ('psi', float),
                 ('area', float), ('polygon', np.ndarray),
                 ('separatrix_area', float), ('separatrix_psi', float),
                 ('psi_norm', float)]
        self.points = np.zeros(n, dtype=dtype)
        index = itertools.count(0)
        for x in self.point_grid['x']:
            for z in self.point_grid['z']:
                i = next(index)
                res = minimize(self.Bpoint_abs, [x, z], method='Nelder-Mead')
                self.points[i]['x'] = res.x[0]
                self.points[i]['z'] = res.x[1]
                self.points[i]['B'] = res.fun
                self.points[i]['psi'] = self.Ppoint([res.x[0], res.x[1]])
        index = geom.unique2D(np.array([self.points['x'], self.points['z']]).T,
                              eps=0.1, bound=self.limit.flatten())[0]
        self.points = self.points[index]  # trim duplicates / edge points
        self.get_bounding_loops()  # get bounding loops
        self.set_Mpoints()  # set Mpoints
        self.set_Xpoints()  # set Xpoints
        if plot:
            self.plot_set_points()
            self.plot_bounding_loops()
        '''

    def set_Mpoints(self):
        npoints = len(self.points)
        self.Mindex = np.zeros(npoints, dtype=bool)  # select index
        for i, point in enumerate(self.points):
            p = shapely.geometry.Point([point['x'], point['z']])
            for loop in self.bounding_loops['exterior']:
                if p.within(loop['poly']):
                    self.Mindex[i] = True
                    self.points['separatrix_area'][i] = loop['area']
                    self.points['separatrix_psi'][i] = loop['psi']
        if sum(self.Mindex) == 0:  # select via interior loop centroid
            nloops = len(self.bounding_loops['exterior'])
            self.points = np.append(self.points,
                                    np.zeros(nloops, dtype=self.points.dtype))
            self.Mindex = np.append(self.Mindex,
                                    np.zeros(nloops, dtype=self.Mindex.dtype))
            for i, (loop_ex, loop_in) in\
                    enumerate(zip(self.bounding_loops['exterior'],
                                  self.bounding_loops['interior'])):
                self.Mindex[i+npoints] = True
                x, z = loop_in['poly'].centroid.xy
                x, z = x[0], z[0]
                self.points[i+npoints]['x'] = x
                self.points[i+npoints]['z'] = z
                self.points[i+npoints]['B'] = self.Bpoint_abs([x, z])
                self.points[i+npoints]['psi'] = self.Ppoint([x, z])
                self.points['separatrix_area'][i+npoints] = loop_ex['area']
                self.points['separatrix_psi'][i+npoints] = loop_ex['psi']
        if sum(self.Mindex) > 0:  # found points within exterior loops
            self.Mpoints = self.points[self.Mindex]
            self.Mpoints = np.sort(self.Mpoints, order='separatrix_area')[::-1]
            self.mo_array = []
            for point in self.Mpoints:
                self.mo_array.append([point['x'], point['z']])
            self.mo = self.mo_array[0]
            if len(self.points) == npoints:  # found via interior points
                self.get_Mpsi(plot=False)
            else:
                self.Mpsi = self.Mpoints[0]['psi']
                self.Mpoint = np.array([self.Mpoints[0]['x'],
                                        self.Mpoints[0]['z']])
        else:  # Mpoints not found
            self.Mpsi = None
            self.Mpoint = None
            self.mo = None

    def set_Xpoints(self):
        if sum(self.Mindex) > 0 and sum(~self.Mindex) > 0 and \
                self.Xloc != 'edge':  # found points
            Xpsi = self.Mpoints['separatrix_psi'][0]  # Xpsi estimate
            self.points['psi_norm'] =\
                (self.points['psi'] - self.Mpsi) / (Xpsi - self.Mpsi)
            self.Xpoints = self.points[~self.Mindex]
            self.Xpoints = np.sort(self.Xpoints, order='psi_norm')
            self.po_array = []
            for Xpoint in self.Xpoints:
                self.po_array.append([Xpoint['x'], Xpoint['z']])
            if len(self.po_array) > 0:
                self.po = self.po_array[0]
            else:
                self.po = None
            if self.po:
                self.Xpsi, self.Xpoint, self.Xpsi_array, self.Xpoint_array,\
                    self.Xloc = self.get_Xpsi()
            self.update_psi_norm()
        elif self.Xloc == 'edge':
            # self.Xloc = 'edge'
            self.Xpsi = self.bounding_loops['exterior']['psi'][0]
            self.get_boundary(alpha=1, set_boundary=True, boundary_cntr=False)
            self.Xpsi_array = np.array([self.Xpsi])
            edge = shapely.geometry.Polygon(
                [(self.limit[0, 0], self.limit[1, 0]),
                 (self.limit[0, 1], self.limit[1, 0]),
                 (self.limit[0, 1], self.limit[1, 1]),
                 (self.limit[0, 0], self.limit[1, 1])])
            edge_ring = shapely.geometry.LinearRing(edge.exterior.coords)
            xloop = self.bounding_loops['exterior']['x'][0]
            zloop = self.bounding_loops['exterior']['z'][0]
            dx = []
            for x, z in zip(xloop, zloop):  # find loop-edge intersect
                point = shapely.geometry.Point(x, z)
                d = edge_ring.project(point)
                p = np.array(list(edge_ring.interpolate(d).coords)[0])
                dx.append(np.sqrt((x - p[0])**2 + (z - p[1])**2))
            index = np.argmin(dx)  # intersection
            self.Xpoint = np.array([xloop[index], zloop[index]])
            self.Xpoint_array = [self.Xpoint]
            self.update_Xpoints()
        else:
            self.Xpsi = None
            self.Xpoint = None
            self.Xpoint_array = [self.Xpoint]
            self.update_Xpoints()
        if self.fw_limit:
            self.Xloc = 'first_wall'
            self.set_Plimit()
            self.Xpsi_array[0] = self.Xpsi
            self.Xpoint_array[0] = self.Xpoint
            self.update_Xpoints()
            self.update_psi_norm()

    def update_Xpoints(self):
        self.Xpoints = np.zeros(1, dtype=self.points.dtype)
        if self.Xpoint is not None:
            self.Xpoints['x'][0] = self.Xpoint[0]
            self.Xpoints['z'][0] = self.Xpoint[1]
        self.Xpoints['psi'][0] = self.Xpsi

    def update_psi_norm(self):
        self.points['psi_norm'] =\
            (self.points['psi'] - self.Mpsi) / (self.Xpsi - self.Mpsi)
        self.Mpoints['psi_norm'] =\
            (self.Mpoints['psi'] - self.Mpsi) / (self.Xpsi - self.Mpsi)
        self.Xpoints['psi_norm'] =\
            (self.Xpoints['psi'] - self.Mpsi) / (self.Xpsi - self.Mpsi)

    def get_bounding_loops(self, n_std=3, n=150,
                           loops_per_level=5, plot=False):
        delta_psi = n_std * np.std(self.psi)
        levels = np.mean(self.psi) + np.linspace(-delta_psi, delta_psi, n)
        dtype = [('valid', bool), ('closed', bool), ('area', float),
                 ('x', np.ndarray), ('z', np.ndarray), ('poly', np.ndarray),
                 ('psi', float)]
        loops = np.zeros(n * loops_per_level, dtype=dtype)  # inital sizing
        k = 0
        contours = self.get_contour(levels)
        for i, psi_line in enumerate(contours):
            for j, line in enumerate(psi_line):
                x, z = line[:, 0], line[:, 1]
                if len(x) > 2:  # loop condition
                    loop_length = geom.length(x, z, norm=False)[-1]
                    gap = np.sqrt((x[0]-x[-1])**2 + (z[0]-z[-1])**2)
                    closed = gap < 0.01*loop_length  # loop is closed
                    if closed:
                        poly = shapely.geometry.Polygon(np.array([x, z]).T)
                        loops[k]['valid'] = True
                        loops[k]['x'] = x
                        loops[k]['z'] = z
                        loops[k]['area'] = poly.area
                        loops[k]['poly'] = poly
                        loops[k]['psi'] = levels[i]
                        k += 1
                        if k >= n * loops_per_level:
                            errtxt = 'loops array undersized\n'
                            errtxt += 'increase number of loops per level'
                            IndexError(errtxt)
        loops = loops[:k]
        self.bounding_loops = {}  # return interior and exterior loops
        for select in ['interior', 'exterior']:
            self.bounding_loops[select] = self.select_loops(loops, select)
        if plot:
            self.plot_bounding_loops()

    def plot_bounding_loops(self):
        for select, ls in zip(self.bounding_loops, ['C6--', 'C6-']):
            for loop in self.bounding_loops[select]:
                plt.plot(loop['x'], loop['z'], ls, lw=2)

    def select_loops(self, loops, select):
        loops = np.copy(loops)
        n = len(loops)
        for i in range(n):  # trim interior loops
            if loops[i]['valid']:
                for j in range(n):
                    if i != j and loops[j]['valid']:
                        if loops[i]['poly'].contains(loops[j]['poly']):
                            if select == 'exterior':
                                loops[j]['valid'] = False
                            elif select == 'interior':
                                loops[i]['valid'] = False
                            else:
                                errtxt = 'select must equal ''exterior'''
                                errtxt += ' or ''interior'''
                                raise ValueError(errtxt)
        loops = loops[loops['valid']]
        return loops

    def plot_set_points(self, ax=None, plot_grid=False):
        if ax is None:
            ax = plt.gca()
        if plot_grid:
            for x in self.point_grid['x']:
                for z in self.point_grid['z']:
                    ax.plot(x, z, 'C1d', ms=2)
        if self.Xpsi is not None and self.Mpsi is not None:
            for point in self.Mpoints:
                self.label_point(ax, point)
            for point in self.Xpoints:
                self.label_point(ax, point)
            for po in self.po_array:
                ax.plot(po[0], po[1], 'o', ms=10, color='gray', alpha=0.5)
            for mo in self.mo_array:
                ax.plot(mo[0], mo[1], 'o', ms=10, color='gray', alpha=0.5)
            self.plot_nulls(labels=['X', 'M'])
            plt.text(self.mo[0], self.mo[1], 'mo  ', ha='right', va='center')
            ax.text(self.po[0], self.po[1], 'xo  ', ha='right', va='center')

#####

    @property
    def polarity(self):
        """
        Return plasma current polarity.

        Returns
        -------
        polarity: int
            Plasma current polarity.

        """
        if self._update_polarity:
            self._polarity = self.source.coilframe.Ip_sign
            self._update_polarity = False
        return self._polarity

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
        # principal curvatures
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

    def _signed_flux(self, x):
        return -1 * self.polarity * self.interpolate('Psi').ev(*x)

    def _signed_flux_gradient(self, x):
        return -1 * self.polarity * np.array(
            [self.interpolate('Psi').ev(*x, dx=1),
             self.interpolate('Psi').ev(*x, dy=1)])

    def get_Opoint(self, xo=None):
        """
        Return coordinates of plasma O-point.

        O-point defined as center of nested flux surfaces.

        Parameters
        ----------
        xo : array-like(float), shape(2,), optional
            Sead coordinates (x, z). The default is None.

            - None: xo set to grid center

        Raises
        ------
        TopologyError
            Failed to find signed flux minimum.

        Returns
        -------
        Opoint, array-like(float), shape(2,)
            Coordinates of O-point.

        """
        if xo is None:
            xo = self.bounds.mean(axis=1)
        res = scipy.optimize.minimize(
            self._signed_flux, xo,
            jac=self._signed_flux_gradient, bounds=self.bounds)
        if not res.success:
            raise TopologyError('Opoint signed flux minimization failure\n\n'
                                f'{res}.')
        return res.x

    @property
    def Opoint(self):
        """
        Return coordinates of center of nested flux surfaces.

        Returns
        -------
        tuple
            Plasma O-point coordinates (x, z).

        """
        if self._update_Opoint or self._Opoint is None:
            self._Opoint = self.get_Opoint(xo=self._Opoint)
            self._update_Opoint = False
        return self._Opoint

    @property
    def Opsi(self):
        """
        Return poloidal flux calculated at O-point.

        Returns
        -------
        Opsi: float
            O-point poloidal flux.

        """
        if self._update_Opsi:
            self._Opsi = float(self.interpolate('Psi').ev(*self.Opoint))
            self._update_Opsi = False
        return self._Opsi

    def _field_null(self, x, grad):
        if grad.size > 0:
            grad[:] = self._field_gradient(x)
        return self.interpolate('B').ev(*x).item()

    def _field_gradient(self, x):
        return  [self.interpolate('B').ev(*x, dx=1),
                         self.interpolate('B').ev(*x, dy=1)]

    def get_Xpoint(self, xo):
        """
        Return X-point coordinates.

        Resolve X-point location based on solution of field minimum in
        proximity to sead location, *xo*.

        Parameters
        ----------
        xo : array-like(float), shape(2,)
            Sead coordinates (x, z).

        Raises
        ------
        TopologyError
            Field minimization failure.

        Returns
        -------
        Xpoint: array-like(float), shape(2,)
            X-point coordinates (x, z).

        """

        """
        opt = nlopt.opt(nlopt.G_MLSL_LDS, 2)
        local = nlopt.opt(nlopt.LD_MMA, 2)
        '''
        local.set_ftol_rel(1e-4)
        local.set_min_objective(self._field_null)
        local.set_lower_bounds([4, -4])
        local.set_upper_bounds([8, 4])
        '''

        opt.set_local_optimizer(local)
        opt.set_min_objective(self._field_null)
        opt.set_ftol_rel(1e-4)
        opt.set_maxeval(50)
        # grid limits
        opt.set_lower_bounds([4, -4])
        opt.set_upper_bounds([8, 4])

        opt.set_population(2)

        x = opt.optimize(xo)

        print(opt)

        #print(self.grid_boundary[1::2])
        #print(x)
        """
        opt = nlopt.opt(nlopt.LD_MMA, 2)
        opt.set_min_objective(self._field_null)
        opt.set_ftol_rel(1e-6)
        opt.set_lower_bounds(self.grid_boundary[::2])
        opt.set_upper_bounds(self.grid_boundary[1::2])
        x = opt.optimize(xo)

        """
        res = scipy.optimize.minimize(
            self._field_null, xo, jac=self._field_gradient,
            # bounds=self.bounds,
            )
        if not res.success:
            raise TopologyError('Xpoint signed |B| minimization failure\n\n'
                                f'{res}.')
        return res.x
        """
        return x

    @property
    def Xpoint(self):
        """
        Manage Xpoint locations.

        Parameters
        ----------
        xo : array-like, shape(n, 2)
            Sead Xpoints.

        Returns
        -------
        Xpoint: ndarray, shape(2)
            Coordinates of primary Xpoint (x, z).

        """
        if self._update_Xpoint or self._Xpoint is None:
            if self._Xpoint is None:  # sead with boundary midsides
                bounds = self.bounds
                self.Xpoint = [[np.mean(bounds[0]), bounds[1][i]]
                               for i in range(2)]
            nX = len(self._Xpoint)
            _Xpoint = np.zeros((nX, 2))
            _Xpsi = np.zeros(nX)
            for i in range(nX):
                _Xpoint[i] = self.get_Xpoint(self._Xpoint[i])
                _Xpsi[i] = self.interpolate('Psi').ev(*_Xpoint[i])
            self._Xpoint = _Xpoint[np.argsort(_Xpsi)]
            if self.source.coilframe.Ip_sign > 0:
                self._Xpoint = self._Xpoint[::-1]
            self._update_Xpoint = False
        return self._Xpoint[0]

    @Xpoint.setter
    def Xpoint(self, xo):
        if not isinstance(xo, np.ndarray):
            xo = np.array(xo)
        if xo.ndim == 1:
            xo = xo.reshape(1, -1)
        if xo.shape[1] != 2:
            raise IndexError(f'shape(xo) {xo.shape} not (n, 2)')
        self._Xpoint = xo


class Stencil:

    def __init__(self, grid_boundary):
        self.grid_boundary = grid_boundary

    def get_bounds(self):
        return (self.grid_boundary[::2], self.grid_boundary[1::2])


class FieldNull(Stencil):

    def __init__(self, grid_boundary, interpolate):
        Stencil.__init__(self, grid_boundary)
        self.interpolate = interpolate

    def fitness(self, x):
        return [self.interpolate('B').ev(*x)]

    def get_nobj(self):
        return 1

    def gradient(self, x):
        return [self.interpolate('B').ev(*x, dx=1), self.interpolate('B').ev(*x, dy=1)]


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    build = True
    filename = 'Xtest'
    cs = CoilSet()
    if build:
        polygon = shapely.geometry.Point(5, 1).buffer(0.5)
        cs.add_plasma(polygon, dPlasma=0.1)
        cs.add_coil(5, -0.5, 0.75, 0.75, dCoil=0.2)
        cs.plasmagrid.generate_grid(expand=1, n=2e4)  # generate plasma grid
        cs.Ic = 15e6 * np.ones(2)
        cs.save_coilset(filename)
    else:
        cs.load_coilset(filename)

    #nl = pg.nlopt('slsqp')
    #nl.xtol_rel = 1E-6




    '''
    algo = pg.algorithm(pg.nlopt('mma'))
    algo.set_verbosity(1)
    prob = pg.problem(FieldNull(cs.plasmagrid.grid_boundary,
                                cs.plasmagrid.interpolate('B'),
                                cs.plasmagrid.interpolate('Psi')))


    #prob.c_tol = [1E-6]  # Set constraints tolerance to 1E-6
    #def ev():
    pop = pg.population(prob, 50)


    pop = algo.evolve(pop)


    plt.figure()
    cs.plot(True)
    cs.plasmagrid.plot_flux()

    plt.plot(*pop.get_x().T, 'o')
    plt.plot(*pop.champion_x, 'X')
    '''

