import numpy as np
from amigo.pyplot import plt
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import InterpolatedUnivariateSpline as sinterp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import nova.geqdsk
from matplotlib._cntr import Cntr as cntr
from collections import OrderedDict
from amigo import geom
from amigo.IO import trim_dir
from amigo.geom import loop_vol, grid
import nova
from amigo.IO import class_dir
from os.path import join
from warnings import warn
from itertools import count
from shapely.geometry import Polygon, Point, LineString
from nova.exceptions import TopologyError
from amigo.geom import poly_inloop


class SF(object):

    def __init__(self, **kwargs):
        self.shape = {}
        self.set_kwargs(kwargs)
        self.load_eqdsk()

    def set_kwargs(self, kwargs):
        self.fw_limit = False  # default
        self.fw_label = 'first_wall'
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def load_eqdsk(self):
        if hasattr(self, 'filename'):  # initalise from file
            self.eqdsk = nova.geqdsk.read(self.filename)
            self.normalise()  # unit normalisation
        if not hasattr(self, 'eqdsk'):
            self.eqdsk = {}  # initalise as empty
        self.update_eqdsk(self.eqdsk)

    def update_eqdsk(self, eqdsk):
        self.set_flux_functions(eqdsk)  # calculate flux profiles
        self.set_TF(eqdsk)  # TF magnetic moment
        self.set_firstwall(eqdsk)  # first wall profile
        self.set_psi(eqdsk)  # flux map + nulls

    def set_psi(self, eqdsk):  # was set_plasma
        required_keys = ['x', 'z', 'psi']
        optional_keys = ['Ipl', 'beta_p', 'Li']
        self.fw_limit = eqdsk.get('fw_limit', self.fw_limit)
        if np.array([key in eqdsk for key in required_keys]).all():
            for key in required_keys:
                setattr(self, key, eqdsk[key])
            for key in optional_keys:
                if key in eqdsk:
                    setattr(self, key, eqdsk[key])
            self.trim_x(xmin=0.5)  # trim zero x-coordinate entries
            self.meshgrid()  # store grid
            self.Pfield()
            self.Bfield()  # compute Bfield from psi input
            self.check_firstwall(eqdsk)  # check for firstwall update
            self.set_contour()  # compute flux map
            self.set_points(n=50, plot=False)  # set X and M points
            self.initalize_field(plot=False)

    def check_firstwall(self, eqdsk):
        optional_keys = ['xlim', 'zlim']
        if np.array([key in eqdsk for key in optional_keys]).all():
            self.set_firstwall(eqdsk)  # reset
        else:  # define as flux boundary
            self.fw_label = 'edge'
            limit = [[eqdsk['x'][0], eqdsk['x'][-1]],
                     [eqdsk['z'][0], eqdsk['z'][-1]]]
            xlim = [limit[0][0], limit[0][1], limit[0][1], limit[0][0]]
            zlim = [limit[1][0], limit[1][0], limit[1][1], limit[1][1]]
            xlim, zlim = geom.xzInterp(xlim, zlim)
            self.set_firstwall({'xlim': xlim, 'zlim': zlim})

    def initalize_field(self, plot=False):
        if self.Xpoint is not None and self.Mpoint is not None:
            try:
                self.get_midplane()
                self.get_boundary(alpha=1-1e-4, set_boundary=True,
                                  plot=plot)
            except TopologyError:
                pass
            except NotImplementedError:
                pass
            try:
                self.get_sol(debug=False)
            except ValueError:
                pass
            except AttributeError:
                pass

    def get_sol(self, plot=False, debug=False):
        if hasattr(self, 'fw'):
            # check that x-point is inside first wall
            if Polygon(self.fw).intersects(Point(self.Xpoint)):
                self.initalize_sol(rcirc=0.2, drcirc=0.15)
                self.sol(plot=plot, debug=debug)

    def initalize_sol(self, rcirc=0.3, drcirc=0.15):
        self.get_sol_psi(dSOL=3e-3, Nsol=15, verbose=False)
        self.rcirc = rcirc * abs(self.Mpoint[1] - self.Xpoint[1])  # radius
        self.drcirc = drcirc * self.rcirc  # leg search width

    def normalise(self):
        if ('Fiesta' in self.eqdsk['name'] or 'Nova' in self.eqdsk['name'] or
                'disr' in self.eqdsk['name'] or 'DINA' in self.eqdsk['name']):
            self.norm = 1
        elif 'tosca' in self.eqdsk['header'].lower() or\
                'TEQ' in self.eqdsk['name']:
            self.eqdsk['Ipl'] *= -1
            self.eqdsk['It'] *= -1
            self.norm = 1
        else:  # CREATE
            self.eqdsk['Ipl'] *= -1
            self.norm = 2*np.pi
        if self.norm != 1:
            for key in ['psi', 'simagx', 'sibdry']:
                self.eqdsk[key] /= self.norm  # Webber/loop to Webber/radian
            for key in ['ffprim', 'pprime']:
                # []/(Webber/loop) to []/(Webber/radian)
                self.eqdsk[key] *= self.norm
        self.b_scale = 1  # flux function scaling

    def trim_x(self, xmin=0.5):
        if self.x[0] < xmin:  # trim zero x-coordinate entries
            i = np.argmin(abs(self.x - xmin))
            self.x = self.x[i:]
            self.psi = self.psi[i:, :]

    def eqwrite(self, pf, CREATE=False, prefix='Nova', config=''):
        if len(config) > 0:
            name = prefix + '_' + config
        else:
            name = prefix
        if CREATE:  # save with create units (Webber/loop, negated Iplasma)
            name = 'CREATE_format_' + name
            norm = 2 * np.pi  # reformat: webber/loop
            Ip_dir = -1  # reformat: reverse plasma current
            psi_offset = self.Xpsi()  # reformat: boundary psi=0
        else:
            norm, Ip_dir, psi_offset = 1, 1, 0  # no change
        nc, xc, zc, dxc, dzc, It = pf.unpack_coils()[:-1]
        psi_ff = np.linspace(0, 1, self.nx)
        pad = np.zeros(self.nx)
        eq = {'name': name,
              # Number of horizontal and vertical points
              'nx': self.nx, 'ny': self.nz,
              'x': self.x, 'z': self.z,  # Location of the grid-points
              'xdim': self.x[-1] - self.x[0],  # Size of the domain in meters
              'zdim': self.z[-1] - self.z[0],  # Size of the domain in meters
              # Reference vacuum toroidal field (m, T)
              'xcentr': self.eqdsk['xcentr'],
              # Reference vacuum toroidal field (m, T)
              'bcentr': self.eqdsk['bcentr'],
              'xgrid1': self.x[0],  # X of left side of domain
              # Z at the middle of the domain
              'zmid': self.z[0] + (self.z[-1] - self.z[0]) / 2,
              'xmagx': self.Mpoint[0],  # Location of magnetic axis
              'zmagx': self.Mpoint[1],  # Location of magnetic axis
              # Poloidal flux at the axis (Weber / rad)
              'simagx': float(self.Mpsi) * norm,
              # Poloidal flux at plasma boundary (Weber / rad)
              'sibdry': self.Xpsi * norm,
              'Ipl': self.eqdsk['Ipl'] * Ip_dir,
              # Poloidal flux in Weber/rad on grid points
              'psi': (np.transpose(self.psi).reshape((-1,)) -
                      psi_offset) * norm,
              # Poloidal current function on uniform flux grid
              'fpol': self.Fpsi(psi_ff),
              # "FF'(psi) in (mT)^2/(Weber/rad) on uniform flux grid"
              'ffprim': self.b_scale * self.FFprime(psi_ff) / norm,
              # "P'(psi) in (N/m2)/(Weber/rad) on uniform flux grid"
              'pprime': self.b_scale * self.Pprime(psi_ff) / norm,
              'pressure': pad,  # Plasma pressure in N/m^2 on uniform flux grid
              'qpsi': pad,  # q values on uniform flux grid
              # Plasma boundary
              'nbdry': self.nbdry, 'xbdry': self.xbdry, 'zbdry': self.zbdry,
              # first wall
              'nlim': self.nlim, 'xlim': self.xlim, 'zlim': self.zlim,
              'ncoil': nc, 'xc': xc, 'zc': zc, 'dxc': dxc,
              'dzc': dzc, 'It': It}  # coils

        eqdir = trim_dir('../../eqdsk')
        filename = eqdir + '/' + config + '.eqdsk'
        print('writing eqdsk', filename)
        nova.geqdsk.write(filename, eq)

    def write_flux(self):
        psi_norm = np.linspace(0, 1, self.nx)
        pprime = self.b_scale * self.Pprime(psi_norm)
        FFprime = self.b_scale * self.FFprime(psi_norm)
        with open('../Data/' + self.dataname + '_flux.txt', 'w') as f:
            f.write(
                'psi_norm\tp\' [Pa/(Weber/rad)]\tFF\' [(mT)^2/(Weber/rad)]\n')
            for psi, p_, FF_ in zip(psi_norm, pprime, FFprime):
                f.write('{:1.4f}\t\t{:1.4f}\t\t{:1.4f}\n'.format(psi, p_, FF_))

    def set_flux_functions(self, eqdsk):
        required_keys = ['fpol', 'pressure', 'ffprim', 'pprime']
        if np.array([key in eqdsk for key in required_keys]).all():
            F_ff = eqdsk['fpol']
            P_ff = eqdsk['pressure']
            n = len(F_ff)
            psi_ff = np.linspace(0, 1, n)
            F_ff = interp1d(psi_ff, F_ff)(psi_ff)
            P_ff = interp1d(psi_ff, P_ff)(psi_ff)
            dF_ff = np.gradient(F_ff, 1 / (n - 1))
            dFF_ff = np.gradient(F_ff**2, 1 / (n - 1))
            dP_ff = np.gradient(P_ff, 1 / (n - 1))
            self.Fpsi = interp1d(psi_ff, F_ff)
            self.Ppsi = interp1d(psi_ff, P_ff)
            self.dFpsi = interp1d(psi_ff, dF_ff)
            self.dFFpsi = interp1d(psi_ff, dFF_ff)
            self.dPpsi = interp1d(psi_ff, dP_ff)
            FFp = spline(psi_ff, eqdsk['ffprim'], s=1e-5)(psi_ff)
            Pp = spline(psi_ff, eqdsk['pprime'], s=1e2)(psi_ff)  # s=1e5
            self.FFprime = interp1d(psi_ff, FFp, fill_value=0,
                                    bounds_error=False)
            self.Pprime = interp1d(psi_ff, Pp, fill_value=0,
                                   bounds_error=False)

    def plot_flux_functions(self):
        ax = plt.subplots(2, 1, sharex=True)[1]
        psi = np.linspace(0, 1, 51)
        ax[0].plot(psi, self.Pprime(psi))
        ax[1].plot(psi, self.FFprime(psi))
        ax[0].set_ylabel('$P''(\Psi)$')
        ax[1].set_ylabel('$F''(\Psi)$')
        ax[1].set_xlabel('$\Psi''$')

    def set_TF(self, eqdsk):
        required_keys = ['xcentr', 'bcentr']
        if np.array([key in eqdsk for key in required_keys]).all():
            for key in required_keys:
                setattr(self, key, eqdsk[key])

    def set_boundary(self, x, z, n=5e2):
        self.nbdry = int(n)
        self.xbdry, self.zbdry = geom.xzSLine(x, z, npoints=n)

    def set_firstwall(self, eqdsk):
        required_keys = ['xlim', 'zlim']
        if np.array([key in eqdsk for key in required_keys]).all():
            xlim = eqdsk['xlim']
            zlim = eqdsk['zlim']
            L = geom.length(xlim, zlim, norm=False)
            dL = np.min(np.diff(L))  # minimum space interpolation
            self.nlim = int(L[-1]/dL)
            self.xlim, self.zlim = geom.xzInterp(xlim, zlim, self.nlim)
            fw_list = [(x, z) for x, z in zip(self.xlim, self.zlim)]
            self.fw = LineString(fw_list)  # shapley fw linestring

    def get_Plimit(self, plot=True, **kwargs):
        xlim = kwargs.get('xlim', self.xlim)
        zlim = kwargs.get('zlim', self.zlim)
        loop = {'x': self.xbdry, 'z': self.zbdry}
        points = {'x': xlim, 'z': zlim}
        xlim, zlim = poly_inloop(loop, points, plot=False)
        if xlim is not None and zlim is not None:  # found intersection
            psi = np.zeros(len(xlim))
            for i, (x, z) in enumerate(zip(xlim, zlim)):  # psi_norm
                psi[i] = (self.Ppoint((x, z))-self.Xpsi)
                psi[i] /= (self.Xpsi-self.Mpsi)
            FWindex = np.argmin(psi)
            FWpoint = np.array([xlim[FWindex], zlim[FWindex]])
            FWpsi = psi[FWindex]
        else:
            FWpoint = None
            FWpsi = None
        if plot:
            self.contour()
            plt.plot(xlim, zlim, 'C3')
            plt.plot(FWpoint[0], FWpoint[1], 'X')
        return FWpoint, FWpsi

    def set_Plimit(self, plot=False):
        self.get_boundary(alpha=1, set_boundary=True)
        poly = Polygon(np.array([self.xbdry, self.zbdry]).T)
        FWlimit = poly.intersects(self.fw)  # Point(FWpoint)
        if FWlimit:  # plasma limited by first wall
            FWpoint, psi = self.get_Plimit(plot=plot)
            self.Xpsi = psi
            self.Xpsi = self.Ppoint(FWpoint)
            self.Xpoint = FWpoint
            self.Xloc = self.fw_label

    def upsample(self, sample):
        if sample > 1:
            '''
            EQ(self,n=sample*self.n)
            self.space()
            '''
            from scipy.interpolate import RectBivariateSpline as rbs
            sample = np.int(np.float(sample))
            interp_psi = rbs(self.x, self.z, self.psi)
            self.nx, self.nz = sample * self.nx, sample * self.nz
            self.x = np.linspace(self.x[0], self.x[-1], self.nx)
            self.z = np.linspace(self.z[0], self.z[-1], self.nz)
            self.psi = interp_psi(self.x, self.z, dx=0, dy=0)
            self.space()

    def meshgrid(self):
        self.nx = len(self.x)
        self.nz = len(self.z)
        self.n = self.nx * self.nz
        self.dx = (self.x[-1] - self.x[0]) / (self.nx - 1)
        self.dz = (self.z[-1] - self.z[0]) / (self.nz - 1)
        self.delta = np.sqrt(self.dx**2 + self.dz**2)
        self.x2d, self.z2d = np.meshgrid(self.x, self.z, indexing='ij')
        self.limit = np.array([[self.x[0], self.x[-1]],
                               [self.z[0], self.z[-1]]])

    def Pfield(self):
        self.Pspline = RectBivariateSpline(self.x, self.z, self.psi)

    def Bfield(self):
        psi_x, psi_z = np.gradient(self.psi, self.dx, self.dz)
        xm = np.array(np.matrix(self.x).T * np.ones([1, self.nz]))
        xm[xm == 0] = 1e-34
        self.Bx = -psi_z / xm
        self.Bz = psi_x / xm
        self.Babs = np.sqrt(self.Bx**2 + self.Bz**2)
        self.Bmap()  # generate smooth field maps

    def bound_point(self, point):
        for j, bound in enumerate(self.limit):
            if point[j] < bound[0]:
                point[j] = bound[0]
            elif point[j] > bound[1]:
                point[j] = bound[1]
        return point

    def Bmap(self):
        self.Bspline = [[], [], []]
        self.Bspline[0] = RectBivariateSpline(self.x, self.z, self.Bx)
        self.Bspline[1] = RectBivariateSpline(self.x, self.z, self.Bz)
        self.Bspline[2] = RectBivariateSpline(self.x, self.z, self.Babs)

    def Bpoint(self, point, check_bounds=False):  # magnetic field at point
        field = np.zeros(2)  # function re-name (was Bcoil)
        if check_bounds:
            inbound = point[0] >= np.min(self.x) and\
                point[0] <= np.max(self.x) \
                and point[1] >= np.min(self.z) and point[1] <= np.max(self.z)
            return inbound
        else:
            point = self.bound_point(point)
            for i in range(2):
                field[i] = self.Bspline[i].ev(point[0], point[1])
            return field

    def Bpoint_abs(self, point):  # absolute polidal field
        point = self.bound_point(point)
        Babs = self.Bspline[2].ev(point[0], point[1])
        return Babs

    def Bpoint_jac(self, point):  # absolute polidal field
        dB = np.zeros(2)
        point = self.bound_point(point)
        dB[0] = self.Bspline[2].ev(point[0], point[1], dx=1, dy=0)
        dB[1] = self.Bspline[2].ev(point[0], point[1], dx=0, dy=1)
        return dB

    def set_points(self, n=50, plot=False):
        x, z, nx, nz = grid(n, self.limit.flatten())[-4:]
        self.point_grid = {'x': x, 'z': z}
        n = nx * nz
        dtype = [('x', float), ('z', float), ('B', float), ('psi', float),
                 ('area', float), ('polygon', np.ndarray),
                 ('separatrix_area', float), ('separatrix_psi', float),
                 ('psi_norm', float)]
        self.points = np.zeros(n, dtype=dtype)
        index = count(0)
        for x in self.point_grid['x']:
            for z in self.point_grid['z']:
                i = next(index)
                res = minimize(self.Bpoint_abs, [x, z], jac=self.Bpoint_jac,
                               method='L-BFGS-B', bounds=self.limit)
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

    def set_Mpoints(self):
        npoints = len(self.points)
        self.Mindex = np.zeros(npoints, dtype=bool)  # select index
        for i, point in enumerate(self.points):
            p = Point([point['x'], point['z']])
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
        if sum(self.Mindex) > 0 and sum(~self.Mindex) > 0:  # found points
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
            if self.fw_limit:
                self.set_Plimit()
                self.Xpoint_array[0] = self.Xpoint
                self.Xpsi_array[0] = self.Xpsi
                self.Xpoints['x'][0] = self.Xpoint[0]
                self.Xpoints['z'][0] = self.Xpoint[1]
                self.Xpoints['psi'][0] = self.Xpsi

            # update psi norm
            self.points['psi_norm'] =\
                (self.points['psi'] - self.Mpsi) / (self.Xpsi - self.Mpsi)
            self.Mpoints['psi_norm'] =\
                (self.Mpoints['psi'] - self.Mpsi) / (self.Xpsi - self.Mpsi)
            self.Xpoints['psi_norm'] =\
                (self.Xpoints['psi'] - self.Mpsi) / (self.Xpsi - self.Mpsi)
        else:
            self.Xpsi = None
            self.Xpoint = None
            self.Xpoint_array = None
            self.po = None
            self.Xloc = None

    def label_point(self, ax, point):
        txt = '  psi_norm: {:1.3f}\n'.format(point['psi_norm'])
        txt += '  B: {:1.3f}'.format(point['B'])
        ax.text(point['x'], point['z'], txt, va='center')

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

    def minimum_field(self, radius, theta):
        X = radius * np.sin(theta) + self.Xpoint[0]
        Z = radius * np.cos(theta) + self.Xpoint[1]
        B = np.zeros(len(X))
        for i, (x, z) in enumerate(zip(X, Z)):
            field = self.Bpoint((x, z))
            B[i] = np.sqrt(field[0]**2 + field[1]**2)
        return np.argmin(B)

    def Ppoint(self, point):
        psi = self.Pspline.ev(point[0], point[1])
        return psi

    def plot_firstwall(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.xlim, self.zlim, 'gray', lw=0.75)

    def plot_separatrix(self, select='both', ax=None):
        if ax is None:
            ax = plt.gca()
        if select == 'both':
            select = ['upper', 'lower']
        else:
            select = [select]
        for select_ in select:
            Xpsi = self.get_Xpsi(select=select_)[0]
            alpha = 1-1e-5
            Spsi = alpha * (Xpsi - self.Mpsi) + self.Mpsi
            psi_line = self.get_contour([Spsi], boundary=False)[0]
            for line in psi_line:
                x, z = line[:, 0], line[:, 1]
                ax.plot(x, z, 'C4', alpha=0.75, zorder=-10)
                
    def get_levels(self, Nlevel=31, Nstd=4, **kwargs):
        level, n = [-Nstd * np.std(self.psi),
                    Nstd * np.std(self.psi)], Nlevel
        levels = np.mean(self.psi) + np.linspace(level[0], level[1], n)
        return levels

    def contour(self, Nlevel=31, Nstd=4, Xnorm=True, lw=1, plot_vac=True,
                boundary=True, separatrix='', plot_points=False, **kwargs):
        if self.Xpsi is None:
            boundary = False
            Xnorm = False
            separatrix = ''
        ax = kwargs.get('ax', plt.gca())
        alpha = np.array([1, 1], dtype=float)
        lw = lw * np.array([2.25, 1.75])
        if boundary:
            x, z = self.get_boundary(alpha=1-1e-4)
            ax.plot(x, z, linewidth=lw[0], color=0.75 * np.ones(3))
            self.set_boundary(x, z)
        if separatrix:
            self.plot_separatrix(select=separatrix, ax=ax)
        if 'levels' not in kwargs.keys():
            levels = self.get_levels(Nlevel=Nlevel, Nstd=Nstd)
            linetype = '-'
        else:
            levels = kwargs['levels']
            linetype = '-'
        color = kwargs.get('color', 'k')
        if 'linetype' in kwargs.keys():
            linetype = kwargs['linetype']
        if color == 'k':
            alpha *= 0.25
        if Xnorm:
            levels = levels + self.Xpsi
        contours = self.get_contour(levels)
        for psi_line, level in zip(contours, levels):
            if Xnorm:
                level = level - self.Xpsi
            for line in psi_line:
                x, z = line[:, 0], line[:, 1]
                if self.inPlasma(x, z) and boundary:
                    pindex = 0
                else:
                    pindex = 1
                if (not plot_vac and pindex == 0) or plot_vac:
                    ax.plot(x, z, linetype, linewidth=lw[pindex],
                            color=color, alpha=alpha[pindex])
        if boundary:
            ax.plot(self.xbdry, self.zbdry, linetype, linewidth=lw[0],
                    color=color, alpha=alpha[0])
        ax.axis('equal')
        ax.axis('off')
        if Xnorm:
            levels = levels - self.Xpsi
        if plot_points:
            self.plot_set_points(ax=ax)
        return levels

    def inPlasma(self, X, Z, delta=0):
        if np.array([hasattr(self, key) for key in ['xbdry', 'zbdry']]).all():
            inside = X.min() >= self.xbdry.min() - delta and \
                     X.max() <= self.xbdry.max() + delta and \
                     Z.min() >= self.zbdry.min() - delta and \
                     Z.max() <= self.zbdry.max() + delta
        else:
            inside = False
        return inside

    def Bcontour(self, axis, Nstd=1.5, color='k'):
        var = 'B' + axis
        if not hasattr(self, var):
            self.Bfield()
        B = getattr(self, var)
        level = [np.mean(B) - Nstd * np.std(B),
                 np.mean(B) + Nstd * np.std(B)]
        CS = plt.contour(self.x2d, self.z2d, B,
                         levels=np.linspace(level[0], level[1], 30),
                         colors=color)
        for cs in CS.collections:
            cs.set_linestyle('solid')

    def Bquiver(self):
        if not hasattr(self, 'Bx'):
            self.Bfield()
        plt.quiver(self.x, self.z, self.Bx.T, self.Bz.T)

    def Bsf(self):
        if not hasattr(self, 'Bx'):
            self.Bfield()
        plt.streamplot(self.x, self.z, self.Bx.T, self.Bz.T,
                       color=self.Bx.T, cmap=plt.cm.RdBu)
        plt.clim([-1.5, 1.5])

    def getX(self, po=None):
        res = minimize(self.Bpoint_abs, np.array(po), jac=self.Bpoint_jac,
                       method='L-BFGS-B', bounds=self.limit)
        return res.x

    def get_Xpsi(self, select='primary', plot=False):
        n_po = len(self.po_array)
        Xpoint = np.zeros((n_po, 2))
        Xpsi = np.zeros(n_po)
        for i, po in enumerate(self.po_array):
            Xpoint[i, :] = self.getX(po=po)
            Xpsi[i] = self.Ppoint(Xpoint[i, :])
        if select == 'primary':
            i = 0  # use prior Xpoint sort
            if Xpoint[i, 1] < self.Mpoint[1]:
                Xloc = 'lower'
            else:
                Xloc = 'upper'
        else:
            Xloc = select
            index = np.argsort(Xpoint[:, 1])
            Xpoint = Xpoint[index, :]
            Xpsi = Xpsi[index]
            if select == 'lower':
                i = 0  # lower Xpoint
            elif select == 'upper':
                i = 1  # upper Xpoint
            else:
                IndexError('select not in [''primary'', ''lower'', ''upper'']')
        if len(Xpsi) > 1:
            self.Xerr = Xpsi[1] - Xpsi[0]  # for double-null
        else:
            self.Xerr = None
        Xpsi_array = Xpsi
        Xpoint_array = Xpoint
        Xpsi = Xpsi[i]
        Xpoint = Xpoint[i]
        if plot:
            self.plot_nulls(labels=['X'])
        return Xpsi, Xpoint, Xpsi_array, Xpoint_array, Xloc

    def getM(self, mo=None):
        if mo is None:
            mo = self.mo
        res = minimize(self.Bpoint_abs, np.array(mo), jac=self.Bpoint_jac,
                       method='L-BFGS-B', bounds=self.limit)
        return res.x

    def plot_nulls(self, labels=['X', 'M'], ax=None):
        if ax is None:
            ax = plt.gca()
        for label in labels:
            if label == 'M':
                ax.plot(self.Mpoint[0], self.Mpoint[1], 'C3*', ms=10,
                        zorder=10, label='Mpoint')
            elif label == 'X':
                for i, (Xpoint, Xpsi) in enumerate(zip(self.Xpoint_array,
                                                       self.Xpsi_array)):
                    if Xpsi:
                        if i == 0:
                            if self.Xloc in ['edge', 'first_wall']:
                                txt = self.Xloc
                            else:
                                txt = 'P'
                        else:
                            txt = 'S{}'.format(i)
                        alpha = 1 if i == 0 else 0.25
                        if Xpsi:
                            label_txt = 'Xpoint-{}'.format(txt)
                            ax.plot(Xpoint[0], Xpoint[1], 'C0*', alpha=alpha,
                                    ms=10, zorder=10, label=label_txt)
        ax.legend(loc=1)

    def get_Mpsi(self, mo=None, plot=False):
        self.Mpoint = self.getM(mo=mo)
        self.Mpsi = float(self.Ppoint(self.Mpoint))
        if plot:
            self.plot_nulls(labels=['M'])
        return (self.Mpsi, self.Mpoint)

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
                        poly = Polygon(np.array([x, z]).T)
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

    def remove_contour(self):
        for key in ['cfield', 'cfield_bndry']:
            if hasattr(self, key):
                delattr(self, key)

    def set_contour(self):
        flux = self.Ppoint((self.limit[0][1], np.mean(self.limit[1])))
        if flux > 0:
            psi_boundary = np.max(self.psi)
        else:
            psi_boundary = np.min(self.psi)
        psi_bndry = np.pad(self.psi[1:-1, 1:-1], (1,),
                           mode='constant', constant_values=psi_boundary)
        self.cfield_bndry = cntr(self.x2d, self.z2d, psi_bndry)
        self.cfield = cntr(self.x2d, self.z2d, self.psi)

    def get_contour(self, levels, boundary=False):
        if boundary:
            def cfield(level): return self.cfield_bndry.trace(level, level, 0)
        else:
            def cfield(level): return self.cfield.trace(level, level, 0)
        lines = []
        for level in levels:
            psi_line = cfield(level)
            psi_line = psi_line[:len(psi_line) // 2]
            lines.append(psi_line)
        return lines

    def get_boundary(self, plot=False, set_boundary=False, **kwargs):
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
            Spsi = alpha * (self.Xpsi - self.Mpsi) + self.Mpsi
        elif 'psi' in kwargs:
            Spsi = kwargs['psi']
        else:
            raise ValueError('requires target alpha or psi')
        boundary = kwargs.get('boundary', True)
        psi_line = self.get_contour([Spsi], boundary=boundary)[0]
        X, Z = [], []
        for line in psi_line:
            x, z = line[:, 0], line[:, 1]
            if self.Xloc == 'lower':  # lower Xpoint
                index = z >= self.Xpoint[1]
            elif self.Xloc == 'upper':  # upper Xpoint
                index = z <= self.Xpoint[1]
            elif self.Xloc in ['edge', 'first_wall']:  # limited
                index = np.ones(len(x), dtype=bool)
            if sum(index) > 0:
                x, z = x[index], z[index]
                loop = np.sqrt((x[0] - x[-1])**2 +
                               (z[0] - z[-1])**2) < 3*self.delta
                # (z > self.Mpoint[1]).any() and\
                # (z < self.Mpoint[1]).any() and
                if loop:
                    X.append(x)
                    Z.append(z)
        try:
            point = Point(self.Mpoint)
            index = False
            for i, (x, z) in enumerate(zip(X, Z)):
                if len(x) > 2:
                    poly = Polygon(np.array([x, z]).T)
                    if point.within(poly):
                        index = i
                        break
            if index is not False:
                X, Z = X[index], Z[index]
            else:
                '''
                for i, (x, z) in enumerate(zip(X, Z)):
                    plt.plot(x, z, 'C6')
                plt.plot(self.Mpoint[0], self.Mpoint[1], 'C7o')
                '''
                raise TopologyError('Mpoint not found within separatrix')
            
        except ValueError:
            for x, z in zip(X, Z):
                plt.plot(x, z, 'C3')
                plt.plot(self.Xpoint[0], self.Xpoint[1], 'o')
                plt.plot(self.Mpoint[0], self.Mpoint[1], 'o')
            try:
                self.contour(boundary=False)
            except AttributeError:
                warn('unable to draw contour - *bdry not set')
                pass
            err_txt = 'failed to find contour'
            err_txt += '\nseperatrix open'
            err_txt += '\ncheck X-point definition'
            raise ValueError(err_txt)
        except IndexError:
            plt.figure()
            try:
                self.contour(boundary=False)
                warn('plot boundary index error')
            except AttributeError:
                warn('unable to draw contour - *bdry not set')
                pass
            err_txt = '\ngeom.clock failed to find contour'
            err_txt += '\nseperatrix open'
            err_txt += '\ncheck X-point definition'
            raise IndexError(err_txt)
        if set_boundary:
            self.set_boundary(X, Z)
        if plot:
            plt.plot(X, Z)
        return X, Z

    def eq_boundary(self, expand=0):  # generate boundary dict for elliptic
        X, Z = self.get_boundary(alpha=1-1e-4)
        boundary = {'X': X, 'Z': Z, 'expand': expand}
        return boundary

    def get_midplane(self, alpha=1-1e-4):  # rename, was get_LFP
        x, z = self.get_boundary(alpha=alpha)
        poly = Polygon(np.array([x, z]).T)
        point = Point(self.Mpoint)
        self.midplane = {}
        if point.within(poly):  # Mpoint within separatrix
            dx = np.diff(self.limit)[0]
            LFline = LineString([self.Mpoint, self.Mpoint + np.array([dx, 0])])
            HFline = LineString([self.Mpoint, self.Mpoint - np.array([dx, 0])])
            LFx, LFz = poly.intersection(LFline).coords[1]
            HFx, HFz = poly.intersection(HFline).coords[1]
            self.midplane['LF'] = {'x': LFx, 'z': LFz}
            self.midplane['HF'] = {'x': HFx, 'z': HFz}
            self.shape['R'] = np.mean([self.midplane['HF']['x'],
                                       self.midplane['LF']['x']])
            self.shape['a'] = (self.midplane['LF']['x'] -
                               self.midplane['HF']['x']) / 2
            self.shape['AR'] = self.shape['R'] / self.shape['a']
        else:
            plt.plot(x, z)  # separatrix
            plt.plot(self.Mpoint[0], self.Mpoint[1], 'o')
            self.contour()
            raise TopologyError('Mpoint located outside separatrix')
        return

    def first_wall_psi(self, trim=True, single_contour=False, **kwargs):
        if 'point' in kwargs:
            xeq, zeq = kwargs.get('point')
            psi = self.Ppoint([xeq, zeq])
        else:
            xeq, zeq = self.LFPx, self.LFPz
            if 'psi_n' in kwargs:  # normalized psi
                psi_n = kwargs.get('psi_n')
                psi = psi_n * (self.Xpsi - self.Mpsi) + self.Mpsi
            elif 'psi' in kwargs:
                psi = kwargs.get('psi')
            else:
                raise ValueError('set point=(x,z) or psi in kwargs')
        contours = self.get_contour([psi])
        X, Z = self.pick_contour(contours, Xpoint=False)
        if single_contour:
            min_contour = np.empty(len(X))
            for i in range(len(X)):
                min_contour[i] = np.min((X[i] - xeq)**2 + (Z[i] - zeq)**2)
            imin = np.argmin(min_contour)
            x, z = X[imin], Z[imin]
        else:
            x, z = np.array([]), np.array([])
            for i in range(len(X)):
                x = np.append(x, X[i])
                z = np.append(z, Z[i])
        if trim:
            if self.Xloc == 'lower':
                x, z = x[z <= zeq], z[z <= zeq]
            elif self.Xloc == 'upper':
                x, z = x[z >= zeq], z[z >= zeq]
            else:
                raise ValueError('Xloc not set (get_Xpsi)')
            if xeq > self.Xpoint[0]:
                x, z = x[x > self.Xpoint[0]], z[x > self.Xpoint[0]]
            else:
                x, z = x[x < self.Xpoint[0]], z[x < self.Xpoint[0]]
            istart = np.argmin((x - xeq)**2 + (z - zeq)**2)
            x = np.append(x[istart + 1:], x[:istart])
            z = np.append(z[istart + 1:], z[:istart])
        istart = np.argmin((x - xeq)**2 + (z - zeq)**2)
        if istart > 0:
            x, z = x[::-1], z[::-1]
        return x, z, psi

    def firstwall_loop(self, plot=False, **kwargs):
        if 'psi_n' in kwargs:
            x, z, psi = self.first_wall_psi(psi_n=kwargs['psi_n'], trim=False)
            psi_lfs = psi_hfs = psi
        elif 'dx' in kwargs:  # geometric offset
            dx = kwargs.get('dx')
            LFfwr, LFfwz = self.LFPx + dx, self.LFPz
            HFfwr, HFfwz = self.HFPx - dx, self.HFPz
            x_lfs, z_lfs, psi_lfs = self.first_wall_psi(point=(LFfwr, LFfwz))
            x_hfs, z_hfs, psi_hfs = self.first_wall_psi(point=(HFfwr, HFfwz))
            x_top, z_top = self.get_offset(dx)
            if self.Xloc == 'lower':
                x_top, z_top = geom.theta_sort(x_top, z_top, po=self.po,
                                               origin='top')
                index = z_top >= self.LFPz
            else:
                x_top, z_top = geom.theta_sort(x_top, z_top, po=self.po,
                                               origin='bottom')
                index = z_top <= self.LFPz
            x_top, z_top = x_top[index], z_top[index]
            istart = np.argmin((x_top - HFfwr)**2 + (z_top - HFfwz)**2)
            if istart > 0:
                x_top, z_top = x_top[::-1], z_top[::-1]
            x = np.append(x_hfs[::-1], x_top)
            x = np.append(x, x_lfs)
            z = np.append(z_hfs[::-1], z_top)
            z = np.append(z, z_lfs)
        else:
            errtxt = 'requre \'psi_n\' or \'dx\' in kwargs'
            raise ValueError(errtxt)
        if plot:
            plt.plot(x, z)
        return x[::-1], z[::-1], (psi_lfs, psi_hfs)

    def get_offset(self, dx, Nsub=0):
        rpl, zpl = self.get_boundary()  # boundary points
        rpl, zpl = geom.offset(rpl, zpl, dx)  # offset from sep
        if Nsub > 0:  # sub-sample
            rpl, zpl = geom.xzSLine(rpl, zpl, Nsub)
        return rpl, zpl

    def midplane_loop(self, x, z):
        index = np.argmin((x - self.LFPx)**2 + (z - self.LFPz)**2)
        if z[index] <= self.LFPz:
            index -= 1
        x = np.append(x[:index + 1][::-1], x[index:][::-1])
        z = np.append(z[:index + 1][::-1], z[index:][::-1])
        L = geom.length(x, z)
        index = np.append(np.diff(L) != 0, True)
        x, z = x[index], z[index]  # remove duplicates
        return x, z

    def get_sol_psi(self, verbose=False, **kwargs):
        for var in ['dSOL', 'Nsol']:
            if var in kwargs:
                setattr(self, var, kwargs[var])
        if verbose:
            print('calculating sol psi', self.Nsol, self.dSOL)
        self.Dsol = np.linspace(0, self.dSOL, self.Nsol)
        x = self.midplane['LF']['x'] + self.Dsol
        z = self.midplane['LF']['z'] * np.ones(len(x))
        self.sol_psi = np.zeros(len(x))
        for i, (rp, zp) in enumerate(zip(x, z)):
            self.sol_psi[i] = self.Ppoint([rp, zp])

    def upsample_sol(self, nmult=10, debug=True):
        k = 1  # smoothing factor
        for i, (x, z) in enumerate(zip(self.Xsol, self.Zsol)):
            le = geom.length(x, z)
            L = np.linspace(0, 1, nmult * len(le))
            self.Xsol[i] = sinterp(le, x, k=k)(L)
            self.Zsol[i] = sinterp(le, z, k=k)(L)
            if debug:
                plt.plot(self.Xsol[i], self.Zsol[i])

    def sol(self, dx=15e-3, Nsol=5, plot=False, update=False, debug=False):
        if update or not hasattr(self, 'sol_psi') or dx > self.dSOL\
                or Nsol > self.Nsol:
            self.get_sol_psi(dSOL=dx, Nsol=Nsol)  # re-calculcate LFP
        elif (Nsol > 0 and Nsol != self.Nsol) or \
                (dx > 0 and dx != self.dSOL):  # update
            if dx > 0:
                self.dSOL = dx
            if Nsol > 0:
                self.Nsol = Nsol
            Dsol = np.linspace(0, self.dSOL, self.Nsol)
            self.sol_psi = interp1d(self.Dsol, self.sol_psi)(Dsol)
            self.Dsol = Dsol
        if self.Xloc in ['lower', 'upper']:
            contours = self.get_contour(self.sol_psi)
            self.Xsol, self.Zsol = \
                self.pick_contour(contours, Xpoint=True,
                                  Midplane=False, Plasma=False)
            self.upsample_sol(nmult=4, debug=debug)  # upsample
            self.get_legs(debug=debug)
            self.trim_legs()  # trim legs to first wall
            if plot:
                self.plot_sol(core=True)

    def plot_sol(self, core=False, ax=None):
        if hasattr(self, 'legs'):
            if ax is None:
                ax = plt.gca()
            for c, leg in enumerate(self.legs):
                if core or 'core' not in leg:
                    for i in np.arange(self.legs[leg]['i'])[::-1]:
                        x, z = self.snip(leg, i)
                        x, z = self.legs[leg]['X'][i], self.legs[leg]['Z'][i]
                        ax.plot(x, z, color='C0', linewidth=0.5)

    def trim_legs(self):
        self.targets = {}
        for c, leg in enumerate(self.legs):
            if 'core' not in leg:
                self.targets[leg] = {}
                self.legs[leg]['strike'] = \
                    np.zeros((self.legs[leg]['i'], 2))
                for i in np.arange(self.legs[leg]['i'])[::-1]:
                    line = LineString([(x, z) for x, z in
                                       zip(self.legs[leg]['X'][i],
                                           self.legs[leg]['Z'][i])])
                    if not self.fw.intersects(line):
                        break
                    point = self.fw.intersection(line)
                    if point.type == 'MultiPoint':
                        point = Point([point[0].x, point[0].y])
                    index = np.argmin((self.legs[leg]['X'][i] - point.x)**2 +
                                      (self.legs[leg]['Z'][i] - point.y)**2)
                    self.legs[leg]['X'][i] = \
                        np.append(self.legs[leg]['X'][i][:index], point.x)
                    self.legs[leg]['Z'][i] = \
                        np.append(self.legs[leg]['Z'][i][:index], point.y)
                    self.legs[leg]['strike'][i] = [point.x, point.y]

                L2D = geom.length(self.legs[leg]['X'][0],
                                  self.legs[leg]['Z'][0], norm=False)[-1]
                self.targets[leg]['L2D'] = L2D
                self.targets[leg]['strike'] = self.legs[leg]['strike'][0]

    def add_core(self):  # refarance from low-field midplane
        for i in range(self.Nsol):
            for leg in ['inner', 'inner1', 'inner2', 'outer',
                        'outer1', 'outer2']:
                if leg in self.legs:
                    if 'inner' in leg:
                        core = 'core1'
                    else:
                        core = 'core2'
                    Xc = self.legs[core]['X'][i][:-1]
                    Zc = self.legs[core]['Z'][i][:-1]
                    self.legs[leg]['X'][i] = np.append(
                        Xc, self.legs[leg]['X'][i])
                    self.legs[leg]['Z'][i] = np.append(
                        Zc, self.legs[leg]['Z'][i])

    def orientate(self, X, Z):
        if X[-1] > X[0]:  # counter clockwise
            X = X[::-1]
            Z = Z[::-1]
        return X, Z

    def pick_contour(self, contours, Xpoint=False,
                     Midplane=True, Plasma=False):
        Xs = []
        Zs = []
        Xp, Mid, Pl = True, True, True
        for psi_line in contours:
            for line in psi_line:
                X, Z = line[:, 0], line[:, 1]
                if Xpoint:  # check Xpoint proximity
                    rX = np.sqrt(
                        (X - self.Xpoint[0])**2 + (Z - self.Xpoint[1])**2)
                    if (min(rX) < self.rcirc):
                        Xp = True
                    else:
                        Xp = False
                if Midplane:  # check lf midplane crossing
                    if (np.max(Z) > self.LFPz) and (np.min(Z) < self.LFPz):
                        Mid = True
                    else:
                        Mid = False
                if Plasma:
                    if (np.max(X) < np.max(self.xbdry)) and\
                        (np.min(X) > np.min(self.xbdry)) and\
                        (np.max(Z) < np.max(self.zbdry)) and\
                            (np.min(Z) > np.min(self.zbdry)):
                        Pl = True
                    else:
                        Pl = False
                if Xp and Mid and Pl:
                    X, Z = self.orientate(X, Z)
                    Xs.append(X)
                    Zs.append(Z)
        return Xs, Zs

    def topolar(self, X, Z):
        x, y = X, Z
        r = np.sqrt((x - self.Xpoint[0])**2 + (y - self.Xpoint[1])**2)
        if self.Xloc == 'lower':
            t = np.arctan2(x - self.Xpoint[0], y - self.Xpoint[1])
        elif self.Xloc == 'upper':
            t = np.arctan2(x - self.Xpoint[0], self.Xpoint[1] - y)
        else:
            raise ValueError('Xloc not set (get_Xpsi)')
        return r, t

    def store_leg(self, rloop, tloop):
        if np.argmin(rloop) > len(rloop) / 2:  # point legs out
            rloop, tloop = rloop[::-1], tloop[::-1]
        ncirc = np.argmin(abs(rloop - self.rcirc))
        tID = np.argmin(abs(tloop[ncirc] - self.tleg))
        legID = self.tID[tID]
        if self.nleg == 6:
            if legID <= 1:
                label = 'inner' + str(legID + 1)
            elif legID >= 4:
                label = 'outer' + str(legID - 3)
            elif legID == 2:
                label = 'core1'
            elif legID == 3:
                label = 'core2'
            else:
                label = ''
        else:
            if legID == 0:
                label = 'inner'
            elif legID == 3:
                label = 'outer'
            elif legID == 1:
                label = 'core1'
            elif legID == 2:
                label = 'core2'
            else:
                label = ''
        if label:
            i = self.legs[label]['i']
            X = rloop * np.sin(tloop) + self.Xpoint[0]
            if self.Xloc == 'lower':
                Z = rloop * np.cos(tloop) + self.Xpoint[1]
            elif self.Xloc == 'upper':
                Z = -rloop * np.cos(tloop) + self.Xpoint[1]
            else:
                raise ValueError('Xloc not set (get_Xpsi)')
            if i > 0:
                if X[0]**2 + Z[0]**2 == (self.legs[label]['X'][i - 1][0]**2 +
                                         self.legs[label]['Z'][i - 1][0]**2):
                    i -= 1
            if 'core' in label:
                X, Z = X[::-1], Z[::-1]
            self.legs[label]['X'][i] = X
            self.legs[label]['Z'][i] = Z
            self.legs[label]['i'] = i + 1

    def min_L2D(self, targets):
        L2D = np.zeros(len(targets.keys()))
        for i, target in enumerate(targets.keys()):
            L2D[i] = targets[target]['L2D'][0]
        return L2D.min()

    def check_legs(self):
        if self.sf.z.min() > self.sf.Xpoint[1] - self.sf.rcirc:
            print('grid out of bounds')

    def get_legs(self, debug=False):
        if debug:
            theta = np.linspace(-np.pi, np.pi, 100)
            x = (self.rcirc - self.drcirc / 2) * np.cos(theta)
            z = (self.rcirc - self.drcirc / 2) * np.sin(theta)
            plt.plot(x + self.Xpoint[0], z + self.Xpoint[1],
                     'k--', alpha=0.5)
            x = (self.rcirc + self.drcirc / 2) * np.cos(theta)
            z = (self.rcirc + self.drcirc / 2) * np.sin(theta)
            plt.plot(x + self.Xpoint[0], z + self.Xpoint[1],
                     'k--', alpha=0.5)
        self.tleg = np.array([])
        for N in range(len(self.Xsol)):
            x, t = self.topolar(self.Xsol[N], self.Zsol[N])
            index = (x > self.rcirc - self.drcirc /
                     2) & (x < self.rcirc + self.drcirc / 2)
            self.tleg = np.append(self.tleg, t[index])
        nbin = 50
        nhist, bins = np.histogram(self.tleg, bins=nbin)
        flag, self.nleg, self.tleg = 0, 0, np.array([])
        for i in range(len(nhist)):
            if nhist[i] > 0:
                if flag == 0:
                    tstart = bins[i]
                    tend = bins[i]
                    flag = 1
                if flag == 1:
                    tend = bins[i]
            elif flag == 1:
                self.tleg = np.append(self.tleg, (tstart + tend) / 2)
                self.nleg += 1
                flag = 0
            else:
                flag = 0
        if nhist[-1] > 0:
            tend = bins[-1]
            self.tleg = np.append(self.tleg, (tstart + tend) / 2)
            self.nleg += 1
        if self.nleg == 6:  # snow-flake
            self.legs = {
                'inner1': {'X': [[] for i in range(self.Nsol)],
                           'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'inner2': {'X': [[] for i in range(self.Nsol)],
                           'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'outer1': {'X': [[] for i in range(self.Nsol)],
                           'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'outer2': {'X': [[] for i in range(self.Nsol)],
                           'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'core1': {'X': [[] for i in range(self.Nsol)],
                          'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'core2': {'X': [[] for i in range(self.Nsol)],
                          'Z': [[] for i in range(self.Nsol)], 'i': 0}}
        else:
            self.legs = {
                'inner': {'X': [[] for i in range(self.Nsol)],
                          'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'outer': {'X': [[] for i in range(self.Nsol)],
                          'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'core1': {'X': [[] for i in range(self.Nsol)],
                          'Z': [[] for i in range(self.Nsol)], 'i': 0},
                'core2': {'X': [[] for i in range(self.Nsol)],
                          'Z': [[] for i in range(self.Nsol)], 'i': 0}}
        self.legs = OrderedDict(sorted(self.legs.items(), key=lambda t: t[0]))
        if self.nleg == 0:
            err_txt = 'legs not found\n'
            self.contour()
            raise ValueError(err_txt)
        self.tID = np.arange(self.nleg)
        self.tID = np.append(self.nleg - 1, self.tID)
        self.tID = np.append(self.tID, 0)
        self.tleg = np.append(-np.pi - (np.pi - self.tleg[-1]), self.tleg)
        self.tleg = np.append(self.tleg, np.pi + (np.pi + self.tleg[1]))

        for N in range(len(self.Xsol)):
            ends, ro = [0, -1], np.zeros(2)
            for i in ends:
                ro[i] = np.sqrt(self.Xsol[N][i]**2 + self.Zsol[N][i]**2)
            x, t = self.topolar(self.Xsol[N], self.Zsol[N])
            post = False
            rpost, tpost = 0, 0
            if ro[0] == ro[-1]:  # cut loops
                if np.min(x * np.cos(t)) > self.drcirc - self.rcirc:
                    nmax = np.argmax(x * np.sin(t))  # LF
                else:
                    nmax = np.argmin(x * np.cos(t))  # minimum z
                x = np.append(x[nmax:], x[:nmax])
                t = np.append(t[nmax:], t[:nmax])
            while len(x) > 0:
                if x[0] > self.rcirc:
                    if np.min(x) < self.rcirc:
                        ncut = np.arange(len(x))[x < self.rcirc][0]
                        rloop, tloop = x[:ncut], t[:ncut]
                        loop = False
                    else:
                        ncut = -1
                        rloop, tloop = x, t
                        loop = True
                    if post:
                        rloop, tloop = np.append(
                            rpost, rloop), np.append(tpost, tloop)
                else:
                    ncut = np.arange(len(x))[x > self.rcirc][0]
                    xin, tin = x[:ncut], t[:ncut]
                    nx = self.minimum_field(xin, tin)  # minimum field
                    rpre, tpre = xin[:nx + 1], tin[:nx + 1]
                    rpost, tpost = xin[nx:], tin[nx:]
                    loop = True
                    post = True
                    rloop, tloop = np.append(
                        rloop, rpre), np.append(tloop, tpre)
                if loop:
                    if rloop[0] < self.rcirc and rloop[-1] < self.rcirc:
                        if np.min(rloop * np.cos(tloop)) >\
                                self.drcirc - self.rcirc:
                            nmax = np.argmax(rloop * np.sin(tloop))  # LF
                        else:
                            nmax = np.argmax(rloop)
                        self.store_leg(rloop[:nmax], tloop[:nmax])
                        self.store_leg(rloop[nmax:], tloop[nmax:])
                    else:
                        self.store_leg(rloop, tloop)
                if ncut == -1:
                    x, t = [], []
                else:
                    x, t = x[ncut:], t[ncut:]

    def strike_point(self, Xi, graze):
        ratio = np.sin(graze) * np.sqrt(Xi[-1]**2 + 1)
        if np.abs(ratio) > 1:
            theta = np.sign(ratio) * np.pi
        else:
            theta = np.arcsin(ratio)
        return theta

    def snip(self, leg, layer_index=0, L2D=0):
        if not hasattr(self, 'Xsol'):
            self.sol()
        Xsol = self.legs[leg]['X'][layer_index]
        Zsol = self.legs[leg]['Z'][layer_index]
        Lsol = geom.length(Xsol, Zsol, norm=False)
        if L2D == 0:
            L2D = Lsol[-1]
        if layer_index != 0:
            Xsolo = self.legs[leg]['X'][0]
            Zsolo = self.legs[leg]['Z'][0]
            Lsolo = geom.length(Xsolo, Zsolo, norm=False)
            indexo = np.argmin(np.abs(Lsolo - L2D))
            index = np.argmin((Xsol - Xsolo[indexo])**2 +
                              (Zsol - Zsolo[indexo])**2)
            L2D = Lsol[index]
        else:
            index = np.argmin(np.abs(Lsol - L2D))
        if Lsol[index] > L2D:
            index -= 1
        if L2D > Lsol[-1]:
            L2D = Lsol[-1]
            print('warning: targent requested outside grid')
        Xend, Zend = interp1d(Lsol, Xsol)(L2D), interp1d(Lsol, Zsol)(L2D)
        Xsol, Zsol = Xsol[:index], Zsol[:index]  # trim to strike point
        Xsol, Zsol = np.append(Xsol, Xend), np.append(Zsol, Zend)
        return (Xsol, Zsol)

    def pick_leg(self, leg, layer_index):
        X = self.legs[leg]['X'][layer_index]
        Z = self.legs[leg]['Z'][layer_index]
        return X, Z

    def Xtrim(self, Xsol, Zsol):
        Xindex = np.argmin((self.Xpoint[0] - Xsol)**2 +
                           (self.Xpoint[1] - Zsol)**2)
        if (Xsol[-1] - Xsol[Xindex])**2 + (Zsol[-1] - Zsol[Xindex])**2 <\
                (Xsol[0] - Xsol[Xindex])**2 + (Zsol[0] - Zsol[Xindex])**2:
            Xsol = Xsol[:Xindex]  # trim to Xpoints
            Zsol = Zsol[:Xindex]
            Xsol = Xsol[::-1]
            Zsol = Zsol[::-1]
        else:
            Xsol = Xsol[Xindex:]  # trim to Xpoints
            Zsol = Zsol[Xindex:]
        return (Xsol, Zsol)

    def get_graze(self, point, target):
        T = target / np.sqrt(target[0]**2 + target[1]**2)  # target vector
        B = self.Bpoint([point[0], point[1]])
        B /= np.sqrt(B[0]**2 + B[1]**2)  # poloidal field line vector
        theta = np.arccos(np.dot(B, T))
        if theta > np.pi / 2:
            theta = np.pi - theta
        Xi = self.expansion([point[0]], [point[1]])
        graze = np.arcsin(np.sin(theta) * (Xi[-1]**2 + 1)**-0.5)
        return graze

    def get_max_graze(self, x, z):
        theta = np.pi / 2  # normal target, maximum grazing angle
        Xi = self.expansion([x], [z])
        graze = np.arcsin(np.sin(theta) * (Xi[-1]**2 + 1)**-0.5)
        return graze

    def expansion(self, Xsol, Zsol):
        Xi = np.array([])
        Bm = np.abs(self.bcentr * self.xcentr)
        for x, z in zip(Xsol, Zsol):
            B = self.Bpoint([x, z])
            Bp = np.sqrt(B[0]**2 + B[1]**2)  # polodial field
            Bphi = Bm / x  # torodal field
            Xi = np.append(Xi, Bphi / Bp)  # field expansion
        return Xi

    def connection(self, leg, layer_index, L2D=0):
        if L2D > 0:  # trim targets to L2D
            Xsol, Zsol = self.snip(leg, layer_index, L2D)
        else:  # rb.trim_sol to trim to targets
            Xsol = self.legs[leg]['X'][layer_index]
            Zsol = self.legs[leg]['Z'][layer_index]
        Lsol = geom.length(Xsol, Zsol)
        index = np.append(np.diff(Lsol) != 0, True)
        Xsol, Zsol = Xsol[index], Zsol[index]  # remove duplicates
        if len(Xsol) < 2:
            L2D, L3D = [0], [0]
        else:
            dXsol = np.diff(Xsol)
            dZsol = np.diff(Zsol)
            L2D = np.append(0, np.cumsum(np.sqrt(dXsol**2 + dZsol**2)))
            dTsol = np.array([])
            Xi = self.expansion(Xsol, Zsol)
            for x, dx, dz, xi in zip(Xsol[1:], dXsol, dZsol, Xi):
                dLp = np.sqrt(dx**2 + dz**2)
                dLphi = xi * dLp
                dTsol = np.append(dTsol, dLphi / (x + dx / 2))
            L3D = np.append(0, np.cumsum(dTsol * np.sqrt((dXsol / dTsol)**2 +
                                                         (dZsol / dTsol)**2 +
                                                         (Xsol[:-1])**2)))
        return L2D, L3D, Xsol, Zsol

    def shape_parameters(self, plot=False):
        x95, z95 = self.get_boundary(alpha=0.95)
        ru = x95[np.argmax(z95)]  # triangularity
        rl = x95[np.argmin(z95)]
        self.shape['del_u'] = (self.shape['R'] - ru) / self.shape['a']
        self.shape['del_l'] = (self.shape['R'] - rl) / self.shape['a']
        self.shape['kappa'] = (np.max(z95) - np.min(z95)
                               ) / (2 * self.shape['a'])
        x, z = self.get_boundary(alpha=1)
        x, z = geom.clock(x, z, reverse=True)
        self.shape['V'] = loop_vol(x, z, plot=plot)
        return self.shape


if __name__ == '__main__':

    directory = join(class_dir(nova), '../eqdsk')
    sf = SF(filename=join(directory, 'DEMO_SN.eqdsk'), fw_limit=True)

    sf.contour()

    sf.plot_sol(core=True)
    sf.plot_firstwall()

