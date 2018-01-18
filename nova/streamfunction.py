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
from amigo.geom import loop_vol


class SF(object):

    def __init__(self, filename, upsample=1, **kwargs):
        self.shape = {}
        self.filename = filename
        self.set_kwargs(kwargs)
        self.eqdsk = nova.geqdsk.read(self.filename)
        self.normalise()  # unit normalisation
        self.set_plasma(self.eqdsk)
        self.set_boundary(self.eqdsk['xbdry'], self.eqdsk['zbdry'])
        self.set_flux(self.eqdsk)  # calculate flux profiles
        self.set_TF(self.eqdsk)
        self.set_current(self.eqdsk)
        xo_arg = np.argmin(self.zbdry)
        self.po = [self.xbdry[xo_arg], self.zbdry[xo_arg]]
        self.mo = [self.eqdsk['xmagx'], self.eqdsk['zmagx']]
        self.upsample(upsample)
        self.get_Xpsi()
        self.get_Mpsi()
        self.set_contour()  # set cfeild
        self.get_LFP()
        # self.get_sol_psi(dSOL=3e-3,Nsol=15,verbose=False)
        # leg search radius
        self.rcirc = 0.3 * abs(self.Mpoint[1] - self.Xpoint[1])
        self.drcirc = 0.15 * self.rcirc  # leg search width
        self.xlim = self.eqdsk['xlim']
        self.ylim = self.eqdsk['ylim']
        self.nlim = self.eqdsk['nlim']

    def set_kwargs(self, kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def normalise(self):
        if ('Fiesta' in self.eqdsk['name'] or 'Nova' in self.eqdsk['name'] or
            'disr' in self.eqdsk['name']) and\
                'CREATE' not in self.eqdsk['name']:
            self.norm = 1
        else:  # CREATE
            self.eqdsk['cpasma'] *= -1
            self.norm = 2 * np.pi
            for key in ['psi', 'simagx', 'sibdry']:
                self.eqdsk[key] /= self.norm  # Webber/loop to Webber/radian
            for key in ['ffprim', 'pprime']:
                # []/(Webber/loop) to []/(Webber/radian)
                self.eqdsk[key] *= self.norm
        self.b_scale = 1  # flux function scaling

    def trim_x(self, rmin=1.5):
        if self.x[0] == 0:  # trim zero x-coordinate entries
            i = np.argmin(abs(self.x - rmin))
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
            psi_offset = self.get_Xpsi()[0]  # reformat: boundary psi=0
        else:
            norm, Ip_dir, psi_offset = 1, 1, 0  # no change
        nc, xc, zc, dxc, dzc, Ic = pf.unpack_coils()[:-1]
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
              'cpasma': self.eqdsk['cpasma'] * Ip_dir,
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
              'nlim': self.nlim, 'xlim': self.xlim, 'ylim': self.ylim,
              'ncoil': nc, 'xc': xc, 'zc': zc, 'dxc': dxc,
              'dzc': dzc, 'Ic': Ic}  # coils

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

    def set_flux(self, eqdsk):
        F_ff = eqdsk['fpol']
        P_ff = eqdsk['pressure']
        n = len(F_ff)
        psi_ff = np.linspace(0, 1, n)
        F_ff = interp1d(psi_ff, F_ff)(psi_ff)
        P_ff = interp1d(psi_ff, P_ff)(psi_ff)
        dF_ff = np.gradient(F_ff, 1 / (n - 1))
        dP_ff = np.gradient(P_ff, 1 / (n - 1))
        self.Fpsi = interp1d(psi_ff, F_ff)
        self.dFpsi = interp1d(psi_ff, dF_ff)
        self.dPpsi = interp1d(psi_ff, dP_ff)
        FFp = spline(psi_ff, eqdsk['ffprim'], s=1e-5)(psi_ff)
        Pp = spline(psi_ff, eqdsk['pprime'], s=1e2)(psi_ff)  # s=1e5
        self.FFprime = interp1d(psi_ff, FFp, fill_value=0, bounds_error=False)
        self.Pprime = interp1d(psi_ff, Pp, fill_value=0, bounds_error=False)

    def set_TF(self, eqdsk):
        for key in ['xcentr', 'bcentr']:
            setattr(self, key, eqdsk[key])

    def set_boundary(self, x, z, n=5e2):
        self.nbdry = int(n)
        self.xbdry, self.zbdry = geom.rzSLine(x, z, npoints=n)

    def set_current(self, eqdsk):
        for key in ['cpasma']:
            setattr(self, key, eqdsk[key])

    def update_plasma(self, eq):  # update requres full separatrix
        for attr in ['Bspline', 'Pspline', 'Xpsi', 'Mpsi', 'Bx', 'Bz', 'LFPx']:
            if hasattr(self, attr):
                delattr(self, attr)
        self.set_plasma(eq)
        self.get_Xpsi()
        self.get_Mpsi()
        self.set_contour()  # calculate cfeild
        self.get_LFP()
        x, z = self.get_boundary()
        self.set_boundary(x, z)
        # self.get_Plimit()  # limit plasma extent
        # self.get_sol_psi()  # re-calculate sol_psi

    def get_Plimit(self):
        psi = np.zeros(self.nlim)
        for i, (x, z) in enumerate(zip(self.xlim, self.ylim)):
            psi[i] = self.Ppoint((x, z))
        self.Xpsi = np.max(psi)

    def set_plasma(self, eq):
        for key in ['x', 'z', 'psi']:
            if key in eq.keys():
                setattr(self, key, eq[key])
        self.trim_x()
        self.space()
        self.Bfeild()

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

    def space(self):
        self.nx = len(self.x)
        self.nz = len(self.z)
        self.n = self.nx * self.nz
        self.dx = (self.x[-1] - self.x[0]) / (self.nx - 1)
        self.dz = (self.z[-1] - self.z[0]) / (self.nz - 1)
        self.delta = np.sqrt(self.dx**2 + self.dz**2)
        self.x2d, self.z2d = np.meshgrid(self.x, self.z, indexing='ij')

    def Bfeild(self):
        psi_x, psi_z = np.gradient(self.psi, self.dx, self.dz)
        xm = np.array(np.matrix(self.x).T * np.ones([1, self.nz]))
        xm[xm == 0] = 1e-34
        self.Bx = -psi_z / xm
        self.Bz = psi_x / xm

    def Bpoint(self, point, check_bounds=False):  # magnetic feild at point
        feild = np.zeros(2)  # function re-name (was Bcoil)
        if not hasattr(self, 'Bspline'):
            self.Bspline = [[], []]
            self.Bspline[0] = RectBivariateSpline(self.x, self.z, self.Bx)
            self.Bspline[1] = RectBivariateSpline(self.x, self.z, self.Bz)
        if check_bounds:
            inbound = point[0] >= np.min(self.x) and\
                point[0] <= np.max(self.x) \
                and point[1] >= np.min(self.z) and point[1] <= np.max(self.z)
            return inbound
        else:
            for i in range(2):
                feild[i] = self.Bspline[i].ev(point[0], point[1])
            return feild

    def minimum_feild(self, radius, theta):
        X = radius * np.sin(theta) + self.Xpoint[0]
        Z = radius * np.cos(theta) + self.Xpoint[1]
        B = np.zeros(len(X))
        for i, (x, z) in enumerate(zip(X, Z)):
            feild = self.Bpoint((x, z))
            B[i] = np.sqrt(feild[0]**2 + feild[1]**2)
        return np.argmin(B)

    def Ppoint(self, point):  # was Pcoil
        if not hasattr(self, 'Pspline'):
            self.Pspline = RectBivariateSpline(self.x, self.z, self.psi)
        psi = self.Pspline.ev(point[0], point[1])
        return psi

    def contour(self, Nstd=1.5, Nlevel=31, Xnorm=True, lw=1, plot_vac=True,
                boundary=True, **kwargs):
        alpha = np.array([1, 1], dtype=float)
        lw = lw * np.array([2.25, 1.75])
        if boundary:
            x, z = self.get_boundary(1 - 1e-3)
            plt.plot(x, z, linewidth=lw[0], color=0.75 * np.ones(3))
            self.set_boundary(x, z)
        if not hasattr(self, 'Xpsi'):
            self.get_Xpsi()
        if not hasattr(self, 'Mpsi'):
            self.get_Mpsi()
        if 'levels' not in kwargs.keys():
            dpsi = 0.01 * (self.Xpsi - self.Mpsi)
            level, n = [self.Mpsi + dpsi, self.Xpsi - dpsi], 17
            level, n = [np.mean(self.psi) - Nstd * np.std(self.psi),
                        np.mean(self.psi) + Nstd * np.std(self.psi)], 15
            level, n = [-Nstd * np.std(self.psi),
                        Nstd * np.std(self.psi)], Nlevel
            if Nstd * np.std(self.psi) < self.Mpsi - self.Xpsi and \
                    self.z.max() > self.Mpoint[1]:
                Nstd = (self.Mpsi - self.Xpsi) / np.std(self.psi)
                level, n = [-Nstd * np.std(self.psi),
                            Nstd * np.std(self.psi)], Nlevel
            levels = np.linspace(level[0], level[1], n)
            linetype = '-'
        else:
            levels = kwargs['levels']
            linetype = '-'
        color = 'k'
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
                    plt.plot(x, z, linetype, linewidth=lw[pindex],
                             color=color, alpha=alpha[pindex])
        if boundary:
            plt.plot(self.xbdry, self.zbdry, linetype, linewidth=lw[pindex],
                     color=color, alpha=alpha[pindex])
        plt.axis('equal')
        plt.axis('off')
        return levels

    def inPlasma(self, X, Z, delta=0):
        return X.min() >= self.xbdry.min() - delta and \
            X.max() <= self.xbdry.max() + delta and \
            Z.min() >= self.zbdry.min() - delta and \
            Z.max() <= self.zbdry.max() + delta

    def plot_cs(self, cs, norm, Plasma=False, color='k',
                pcolor='w', linetype='-'):
        alpha = np.array([1, 0.5])
        lw = 0.75
        if not Plasma:
            norm = 0
        if color == 'k':
            alpha *= 0.25

        for p in cs.get_paths():
            v = p.vertices
            X, Z, delta = v[:, 0][:], v[:, 1][:], 0.5
            inPlasma = X.min() >= self.xbdry.min() - delta and \
                X.max() <= self.xbdry.max() + delta and \
                Z.min() >= self.zbdry.min() - delta and \
                Z.max() <= self.zbdry.max() + delta
            if inPlasma:
                plt.plot(X, Z, linetype, linewidth=1.25 * lw,
                        color=norm * np.array([1, 1, 1]), alpha=alpha[0])
            else:
                plt.plot(X, Z, linetype, linewidth=lw,
                        color=color, alpha=alpha[1])

    def Bcontour(self, axis, Nstd=1.5, color='x'):
        var = 'B' + axis
        if not hasattr(self, var):
            self.Bfeild()
        B = getattr(self, var)
        level = [np.mean(B) - Nstd * np.std(B),
                 np.mean(B) + Nstd * np.std(B)]
        CS = plt.contour(self.x, self.z, B,
                        levels=np.linspace(level[0], level[1], 30),
                        colors=color)
        for cs in CS.collections:
            cs.set_linestyle('solid')

    def Bquiver(self):
        if not hasattr(self, 'Bx'):
            self.Bfeild()
        plt.quiver(self.x, self.z, self.Bx.T, self.Bz.T)

    def Bsf(self):
        if not hasattr(self, 'Bx'):
            self.Bfeild()
        plt.streamplot(self.x, self.z, self.Bx.T, self.Bz.T,
                      color=self.Bx.T, cmap=plt.cm.RdBu)
        plt.clim([-1.5, 1.5])

    def getX(self, po=None):
        def feild(p):
            B = self.Bpoint(p)
            return sum(B * B)**0.5
        res = minimize(feild, np.array(po), method='nelder-mead',
                       options={'xtol': 1e-7, 'disp': False})
        return res.x

    def get_Xpsi(self, po=None, select='primary'):
        if po is None:
            if hasattr(self, 'po'):
                po = self.po
            else:
                xo_arg = np.argmin(self.eqdsk['zbdry'])
                po = [self.eqdsk['rbdry'][xo_arg],
                      self.eqdsk['zbdry'][xo_arg]]
        Xpoint = np.zeros((2, 2))
        Xpsi = np.zeros(2)
        for i, flip in enumerate([1, -1]):
            po[1] *= flip
            Xpoint[:, i] = self.getX(po=po)
            Xpsi[i] = self.Ppoint(Xpoint[:, i])
        index = np.argsort(Xpoint[1, :])
        Xpoint = Xpoint[:, index]
        Xpsi = Xpsi[index]
        if select == 'lower':
            i = 0  # lower Xpoint
        elif select == 'upper':
            i = 1  # upper Xpoint
        elif select == 'primary':
            i = np.argmax(Xpsi)  # primary Xpoint
        self.Xerr = Xpsi[1] - Xpsi[0]
        self.Xpsi = Xpsi[i]
        self.Xpoint = Xpoint[:, i]
        self.Xpoint_array = Xpoint
        if i == 0:
            po[1] *= -1  # re-flip
        if self.Xpoint[1] < self.mo[1]:
            self.Xloc = 'lower'
        else:
            self.Xloc = 'upper'
        return (self.Xpsi, self.Xpoint)

    def getM(self, mo=None):
        if mo is None:
            mo = self.mo

        def psi(m):
            return -self.Ppoint(m)
        res = minimize(psi, np.array(mo), method='nelder-mead',
                       options={'xtol': 1e-7, 'disp': False})
        return res.x

    def get_Mpsi(self, mo=None):
        self.Mpoint = self.getM(mo=mo)
        self.Mpsi = self.Ppoint(self.Mpoint)
        return (self.Mpsi, self.Mpoint)

    def remove_contour(self):
        for key in ['cfeild', 'cfeild_bndry']:
            if hasattr(self, key):
                delattr(self, key)

    def set_contour(self):
        psi_boundary = 1.1 * (self.Xpsi - self.Mpsi) + self.Mpsi
        psi_bndry = np.pad(self.psi[1:-1, 1:-1], (1,),
                           mode='constant', constant_values=psi_boundary)
        self.cfeild = cntr(self.x2d, self.z2d, self.psi)
        self.cfeild_bndry = cntr(self.x2d, self.z2d, psi_bndry)

    def get_contour(self, levels, boundary=False):
        if boundary:
            def cfeild(level): return self.cfeild_bndry.trace(level, level, 0)
        else:
            def cfeild(level): return self.cfeild.trace(level, level, 0)
        lines = []
        for level in levels:
            psi_line = cfeild(level)
            psi_line = psi_line[:len(psi_line) // 2]
            lines.append(psi_line)
        return lines

    def get_boundary(self, alpha=1-1e-3, plot=False):
        self.Spsi = alpha * (self.Xpsi - self.Mpsi) + self.Mpsi

        psi_line = self.get_contour([self.Spsi], boundary=True)[0]
        X, Z = np.array([]), np.array([])
        for line in psi_line:
            x, z = line[:, 0], line[:, 1]
            if self.Xloc == 'lower':  # lower Xpoint
                index = z >= self.Xpoint[1]
            elif self.Xloc == 'upper':  # upper Xpoint
                index = z <= self.Xpoint[1]
            if sum(index) > 0:
                x, z = x[index], z[index]
                loop = np.sqrt((x[0] - x[-1])**2 +
                               (z[0] - z[-1])**2) < 5*self.delta
                if (z > self.Mpoint[1]).any() and\
                        (z < self.Mpoint[1]).any() and loop:
                    X, Z = np.append(X, x), np.append(Z, z)
        try:
            X, Z = geom.clock(X, Z)
        except IndexError:
            plt.figure()
            self.contour(boundary=False)
            raise IndexError('seperatrix open')
        if plot:
            plt.plot(X, Z)
        return X, Z

    def eq_boundary(self, expand=0):  # generate boundary dict for elliptic
        X, Z = self.get_boundary()
        boundary = {'X': X, 'Z': Z, 'expand': expand}
        return boundary

    def get_midplane(self, x, z):
        def psi_err(x, *args):
            z = args[0]
            psi = self.Ppoint((x, z))
            return abs(psi - self.Xpsi)
        res = minimize(psi_err, np.array(x), method='nelder-mead',
                       args=(z), options={'xtol': 1e-7, 'disp': False})
        return res.x[0]

    def get_LFP(self, alpha=1-1e-3):
        x, z = self.get_boundary(alpha=alpha)
        if self.Xpoint[1] < self.Mpoint[1]:
            index = z > self.Xpoint[1]
        else:  # alowance for upper Xpoint
            index = z < self.Xpoint[1]
        x_loop, z_loop = x[index], z[index]
        xc, zc = self.Mpoint
        radius = ((x_loop - xc)**2 + (z_loop - zc)**2)**0.5
        theta = np.arctan2(z_loop - zc, x_loop - xc)
        index = theta.argsort()
        radius, theta = radius[index], theta[index]
        theta = np.append(theta[-1] - 2 * np.pi, theta)
        radius = np.append(radius[-1], radius)
        x = xc + radius * np.cos(theta)
        z = zc + radius * np.sin(theta)
        fLFSx = interp1d(theta, x)
        fLFSz = interp1d(theta, z)
        self.LFPx, self.LFPz = fLFSx(0), fLFSz(0)
        self.LFPx = self.get_midplane(self.LFPx, self.LFPz)
        self.HFPx, self.HFPz = fLFSx(-np.pi), fLFSz(-np.pi)
        self.HFPx = self.get_midplane(self.HFPx, self.HFPz)
        self.shape['R'] = np.mean([self.HFPx, self.LFPx])
        self.shape['a'] = (self.LFPx - self.HFPx) / 2
        self.shape['AR'] = self.shape['R'] / self.shape['a']
        return (self.LFPx, self.LFPz, self.HFPx, self.HFPz)

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
        if not hasattr(self, 'LFPx'):
            self.get_LFP()
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
            rpl, zpl = geom.rzSLine(rpl, zpl, Nsub)
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
        self.get_LFP()
        self.Dsol = np.linspace(0, self.dSOL, self.Nsol)
        x = self.LFPx + self.Dsol
        z = self.LFPz * np.ones(len(x))
        self.sol_psi = np.zeros(len(x))
        for i, (rp, zp) in enumerate(zip(x, z)):
            self.sol_psi[i] = self.Ppoint([rp, zp])

    def upsample_sol(self, nmult=10):
        k = 1  # smoothing factor
        for i, (x, z) in enumerate(zip(self.Xsol, self.Zsol)):
            l = geom.length(x, z)
            L = np.linspace(0, 1, nmult * len(l))
            self.Xsol[i] = sinterp(l, x, k=k)(L)
            self.Zsol[i] = sinterp(l, z, k=k)(L)

    # dx [m]
    def sol(self, dx=3e-3, Nsol=5, plot=False, update=False, debug=False):
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
        contours = self.get_contour(self.sol_psi)
        self.Xsol, self.Zsol = self.pick_contour(contours, Xpoint=True,
                                                 Midplane=False, Plasma=False)
        self.upsample_sol(nmult=4)  # upsample
        self.get_legs(debug=debug)
        if plot:
            color = sns.color_palette('Set2', 6)
            for c, leg in enumerate(self.legs): # enumerate(['inner', 'outer']):
                for i in np.arange(self.legs[leg]['i'])[::-1]:
                    x, z = self.snip(leg, i)
                    x, z = self.legs[leg]['X'][i], self.legs[leg]['Z'][i]
                    plt.plot(x, z, color=color[c], linewidth=0.5)

    def add_core(self):  # refarance from low-feild midplane
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
        if self.nleg == 6:  # snow flake
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
                    nx = self.minimum_feild(xin, tin)  # minimum feild
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
        B /= np.sqrt(B[0]**2 + B[1]**2)  # poloidal feild line vector
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
            Bp = np.sqrt(B[0]**2 + B[1]**2)  # polodial feild
            Bphi = Bm / x  # torodal field
            Xi = np.append(Xi, Bphi / Bp)  # feild expansion
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
        self.get_LFP()
        x95, z95 = self.get_boundary(alpha=0.95)
        ru = x95[np.argmax(z95)]  # triangularity
        rl = x95[np.argmin(z95)]
        self.shape['del_u'] = (self.shape['R'] - ru) / self.shape['a']
        self.shape['del_l'] = (self.shape['R'] - rl) / self.shape['a']
        self.shape['kappa'] = (np.max(z95) - np.min(z95)
                               ) / (2 * self.shape['a'])
        x, z = self.get_boundary(alpha=1 - 1e-4)
        x, z = geom.clock(x, z, reverse=True)
        self.shape['V'] = loop_vol(x, z, plot=plot)
        return self.shape
