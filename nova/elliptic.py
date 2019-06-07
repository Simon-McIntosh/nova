from amigo.pyplot import plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
import nova.cross_coil as cc
from scipy.optimize import minimize
from matplotlib.colors import Normalize
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.linalg import lstsq
from amigo import geom
from scipy.optimize import newton
import sys
from warnings import warn
from time import time
from amigo.geom import poly_inloop
from nova.streamfunction import SF
from nova.coil_class import CoilClass


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class EQ(object):

    def __init__(self, coilset, eqdsk, sigma=0, **kwargs):
        self.mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]
        self.cc = CoilClass(coilset)
        self.sf = SF(eqdsk=eqdsk)
        self.resample(sigma=sigma, **kwargs)
        self.get_plasma()  # plasma coils
        self.select_control_coils()  # identify control coils

    def resample(self, sigma=0, **kwargs):  # resample current density
        self.x, self.z = self.sf.x, self.sf.z
        self.x2d, self.z2d = self.sf.x2d, self.sf.z2d
        self.dx, self.dz = self.sf.dx, self.sf.dz
        self.dA = self.sf.dx * self.sf.dz
        self.psi = self.sf.psi  # copy psi
        self.GSoper()  # apply GS opperator to psi
        # construct spline interpolator
        GSspline = RBS(self.x, self.z, self.GS)
        update_edge = self.grid(**kwargs)  # update solution grid
        self.GS = GSspline.ev(self.x2d, self.z2d)  # interpolate b
        # post interpolation filter
        self.GS = self.convolve(self.GS, sigma=sigma)
        self.b = np.copy(np.reshape(self.GS, self.nx * self.nz))  # set core
        self.edgeBC(update_edge=update_edge)  # set edge
        self.psi = self.solve()  # re-solve
        # self.set_eq_psi()  # pass grid and psi back to sf

    def limits(self, boundary):
        X, Z = boundary.get('X'), boundary.get('Z')
        for key in ['xmin', 'xmax', 'zmin', 'zmax']:
            if key in boundary.keys():
                if 'x' in key:
                    var = X
                else:
                    var = Z
                if 'min' in key:
                    index = var > boundary.get(key)
                else:
                    index = var < boundary.get(key)
                X, Z = X[index], Z[index]
        lim = np.array([X.min(), X.max(), Z.min(), Z.max()])
        if 'expand' in boundary.keys():  # expansion from boundary in meters
            expand = boundary.get('expand')
        else:
            expand = 0.5
        for i, direction in enumerate([-1, 1, -1, 1]):
            lim[i] += direction * expand
        return lim

    def grid(self, **kwargs):
        update_edge = True
        if 'boundary' in kwargs:
            limit = self.limits(kwargs.get('boundary'))
            for i, lim in enumerate(['xmin', 'xmax', 'zmin', 'zmax']):
                if lim in kwargs:
                    limit[i] = kwargs[lim]
        elif 'limit' in kwargs:
            limit = kwargs.get('limit')
        elif 'delta' in kwargs:
            delta = kwargs.get('delta')
            limit = np.array([self.sf.x[0] + delta, self.sf.x[-1] - delta,
                              self.sf.z[0] + delta, self.sf.z[-1] - delta])
        else:
            limit = np.array([self.sf.x[0], self.sf.x[-1],
                              self.sf.z[0], self.sf.z[-1]])
            update_edge = False
        if 'n' in kwargs.keys():
            n = kwargs.get('n')
        else:
            n = self.sf.nx * self.sf.nz
        self.limit_o = np.array([self.x[0], self.x[-1],
                                 self.z[0], self.z[-1]])
        self.limit, self.n = limit, n
        xo, zo = limit[:2], limit[2:]
        dro, dzo = (xo[-1] - xo[0]), (zo[-1] - zo[0])
        ar = dro / dzo
        self.nz = int(np.sqrt(n / ar))
        self.nx = int(n / self.nz)
        self.x = np.linspace(xo[0], xo[1], self.nx)
        self.z = np.linspace(zo[0], zo[1], self.nz)
        self.x2d, self.z2d = np.meshgrid(self.x, self.z, indexing='ij')
        self.N = self.nx * self.nz
        self.b = np.zeros(self.N)
        self.bp = np.zeros(self.N)
        self.psi_plasma = np.zeros((self.nx, self.nz))
        self.dx = (self.x[-1] - self.x[0]) / (self.nx - 1)
        self.dz = (self.z[-1] - self.z[0]) / (self.nz - 1)
        self.rc = np.linspace(xo[0] - self.dx / 2,
                              xo[1] + self.dx / 2, self.nx + 1)
        self.zc = np.linspace(zo[0] - self.dz / 2,
                              zo[1] + self.dz / 2, self.nz + 1)
        self.x2dc, self.z2dc = np.meshgrid(self.rc, self.zc, indexing='ij')
        self.dA = self.dx * self.dz
        self.A = self.matrix()
        self.boundary()
        self.edge()
        self.core_index()
        return update_edge

    def core_index(self):
        self.core_indx = np.ones(self.N, dtype=bool)
        self.core_indx[self.indx(0, np.arange(self.nz))] = 0
        self.core_indx[self.indx(np.arange(self.nx), 0)] = 0
        self.core_indx[self.indx(self.nx - 1, np.arange(self.nz))] = 0
        self.core_indx[self.indx(np.arange(self.nx), self.nz - 1)] = 0

    def select_control_coils(self):
        self.ccoil = {'vertical': {}, 'horizontal': {}}
        PF = self.cc.coil.loc[self.cc.coil['part'] == 'PF', :]
        CS = self.cc.coil.loc[self.cc.coil['part'] == 'CS', :]
        self.ccoil['vertical'] = np.zeros(
                PF.nC, dtype=[('name', '|S10'), ('value', 'float'),
                              ('Io', 'float'), ('Ip', 'float'),
                              ('Ii', 'float'), ('z', 'float')])
        self.ccoil['horizontal'] = np.zeros(
                CS.nC, dtype=[('name', '|S10'), ('value', 'float'),
                              ('Io', 'float'), ('Ip', 'float'),
                              ('Ii', 'float'), ('z', 'float')])
        nV, nH = -1, -1
        index = list(PF.index) + list(CS.index)
        for name in index:
            x = self.cc.coil.at[name, 'x']
            z = self.cc.coil.at[name, 'z']
            field = cc.green_field(self.sf.Mpoint[0], self.sf.Mpoint[1], x, z)
            if name in PF.index:
                nV += 1
                direction, index, iB = 'vertical', nV, 0
            elif name in CS.index:
                nH += 1
                direction, index, iB = 'horizontal', nH, 1
            self.ccoil[direction]['name'][index] = name
            self.ccoil[direction]['z'][index] = z
            self.ccoil[direction]['value'][index] = field[iB]
            self.ccoil[direction]['Io'][index] = \
                self.cc.coil.at[name, 'It']
        for direction in ['vertical', 'horizontal']:
            self.ccoil[direction] = np.sort(self.ccoil[direction],
                                            order='value')  # order='z'
        self.ccoil['active'] = []
        for index in [0, -1]:
            self.ccoil['active'].append(
                self.ccoil['vertical']['name'][index].decode())
        for index in range(2):
            self.ccoil['active'].append(
                self.ccoil['horizontal']['name'][index].decode())

    def indx(self, i, j):
        return i * self.nz + j

    def ij(self, indx):
        j = np.mod(indx, self.nz)
        i = int((indx - j) / self.nz)
        return i, j

    def matrix(self):
        A = lil_matrix((self.N, self.N))
        A.setdiag(np.ones(self.N))
        for i in range(1, self.nx - 1):
            for j in range(1, self.nz - 1):
                rp = 0.5 * (self.x[i + 1] + self.x[i])  # r_{i+1/2}
                rm = 0.5 * (self.x[i] + self.x[i - 1])  # r_{i-1/2}
                ind = self.indx(i, j)
                A[ind, ind] = -(self.x[i] / self.dx**2) *\
                    (1 / rp + 1 / rm) - 2 / self.dz**2
                A[ind, self.indx(i + 1, j)] = (self.x[i] / self.dx**2) / rp
                A[ind, self.indx(i - 1, j)] = (self.x[i] / self.dx**2) / rm
                A[ind, self.indx(i, j + 1)] = 1 / self.dz**2
                A[ind, self.indx(i, j - 1)] = 1 / self.dz**2
        return A.tocsr()

    def resetBC(self):
        self.b[self.core_indx] *= 0
        self.b *= 0

    def ingrid(self, x, z):
        if x > self.x[0] + self.dx and x < self.x[-1] - self.dx\
                and z > self.z[0] + self.dz and z < self.z[-1] - self.dz:
            return True
        else:
            return False

    def psi_ex(self):
        psi = np.zeros(self.Ne)
        subcoil = self.cc.subset('Plasma', invert=True).subcoil
        x, z, It = subcoil.loc[:, ['x', 'z', 'It']].values.T
        for i in range(subcoil.nC):
            if not self.ingrid(x[i], z[i]):
                psi += cc.mu_o * It[i] * cc.green(self.Xe, self.Ze, x[i], z[i])
        self.psi_external = psi
        return psi

    def psi_pl(self):
        self.b[~self.core_indx] = 0  # zero edge BC
        psi_core = self.solve()  # solve with zero edgeBC
        dgdn = self.boundary_normal(psi_core)
        psi = np.zeros(self.Ne)
        for i, (x, z) in enumerate(zip(self.Xe, self.Ze)):
            circ = -cc.green(x, z, self.Xb, self.Zb) * dgdn / self.Xb
            psi[i] = np.trapz(circ, self.Lb)
        self.psi_e_plasma = psi
        return psi

    def psi_edge(self):
        psi = np.zeros(self.Ne)
        for i, (x, z) in enumerate(zip(self.Xe, self.Ze)):
            psi[i] = self.sf.Ppoint((x, z))
        return psi

    def coil_core(self):
        subcoil = self.cc.subset('Plasma', invert=True).subcoil
        x, z, It = subcoil.loc[:, ['x', 'z', 'It']].values.T
        for index in range(subcoil.nC):
            if self.ingrid(x[index], z[index]):
                i = np.argmin(np.abs(x[index] - self.x))
                j = np.argmin(np.abs(z[index] - self.z))
                self.b[self.indx(i, j)] += -self.mu_o * \
                    It[index] * self.x[i] / (self.dA)

    def set_vacuum(self, GS):
        X, Z = geom.inloop(self.sf.xbdry, self.sf.zbdry,
                           self.x2d.flatten(), self.z2d.flatten())[:2]
        GSo = np.copy(GS)
        GS *= 0  # clear all
        for x, z in zip(X, Z):  # backfill plasma
            i = np.argmin(np.abs(x - self.x))
            j = np.argmin(np.abs(z - self.z))  # plasma vertical offset
            GS = GSo[i, j]

    def index_plasma_core(self):
        xbdry, zbdry = self.sf.get_boundary(alpha=1-1e-4)
        loop = {'x': xbdry, 'z': zbdry}
        points = {'x': self.x2d.flatten(), 'z': self.z2d.flatten()}
        X, Z = poly_inloop(loop, points, plot=False)
        self.Nplasma = len(X)
        self.plasma_index = np.zeros(self.Nplasma, dtype=int)
        self.psi_norm = np.zeros(self.Nplasma, dtype=float)
        for k, (x, z) in enumerate(zip(X, Z)):
            i = np.argmin(np.abs(x - self.x))
            j = np.argmin(np.abs(z - self.z))  # plasma vertical offset
            self.plasma_index[k] = self.indx(i, j)
            self.psi_norm[k] =\
                (self.psi[i, j] - self.sf.Mpsi) / (self.sf.Xpsi - self.sf.Mpsi)

        # calculate self-inductance coupling matrix
        '''
        #  slow
        self.xB = np.zeros((self.Nplasma, self.Nplasma))  # [xB][I] = x|B|**2
        for p in range(self.Nplasma):  # sink
            ip, jp = self.ij(self.plasma_index[p])
            x, z = self.x[ip], self.z[jp]
            for q in range(self.Nplasma):  # source
                iq, jq = self.ij(self.plasma_index[q])
                xi, zi = self.x[iq], self.z[jq]
                B = 2 * np.pi * cc.mu_o * cc.get_green_field(x, z, xi, zi,
                                                             self.dx/2)
                self.xB[p, q] = x * (B[0]**2 + B[1]**2)
        '''

    def plasma_core(self, update=True):
        if update:  # calculate plasma contribution
            self.index_plasma_core()
            self.Ip = 0
            for index, psi_norm in zip(self.plasma_index, self.psi_norm):
                x = self.x[self.ij(index)[0]]
                self.bp[index] = -self.mu_o * x**2 * self.sf.Pprime(psi_norm)\
                    - self.sf.FFprime(psi_norm)
                self.Ip -= self.dA * self.bp[index] / (self.mu_o * x)
            # li = np.dot(self.xB, I.reshape(-1, 1))  # internal inductance
            scale_plasma = self.sf.Ip / self.Ip
            self.sf.b_scale = scale_plasma
        for i, index in enumerate(self.plasma_index):
            self.b[index] = self.bp[index] * self.sf.b_scale

    def set_plasma(self):
        Xp, Zp = np.zeros(self.Nplasma), np.zeros(self.Nplasma)
        Ip = np.zeros(self.Nplasma)
        for index in range(self.Nplasma):
            i, j = self.ij(self.plasma_index[index])
            x, z = self.x[i], self.z[j]
            It = -self.dA * self.b[self.plasma_index[index]] / (self.mu_o * x)
            Xp[index], Zp[index] = x, z
            Ip[index] = It
        index = -1
        self.cc.add_plasma(Xp, Zp, self.dx, self.dz, It=Ip)

    def get_plasma(self):
        self.plasma_core()
        self.set_plasma()

    def coreBC(self, update=True):
        self.plasma_core(update=update)
        self.coil_core()

    def edgeBC(self, update_edge=True, external_coils=True):
        if update_edge:
            psi = self.psi_pl()  # plasma component
            if external_coils:
                psi += self.psi_ex()  # coils external to grid
        else:  # edge BC from sf  #  or self.sf.eqdsk['ncoil'] == 0:
            psi = self.psi_edge()
        self.b[self.indx(0, np.arange(self.nz))] = \
            psi[2 * self.nx + self.nz:2 * self.nx + 2 * self.nz][::-1]
        self.b[self.indx(np.arange(self.nx), 0)] = psi[:self.nx]
        self.b[self.indx(self.nx - 1, np.arange(self.nz))] = \
            psi[self.nx:self.nx + self.nz]
        self.b[self.indx(np.arange(self.nx), self.nz - 1)] = \
            psi[self.nx + self.nz:2 * self.nx + self.nz][::-1]

    def solve(self):
        psi = spsolve(self.A, self.b)
        return np.reshape(psi, (self.nx, self.nz))

    def run(self, update=True):
        self.resetBC()
        self.coreBC(update=update)
        self.edgeBC()
        self.psi = self.solve()
        self.set_eq_psi()

        self.coreBC(update=update)
        self.set_plasma()

    def set_eq_psi(self):  # set psi from eq
        eqdsk = {'x': self.x, 'z': self.z, 'psi': self.psi, 'Ip': self.sf.Ip}
        self.sf.update_eqdsk(eqdsk)  # include boundary update

    def set_control_current(self, name, **kwargs):  # set filliment currents
        if 'It' in kwargs:
            It = kwargs['It']
        elif 'factor' in kwargs:
            It = (1 + kwargs['factor']) * self.cc.coil.at[name, 'Io']
        else:
            errtxt = '\n'
            errtxt += 'kw input \'It\' or \'factor\'\n'
            raise ValueError(errtxt)
        self.cc.coil.at[name, 'It'] = It
        Nf = self.cc.coil.at[name, 'Nf']
        for subcoil in range(Nf):
            subname = '{}_{:1.0f}'.format(name, subcoil)
            self.cc.subcoil.at[subname, 'It'] = It / Nf

    def reset_control_current(self):
        self.cc = 0
        for name in self.ccoil['active']:
            self.set_control_current(name, factor=0)

    def gen(self, ztarget=None, rtarget=None, Zerr=1e-3, kp=-0.2, ki=-0.02,
            Nmax=50, **kwargs):
        if not hasattr(self, 'to'):
            self.to = time()  # start clock for single gen run
        if ztarget is None:  # sead plasma magnetic centre vertical target
            self.ztarget = self.sf.Mpoint[1]
        else:
            self.ztarget = ztarget
        '''
        if rtarget is None:  # sead plasma major radius horizontal target
            self.rtarget = self.ccoil['rtarget']
        else:
            self.rtarget = rtarget
        '''
        plt.plot(self.sf.xbdry, self.sf.zbdry)
        Mflag = False
        self.Zerr, self.Xerr = np.zeros(Nmax), np.zeros(Nmax)
        self.dIc = np.zeros((Nmax, 2))
        self.reset_control_current()
        to = time()
        for i in range(Nmax):
            # print('genopp', i)
            # print(self.sf.Mpoint)
            self.run()
            # self.get_Mpsi()  # high res
            self.Zerr[i] = self.sf.Mpoint[1] - self.ztarget
            # self.Xerr[i] = -(self.sf.shape['X'] - self.rtarget)
            # print(i, self.sf.Mpoint, self.Xerr[i], self.Zerr[i])
            if i > 1:
                if abs(self.Zerr[i - 1]) <= Zerr and abs(self.Zerr[i]) <= Zerr:
                    if Mflag:
                        progress = '\ri:{:1.0f} '.format(i)
                        progress += 'z {:1.3f}m'.format(self.ztarget)
                        progress += '{:+1.3f}mm '.format(1e3 * self.Zerr[i])
                        # progress += 'X {:1.3f}m'.format(self.sf.shape['X'])
                        # progress += '{:+1.3f}mm '.format(1e3 * self.Xerr[i])
                        progress += 'Ipf {:1.3f}KA '.format(
                            1e-3 * self.Ic['v'])
                        progress += 'Ics {:1.3f}MA '.format(
                            1e-6 * self.Ic['h'])
                        progress += 'gen {:1.0f}s '.format(time() - to)
                        progress += 'total {:1.0f}s '.format(time() - self.to)
                        progress += '\t\t\t'  # white space
                        sys.stdout.write(progress)
                        sys.stdout.flush()
                        break
                    Mflag = True
            elif i == Nmax - 1:
                warn('gen vertical position itteration limit reached')
            else:
                Mflag = False
            self.Ic = {'v': 0, 'h': 0}  # control current
            # stability coil pair [0,-1],[1,-1]
            for index, sign in zip([0, -1], [1, -1]):
                dIc = self.PID(self.Zerr[i], 'vertical', index, kp=kp, ki=ki)
                self.Ic['v'] += sign * (dIc)
                self.dIc[i, index] = dIc
            '''
            for index in range(1):
                dIc = self.PID(self.Xerr[i], 'horizontal',
                               index, kp=0.5 * kp, ki=2 * ki)
                self.Ic['h'] += dIc
            '''
        plt.figure()
        plt.plot(self.Zerr)
        plt.figure()
        plt.plot(self.dIc)
        plt.figure()
        # self.set_plasma()  # for independance + Vcoil at start
        return self.Ic['v']

    def PID(self, error, field, i, kp=1.5, ki=0.05):
        name = self.ccoil[field]['name'][i].decode()
        gain = 1e5 / self.ccoil[field]['value'][i]
        self.ccoil[field]['Ip'][i] = gain * kp * error  # proportional
        self.ccoil[field]['Ii'][i] += gain * ki * error  # intergral
        dIc = self.ccoil[field]['Ip'][i] + self.ccoil[field]['Ii'][i]
        Ic = self.ccoil[field]['Io'][i] + dIc
        self.set_control_current(name, field, i, Ic=Ic)
        return dIc

    '''
    def setDC(self):  # store DC coil current for control coils
        for field in ['horizontal', 'vertical']:
            for i in range(len(self.ccoil[field]['name'])):
                name = self.ccoil[field]['name'][i].decode()
                self.ccoil[field]['Io'][i] = self.coil.at[name, 'It']
    '''

    def gen_opp(self, z=None, Zerr=5e-4, Nmax=100, **kwargs):
        self.to = time()  # time at start of gen opp loop
        if z is None:  # sead plasma magnetic center vertical target
            z = self.sf.Mpoint[1]
        f, zt, dzdf = np.zeros(Nmax), np.zeros(Nmax), -0.7e-7  # -2.2e-7
        zt[0] = z
        self.select_control_coils()  # or set_Vcoil for virtual pair
        for i in range(Nmax):
            f[i] = self.gen(zt[i], Zerr=Zerr / 2, **kwargs)
            if i == 0:
                zt[i + 1] = zt[i] - f[i] * dzdf  # estimate second step
            elif i < Nmax - 1:
                fdash = (f[i] - f[i - 1]) / \
                    (zt[i] - zt[i - 1])  # Newtons method
                dz = f[i] / fdash
                if abs(dz) < Zerr:
                    dz_txt = '\nvertical position converged'
                    dz_txt += '< {:1.3f}mm'.format(1e3 * Zerr)
                    print(dz_txt)
                    self.set_plasma()
                    break
                else:
                    zt[i + 1] = zt[i] - dz  # Newton's method
            else:
                errtxt = 'gen_opp vertical position itteration limit reached'
                raise ValueError(errtxt)
        print('')  # escape \x line

    def gen_bal(self, ztarget=None, Zerr=5e-4, tol=1e-4):
        # balance Xpoints (for double null)
        self.to = time()  # time at start of gen bal loop
        print('balancing DN Xpoints:')
        if ztarget is None:  # sead plasma magnetic center vertical target
            ztarget = self.sf.Mpoint[1]
        self.select_control_coils()  # or set_Vcoil for virtual pair

        def gen_err(ztarget):  # Xpoint pair error (balence)
            self.gen(ztarget, Zerr=Zerr)
            return self.sf.Xerr
        newton(gen_err, ztarget, tol=tol)  # find root
        print('')  # escape line

    def GSoper_spline(self):  # apply GS operator
        psi_sp = RBS(self.x, self.z, self.psi)  # construct spline interpolator
        dpsi_r = psi_sp.ev(self.x2d, self.z2d, dx=1, dy=0)
        dpsi_z = psi_sp.ev(self.x2d, self.z2d, dx=0, dy=1)
        GSr = self.x2d * RBS(self.x, self.z, dpsi_r / self.x2d).\
            ev(self.x2d, self.z2d, dx=1, dy=0)
        GSz = RBS(self.x, self.z, dpsi_z).ev(self.x2d, self.z2d, dx=0, dy=1)
        self.GS = GSr + GSz
        '''
        edge_order = 2
        dpsi = np.gradient(self.psi,edge_order=edge_order)
        dpsi_r,dpsi_z = dpsi[0]/self.dx,dpsi[1]/self.dz
        GSr = self.x2d*np.gradient(dpsi_r/self.x2d,
                                   edge_order=edge_order)[0]/self.dx
        GSz = np.gradient(dpsi_z,edge_order=edge_order)[1]/self.dz
        '''

    def GSoper(self):  # apply GS operator
        edge_order = 2
        dpsi = np.gradient(self.psi, edge_order=edge_order)
        dpsi_r, dpsi_z = dpsi[0] / self.dx, dpsi[1] / self.dz
        GSr = self.x2d * \
            np.gradient(dpsi_r / self.x2d, edge_order=edge_order)[0] / self.dx
        GSz = np.gradient(dpsi_z, edge_order=edge_order)[1] / self.dz
        self.GS = GSr + GSz

    def convolve(self, var, sigma=0):
        if sigma > 0:
            # convolution filter
            var = gaussian_filter(var, sigma * self.dA**-0.5)
        return var

    def getj(self, sigma=0):
        j = -self.GS / (self.mu_o * self.x2d)  # current density from psi
        return self.convolve(j, sigma=sigma)

    def sparseBC(self, sigma=0):  # set sparse BC
        b = self.convolve(self.GS, sigma=sigma)
        self.b = np.reshape(b, self.nx * self.nz)

    def set_plasma_current(self):  # plasma current from sf line intergral
        Ip = 0
        X, Z = self.sf.get_boundary()
        tX, tZ, X, Z = cc.tangent(X, Z, norm=False)
        for x, z, tx, tz in zip(X, Z, tX, tZ):
            B = self.sf.Bcoil((x, z))
            t = np.array([tx, tz])
            Ip += np.dot(B, t)
        Ip /= cc.mu_o
        self.sf.Ip = Ip

    def fluxfunctions(self, npsi=100, sigma=0):
        FFp, Pp = np.ones(npsi), np.ones(npsi)  # flux functions
        psi_norm = np.linspace(1e-3, 1 - 1e-3, npsi)
        j = self.getj(sigma=sigma)  # current denstiy
        bspline = RBS(self.x, self.z, j)
        for i, psi in enumerate(psi_norm):
            rb, zb = self.sf.get_boundary(alpha=psi)
            Lb = self.sf.length(rb, zb)
            L, dL = np.linspace(Lb.min(), Lb.max(), len(Lb), retstep=True)
            x, z = np.interp(L, Lb, rb), np.interp(L, Lb, zb)  # even spacing
            js = bspline.ev(x, z)
            A = np.matrix(np.ones((len(x), 2)))
            A[:, 0] = np.reshape(1 / (self.mu_o * x), (-1, 1))
            A[:, 1] = np.reshape(x, (-1, 1))
            FFp[i], Pp[i] = lstsq(A, js)[0]  # fit flux functions
            js_lstsq = x * Pp[i] + FFp[i] / (self.mu_o * x)  # evaluate
            jfact = sum(js) / sum(js_lstsq)
            FFp[i] *= jfact  # normalize
            Pp[i] *= jfact  # normalize
        return FFp, Pp, psi_norm

    # sigma==smoothing sd [m]
    def set_fluxfunctions(self, update=True, sigma=0):
        self.FFp, self.Pp, self.psi_norm = self.fluxfunctions(sigma=sigma)
        self.psi_norm = np.append(0, self.psi_norm)  # bounds 0-1
        self.psi_norm[-1] = 1
        self.FFp = np.append(self.FFp[0], self.FFp)
        self.Pp = np.append(self.Pp[0], self.Pp)
        self.sf.FFprime_o = self.sf.FFprime  # store previous
        self.sf.Pprime_o = self.sf.Pprime
        if update:  # replace with new functions
            self.sf.FFprime = interp1d(self.psi_norm, self.FFp)
            self.sf.Pprime = interp1d(self.psi_norm, self.Pp)
        else:  # retain old function shapes and scale to fit
            FFscale = sum(self.FFp) / sum(self.sf.FFprime_o(self.psi_norm))
            Pscale = sum(self.Pp) / sum(self.sf.Pprime_o(self.psi_norm))
            self.sf.FFprime = interp1d(self.psi_norm, FFscale *
                                       self.sf.FFprime_o(self.psi_norm))
            self.sf.Pprime = interp1d(self.psi_norm, Pscale *
                                      self.sf.Pprime_o(self.psi_norm))

    def edge(self):
        X = np.zeros(2 * (self.nx + self.nz))
        Z = np.zeros(2 * (self.nx + self.nz))
        X[:self.nx] = self.x
        X[self.nx:self.nx + self.nz] = self.x[-1]
        X[self.nx + self.nz:2 * self.nx + self.nz] = self.x[::-1]
        X[2 * self.nx + self.nz:2 * self.nx + 2 * self.nz] = self.x[0]
        Z[:self.nx] = self.z[0]
        Z[self.nx:self.nx + self.nz] = self.z
        Z[self.nx + self.nz:2 * self.nx + self.nz] = self.z[-1]
        Z[2 * self.nx + self.nz:2 * self.nx + 2 * self.nz] = self.z[::-1]
        self.Xe, self.Ze, self.Ne = X, Z, len(X)

    def boundary(self):
        X = np.zeros(2 * (self.nx - 1) + 2 * (self.nz - 1))
        Z = np.zeros(2 * (self.nx - 1) + 2 * (self.nz - 1))
        dL = np.zeros(2 * (self.nx - 1) + 2 * (self.nz - 1))
        L = np.zeros(2 * (self.nx - 1) + 2 * (self.nz - 1))
        X[:self.nx - 1] = self.x[:-1] + self.dx / 2
        X[self.nx - 1:self.nx + self.nz - 2] = self.x[-1]
        X[self.nx + self.nz - 2:2 * self.nx + self.nz -
            3] = self.x[:-1][::-1] + self.dx / 2
        X[2 * self.nx + self.nz - 3:2 * self.nx + 2 * self.nz - 4] = self.x[0]
        Z[:self.nx - 1] = self.z[0]
        Z[self.nx - 1:self.nx + self.nz - 2] = self.z[:-1] + self.dz / 2
        Z[self.nx + self.nz - 2:2 * self.nx + self.nz - 3] = self.z[-1]
        Z[2 * self.nx + self.nz - 3:2 * self.nx + 2 * self.nz - 4] = \
            self.z[:-1][::-1] + self.dz / 2
        dL[:self.nx - 1] = self.dx
        dL[self.nx - 1:self.nx + self.nz - 2] = self.dz
        dL[self.nx + self.nz - 2:2 * self.nx + self.nz - 3] = self.dx
        dL[2 * self.nx + self.nz - 3:2 * self.nx + 2 * self.nz - 4] = self.dz
        L[:self.nx - 1] = self.dx / 2 + \
            np.cumsum(self.dx * np.ones(self.nx - 1)) - self.dx
        L[self.nx - 1:self.nx + self.nz - 2] =\
            L[self.nx - 2] + (self.dx + self.dz) / 2 +\
            np.cumsum(self.dz * np.ones(self.nz - 1)) - self.dz
        L[self.nx + self.nz - 2:2 * self.nx + self.nz - 3] = \
            L[self.nx + self.nz - 3] + (self.dz + self.dx) / 2 +\
            np.cumsum(self.dx * np.ones(self.nx - 1)) - self.dx
        L[2 * self.nx + self.nz - 3:2 * self.nx + 2 * self.nz - 4] = \
            L[2 * self.nx + self.nz - 4] + (self.dx + self.dz) / 2 +\
            np.cumsum(self.dz * np.ones(self.nz - 1)) - self.dz
        L[-1] += self.dz / 2
        self.Xb, self.Zb, self.dL, self.Lb = X, Z, dL, L

    def boundary_normal(self, psi):
        dgdn = np.zeros(2 * (self.nx - 1) + 2 * (self.nz - 1))
        dgdn[:self.nx - 1] = -(psi[1:, 1] + psi[:-1, 1]) / (2 * self.dz)
        dgdn[self.nx - 1:self.nx + self.nz - 2] = \
            -(psi[-2, 1:] + psi[-2, :-1]) / (2 * self.dx)
        dgdn[self.nx + self.nz - 2:2 * self.nx + self.nz - 3] = \
            -(psi[1:, -2][::-1] + psi[:-1, -2][::-1]) / (2 * self.dz)
        dgdn[2 * self.nx + self.nz - 3:2 * self.nx + 2 * self.nz - 4] = \
            -(psi[1, 1:][::-1] + psi[1, :-1][::-1]) / (2 * self.dx)
        return dgdn

    def plotb(self, trim=True, alpha=1):
        b = np.copy(np.reshape(self.b, (self.nx, self.nz))).T
        if trim:  # trim edge
            b = b[1:-1, 1:-1]
            rlim, zlim = [self.x[1], self.x[-2]], [self.z[1], self.z[-2]]
        else:
            rlim, zlim = [self.x[0], self.x[-1]], [self.z[0], self.z[-1]]
        cmap = plt.get_cmap('Purples_r')
        # cmap = plt.get_cmap('RdBu')
        cmap._init()  # create the _lut array, with rgba values
        cmap._lut[-4, -1] = 1.0  # set zero value clear
        cmap._lut[0, :-1] = 1  # set zero value clear

        norm = MidpointNormalize(midpoint=0)
        plt.imshow(b, cmap=cmap, norm=norm, vmin=b.min(), vmax=1.1*b.max(),
                   extent=[rlim[0], rlim[-1], zlim[0], zlim[-1]],
                   interpolation='nearest', origin='lower', alpha=alpha)
        # c = plt.colorbar(orientation='horizontal')
        # c.set_xlabel(x'$j$ MAm$^{-1}$')

    def plotj(self, sigma=0, trim=False):
        self.GSoper()
        j = self.getj(sigma=sigma)
        self.plot_matrix(j, scale=1e-6, trim=trim)

    def plot_matrix(self, m, midpoint=0, scale=1, trim=True):
        if trim:  # trim edge
            m = m[1:-1, 1:-1]
            rlim, zlim = [self.x[1], self.x[-2]], [self.z[1], self.z[-2]]
        else:
            rlim, zlim = [self.x[0], self.x[-1]], [self.z[0], self.z[-1]]
        '''
        cmap = plt.get_cmap('RdBu_r')
        caxis = np.round([scale*m.min(),scale*m.max()],decimals=3)
        norm = MidpointNormalize(midpoint=midpoint,vmin=caxis[0],vmax=caxis[1])
        cmap.set_over(color='x',alpha=1)

        plt.imshow(scale*m.T,cmap=cmap,norm=norm,
                  extent=[rlim[0],rlim[-1],zlim[0],zlim[-1]],
                  interpolation='nearest',origin='lower',alpha=1)
        c = plt.colorbar(orientation='horizontal',shrink=.6,
                        pad=0.025, aspect=15)
        c.ax.set_xlabel(x'$J$ MAm$^{-2}$')
        c.set_ticks([caxis[0],0,caxis[1]])
        '''
        cmap = plt.get_cmap('Purples_r')
        # cmap = plt.get_cmap('RdBu')
        cmap.set_under(color='w', alpha=0)
        plt.imshow(scale * m.T, cmap=cmap, vmin=0.1,
                   extent=[rlim[0], rlim[-1], zlim[0], zlim[-1]],
                   interpolation='nearest', origin='lower', alpha=1)

    def plot(self, levels=[]):
        if not list(levels):
            Nstd = 2.5
            level, n = [np.mean(self.psi) - Nstd * np.std(self.psi),
                        np.mean(self.psi) + Nstd * np.std(self.psi)], 15
            levels = np.linspace(level[0], level[1], n)
        self.cs = plt.contourf(self.x2d, self.z2d, self.psi,
                               levels=levels, cmap=plt.cm.RdBu)
        plt.colorbar()
        cs = plt.contour(self.x2d, self.z2d, self.psi, levels=self.cs.levels)
        plt.clabel(cs, inline=1, fontsize=10)

    def set_sf_psi(self):
        self.psi = np.zeros(np.shape(self.x2d))
        for i in range(self.nx):
            for j in range(self.nz):
                self.psi[i, j] = self.sf.Ppoint((self.x[i], self.z[j]),
                                                self.pf,
                                                plasma=self.plasma)

    def set_coil_psi(self):
        psi = cc.get_coil_psi(self.x2d, self.z2d, self.pf,
                              plasma=self.plasma)
        self.sf.set_plasma({'x': self.x, 'z': self.z, 'psi': psi})
        self.get_Xpsi()  # high-res Xpsi

    def update_coil_psi(self):
        psi = cc.get_coil_psi(self.x2d, self.z2d, self.pf,
                              plasma=self.plasma)
        self.sf.update_plasma({'x': self.x, 'z': self.z, 'psi': psi})
        self.get_Xpsi()  # high-res Xpsi

    def set_plasma_psi(self):
        self.psi = np.zeros(np.shape(self.x2d))
        for name in self.plasma.keys():
            self.psi += cc.add_Pcoil(self.x2d,
                                     self.z2d, self.plasma[name])
        self.sf.update_plasma({'x': self.x, 'z': self.z, 'psi': self.psi})

    def plot_psi(self, **kwargs):
        if 'levels' in kwargs:
            CS = plt.contour(self.x2d, self.z2d, self.psi,
                             levels=kwargs['levels'], colors=[[0.5, 0.5, 0.5]])
        elif hasattr(self, 'cs'):
            levels = self.cs.levels
            CS = plt.contour(self.x2d, self.z2d, self.psi, levels=levels)
        else:
            CS = plt.contour(self.x2d, self.z2d, self.psi, 31,
                             colors=[[0.5, 0.5, 0.5]])
        for cs in CS.collections:
            cs.set_linestyle('solid')
            cs.set_alpha(0.5)

    def get_Xpsi(self):
        self.sf.Xpoint = minimize(cc.Bmag, np.array(self.sf.Xpoint),
                                  method='nelder-mead',
                                  args=(self.pf, self.plasma),
                                  options={'xtol': 1e-4, 'disp': False}).x
        self.sf.Xpsi = cc.Ppoint(self.sf.Xpoint, self.pf,
                                 plasma=self.plasma)

    def get_Mpsi(self):
        self.sf.Mpoint = minimize(cc.Bmag, np.array(self.sf.Mpoint),
                                  method='nelder-mead',
                                  args=(self.pf, self.plasma),
                                  options={'xtol': 1e-4, 'disp': False}).x
        self.sf.Mpsi = cc.Ppoint(self.sf.Mpoint, self.pf,
                                 plasma=self.plasma)
