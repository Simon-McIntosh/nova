import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
from itertools import count
import seaborn as sns
import matplotlib
import collections
import amigo.geom as geom
from nova.loops import Profile, get_oppvar, get_value
from nova.config import Setup
from nova.streamfunction import SF
import nova.cross_coil as cc
from nova.coil_cage import coil_cage
from nova.shape import Shape
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from nova.DEMOxlsx import DEMO
from warnings import warn
from nova.inverse import INV
from copy import deepcopy

colors = sns.color_palette('Paired', 12)


class PF(object):
    def __init__(self, eqdsk):
        self.nC = count(0)
        self.set_coils(eqdsk)
        self.plasma_coil = collections.OrderedDict()

    def set_coils(self, eqdsk):
        self.xo = [eqdsk['xcentr'], eqdsk['zmid']]
        self.coil = collections.OrderedDict()
        if eqdsk['ncoil'] > 0:
            CSindex = np.argmin(eqdsk['xc'])  # CS radius and width
            self.rCS, self.drCS = eqdsk['xc'][CSindex], eqdsk['dxc'][CSindex]
            for i, (x, z, dx, dz, I) in enumerate(
                   zip(eqdsk['xc'], eqdsk['zc'], eqdsk['dxc'],
                       eqdsk['dzc'], eqdsk['Ic'])):
                self.add_coil(x, z, dx, dz, I, categorize=False)
                if eqdsk['ncoil'] > 100 and i >= eqdsk['ncoil'] - 101:
                    print('exit set_coil loop - coils')
                    break
        self.categorize_coils()

    def categorize_coils(self):
        catogory = np.zeros(len(self.coil), dtype=[('x', 'float'),
                            ('z', 'float'), ('theta', 'float'),
                            ('index', 'int'), ('name', 'object')])
        for i, name in enumerate(self.coil):
            catogory[i]['x'] = self.coil[name]['x']
            catogory[i]['z'] = self.coil[name]['z']
            catogory[i]['theta'] = np.arctan2(self.coil[name]['z']-self.xo[1],
                                              self.coil[name]['x']-self.xo[0])
            catogory[i]['index'] = i
            catogory[i]['name'] = name
        CSsort = np.sort(catogory, order=['x', 'z'])  # sort CS, x then z
        CSsort = CSsort[CSsort['x'] < self.rCS + self.drCS]
        PFsort = np.sort(catogory, order='theta')  # sort PF,  z
        PFsort = PFsort[PFsort['x'] > self.rCS + self.drCS]
        self.index = {'PF': {'n': len(PFsort['index']),
                             'index': PFsort['index'], 'name': PFsort['name']},
                      'CS': {'n': len(CSsort['index']),
                             'index': CSsort['index'], 'name': CSsort['name']}}
        self.sort()  # sort pf.coil dict [PF,CS]

    def sort(self):  # order coil dict for use by inverse.py
        coil = deepcopy(self.coil)
        self.coil = collections.OrderedDict()
        for name in np.append(self.index['PF']['name'],
                              self.index['CS']['name']):
            self.coil[name] = coil[name]
        nPF, nCS = self.index['PF']['n'], self.index['CS']['n']
        self.index['PF']['index'] = np.arange(0, nPF)
        self.index['CS']['index'] = np.arange(nPF, nPF+nCS)

    def add_coil(self, x, z, dx, dz, I, categorize=True):
        name = 'Coil{:1.0f}'.format(next(self.nC))
        self.coil[name] = {'x': x, 'z': z, 'dx': dx, 'dz': dz, 'I': I,
                           'rc': np.sqrt(dx**2 + dz**2) / 2}
        if categorize:
            self.categorize_coils()

    def remove_coil(self, Clist):
        for c in Clist:
            coil = 'Coil{:1.0f}'.format(c)
            self.coil.pop(coil)
        self.categorize_coils()

    def unpack_coils(self):
        nc = len(self.coil.keys())
        Ic = np.zeros(nc)
        xc, zc, dxc, dzc = np.zeros(nc), np.zeros(
            nc), np.zeros(nc), np.zeros(nc)
        names = []
        for i, name in enumerate(self.coil.keys()):
            xc[i] = self.coil[name]['x']
            zc[i] = self.coil[name]['z']
            dxc[i] = self.coil[name]['dx']
            dzc[i] = self.coil[name]['dz']
            Ic[i] = self.coil[name]['I']
            names.append(name)
        return nc, xc, zc, dxc, dzc, Ic, names

    def mesh_coils(self, dCoil=-1):
        if dCoil < 0:  # dCoil not set, use stored value
            if not hasattr(self, 'dCoil'):
                self.dCoil = 0
        else:
            self.dCoil = dCoil
        if self.dCoil == 0:
            self.sub_coil = self.coil
            for name in self.coil.keys():
                self.sub_coil[name]['rc'] = np.sqrt(self.coil[name]['dx']**2 +
                                                    self.coil[name]['dx']**2)
        else:
            self.sub_coil = {}
            for name in self.coil.keys():
                self.size_coil(name, self.dCoil)

    def size_coil(self, name, dCoil):
        xc, zc = self.coil[name]['x'], self.coil[name]['z']
        Dx, Dz = self.coil[name]['dx'], self.coil[name]['dz']
        Dx = abs(Dx)
        Dz = abs(Dz)
        if self.coil[name]['I'] != 0:
            if Dx > 0 and Dz > 0 and 'plasma' not in name:
                nx, nz = np.ceil(Dx / dCoil), np.ceil(Dz / dCoil)
                dx, dz = Dx / nx, Dz / nz
                x = xc + np.linspace(dx / 2, Dx - dx / 2, nx) - Dx / 2
                z = zc + np.linspace(dz / 2, Dz - dz / 2, nz) - Dz / 2
                X, Z = np.meshgrid(x, z, indexing='ij')
                X, Z = np.reshape(X, (-1, 1)), np.reshape(Z, (-1, 1))
                Nf = len(X)  # filament number
                # self.coil_o[name]['Nf'] = Nf
                # self.coil_o[name]['Io'] = self.pf.pf_coil[name]['I']
                I = self.coil[name]['I'] / Nf
                bundle = {'x': np.zeros(Nf), 'z': np.zeros(Nf),
                          'dx': dx * np.ones(Nf), 'dz': dz * np.ones(Nf),
                          'I': I * np.ones(Nf), 'sub_name': np.array([]),
                          'Nf': 0}
                for i, (x, z) in enumerate(zip(X, Z)):
                    sub_name = name + '_{:1.0f}'.format(i)
                    self.sub_coil[sub_name] = {'x': x, 'z': z,
                                               'dx': dx, 'dz': dz,
                                               'I': I, 'Nf': Nf,
                                               'rc': np.sqrt(dx**2 + dz**2)/2}
                    bundle['x'][i], bundle['z'][i] = x, z
                    bundle['sub_name'] = np.append(
                        bundle['sub_name'], sub_name)
                bundle['Nf'] = i + 1
                bundle['Xo'] = np.mean(bundle['x'])
                bundle['Zo'] = np.mean(bundle['z'])
            else:
                print('coil bundle not found', name)
                self.sub_coil[name] = self.coil[name]
                bundle = self.coil[name]
        return bundle

    def plot_coil(self, coils, label=False, current=False, coil_color=None,
                  fs=12, alpha=1):
        if coil_color is None:
            color = colors
        else:
            color = coil_color  # color itterator
        if len(np.shape(color)) == 1:
            color = color * np.ones((6, 1))

        for i, name in enumerate(coils.keys()):
            coil = coils[name]
            x, z, dx, dz = coil['x'], coil['z'], coil['dx'], coil['dz']
            Xfill = [x + dx / 2, x + dx / 2, x - dx / 2, x - dx / 2]
            Zfill = [z - dz / 2, z + dz / 2, z + dz / 2, z - dz / 2]
            if coil['I'] != 0:
                edgecolor = 'k'
            else:
                edgecolor = 'x'
            coil_color = color[4]
            if name.split('_')[0] in self.index['CS']['name']:
                drs = -2.5 / 3 * dx
                ha = 'right'
                coil_color = color[5]
            elif name.split('_')[0] in self.index['PF']['name']:
                drs = 2.5 / 3 * dx
                ha = 'left'
                coil_color = color[4]
            pl.fill(Xfill, Zfill, facecolor=coil_color, alpha=alpha,
                    edgecolor=edgecolor)
            if label and current:
                zshift = max([coil['dz'] / 4, 0.4])
            else:
                zshift = 0
            if label:
                pl.text(x + drs, z + zshift, name, fontsize=fs * 1.1,
                        ha=ha, va='center', color=0.2 * np.ones(3))
            if current:
                pl.text(x + drs, z - zshift,
                        '{:1.1f}MA'.format(coil['I'] * 1e-6),
                        fontsize=fs * 1.1, ha=ha, va='center',
                        color=0.2 * np.ones(3))

    def plot(self, color=None, subcoil=False, label=False, plasma=False,
             current=False, alpha=1):
        fs = matplotlib.rcParams['legend.fontsize']
        if subcoil:
            coils = self.sub_coil
        else:
            coils = self.coil
        self.plot_coil(coils, label=label, current=current, fs=fs,
                       coil_color=color, alpha=alpha)
        if plasma:
            coils = self.plasma_coil
            self.plot_coil(coils, coil_color=color, alpha=alpha)

    def inductance(self, dCoil=0.5, Iscale=1):
        pf = deepcopy(self)
        inv = INV(pf, Iscale=Iscale, dCoil=dCoil)
        Nf = np.array([1 / inv.coil['active'][coil]['Nf']
                       for coil in inv.coil['active']])
        for i, coil in enumerate(inv.adjust_coils):
            x, z = inv.pf.coil[coil]['x'], inv.pf.coil[coil]['z']
            inv.add_psi(1, point=(x, z))
        inv.set_foreground()
        fillaments = np.dot(np.ones((len(Nf), 1)), Nf.reshape(1, -1))
        self.M = 2 * np.pi * inv.G * fillaments  # PF/CS inductance matrix

    def coil_corners(self, coils):
        X, Z = np.array([]), np.array([])
        Nc = len(coils['id'])
        dX, dZ = np.zeros(Nc), np.zeros(Nc)
        if len(coils['dX']) > 0:
            dX[Nc - len(coils['dX']):] = coils['dX']
        if len(coils['dZ']) > 0:
            dZ[Nc - len(coils['dZ']):] = coils['dZ']
        for Cid, Cdr, Cdz in zip(coils['id'], dX, dZ):
            x = self.coil['Coil' + str(Cid)]['x']
            z = self.coil['Coil' + str(Cid)]['z']
            dx = self.coil['Coil' + str(Cid)]['dx']
            dz = self.coil['Coil' + str(Cid)]['dz']
            if Cdr == 0 and Cdz == 0:
                X = np.append(
                    X, [x + dx / 2, x + dx / 2, x - dx / 2, x - dx / 2])
                Z = np.append(
                    Z, [z + dz / 2, z - dz / 2, z + dz / 2, z - dz / 2])
            else:
                X = np.append(X, x + Cdr)
                Z = np.append(Z, z + Cdz)
        return X, Z

    def fit_coils(self, Cmove, dLo=0.1):
        coils = collections.OrderedDict()
        for side in Cmove.keys():
            if isinstance(dLo, (list, tuple)):
                if 'in' not in side:
                    dL = dLo[0]
                else:
                    dL = dLo[-1]
            else:
                dL = dLo
            for index in Cmove[side]:
                coil = 'Coil' + str(index)
                dx, dz = self.Cshift(self.sf.coil[coil], side, dL)
                coils[coil] = self.sf.coil[coil]
                coils[coil]['x'] = coils[coil]['x'] + dx
                coils[coil]['z'] += dz
                coils[coil]['shiftr'] = dx
                coils[coil]['shiftz'] = dz

        with open('../Data/' + self.conf.config + '_coil_shift.txt', 'w') as f:
            f.write('Ncoils = {:1.0f}\n\n'.format(len(coils.keys())))
            f.write('Name\tX[m]\t\tZ[m]\t\tshiftX[m]\tshiftZ[m]\n')
            index = sorted(map((lambda s: int(s.strip('Coil'))), coils.keys()))
            for i in index:
                coil = 'Coil' + str(i)
                f.write(coil + '\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(
                    coils[coil]['x'], coils[coil]['z'],
                    coils[coil]['shiftr'], coils[coil]['shiftz']))
        return coils


class TF(object):

    def __init__(self, **kwargs):
        self.initalise_loops()  # initalise loop family
        self.initalise_cage(**kwargs)
        if 'sf' in kwargs:
            self.sf = kwargs['sf']
        if 'profile' in kwargs:
            self.profile = kwargs['profile']
            self.update_profile()
            self.nTF = self.profile.nTF
        elif 'p_in' in kwargs and 'nTF' in kwargs:
            self.p_in = kwargs['p_in']
            self.nTF = kwargs['nTF']
        else:
            err_txt = 'insurficent key word inputs\n'
            err_txt += 'set \'profile\' or \'p_in\' and \'nTF\''
            raise ValueError(err_txt)
        self.set_inner_loop()
        self.set_cage()  # initalise cage

    def set_inner_loop(self):
        self.ro = np.min(self.p_in['x'])
        self.cross_section()  # coil cross-sections
        self.get_loops(self.p_in)

    def update_profile(self):
        self.p_in = self.profile.loop.draw()  # inner loop profile
        self.loop = self.profile.loop
        if hasattr(self.profile, 'nTF'):
            self.nTF = self.profile.nTF
        else:
            self.nTF = 18
            warn('using default nTF: {1.0f}'.format(self.nTF))

    def initalise_cage(self, **kwargs):
        self.sep = kwargs.get('sep', 'unset')
        self.alpha = kwargs.get('alpha', 1-1e-4)
        self.ripple = kwargs.get('ripple', True)
        self.ripple_limit = kwargs.get('ripple_limit', 0.6)
        self.nr = kwargs.get('nr', 1)
        self.ny = kwargs.get('ny', 1)
        self.update_cage = False

    def set_cage(self):  # requires profile loop definition for TF
        if hasattr(self, 'sf'):  # streamfunction set
            plasma = {'sf': self.sf}
        elif self.sep is not 'unset':  # seperatrix set
            plasma = {'x': self.sep['x'], 'z': self.sep['z']}
        else:
            errtxt = 'TF cage requires ether:\n'
            errtxt += 'streamfunction opject \'sf\' or\n'
            errtxt += 'seperatrix boundary dict \'sep[\'x\'] and sep[\'z\']\''
            raise ValueError(errtxt)
        self.cage = coil_cage(nTF=self.nTF, rc=self.rc, plasma=plasma,
                              ny=self.ny, nr=self.nr, alpha=self.alpha,
                              wp=self.section['winding_pack'])
        self.update_cage = True
        self.initalise_loop()

    def initalise_loop(self):
        x = get_value(self.loop.xo)
        p_in = self.loop.draw(x=x)
        xloop = self.get_loops(p_in)  # update tf
        if self.update_cage:
            self.cage.set_TFcoil(xloop['cl'], smooth=False)  # update coil cage
        return xloop

    def update_attr(self, **kwargs):
        for attr in ['sep', 'alpha', 'ripple', 'ripple_limit', 'ny', 'nr']:
            setattr(self, attr, kwargs.get(attr, getattr(self, attr)))

    def adjust_xo(self, name, **kwargs):
        self.profile.loop.adjust_xo(name, **kwargs)
        self.update_profile()
        self.set_inner_loop()

    def cross_section(self, J=18.25, twall=0.045):  # MA/m2 TF current density
        self.section = {}
        self.section['case'] = {'side': 0.1, 'nose': 0.51, 'inboard': 0.04,
                                'outboard': 0.19, 'external': 0.225}
        if hasattr(self, 'sf'):  # TF object initalised with sf
            BX = self.sf.eqdsk['bcentr'] * self.sf.eqdsk['xcentr']
            Iturn = 1e-6 * abs(2 * np.pi * BX / (self.nTF * cc.mu_o))
            Acs = Iturn / J
            rwp1 = self.ro - self.section['case']['inboard']
            theta = np.pi / self.nTF
            rwall = twall / np.sin(theta)
            depth = np.tan(theta) * (rwp1 - rwall +
                                     np.sqrt((rwall - rwp1)**2 - 4 *
                                             Acs / (2 * np.tan(theta))))
            width = Acs / depth
            self.section['winding_pack'] = {'width': width, 'depth': depth}
        else:
            warntxt = 'using default winding pack dimensions'
            warntxt += ('pass sf object to tf to enable wp sizing')
            warn(warntxt)
            self.section['winding_pack'] = {'width': 0.625, 'depth': 1.243}
        self.rc = self.section['winding_pack']['width'] / 2

    def initalise_loops(self):
        self.p = {}
        for loop in ['in', 'wp_in', 'cl', 'wp_out', 'out', 'nose', 'loop',
                     'trans_lower', 'trans_upper']:
            self.p[loop] = {'x': [], 'z': []}

    def transition_index(self, r_in, z_in, eps=1e-4):
        npoints = len(r_in)
        r_cl = r_in[0] + eps
        upper = npoints - \
            next((i for i, r_in_ in enumerate(r_in) if r_in_ > r_cl))  # +1
        lower = next((i for i, r_in_ in enumerate(r_in) if r_in_ > r_cl))
        top, bottom = np.argmax(z_in), np.argmin(z_in)
        index = {'upper': upper, 'lower': lower, 'top': top, 'bottom': bottom}
        return index

    def loop_dt(self, x, z, dt_in, dt_out, index):
        l = geom.length(x, z)
        L = np.array([0, l[index['lower']], l[index['bottom']],
                      l[index['top']], l[index['upper']], 1])
        dX = np.array([dt_in, dt_in, dt_out, dt_out, dt_in, dt_in])
        dt = interp1d(L, dX)(l)
        return dt

    def get_loops(self, p_in):
        x, z = p_in['x'], p_in['z']
        wp = self.section['winding_pack']
        case = self.section['case']
        inboard_dt = [case['inboard'], wp['width'] /
                      2, wp['width'] / 2, case['nose']]
        outboard_dt = [case['outboard'], wp['width'] / 2, wp['width'] / 2,
                       case['external']]
        loops = ['wp_in', 'cl', 'wp_out', 'out']
        self.p['in']['x'], self.p['in']['z'] = x, z
        index = self.transition_index(self.p['in']['x'], self.p['in']['z'])
        for loop, dt_in, dt_out in zip(loops, inboard_dt, outboard_dt):
            dt = self.loop_dt(x, z, dt_in, dt_out, index)
            x, z = geom.offset(x, z, dt, close_loop=True)
            self.p[loop]['x'], self.p[loop]['z'] = x, z
        return self.p

    def split_loop(self, plot=False):  # split inboard/outboard for fe model
        x, z = self.p['cl']['x'], self.p['cl']['z']
        index = self.transition_index(x, z)
        upper, lower = index['upper'], index['lower']
        top, bottom = index['top'], index['bottom']
        self.p['nose']['x'] = np.append(x[upper-1:], x[1:lower + 1])
        self.p['nose']['z'] = np.append(z[upper-1:], z[1:lower + 1])
        self.p['trans_lower']['x'] = x[lower:bottom]
        self.p['trans_lower']['z'] = z[lower:bottom]
        self.p['trans_upper']['x'] = x[top:upper]
        self.p['trans_upper']['z'] = z[top:upper]
        self.p['loop']['x'] = x[bottom-1:top+1]
        self.p['loop']['z'] = z[bottom-1:top+1]

        if plot:
            pl.plot(self.p['cl']['x'], self.p['cl']['z'], 'o')
            for name in ['nose', 'loop', 'trans_lower', 'trans_upper']:
                x, z = self.p[name]['x'], self.p[name]['z']
                pl.plot(x, z)

    # outer loop coordinate intexpolators
    def loop_interpolators(self, trim=[0, 1], offset=0.75, full=False):
        x, z = self.p['cl']['x'], self.p['cl']['z']
        self.fun = {'in': {}, 'out': {}}
        # inner/outer loop offset
        for side, sign in zip(['in', 'out', 'cl'], [-1, 1, 0]):
            x, z = self.p[side]['x'], self.p[side]['z']
            index = self.transition_index(x, z)
            x = x[index['lower'] + 1:index['upper']]
            z = z[index['lower'] + 1:index['upper']]
            x, z = geom.offset(x, z, sign * offset)
            if full:  # full loop (including nose)
                rmid, zmid = np.mean([x[0], x[-1]]), np.mean([z[0], z[-1]])
                x = np.append(rmid, x)
                x = np.append(x, rmid)
                z = np.append(zmid, z)
                z = np.append(z, zmid)
            l = geom.length(x, z)
            lt = np.linspace(trim[0], trim[1], int(np.diff(trim) * len(l)))
            x, z = interp1d(l, x)(lt), interp1d(l, z)(lt)
            l = np.linspace(0, 1, len(x))
            self.fun[side] = {'x': IUS(l, x), 'z': IUS(l, z)}
            self.fun[side]['L'] = geom.length(x, z, norm=False)[-1]
            self.fun[side]['dx'] = self.fun[side]['x'].derivative()
            self.fun[side]['dz'] = self.fun[side]['z'].derivative()

    def norm(self, L, loop, point):
        return (loop['x'](L) - point[0])**2 + (loop['z'](L) - point[1])**2

    def Cshift(self, coil, side, dL):  # shift pf coils to tf track
        if 'in' in side:
            X, Z = self.p['in']['x'], self.p['in']['z']
        else:
            X, Z = self.p['out']['x'], self.p['out']['z']
        xc, zc = coil['x'], coil['z']
        dxc, dzc = coil['dx'], coil['dz']
        xp = xc + np.array([-dxc, dxc, dxc, -dxc]) / 2
        zp = zc + np.array([-dzc, -dzc, dzc, dzc]) / 2
        nX, nZ = geom.normal(X, Z)[:2]
        mag = np.sqrt(nX**2 + nZ**2)
        nX /= mag
        nZ /= mag
        i = []
        L = np.empty(len(xp))
        dn = np.empty((2, len(xp)))
        for j, (x, z) in enumerate(zip(xp, zp)):
            i.append(np.argmin((X - x)**2 + (Z - z)**2))
            dx = [x - X[i[-1]], z - Z[i[-1]]]
            dn[:, j] = [nX[i[-1]], nZ[i[-1]]]
            L[j] = np.dot(dx, dn[:, j])
            if 'in' in side:
                L[j] *= -1
        jc = np.argmin(L)
        fact = dL - L[jc]
        if 'in' in side:
            fact *= -1
        delta = fact * dn[:, jc]
        return delta[0], delta[1]

    def get_loop(self, expand=0):  # generate boundary dict for elliptic
        X, Z = self.p['cl']['x'], self.p['cl']['z']
        boundary = {'X': X, 'Z': Z, 'expand': expand}
        return boundary

    def fill(self, write=False, plot=True, alpha=1, plot_cl=False):
        geom.polyparrot(self.p['in'], self.p['wp_in'],
                        color=0.4 * np.ones(3), alpha=alpha)
        geom.polyparrot(self.p['wp_in'], self.p['wp_out'],
                        color=0.6 * np.ones(3), alpha=alpha)
        geom.polyparrot(self.p['wp_out'], self.p['out'],
                        color=0.4 * np.ones(3), alpha=alpha)
        if plot_cl:  # plot winding pack centre line
            pl.plot(self.p['cl']['x'], self.p['cl']['z'],
                    '-.', color=0.5 * np.ones(3))
        #pl.axis('equal')
        #pl.axis('off')

    def support(self, **kwargs):
        self.rzGet()
        self.fill(**kwargs)

    def add_vessel(self, vessel, npoint=80, offset=[0.12, 0.2]):
        rvv, zvv = geom.rzSLine(vessel['x'], vessel['z'], npoint)
        rvv, zvv = geom.offset(rvv, zvv, offset[1])
        rmin = np.min(rvv)
        rvv[rvv <= rmin + offset[0]] = rmin + offset[0]
        self.shp.add_bound({'x': rvv, 'z': zvv}, 'internal')  # vessel
        self.shp.add_bound({'x': np.min(rvv) - 5e-3, 'z': 0}, 'interior')

    def update_loop(self, xnorm, *args):
        x = get_oppvar(self.loop.xo, self.loop.oppvar, xnorm)
        xloop = self.get_loops(self.loop.draw(x=x))  # update tf
        self.loop.set_input(x=x)  # inner loop
        if self.update_cage:
            self.cage.set_TFcoil(xloop['cl'], smooth=False)  # update coil cage
        return xloop

    def constraints(self, xnorm, *args):
        ripple, ripple_limit = args
        # de-normalize
        if ripple:  # constrain ripple contour
            xloop = self.update_loop(xnorm, *args)
            constraint = np.array([])
            for side, key in zip(['internal', 'interior', 'external'],
                                 ['in', 'in', 'out']):
                constraint = np.append(constraint,
                                       self.shp.dot_diffrence(xloop[key],
                                                              side))
            max_ripple = self.cage.get_ripple()
            edge_ripple = self.cage.edge_ripple(npoints=10)
            constraint = np.append(constraint, ripple_limit - edge_ripple)
            constraint = np.append(constraint, ripple_limit - max_ripple)
        else:  # constraint from shape
            constraint = self.shp.geometric_constraints(xnorm, *args)
        return constraint

    def objective(self, xnorm, *args):
        # loop length or loop volume (torus)
        if self.profile.obj == 'L' or self.profile.obj == 'V':
            objF = self.shp.geometric_objective(xnorm, *args)
        elif self.profile.obj == 'E':
            objF = self.energy(xnorm, *args)
        return objF

    def energy(self, xnorm, *args):
        xloop = self.update_loop(xnorm, *args)
        self.cage.set_TFcoil({'x': xloop['cl']['x'], 'z': xloop['cl']['z']})
        E = self.cage.energy()
        return 1e-9*E

    def minimise(self, vessel, verbose=False, **kwargs):
        if not hasattr(self, 'profile'):
            raise ValueError('minimisation requires profile object')
        self.update_attr(**kwargs)
        # call shape called from within tf (dog wags tail)
        self.shp = Shape(self.profile, objective='L')
        # tailor limits on loop parameters (l controls loop tension)
        self.adjust_xo('upper', lb=0.6)
        self.adjust_xo('lower', lb=0.6)
        self.adjust_xo('l', lb=0.5)  # don't go too high (<1.2)
        self.add_vessel(vessel)  # add vessel constraints

        # pass constraint array and objective to loop optimiser
        self.shp.args = (self.ripple, self.ripple_limit)
        self.shp.constraints = self.constraints
        self.shp.objective = self.objective  # objective function
        self.shp.update = self.update_loop  # called on exit from minimizer

        self.shp.minimise(verbose=verbose)

if __name__ is '__main__':  # test functions

    nPF, nTF = 6, 16
    config = {'TF': 'SN', 'eq': 'SN_{:d}PF_{:d}TF'.format(nPF, nTF)}
    setup = Setup(config['eq'])
    sf = SF(setup.filename)
    profile = Profile(config['TF'], family='S', part='TF', nTF=nTF,
                      obj='L', load=True, npoints=60)

    # profile.loop.plot()
    tf = TF(profile=profile, sf=sf, nr=1, ny=1)

    demo = DEMO()
    demo.fill_part('Vessel')
    demo.fill_part('Blanket')
    demo.plot_ports()

    tf.minimise(demo.parts['Vessel']['out'], verbose=True, ripple=False)
    tf.fill()

    pf = PF(sf.eqdsk)
    pf.plot()
    sf.contour()

    tf.cage.output()

    '''
    # tf.coil.set_input()
    tic = time.time()
    print('energy {:1.3f}GJ'.format(1e-9 * cage.energy()))
    print('time A {:1.3f}s'.format(time.time() - tic))
    '''

    '''
    B = np.zeros((tf.npoints,3))
    for i,(x,z) in enumerate(zip(tf.x['cl']['x'],tf.x['cl']['z'])):
        B[i,:] = cage.Iturn*cage.point((x,0,z),variable='feild')

    npoints = 200
    xcl = np.linspace(np.min(tf.x['cl']['x']),np.max(tf.x['cl']['x']),npoints)
    zcl = cage.eqdsk['zmagx']*np.ones(npoints)
    Bcl = np.zeros((npoints,3))

    for i,(x,z) in enumerate(zip(xcl,zcl)):
        Bcl[i,:] = cage.Iturn*cage.point((x,0,z),variable='feild')

    pl.figure(figsize=(8,6))
    pl.plot(tf.x['cl']['x'],abs(B[:,1]))
    pl.plot(xcl,abs(Bcl[:,1]))
    pl.plot(cage.eqdsk['xcentr'],abs(cage.eqdsk['bcentr']),'o')
    sns.despine()
    '''

    # rp.plot_loops()


