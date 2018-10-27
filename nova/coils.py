import numpy as np
from amigo.pyplot import plt
from scipy.interpolate import interp1d
import matplotlib
import collections
from amigo import geom
from nova.loops import Profile, get_oppvar, get_value
from nova.config import Setup
from nova.streamfunction import SF
import nova.cross_coil as cc
from nova.coil_cage import coil_cage
from nova.shape import Shape
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from warnings import warn
from copy import deepcopy
from scipy.optimize import minimize_scalar
from matplotlib.collections import PatchCollection
from matplotlib import patches


class PF:

    def __init__(self, eqdsk=None, **kwargs):
        self.initalize_collection()
        self.set_coils(eqdsk, **kwargs)
        if 'coilset' in kwargs:
            self.__call__(kwargs['coilset'])
        if 'coil' in kwargs:
            coil = kwargs['coil']
            subcoil = kwargs.get('subcoil', None)
            label = kwargs.get('label', None)
            self.add_coils(coil, subcoil=subcoil, label=label)

    def __call__(self, coilset):
        self.coilset = coilset  # reset instance

    def initalize_coils(self):
        self.coilset = {'nC': 0, 'index': {},
                        'coil': collections.OrderedDict(),
                        'subcoil': collections.OrderedDict(),
                        'plasma_coil': collections.OrderedDict()}
        for coil in ['PF', 'CS']:
            self.coilset['index'][coil] = {'n': 0, 'name': [], 'index': []}

    def set_coils(self, eqdsk, **kwargs):
        self.initalize_coils()
        if eqdsk is None:
            self.xo = kwargs.get('xo', [0, 0])
            self.rCS = kwargs.get('rCS', 0)
            self.drCS = kwargs.get('drCS', 0)
        else:
            self.xo = [eqdsk['xcentr'], eqdsk['zmid']]
            if eqdsk['ncoil'] > 0:
                CSindex = np.argmin(eqdsk['xc'])  # CS radius and width
                self.rCS = eqdsk['xc'][CSindex]
                self.drCS = eqdsk['dxc'][CSindex]
                for i, (x, z, dx, dz, It) in enumerate(
                       zip(eqdsk['xc'], eqdsk['zc'], eqdsk['dxc'],
                           eqdsk['dzc'], eqdsk['It'])):
                    self.add_coil(x, z, dx, dz, It, categorize=False)
                    if eqdsk['ncoil'] > 100 and i >= eqdsk['ncoil'] - 101:
                        print('exit set_coil loop - coils')
                        break
            self.categorize_coils()

    def get_coil_current(self):
        It = {}
        for name in self.coilset['coil']:
            It[name] = self.coilset['coil'][name]['It']
        return It

    def update_current(self, It, coilset=None):  # new current passed as dict
        if coilset is None:
            coilset = self.coilset
        for i, name in enumerate(It):
            if name in coilset['coil']:  # skip removed DINA fillaments
                coilset['coil'][name]['It'] = It[name]  # AT
                if 'Nt' in coilset['coil'][name]:
                    Nt = coilset['coil'][name]['Nt']  # turn number
                    coilset['coil'][name]['Ic'] =\
                        coilset['coil'][name]['It'] / Nt
                if 'Nf' not in coilset['coil'][name]:
                    coilset['coil'][name]['Nf'] = 1
                Nf = coilset['coil'][name]['Nf']
                for n in range(Nf):
                    subcoil = '{}_{}'.format(name, n)
                    coilset['subcoil'][subcoil]['If'] = It[name] / Nf

    def categorize_coils(self):
        catogory = np.zeros(len(self.coilset['coil']), dtype=[('x', 'float'),
                            ('z', 'float'), ('theta', 'float'),
                            ('index', 'int'), ('name', 'object')])
        for i, name in enumerate(self.coilset['coil']):
            catogory[i]['x'] = self.coilset['coil'][name]['x']
            catogory[i]['z'] = self.coilset['coil'][name]['z']
            catogory[i]['theta'] =\
                np.arctan2(self.coilset['coil'][name]['z']-self.xo[1],
                           self.coilset['coil'][name]['x']-self.xo[0])
            catogory[i]['index'] = i
            catogory[i]['name'] = name
        CSsort = np.sort(catogory, order=['x', 'z'])  # sort CS, x then z
        CSsort = CSsort[CSsort['x'] < self.rCS + self.drCS]
        PFsort = np.sort(catogory, order='theta')  # sort PF,  z
        PFsort = PFsort[PFsort['x'] > self.rCS + self.drCS]
        self.coilset['index'] = {'PF': {'n': len(PFsort['index']),
                                        'index': PFsort['index'],
                                        'name': PFsort['name']},
                                 'CS': {'n': len(CSsort['index']),
                                        'index': CSsort['index'],
                                        'name': CSsort['name']}}
        self.sort()  # sort pf.coil dict [PF,CS]

    def sort(self):  # order coil dict for use by inverse.py
        coil = deepcopy(self.coilset['coil'])
        self.coilset['coil'] = collections.OrderedDict()
        for name in np.append(self.coilset['index']['PF']['name'],
                              self.coilset['index']['CS']['name']):
            self.coilset['coil'][name] = coil[name]
        nPF = self.coilset['index']['PF']['n']
        nCS = self.coilset['index']['CS']['n']
        self.coilset['index']['PF']['index'] = np.arange(0, nPF)
        self.coilset['index']['CS']['index'] = np.arange(nPF, nPF+nCS)

    def add_coilsets(self, coilset):
        for label in coilset:
            self.add_coils(coilset[label]['coil'],
                           subcoil=coilset[label]['subcoil'], label=label)

    def add_coils(self, coil, subcoil=None, label=None):
        nCo = self.coilset['nC']
        for name in coil:
            x, z = coil[name]['x'], coil[name]['z']
            dx, dz = coil[name]['dx'], coil[name]['dz']
            It = coil[name]['It']
            oparg = {}  # optional keys
            for key in ['R', 'index', 'sign', 'Nt']:
                if key in coil[name]:
                    oparg[key] = coil[name][key]
            self.add_coil(x, z, dx, dz, It, name=name, categorize=False,
                          **oparg)
            if subcoil:
                Nf = coil[name]['Nf']
                for i in range(Nf):
                    subname = name+'_{}'.format(i)
                    self.coilset['subcoil'][subname] = subcoil[subname]
            else:
                coil[name]['Nf'] = 1
                self.coilset['subcoil'][name+'_0'] = coil[name]
        if label:
            self.coilset['index'][label] =\
                {'name': list(coil.keys()),
                 'index': np.arange(nCo, self.coilset['nC'])}

    def add_coil(self, x, z, dx, dz, It, categorize=True, **kwargs):
        self.coilset['nC'] += 1
        name = kwargs.get('name', 'Coil{:1.0f}'.format(self.coilset['nC']))
        self.coilset['coil'][name] = {'x': x, 'z': z, 'dx': dx, 'dz': dz,
                                      'It': It, 'Nf': 1}
        rc = kwargs.get('rc', np.sqrt(dx**2 + dz**2) / 2)
        self.coilset['coil'][name]['rc'] = rc
        for key in ['R', 'index', 'sign', 'Nt']:  # optional keys
            if key in kwargs:
                self.coilset['coil'][name][key] = kwargs.get(key)
        if categorize:
            self.categorize_coils()

    def join_coils(self, name, names):
        coil_o = deepcopy(self.coilset['coil'])
        subcoil_o = deepcopy(self.coilset['subcoil'])
        self.initalize_coils()  # clear coil dicts

        coils = {name_: coil_o[name_] for name_ in names}
        x, z, dx, dz, It, __, Nt, Nf = self.unpack_coils(coil=coils)[1:]
        xbb = [np.min(x-dx/2), np.max(x+dx/2)]  # x bounding box
        zbb = [np.min(z-dz/2), np.max(z+dz/2)]  # z bounding box
        dx = xbb[1]-xbb[0]  # width
        dz = zbb[1]-zbb[0]  # height
        x, z = np.mean(xbb), np.mean(zbb)  # center
        It = np.sum(It)  # Amp-turns
        Nt = np.sum(Nt)  # turns
        Nf = int(np.sum(Nf))  # turns
        insert = {name: {'It': It, 'Nt': Nt, 'Nf': Nf, 'dx': dx, 'dz': dz,
                         'x': x, 'z': z}}
        sub_index = 0
        for name_ in coil_o:
            if name_ in names:
                if name not in self.coilset['coil']:  # replace coil
                    x, z, dx, dz, It = self.unpack_coils(coil=insert)[1:-3]
                    self.add_coil(x[0], z[0], dx[0], dz[0], It[0],
                                  name=name, categorize=False)
                    # turn number
                    self.coilset['coil'][name]['Nt'] = insert[name]['Nt']
                    self.coilset['coil'][name]['Nf'] = insert[name]['Nf']
                for i in range(coil_o[name_]['Nf']):  # filliment number
                    subname = name+'_{}'.format(sub_index)
                    self.coilset['subcoil'][subname] =\
                        subcoil_o[name_+'_{}'.format(i)]
                    sub_index += 1
            else:  # re-insert
                coil_ = {name_: coil_o[name_]}
                x, z, dx, dz, It = self.unpack_coils(coil=coil_)[1:-3]
                self.add_coil(x[0], z[0], dx[0], dz[0], It[0],
                              name=name_, categorize=False)
                # turn number
                self.coilset['coil'][name_]['Nt'] = coil_o[name_]['Nt']
                self.coilset['coil'][name_]['Nf'] = coil_o[name_]['Nf']
                for i in range(coil_o[name_]['Nf']):  # sub-coils
                    subname = name_+'_{}'.format(i)
                    self.coilset['subcoil'][subname] = subcoil_o[subname]
        self.categorize_coils()

    def remove_coil(self, name):
        Nf = self.coilset['coil'][name]['Nf']
        self.coilset['coil'].pop(name)  # remove main coil
        if name+'_0' in self.coilset['subcoil']:
            for i in range(Nf):
                self.coilset['subcoil'].pop('{}_{}'.format(name, i))

    def unpack_coils(self, **kwargs):
        coil = kwargs.get('coil', self.coilset['coil'])
        nc = len(coil.keys())
        It = np.zeros(nc)
        xc, zc = np.zeros(nc), np.zeros(nc)
        dxc, dzc = np.zeros(nc), np.zeros(nc)
        Nt, Nf = np.zeros(nc), np.zeros(nc)
        names = []
        for i, name in enumerate(coil.keys()):
            xc[i] = coil[name]['x']
            zc[i] = coil[name]['z']
            dxc[i] = coil[name]['dx']
            dzc[i] = coil[name]['dz']
            It[i] = coil[name]['It']
            if 'Nt' in coil[name]:
                Nt[i] = coil[name]['Nt']
            if 'Nf' in coil[name]:
                Nf[i] = coil[name]['Nf']
            names.append(name)
        return nc, xc, zc, dxc, dzc, It, names, Nt, Nf

    def mesh_coils(self, dCoil=-1):
        if dCoil < 0:  # dCoil not set, use stored value
            if not hasattr(self, 'dCoil'):
                self.dCoil = 0
        else:
            self.dCoil = dCoil
        self.coilset['subcoil'] = collections.OrderedDict()
        if self.dCoil == 0:
            for name in self.coilset['coil'].keys():
                subname = name + '_0'
                self.coilset['subcoil'][subname] = self.coilset['coil'][name]
                self.coilset['subcoil'][subname]['rc'] = \
                    np.sqrt(self.coilset['coil'][name]['dx']**2 +
                            self.coilset['coil'][name]['dz']**2)
        else:
            self.coilset['subcoil'].clear()  # clear dict but retain referance
            for name in self.coilset['coil'].keys():
                subcoil = self.mesh_coil(
                        self.coilset['coil'][name], self.dCoil)
                for i, subcoil_ in enumerate(subcoil):
                    subname = '{}_{}'.format(name, i)
                    self.coilset['subcoil'][subname] = subcoil_

    @staticmethod
    def mesh_coil(coil, dCoil=None):  # returns subcoil list, mutates coil
        xc, zc = coil['x'], coil['z']
        Dx, Dz = abs(coil['dx']), abs(coil['dz'])
        if dCoil is None:
            dCoil = np.max([Dx, Dz])
        nx = int(np.ceil(Dx / dCoil))
        nz = int(np.ceil(Dz / dCoil))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx, dz = Dx / nx, Dz / nz
        x = xc + np.linspace(dx / 2, Dx - dx / 2, nx) - Dx / 2
        z = zc + np.linspace(dz / 2, Dz - dz / 2, nz) - Dz / 2
        X, Z = np.meshgrid(x, z, indexing='ij')
        X, Z = np.reshape(X, (-1, 1)), np.reshape(Z, (-1, 1))
        Nf = len(X)  # filament number
        coil['Nf'] = Nf  # store filament number
        If = coil['It'] / Nf
        subcoil = [[] for __ in range(Nf)]
        for i, (x, z) in enumerate(zip(X, Z)):
            subcoil[i] = {'x': x, 'z': z, 'dx': dx, 'dz': dz, 'If': If,
                          'rc': np.sqrt(dx**2 + dz**2) / 4}
        return subcoil

    def initalize_collection(self, *args):
        self.coil_patch = {'patch': [], 'Icoil': [], 'Jc': []}
        if len(args) > 0:  # list of additional tracked varibles
            for var in args:
                self.coil_patch[var] = []

    def append_patch(self, patch, **kwargs):
        self.coil_patch['patch'].append(patch)
        for var in kwargs:
            try:
                self.coil_patch[var].append(kwargs[var])
            except KeyError:
                txt = '\ninitalisze kwarg {} '.format(var)
                txt += ' in initalize collection'
                raise KeyError(txt)

    def plot_patch(self, c=None, reset=False, **kwargs):
        cmap = kwargs.get('cmap', plt.cm.RdBu)
        clim = kwargs.get('clim', None)
        pc = PatchCollection(self.coil_patch['patch'], cmap=cmap)
        if c is not None:
            if c in self.coil_patch:  # c == It or Jc
                carray = np.array(self.coil_patch[c])
            else:  # c is np.array len == len(patch)
                npatch = len(self.coil_patch['patch'])
                if isinstance(c, np.ndarray) and len(c) == npatch:
                    carray = c
                else:
                    raise ValueError('color array incompatable with patches')
            if not clim:
                cmax = np.max(abs(carray))
                clim = [-cmax, cmax]
            pc.set_clim(vmin=clim[0], vmax=clim[1])
            pc.set_array(carray)

        ax = plt.gca()
        im = ax.add_collection(pc)
        cb = plt.colorbar(im, orientation='horizontal', aspect=40, shrink=0.6,
                          fraction=0.05)
        if c == 'It':
            cb.set_label('$I_c$ kA')
        elif c == 'Jc':
            cb.set_label('$J_c$ MAm$^{-2}$')
        if reset:  # reset patch collection
            self.initalize_collection()

    def get_coil_label(self, name):
        sname = name.split('_')
        if len(sname) == 1:
            label = name
        elif len(sname) == 2:
            label = sname[0]
        else:
            label = '_'.join(sname[:-1])
        return label

    def patch_coil(self, coils, alpha=1, **kwargs):
        patch = [[] for _ in range(len(coils))]
        color_key = ['CS', 'skip', 'VS3', 'vv_DINA', 'plasma',
                     'PF', 'skip', 'bb_DINA', 'trs', 'skip',
                     'skip', 'skip', 'skip', 'vv', 'skip']
        for i, name in enumerate(coils.keys()):
            coil = coils[name]
            x, z, dx, dz = coil['x'], coil['z'], coil['dx'], coil['dz']
            if 'It' in coil:
                Icoil = coil['It']
            else:
                Icoil = coil['If']
            Jc = Icoil / (dx*dz)
            if Icoil != 0:
                edgecolor = 'k'
            else:
                edgecolor = 'k'
            label = self.get_coil_label(name)
            coil_color = 'C3'  # default
            for group in self.coilset['index']:
                if label in self.coilset['index'][group]['name']\
                        and group in color_key:
                    ic = color_key.index(group)
                    icolor = ic % 5
                    coil_color = 'C{}'.format(icolor)
                    break
            coil_color = kwargs.get('coil_color', coil_color)
            if 'VS' in name:
                patch[i] = patches.Circle(
                        (x, z), dx/2, facecolor=coil_color, alpha=alpha,
                        edgecolor=edgecolor)
            else:
                patch[i] = patches.Rectangle(
                        (x-dx/2, z-dz/2), dx, dz,
                        facecolor=coil_color, alpha=alpha, edgecolor=edgecolor)
            self.append_patch(patch[i], Icoil=1e-3*Icoil, Jc=1e-6*Jc)
        return patch

    def label_coils(self, coils, label, current, fs=12):
        for i, name in enumerate(coils.keys()):
            coil = coils[name]
            x, z, dx, dz = coil['x'], coil['z'], coil['dx'], coil['dz']
            if hasattr(self.coilset, 'index'):
                if name in self.coilset['index']['CS']['name']:
                    drs = -2.5 / 3 * dx
                    ha = 'right'
                else:
                    drs = 2.5 / 3 * dx
                    ha = 'left'
            else:
                drs = 2.5 / 3 * dx
                ha = 'left'

            if label and current:
                zshift = max([dz / 10, 0.2])
            else:
                zshift = 0
            if label:
                plt.text(x + drs, z + zshift, name, fontsize=fs,
                         ha=ha, va='center', color=0.2 * np.ones(3))
            if current:
                if current == 'A' and 'Nt' in coil:  # amps
                    Nt = coil['Nt']
                    linecurrent = coil['It'] / Nt
                    txt = '{:1.1f}kA'.format(linecurrent * 1e-3)
                else:  # amp turns
                    if abs(coil['It']) < 0.1e6:  # display as kA.t
                        txt = '{:1.1f}kA$\cdot$T'.format(coil['It'] * 1e-3)
                    else:  # MA.t
                        txt = '{:1.1f}MA$\cdot$T'.format(coil['It'] * 1e-6)
                plt.text(x + drs, z - zshift, txt,
                         fontsize=fs, ha=ha, va='center',
                         color=0.2 * np.ones(3))

    def plot_coil(self, coils, label=False, current=False,
                  fs=12, alpha=1, plot=True, ax=None, patch=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        if patch:
            patch = self.patch_coil(coils, alpha=alpha, **kwargs)
            pc = PatchCollection(patch[::-1], match_original=True)
            ax.add_collection(pc)
        if label or current:
            self.label_coils(coils, label, current, fs=fs)

    def plot(self, subcoil=False, label=False, plasma=False,
             current=False, alpha=1, ax=None, patch=True):
        if ax is None:
            ax = plt.gca()
        fs = matplotlib.rcParams['legend.fontsize']
        if subcoil:
            coils = self.coilset['subcoil']
        else:
            coils = self.coilset['coil']
        self.plot_coil(coils, label=label, current=current, fs=fs,
                       alpha=alpha, ax=ax, patch=patch)
        if plasma:
            self.plot_coil(self.coilset['plasma_coil'], alpha=alpha,
                           coil_color='C4', ax=ax)
        ax.axis('equal')
        ax.axis('off')

    '''
    def inductance(self, dCoil=0.5, Iscale=1):
        pf = deepcopy(self)
        inv = INV(pf, Iscale=Iscale, dCoil=dCoil)
        pf.mesh_coils(dCoil=dCoil)  # multi-filiment coils
        inv.update_coils()
        Nf = np.array([1 / inv.coil['active'][coil]['Nf']
                       for coil in inv.coil['active']])
        for i, coil in enumerate(inv.adjust_coils):
            x, z = inv.pf.coil[coil]['x'], inv.pf.coil[coil]['z']
            inv.add_psi(1, point=(x, z))
        inv.set_foreground()
        fillaments = np.dot(np.ones((len(Nf), 1)), Nf.reshape(1, -1))
        self.M = 2 * np.pi * inv.G * fillaments  # PF/CS inductance matrix
    '''

    def coil_corners(self, coils):
        X, Z = np.array([]), np.array([])
        Nc = len(coils['id'])
        dX, dZ = np.zeros(Nc), np.zeros(Nc)
        if len(coils['dX']) > 0:
            dX[Nc - len(coils['dX']):] = coils['dX']
        if len(coils['dZ']) > 0:
            dZ[Nc - len(coils['dZ']):] = coils['dZ']
        for Cid, Cdr, Cdz in zip(coils['id'], dX, dZ):
            x = self.coilset['coil']['Coil' + str(Cid)]['x']
            z = self.coilset['coil']['Coil' + str(Cid)]['z']
            dx = self.coilset['coil']['Coil' + str(Cid)]['dx']
            dz = self.coilset['coil']['Coil' + str(Cid)]['dz']
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
        self.initalise_cloop()

    def initalise_cloop(self):  # previously initalise_loop
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
                     'trans_lower', 'trans_upper', 'cl_fe']:
            self.p[loop] = {'x': [], 'z': []}

    def transition_index(self, x_in, z_in, eps=1e-12):
        npoints = len(x_in)
        x_cl = x_in[0] + eps
        upper = npoints - \
            next((i for i, x_in_ in enumerate(x_in) if x_in_ > x_cl))  # +1
        lower = next((i for i, x_in_ in enumerate(x_in) if x_in_ > x_cl))
        top, bottom = np.argmax(z_in), np.argmin(z_in)
        index = {'upper': upper, 'lower': lower, 'top': top, 'bottom': bottom}
        return index

    def loop_dt(self, x, z, dt_in, dt_out, index):
        le = geom.length(x, z)
        L = np.array([0, le[index['lower']], le[index['bottom']],
                      le[index['top']], le[index['upper']], 1])
        dX = np.array([dt_in, dt_in, dt_out, dt_out, dt_in, dt_in])
        dt = interp1d(L, dX)(le)
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
            # TODO check that unwind does not effect stored energy calculation
            # self.p[loop] = unwind({'x': x, 'z': z})
            self.p[loop]['x'], self.p[loop]['z'] = x, z
        return self.p

    def split_loop(self, N=100, plot=False):  # split loop for fe model
        x, z = self.p['cl']['x'], self.p['cl']['z']
        index = self.transition_index(x, z)
        upper, lower = index['upper'], index['lower']
        top, bottom = index['top'], index['bottom']

        p = {}
        for part in ['trans_lower', 'loop', 'trans_upper', 'nose']:
            p[part] = {}
        p['nose']['x'] = np.append(x[upper-1:-1], x[:lower + 1])
        p['nose']['z'] = np.append(z[upper-1:-1], z[:lower + 1])
        p['trans_lower']['x'] = x[lower:bottom]
        p['trans_lower']['z'] = z[lower:bottom]
        p['trans_upper']['x'] = x[top:upper]
        p['trans_upper']['z'] = z[top:upper]
        p['loop']['x'] = x[bottom-1:top+1]
        p['loop']['z'] = z[bottom-1:top+1]
        Lcl = geom.length(x, z, norm=False)[-1]
        nL = N/Lcl  # nodes per length
        self.p['cl_fe']['x'] = np.array([p['nose']['x'][-1]])
        self.p['cl_fe']['z'] = np.array([p['nose']['z'][-1]])
        Nstart = 0
        for part in ['trans_lower', 'loop', 'trans_upper', 'nose']:
            Lpart = geom.length(p[part]['x'], p[part]['z'], norm=False)[-1]
            Npart = int(nL * Lpart)+1
            self.p[part]['x'], self.p[part]['z'] = \
                geom.space(p[part]['x'], p[part]['z'], Npart)
            self.p[part]['nd'] = np.arange(Nstart, Nstart+Npart)
            for var in ['x', 'z']:  # append centerline
                self.p['cl_fe'][var] = np.append(self.p['cl_fe'][var],
                                                 self.p[part][var][1:])
            Nstart += Npart - 1
        self.p['nose']['nd'][-1] = 0
        self.p['cl_fe']['x'] = self.p['cl_fe']['x'][:-1]
        self.p['cl_fe']['z'] = self.p['cl_fe']['z'][:-1]

        '''
        p['nose']['x'] = np.append(x[upper-1:-1], x[:lower + 1])
        p['nose']['z'] = np.append(z[upper-1:-1], z[:lower + 1])
        self.p['nose']['nd'] = np.append(np.arange(upper-1, len(x)-1),
                                         np.arange(0, lower+1))
        self.p['trans_lower']['x'] = x[lower:bottom]
        self.p['trans_lower']['z'] = z[lower:bottom]
        self.p['trans_lower']['nd'] = np.arange(lower, bottom)
        self.p['trans_upper']['x'] = x[top:upper]
        self.p['trans_upper']['z'] = z[top:upper]
        self.p['trans_upper']['nd'] = np.arange(top, upper)
        self.p['loop']['x'] = x[bottom-1:top+1]
        self.p['loop']['z'] = z[bottom-1:top+1]
        self.p['loop']['nd'] = np.arange(bottom-1, top+1)
        '''

        if plot:
            plt.plot(self.p['cl_fe']['x'], self.p['cl_fe']['z'], 'o')
            for name in ['nose', 'loop', 'trans_lower', 'trans_upper']:
                x, z = self.p[name]['x'], self.p[name]['z']
                plt.plot(x, z)

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
            le = geom.length(x, z)
            lt = np.linspace(trim[0], trim[1], int(np.diff(trim) * len(le)))
            x, z = interp1d(le, x)(lt), interp1d(le, z)(lt)
            le = np.linspace(0, 1, len(x))
            self.fun[side] = {'x': IUS(le, x), 'z': IUS(le, z)}
            self.fun[side]['L'] = geom.length(x, z, norm=False)[-1]
            self.fun[side]['dx'] = self.fun[side]['x'].derivative()
            self.fun[side]['dz'] = self.fun[side]['z'].derivative()

    def xzL(self, points):
        ''' translate list of x,z points into normalised TF loop lengths '''
        self.loop_interpolators(offset=0)  # construct TF interpolators
        TFloop = self.fun['out']  # outer loop
        L = np.zeros(len(points))
        for i, p in enumerate(points):
            L[i] = minimize_scalar(self.norm, method='bounded',
                                   args=(TFloop, p), bounds=[0, 1]).x
        return L

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

    def eq_boundary(self, expand=0):  # generate boundary dict for elliptic
        X, Z = self.p['cl']['x'], self.p['cl']['z']
        boundary = {'X': X, 'Z': Z, 'expand': expand}
        return boundary

    def fill(self, write=False, alpha=1, plot=True, plot_cl=False,
             color=[0.4*np.ones(3), 0.6*np.ones(3)]):
        # TODO: write and plot?
        geom.polyparrot(self.p['in'], self.p['wp_in'],
                        color=color[0], alpha=alpha)
        geom.polyparrot(self.p['wp_in'], self.p['wp_out'],
                        color=color[1], alpha=alpha)
        geom.polyparrot(self.p['wp_out'], self.p['out'],
                        color=color[0], alpha=alpha)
        if plot_cl:  # plot winding pack centre line
            plt.plot(self.p['cl']['x'], self.p['cl']['z'],
                     '-.', color=0.5 * np.ones(3))
        plt.axis('equal')
        plt.axis('off')

    def plot_XZ(self, alpha=1):
        ax = plt.gca()
        for p in [self.p['in'], self.p['wp_in'],
                  self.p['wp_out'], self.p['out']]:
            ax.plot(p['x'], p['z'], color='k')
        geom.polyparrot(self.p['in'], self.p['wp_in'],
                        color='C1', alpha=alpha)
        geom.polyparrot(self.p['wp_in'], self.p['wp_out'],
                        color='C2', alpha=alpha)
        geom.polyparrot(self.p['wp_out'], self.p['out'],
                        color='C3', alpha=alpha)

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
        # tailor limits on loop parameters (l -> loop tension)
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
    sf = SF(filename=setup.filename)
    sf.contour()
    profile = Profile(config['TF'], family='S', part='TF', nTF=nTF,
                      obj='L', load=True, npoints=1000)

    tf = TF(profile=profile, sf=sf, nr=1, ny=1)
    tf.split_loop(plot=False)
    tf.fill()

    pf = PF(sf.eqdsk)
    pf.mesh_coils(dCoil=0.25)
    pf.plot(subcoil=True)

    '''
    demo = DEMO()
    demo.fill_part('Vessel')
    demo.fill_part('Blanket')
    demo.plot_ports()

    # tf.minimise(demo.parts['Vessel']['out'], verbose=True, ripple=True)
    tf.fill()

    pf = PF(sf.eqdsk)
    pf.plot()
    sf.contour()

    # tf.plot_XZ()

    # tf.cage.output()
    '''

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

    plt.plot(tf.x['cl']['x'],abs(B[:,1]))
    plt.plot(xcl,abs(Bcl[:,1]))
    plt.plot(cage.eqdsk['xcentr'],abs(cage.eqdsk['bcentr']),'o')
    sns.despine()
    '''

    # rp.plot_loops()
