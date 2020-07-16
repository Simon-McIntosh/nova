import copy
import time
import multiprocessing
from itertools import cycle
from warnings import warn
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp1d
import scipy.optimize as op
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
import nlopt

from amigo import geom
from amigo.time import clock
from amigo.pyplot import plt
#from nova import loops
#import nova.cross_coil as cc
#from nova.coils import PF
from nova.electromagnetic.streamfunction import SF
#from nova.force import force_field
from nep.DINA.tie_plate import get_tie_plate


class INV(object):

    def __init__(self, coilset, tf=None, Jmax=12.5, offset=0.3, svd=False,
                 Iscale=1e6, dCoil=None, eqdsk={}, boundary='tf',
                 set_force=False, wp='Soft'):
        self.coilset = coilset  # requires update
        if set_force:
            self.ff = force_field(self.coilset)
        self.sf = SF(eqdsk=eqdsk)
        self.wsqrt = 1
        self.svd = svd  # coil current singular value decomposition flag
        self.Iscale = Iscale  # set current units to MA
        self.dCoil = dCoil  # coil mesh length
        self.boundary = boundary  # set boundary generator 'sf' or 'tf'
        self.ncpu = multiprocessing.cpu_count()
        self.tf = tf
        self.add_active([])  # all coils active
        self.initalise_coil()
        self.initalise_limits()
        self.offset = offset  # offset coils from outer TF loop
        if tf is not None:
            dx = [self.coilset['coil'][name]['dx'] for
                  name in self.coilset['coil']]
            self.norm = np.sqrt(2)*np.mean(dx)/2
            self.tf.loop_interpolators(offset=self.offset)
            self.unwrap()  # store relitive coil posions around TF
        self.Jmax = Jmax  # MAm-2
        self.initalize_fix()
        self.cnstr = {'x': np.array([]), 'z': np.array([]), 'BC': np.array([]),
                      'value': np.array([]), 'condition': np.array([])}
        self.set_swing()
        self.update_coils()
        self.log_scalar = ['rss', 'rssID']
        # 'Fx_max','Fz_max','F_max','FzCS','Fsep','rss_bndry','Isum',
        # 'Imax','z_offset','FxPF','FzPF','dTpol'
        self.log_array = ['Lo']  # ,'Iswing'
        self.log_iter = ['current_iter', 'plasma_iter', 'position_iter']
        self.log_plasma = ['plasma']
        self.CS_Lnorm, self.CS_Znorm = 2, 1
        self.rhs = False  # colocation status
        self.tie_plate = get_tie_plate()

    def initalize_fix(self):
        self.fix = {'x': np.array([]), 'z': np.array([]), 'BC': np.array([]),
                    'value': np.array([]), 'Bdir': np.array([[], []]).T,
                    'factor': np.array([]), 'n': 0}

    def colocate(self, eqdsk=None, psi=True, field=True, Xpoint=True,
                 targets=True, SX=False, npoint=25):
        self.initialize_log()
        if eqdsk:
            self.sf.update_eqdsk(eqdsk)  # update streamfunction
        self.psi_spline()  # generate psi interpolators
        self.plasma_spline()  # generate plasma interpolators
        self.fix_boundary(psi, field, Xpoint, npoint=npoint)
        self.fix_SX_outer_target(SX)
        self.fix_divertor_targets(targets)
        self.get_weight()
        self.set_background()
        self.set_foreground()
        self.rhs = True

    def fix_SX_outer_target(self, SX, **kwargs):
        Rex = kwargs.get('Rex', 1.5)  # radius from x-point
        arg = kwargs.get('arg', 40)  # angle from x-point
        if SX:
            self.sf.legs.pop(-1)  # replace outer
            self.fix_SX(Rex=Rex, arg=arg)

    def fix_boundary(self, psi=True, field=False, Xpoint=False,
                     npoint=25):
        if psi:  # add boundary points
            self.fix_boundary_psi(npoint=npoint, alpha=1-1e-4, factor=1)
        if field:  # add boundary field
            self.fix_boundary_field(npoint=npoint, alpha=1-1e-4, factor=1)
        if Xpoint:  # Xpoints
            self.fix_null(factor=1, point=self.sf.Xpoint_array[0])
            if len(self.sf.Xpoint_array) == 2:
                self.fix_null(factor=1, point=self.sf.Xpoint_array[1])

    def fix_divertor_targets(self, targets, factor=1):
        if targets:
            for target in self.sf.targets:
                L2D = self.sf.targets[target]['L2D']
                Xsol, Zsol = self.sf.snip(target, 0, L2D)
                if 'strike' in self.sf.targets[target]:
                    point = self.sf.targets[target]['strike']
                else:
                    point = (Xsol[-1], Zsol[-1])
                field_angle = np.arctan2(Zsol[-1]-Zsol[-2], Xsol[-1]-Xsol[-2])
                Bdir = np.array([-np.sin(field_angle), np.cos(field_angle)])
                Bdir /= np.linalg.norm(Bdir)
                self.add_alpha(1, factor=factor, point=point, Bdir=Bdir)
                self.add_B(0, [field_angle*180/np.pi], factor=factor,
                           point=point)

    def fix_SX(self, Rex=1.5, arg=40):  # fix outer leg
        R = self.eq.sf.Xpoint[0] * (Rex-1) / np.sin(arg*np.pi/180)
        target = (R, arg)
        self.add_alpha(1, factor=1, polar=target)  # target psi
        self.add_B(0, [-20], factor=3, polar=target)  # target alignment field)

    def get_flux(self, centre=0, width=363/(2*np.pi), array=[-0.5, 0, 0.5],
                 **kwargs):
        flux = centre - np.array(array) * width  # Webber/rad
        return flux

    def set_swing(self, **kwargs):
        if 'flux' in kwargs:
            flux = kwargs['flux']
        else:
            flux = self.get_flux(**kwargs)
        self.nS = len(flux)
        self.swing = {'flux': flux, 'It': np.zeros((self.nS, self.nC)),
                      'rss': np.zeros(self.nS),
                      'FxPF': np.zeros(self.nS), 'FzPF': np.zeros(self.nS),
                      'FsepCS': np.zeros(self.nS), 'FzCS': np.zeros(self.nS),
                      'Isum': np.zeros(self.nS),
                      'IsumCS': np.zeros(self.nS), 'IsumPF': np.zeros(self.nS),
                      'Imax': np.zeros(self.nS),
                      'IcCSmax': np.zeros(self.nS),
                      'Tpol': np.zeros(self.nS),
                      'rss_bndry': np.zeros(self.nS),
                      'z_offset': np.zeros(self.nS), 'dTpol': 0,
                      'Fcoil': [[] for _ in range(self.nS)]}

    def update_swing(self):
        # resize current vector
        self.swing['If'] = np.zeros((self.nS, self.nC))

    def initalise_limits(self):
        self.limit = {'I': {}, 'L': {}, 'F': {}}

    def set_PF_limit(self):
        for coil in self.PFcoils:
            self.limit['L'][coil] = self.limit['L']['PF']

    def set_limit(self, side='both', eps=1e-2, **kwargs):
        # set as ICSsum for [I][CSsum] etc...
        if side == 'both' or side == 'equal':
            index = [0, 1]
        elif side == 'lower':
            index = [0]
        elif side == 'upper':
            index = [1]
        else:
            errtxt = 'invalid side parameter [both, lower, upper]'
            errtxt += ': {}'.format(side)
            raise IndexError(errtxt)
        for key in kwargs:
            variable = key[0]
            if key[1:] not in self.limit[variable]:  # initalize limit
                self.limit[variable][key[1:]] = [-1e16, 1e16]
            if kwargs[key] is None:
                for i in index:
                    sign = self.sign_limit(i, side)
                    self.limit[variable][key[1:]][i] = sign * 1e16
            else:  # set limit(s)
                for i in index:
                    sign = self.sign_limit(i, side)
                    if side == 'equal':
                        value = sign * eps + kwargs[key]
                    else:
                        value = sign * kwargs[key]
                    self.limit[variable][key[1:]][i] = value
        if 'LPF' in kwargs:
            self.set_PF_limit()

    def sign_limit(self, index, side):
        # only apply sign to symetric limits (side==both)
        if side in ['both', 'equal']:
            sign = 2*index-1
        else:
            sign = 1
        return sign

    def Cname(self, coil):
        if isinstance(coil, str):
            name = coil
        else:  # coil number in place of name
            name = 'Coil{:1.0f}'.format(coil)  # convert to name
        return name

    def initalise_coil(self):
        self.nC = len(self.adjust_coils)  # number of active coils
        self.nPF = len(self.PFcoils)
        self.nCS = len(self.CScoils)
        self.coil = {'active': OrderedDict(), 'passive': OrderedDict()}
        names = list(self.coilset['coil'].keys())
        self.all_coils = names.copy()
        self.adjust_coils = names.copy()
        names = np.append(names, 'Plasma')
        for name in names:
            if name in self.adjust_coils:
                state = 'active'
            else:
                state = 'passive'
            self.coil[state][name] = {'x': np.array([]), 'z': np.array([]),
                                      'dx': np.array([]), 'dz': np.array([]),
                                      'subname': np.array([]),
                                      'If': np.array([]), 'Fx': np.array([]),
                                      'Fz': np.array([]),
                                      'Fx_sum': 0, 'Fz_sum': 0, 'Isum': 0,
                                      'Xo': 0, 'Zo': 0}

    def initalise_current(self):
        adjust_coils = self.coil['active'].keys()
        self.If = np.zeros((len(adjust_coils)))
        for i, name in enumerate(adjust_coils):
            self.If[i] = self.coilset['subcoil'][name + '_0']['If']
            self.If[i] /= self.Iscale
        self.alpha = np.zeros(self.nC)  # initalise alpha

    def update_coils(self, plot=False, regrid=False):
        self.initalise_coil()
        self.append_subcoil(self.coilset['subcoil'])
        self.append_subcoil(self.coilset['plasma'])
        if regrid:
            for coil in self.all_coils:
                self.update_bundle(coil)  # re-grid subcoils
        self.update_swing()
        self.initalise_current()
        if plot:
            self.plot_coils(plasma=True)

    def inductance(self):
        fix_o = copy.deepcopy(self.fix)  # store current BCs
        if hasattr(self, 'G'):  # store coupling matrix
            self.Go = self.G.copy()
        self.initalize_fix()  # reinitalize BC vector
        for i, name in enumerate(self.adjust_coils):
            coil = self.coil['active'][name]
            X, Z = coil['x'], coil['z']
            for x, z in zip(X, Z):
                x = self.coilset['coil'][name]['x']
                z = self.coilset['coil'][name]['z']
                self.add_psi(1, point=(x, z))
        self.set_foreground()
        Gi = np.zeros((self.nC, self.nC))  # inductance coupling matrix
        Ncount = 0
        for i, name in enumerate(self.adjust_coils):
            Nf = self.coilset['coil'][name]['Nf']
            Gi[i, :] = np.sum(self.G[Ncount:Ncount+Nf, :], axis=0)
            Ncount += Nf
        Gi /= self.Iscale  # ensure coil currents [A]
        turns = np.array([self.coilset['coil'][name]['Nt']
                          for name in self.coil['active']])
        turns = np.dot(turns.reshape(-1, 1), turns.reshape(1, -1))
        fillaments = np.array([self.coilset['coil'][name]['Nf']
                               for name in self.coil['active']])
        fillaments = np.dot(fillaments.reshape(-1, 1),
                            fillaments.reshape(1, -1))
        # PF/CS inductance matrix
        self.Mc = 2 * np.pi * Gi / fillaments  # inductance [H]
        self.Mt = self.Mc * turns  # inductance [H]
        self.fix = fix_o  # reset BC vector
        if hasattr(self, 'Go'):  # reset coupling matrix
            self.G = self.Go
            del self.Go

    def update_plasma(self):
        self.clear_plasma()
        self.append_subcoil(self.coilset['plasma'])

    def clear_plasma(self):
        for key in ['x', 'z', 'If', 'dx', 'dz', 'subname', 'Fx', 'Fz']:
            self.coil['passive']['Plasma'][key] = np.array([])

    def append_subcoil(self, subcoil):
        for subname in subcoil.keys():
            name = '_'.join(subname.split('_')[:-1])
            if name in self.adjust_coils:
                state = 'active'
            else:
                state = 'passive'
            coil = subcoil[subname]
            for key, var in zip(['x', 'z', 'If', 'dx', 'dz', 'subname'],
                                [coil['x'], coil['z'], coil['If'],
                                 coil['dx'], coil['dz'], subname]):
                self.coil[state][name][key] = \
                    np.append(self.coil[state][name][key], var)
            self.coil[state][name]['It'] = \
                np.sum(self.coil[state][name]['If'])
            self.coil[state][name]['Xo'] = np.mean(self.coil[state][name]['x'])
            self.coil[state][name]['Zo'] = np.mean(self.coil[state][name]['z'])

    def add_active(self, Clist, Ctype=None, empty=False):  # list of coil names
        if empty:
            self.adjust_coils = []
            self.PFcoils = []
            self.CScoils = []
        if Clist:
            for coil in Clist:
                name = self.Cname(coil)
                if name not in self.adjust_coils:
                    self.adjust_coils.append(name)
                if Ctype == 'PF' and name not in self.PFcoils:
                    self.PFcoils.append(name)
                elif Ctype == 'CS' and name not in self.CScoils:
                    self.CScoils.append(name)
        else:
            self.adjust_coils = list(self.coilset['coil'].keys())  # add all
            self.PFcoils = list(self.coilset['index']['PF']['name'])
            self.CScoils = list(self.coilset['index']['CS']['name'])

    def remove_active(self, Clist=[], Ctype='all', full=False):
        # Clist == list of coil names
        if full:
            self.adjust_coils = list(self.coilset['coil'].keys())  # add all
            if Ctype == 'PF':
                self.PFcoils = list(self.coilset['coil'].keys())
            elif Ctype == 'CS':
                self.CScoils = list(self.coilset['coil'].keys())
        if len(Clist) > 0:
            for name in Clist.copy():
                name = self.Cname(name)  # prepend 'Coil' if name not str
                if (Ctype == 'PF' or Ctype == 'all') and name in self.PFcoils:
                    self.PFcoils.pop(self.PFcoils.index(name))
                if (Ctype == 'CS' or Ctype == 'all') and name in self.CScoils:
                    self.CScoils.pop(self.CScoils.index(name))
                if name in self.adjust_coils:
                    self.adjust_coils.pop(self.adjust_coils.index(name))

        else:
            self.adjust_coils = []  # remove all
            self.PFcoils = []
            self.CScoils = []

    def remove_coil(self, Clist):
        for coil in Clist:
            name = self.Cname(coil)
            if name in self.coilset['coil'].keys():
                del self.coilset['coil'][name]
                for i in range(self.coilset['subcoil'][name + '_0']['Nf']):
                    del self.coilset['subcoil'][name + '_{:1.0f}'.format(i)]
        self.remove_active(Clist)
        self.update_coils()

    def get_point(self, **kwargs):
        keys = kwargs.keys()
        if 'point' in keys:
            x, z = kwargs['point']
        elif 'polar' in keys:
            mag, arg = kwargs['polar']
            x = self.eq.sf.Xpoint[0] + mag * np.sin(arg * np.pi / 180)
            z = self.eq.sf.Xpoint[1] - mag * np.cos(arg * np.pi / 180)
        elif 'Lout' in keys:
            L = kwargs['Lout']
            x, z = self.tf.fun['out']['x'](L), self.tf.fun['out']['z'](L)
            dx, dz = self.tf.fun['out']['dx'](L), self.tf.fun['out']['dz'](L)
        elif 'Lin' in keys:
            L = kwargs['Lin']
            x, z = self.tf.fun['in']['x'](L), self.tf.fun['in']['z'](L)
            dx, dz = self.tf.fun['in']['dx'](L), self.tf.fun['in']['dz'](L)
        if 'norm' in keys and 'point' not in keys:
            delta = kwargs['norm'] * \
                np.array([dx, dz]) / np.sqrt(dx**2 + dz**2)
            x += delta[1]
            z -= delta[0]
        return x, z

    def plot_coils(self, plasma=True, ax=None):
        if ax is None:
            ax = plt.gca()
        ic = cycle(range(10))
        for state, marker in zip(['passive', 'active'], ['o', '*']):
            for name in self.coil[state].keys():
                if name != 'Plasma' or plasma:
                    color = 'C{}'.format(next(ic))
                    for x, z in zip(self.coil[state][name]['x'],
                                    self.coil[state][name]['z']):
                        ax.plot(x, z, marker, color='w', markersize=5)
                        ax.plot(x, z, marker, color=color, markersize=2)

    def add_fix(self, x, z, value, Bdir, BC, factor):
        var = {'x': x, 'z': z, 'value': value,
               'Bdir': Bdir, 'BC': BC, 'factor': factor}
        nvar = len(x)
        self.fix['n'] += nvar
        for name in ['value', 'Bdir', 'BC', 'factor']:
            if np.shape(var[name])[0] != nvar:
                var[name] = np.array([var[name]] * nvar)
        for name in var.keys():
            if name == 'Bdir':
                for i in range(nvar):
                    norm = np.sqrt(var[name][i][0]**2 + var[name][i][1]**2)
                    if norm != 0:
                        var[name][i] /= norm  # normalise tangent vectors
                self.fix[name] = np.append(self.fix[name], var[name], axis=0)
            else:
                self.fix[name] = np.append(self.fix[name], var[name])

    def fix_flux(self, flux):
        if not hasattr(self, 'fix_o'):  # set once
            self.fix_o = copy.deepcopy(self.fix)
        for i, (bc, value) in enumerate(zip(self.fix_o['BC'],
                                            self.fix_o['value'])):
            if 'psi' in bc:
                self.fix['value'][i] = value + flux
        self.set_target()  # adjust target flux

    def get_boundary(self, npoint=21, alpha=0.995):
        x, z = self.sf.get_boundary(alpha=alpha, boundary_cntr=False)
        Xindex = np.argmin((x-self.sf.Xpoint[0])**2 + (z-self.sf.Xpoint[1])**2)
        x = np.append(x[Xindex:], x[:Xindex])
        z = np.append(z[Xindex:], z[:Xindex])
        psi_x = -self.psi.ev(x, z, dx=1, dy=0)
        psi_z = -self.psi.ev(x, z, dx=0, dy=1)
        L = geom.length(x, z)
        Lc = np.linspace(0, 1, npoint + 2)[1:-1]
        xb, zb = interp1d(L, x)(Lc), interp1d(L, z)(Lc)
        Bdir = np.array([interp1d(L, psi_x)(Lc), interp1d(L, psi_z)(Lc)]).T
        return xb, zb, Bdir

    def get_psi(self, alpha):
        Xpsi = self.sf.Xpsi
        Mpsi = self.sf.Mpsi
        psi = Mpsi + alpha * (Xpsi - Mpsi)
        return psi

    def fix_boundary_psi(self, npoint=21, alpha=0.995, factor=1):
        x, z, Bdir = self.get_boundary(npoint=npoint, alpha=alpha)
        psi = self.get_psi(alpha) * np.ones(npoint)
        # psi -= self.sf.Xpsi  # normalise
        self.add_fix(x, z, psi, Bdir, ['psi_bndry'], [factor])

    def fix_boundary_field(self, npoint=21, alpha=0.995, factor=1):
        x, z, Bdir = self.get_boundary(npoint=npoint, alpha=alpha)
        self.add_fix(x, z, [0.0], Bdir, ['Bdir'], [factor])

    def fix_null(self, factor=1, **kwargs):
        x, z = self.get_point(**kwargs)
        self.add_fix([x], [z], [0.0], np.array(
            [[1.0], [0.0]]).T, ['Bx'], [factor])
        self.add_fix([x], [z], [0.0], np.array(
            [[0.0], [1.0]]).T, ['Bz'], [factor])
        psi = self.sf.Ppoint((x, z))
        # psi -= self.sf.Xpsi  # normalise
        self.add_psi(psi, factor=factor, label='psi_x', **kwargs)

    def add_Bxo(self, factor=1, **kwargs):
        x, z = self.get_point(**kwargs)
        self.add_fix([x], [z], [0.0], np.array(
            [[1.0], [0.0]]).T, ['Bx'], [factor])

    def add_Bzo(self, factor=1, **kwargs):
        x, z = self.get_point(**kwargs)
        self.add_fix([x], [z], [0.0], np.array(
            [[0.0], [1.0]]).T, ['Bz'], [factor])

    def add_B(self, B, Bdir, factor=1, zero_norm=False, **kwargs):
        x, z = self.get_point(**kwargs)
        if len(Bdir) == 1:  # normal angle from horizontal in degrees
            arg = Bdir[0]
            Bdir = [-np.sin(arg * np.pi / 180), np.cos(arg * np.pi / 180)]
        Bdir /= np.sqrt(Bdir[0]**2 + Bdir[1]**2)
        self.add_fix([x], [z], [B], np.array([[Bdir[0]], [Bdir[1]]]).T,
                     ['Bdir'], [factor])
        if zero_norm:
            self.add_fix([x], [z], [0], np.array([[-Bdir[1]], [Bdir[0]]]).T,
                         ['Bdir'], [factor])

    def add_theta(self, theta, factor=1, graze=1.5, **kwargs):
        x, z = self.get_point(**kwargs)
        Bm = np.abs(self.eq.sf.bcentr * self.eq.sf.rcentr)  # toroidal moment
        Bphi = Bm / x  # torodal field
        Bp = Bphi / np.sqrt((np.sin(theta * np.pi / 180) /
                             np.sin(graze * np.pi / 180))**2)
        self.add_fix([x], [z], [Bp], np.array([[0], [0]]).T, ['Bp'], [factor])

    def add_psi(self, psi, factor=1, **kwargs):
        x, z = self.get_point(**kwargs)
        label = kwargs.get('label', 'psi')
        Bdir = kwargs.get('Bdir', np.array([0, 0]))
        self.add_fix([x], [z], [psi], Bdir, [label], [factor])

    def add_alpha(self, alpha, factor=1, **kwargs):
        psi = self.get_psi(alpha)
        # psi -= self.sf.Xpsi  # normalise
        self.add_psi(psi, factor=factor, **kwargs)

    def add_Bcon(self, B, **kwargs):
        x, z = self.get_point(**kwargs)
        self.add_cnstr([x], [z], ['B'], ['gt'], [B])

    def plot_fix(self, tails=True):
        self.get_weight()
        if self.fix['n'] > 0:
            if hasattr(self, 'wsqrt'):
                weight = self.wsqrt / np.mean(self.wsqrt)
            else:
                weight = np.ones(len(self.fix['BC']))
            psi, Bdir, Bxz = [], [], []
            tail_length = 0.75
            for bc, w in zip(self.fix['BC'], weight):
                if 'psi' in bc:
                    psi.append(w)
                elif bc == 'Bdir':
                    Bdir.append(w)
                elif bc == 'Bx' or bc == 'Bz':
                    Bxz.append(w)
            if len(psi) > 0:
                psi_norm = tail_length / np.mean(psi)
            if len(Bdir) > 0:
                Bdir_norm = tail_length / np.mean(Bdir)
            if len(Bxz) > 0:
                Bxz_norm = tail_length / np.mean(Bxz)
            for x, z, bc, bdir, w in zip(self.fix['x'], self.fix['z'],
                                         self.fix['BC'], self.fix['Bdir'],
                                         weight):
                if bdir[0]**2 + bdir[1]**2 == 0:  # tails
                    direction = [0, -1]
                else:
                    direction = bdir
                # else:
                #    d_dx,d_dz = self.get_gradients(bc,x,z)
                #    direction = np.array([d_dx,d_dz])/np.sqrt(d_dx**2+d_dz**2)
                if 'psi' in bc:
                    norm = psi_norm
                    marker, size, color = 'o', 7.5, 'C0'
                    plt.plot(x, z, marker, color=color, markersize=size)
                    plt.plot(x, z, marker, color=[1, 1, 1],
                             markersize=0.3 * size)
                else:
                    if bc == 'Bdir':
                        norm = Bdir_norm
                        marker, size, color, mew = 'o', 4, 'C1', 0.0
                    elif bc == 'null':
                        norm = Bxz_norm
                        marker, size, color, mew = 'o', 2, 'C2', 0.0
                    elif bc == 'Bx':
                        norm = Bxz_norm
                        marker, size, color, mew = '_', 10, 'C2', 1.75
                    elif bc == 'Bz':
                        norm = Bxz_norm
                        marker, size, color, mew = '|', 10, 'C2', 1.75
                    plt.plot(x, z, marker, color=color, markersize=size,
                             markeredgewidth=mew)
                if tails:
                    plt.plot([x, x + direction[0] * norm * w],
                             [z, z + direction[1] * norm * w],
                             color=color, linewidth=2)
            plt.axis('equal')

    def set_target(self):
        self.T = (self.fix['value'] - self.BG).reshape((len(self.BG), 1))
        self.wT = self.wsqrt * self.T

    def set_background(self):
        self.BG = np.zeros(len(self.fix['BC']))  # background
        self.add_value('passive')

    def set_foreground(self):
        self.G = np.zeros((self.fix['n'], self.nC))  # [G][If] = [T]
        self.add_value('active')
        self.wG = self.wsqrt * self.G
        if self.svd:  # singular value dec.
            self.U, self.S, self.V = np.linalg.svd(self.wG)
            self.wG = np.dot(self.wG, self.V)  # solve in terms of alpha

    def add_value(self, state):
        Xf, Zf, BC, Bdir, nfix = self.unpack_fix()
        tick = clock(nfix)
        for n, (xf, zf, bc, bdir) in enumerate(zip(Xf, Zf, BC, Bdir)):
            for m, name in enumerate(self.coil[state].keys()):
                coil = self.coil[state][name]
                X, Z, If = coil['x'], coil['z'], coil['If']
                dX, dZ = coil['dx'], coil['dz']
                for x, z, i, dx, dz in zip(X, Z, If, dX, dZ):
                    value = self.add_value_coil(bc, xf, zf, x, z, bdir, dx, dz)
                    if state == 'active':
                        self.G[n, m] += value
                    elif state == 'passive':
                        self.BG[n] += i * value / self.Iscale
                    else:
                        errtxt = 'specify coil state'
                        errtxt += '\'active\', \'passive\'\n'
                        raise ValueError(errtxt)
            if state == 'active' and nfix > 500:
                if n == 0:
                    print('computing foreground coupling')
                tick.tock()

    def add_value_coil(self, bc, xf, zf, x, z, bdir, dx, dz):
        if 'psi' in bc:
            # flux Wb/rad.A
            value = self.Iscale * cc.mu_o * \
                cc.green(xf, zf, x, z, dXc=dx, dZc=dz)
        else:
            # field T/A
            B = 2 * np.pi * cc.mu_o * cc.green_field(xf, zf, x, z)
            B *= self.Iscale
            value = self.Bfield(bc, B[0], B[1], bdir)
        return value

    def add_value_plasma(self, bc, xf, zf, bdir):
        if 'psi' in bc:
            value = self.psi_plasma.ev(xf, zf)  # interpolate from GS
        else:
            Bx, Bz = self.B_plasma[0].ev(xf, zf), self.B_plasma[1].ev(xf, zf)
            value = self.Bfield(bc, Bx, Bz, bdir)
        return value

    def Bfield(self, bc, Bx, Bz, Bdir):
        if bc == 'Bx':
            value = Bx
        elif bc == 'Bz':
            value = Bz
        elif bc == 'null':
            value = np.sqrt(Bz**2 + Bz**2)
        elif bc == 'Bdir':
            nhat = Bdir / np.sqrt(Bdir[0]**2 + Bdir[1]**2)
            value = np.dot([Bx, Bz], nhat)
        elif bc == 'Bp':
            value = Bx - Bz / 2
        return value

    def psi_spline(self):
        self.psi = RBS(self.sf.x, self.sf.z, self.sf.psi)
        psi_x = self.psi.ev(self.sf.x2d, self.sf.z2d, dx=1, dy=0)
        psi_z = self.psi.ev(self.sf.x2d, self.sf.z2d, dx=0, dy=1)
        Bx, Bz = -psi_z / self.sf.x2d, psi_x / self.sf.x2d
        B = np.sqrt(Bx**2 + Bz**2)
        self.B = RBS(self.sf.x, self.sf.z, B)

    def plasma_spline(self):
        self.B_plasma = [[], []]
        psi_pl = cc.get_coil_psi(self.sf.x2d, self.sf.z2d,
                                 self.coilset['subcoil'],
                                 self.coilset['plasma'], set_pf=False)
        self.psi_plasma = RBS(self.sf.x, self.sf.z, psi_pl)
        psi_plasma_r = self.psi_plasma.ev(self.sf.x2d, self.sf.z2d, dx=1, dy=0)
        psi_plasma_z = self.psi_plasma.ev(self.sf.x2d, self.sf.z2d, dx=0, dy=1)
        Bplasma_r = -psi_plasma_z / self.sf.x2d
        Bplasma_z = psi_plasma_r / self.sf.x2d
        self.B_plasma[0] = RBS(self.sf.x, self.sf.z, Bplasma_r)
        self.B_plasma[1] = RBS(self.sf.x, self.sf.z, Bplasma_z)

    def unpack_fix(self):
        Xf, Zf = self.fix['x'], self.fix['z']
        BC, Bdir = self.fix['BC'], self.fix['Bdir']
        n = self.fix['n']
        return Xf, Zf, BC, Bdir, n

    def get_gradients(self, bc, xf, zf):
        try:
            if 'psi' in bc:
                d_dx = self.psi.ev(xf, zf, dx=1, dy=0)
                d_dz = self.psi.ev(xf, zf, dx=0, dy=1)
            else:
                d_dx = self.B.ev(xf, zf, dx=1, dy=0)
                d_dz = self.B.ev(xf, zf, dx=0, dy=1)
        except:
            warn('gradient evaluation failed')
            d_dx, d_dz = np.ones(len(bc)), np.ones(len(bc))
        return d_dx, d_dz

    def get_weight(self):
        Xf, Zf, BC, Bdir, n = self.unpack_fix()
        weight = np.zeros(n)
        if n > 0:
            for i, (xf, zf, bc, bdir, factor) in \
                    enumerate(zip(Xf, Zf, BC, Bdir, self.fix['factor'])):
                d_dx, d_dz = self.get_gradients(bc, xf, zf)
                if 'psi' not in bc:  # (Bx,Bz)
                    weight[i] = 1 / abs(np.sqrt(d_dx**2 + d_dz**2))
                elif bc == 'psi_bndry':
                    weight[i] = 1 / abs(np.dot([d_dx, d_dz], bdir))
            if 'psi_bndry' in self.fix['BC']:
                wbar = np.mean([weight[i]
                                for i, bc in enumerate(self.fix['BC'])
                                if bc == 'psi_bndry'])
            else:
                wbar = np.mean(weight)
            for i, bc in enumerate(BC):
                if bc == 'psi_x' or bc == 'psi':  # psi point weights
                    weight[i] = wbar  # mean boundary weight
            if (weight == 0).any():
                warn('fix weight entry not set')
        factor = np.reshape(self.fix['factor'], (-1, 1))
        weight = np.reshape(weight, (-1, 1))
        self.wsqrt = np.sqrt(factor * weight)

    def arange_CS(self, Z):  # ,dZmin=0.01
        Z = np.sort(Z)
        dZ = abs(np.diff(Z) - self.gap)
        Zc = np.zeros(self.nCS)
        for i in range(self.nCS):
            Zc[i] = Z[i] + self.gap / 2 + dZ[i] / 2
        return Zc, dZ

    def add_coil(self, Ctype=None, **kwargs):
        x, z = self.get_point(**kwargs)
        index, i = np.zeros(len(self.coilset['coil'].keys())), -1
        for i, name in enumerate(self.coilset['coil'].keys()):
            index[i] = name.split('Coil')[-1]
        try:
            Cid = index.max() + 1
        except:
            Cid = 0
        name = 'Coil{:1.0f}'.format(Cid)
        self.add_active([Cid], Ctype=Ctype)
        delta = np.ones(2)
        for i, dx in enumerate(['dx', 'dz']):
            if dx in kwargs.keys():
                delta[i] = kwargs.get(dx)
        It = kwargs.get('It', 1e6)
        self.coilset['coil'][name] = \
            {'x': x, 'z': z, 'dx': delta[0], 'dz': delta[1], 'It': It,
             'rc': np.sqrt(delta[0]**2 + delta[1]**2) / 2}

    def grid_CS(self, n, Xo=2.9, Zbound=[-10, 9],
                dx=0.818, gap=0.1, fdr=1):
        # dx=1.0, dx=1.25, Xo=3.2, dx=0.818
        nCS = n
        self.gap = gap
        dx *= fdr  # thicken CS
        Xo -= dx * (fdr - 1) / 2  # shift centre inwards
        dz = (np.diff(Zbound) - gap * (nCS - 1)) / nCS  # coil height
        Zc = np.linspace(Zbound[0] + dz / 2,
                         Zbound[-1] - dz / 2, nCS)  # coil centres
        self.remove_coil(self.CScoils)
        for zc in Zc:
            self.add_coil(point=(Xo, zc), dx=dx, dz=dz, Ctype='CS')
        Le = np.linspace(Zbound[0] - gap / 2,
                         Zbound[-1] + gap / 2, nCS + 1)  # coil edges
        self.update_coils()
        L = np.append(self.Lo['value'][:self.nPF], Le)
        self.set_Lo(L)
        self.set_force_field()

    def grid_PF(self, n):
        nPF = n
        dL = 1 / nPF
        Lpf = np.linspace(dL / 2, 1 - dL / 2, nPF)
        self.remove_coil(self.PFcoils)
        for lpf in Lpf:
            self.add_coil(Lout=lpf, Ctype='PF', norm=self.norm)
        self.update_coils()
        L = np.append(Lpf, self.Lo['value'][-(self.nCS+1):])
        self.set_Lo(L)
        self.set_force_field()

    def unwrap(self):
        '''
        unwrap PF and CS coils from TF 'track'
        store values in self.Lo dict
        function designed to store relitive normalized coil positions
        funciton can accomidate large departures in TF shape
        when used with partner function self.wrap
        '''
        Lpf = np.zeros(self.coilset['index']['PF']['n'])
        for i, name in enumerate(self.coilset['index']['PF']['name']):
            c = self.coilset['coil'][name]
            Lpf[i] = minimize_scalar(self.tf.norm, method='bounded',
                                     args=(self.tf.fun['out'],
                                           (c['x'], c['z'])), bounds=[0, 1]).x
        Lcs = np.zeros(self.coilset['index']['CS']['n'] + 1)
        z = np.zeros(self.coilset['index']['CS']['n'])
        dz = np.zeros(self.coilset['index']['CS']['n'])
        for i, name in enumerate(self.coilset['index']['CS']['name']):
            z[i] = self.coilset['coil'][name]['z']
            dz[i] = self.coilset['coil'][name]['dz']
        self.gap = np.mean((z[1:]-dz[1:]/2)-(z[:-1]+dz[:-1]/2))
        for i, name in enumerate(self.coilset['index']['CS']['name']):
            c = self.coilset['coil'][name]
            if i == 0:
                Lcs[0] = c['z'] - c['dz']/2 - self.gap/2
            Lcs[i + 1] = c['z'] + c['dz']/2 + self.gap/2
        L = np.append(Lpf, Lcs)
        self.set_Lo(L)

    def wrap_PF(self, solve=True):  # wrap PF coils around TF based on self.Lo
        for L, name in zip(self.Lo['value'][:self.nPF], self.PFcoils):
            self.move_PF(name, Lout=L, norm=self.norm)
        self.fit_PF()  # fine-tune offset
        if solve:
            self.ff.set_force_field()
            Lnorm = loops.normalize_variables(self.Lo)
            self.update_position(Lnorm, update_area=True)
            self.eq.run(update=True)  # update psi map

    def get_rss_bndry(self):
        psi_o = self.fix['value'][0]  # target boundary psi
        psi_line = self.sf.get_contour([psi_o])[0]
        dx_bndry, dx_min = np.array([]), 0
        dz_bndry, dz_min = np.array([]), 0
        Xf, Zf, BC = self.unpack_fix()[:3]
        for xf, zf, bc, psi in zip(Xf, Zf, BC, self.fix['value']):
            if bc == 'psi_bndry':
                if psi != psi_o:  # update contour
                    psi_o = psi
                    psi_line = self.sf.get_contour([psi_o])[0]
                for j, line in enumerate(psi_line):
                    x, z = line[:, 0], line[:, 1]
                    dx = np.sqrt((xf - x)**2 + (zf - z)**2)
                    xmin_index = np.argmin(dx)
                    # update boundary error
                    if j == 0 or dx[xmin_index] < dx_min:
                        dx_min = dx[xmin_index]
                    dz = zf - z
                    zmin_index = np.argmin(np.abs(dz))
                    # update boundary error
                    if j == 0 or np.abs(dz[zmin_index]) < np.abs(dz_min):
                        dz_min = dz[zmin_index]
                dx_bndry = np.append(dx_bndry, dx_min)
                dz_bndry = np.append(dz_bndry, dz_min)
        rss_bndry = np.sum(dx_bndry**2)  # calculate rss
        z_offset = np.mean(dz_bndry)
        return rss_bndry, z_offset

    def move_PF(self, name, AR=0, **kwargs):
        name = self.Cname(name)  # convert to name
        if 'point' in kwargs.keys() or 'Lout' in kwargs.keys()\
                or 'Lin' in kwargs.keys():
            x, z = self.get_point(**kwargs)
        elif 'delta' in kwargs.keys():
            ref, dx, dz = kwargs['delta']
            coil = self.coilset['coil'][self.Cname(ref)]
            xc, zc = coil['x'], coil['z']
            x, z = xc + dx, zc + dz
        elif 'L' in kwargs.keys():
            ref, dL = kwargs['dL']
            coil = self.coilset['coil'][self.Cname(ref)]
            xc, zc = coil['x'], coil['z']
        ARmax = 3
        if abs(AR) > ARmax:
            AR = ARmax * np.sign(AR)
        if AR > 0:
            AR = 1 + AR
        else:
            AR = 1 / (1 - AR)
        dA = self.coilset['coil'][name]['dx'] *\
            self.coilset['coil'][name]['dz']
        self.coilset['coil'][name]['dx'] = np.sqrt(AR * dA)
        self.coilset['coil'][name]['dz'] = self.coilset['coil'][name]['dx']
        self.coilset['coil'][name]['dz'] /= AR
        dx = x - self.coilset['coil'][name]['x']
        dz = z - self.coilset['coil'][name]['z']
        self.shift_coil(name, dx, dz)

    def fit_PF(self, **kwargs):
        '''
        offset PF coils from TF (minor adjustments)
        coil shifted along TF normal
        function designed to accomidate changes in PF area
        due to canges in coil current driven by current optimiser
        '''
        offset = kwargs.get('offset', self.offset)
        dl = 0
        for name in self.PFcoils:
            dx, dz = self.tf.Cshift(self.coilset['coil'][name], 'out', offset)
            dl += dx**2 + dz**2
            self.shift_coil(name, dx, dz)
        return np.sqrt(dl / self.nPF)

    def shift_coil(self, name, dx, dz):
        self.coilset['coil'][name]['x'] += dx
        self.coilset['coil'][name]['z'] += dz
        self.coil['active'][name]['x'] += dx
        self.coil['active'][name]['z'] += dz
        for i in range(self.coilset['coil'][name]['Nf']):
            subname = name + '_{:1.0f}'.format(i)
            self.coilset['subcoil'][subname]['x'] += dx
            self.coilset['subcoil'][subname]['z'] += dz
        self.update_bundle(name)

    def move_CS(self, name, z, dz):
        self.coilset['coil'][name]['z'] = z
        self.coilset['coil'][name]['dz'] = dz
        self.update_bundle(name)

    def reset_current(self):
        for j, name in enumerate(self.coil['active'].keys()):
            self.coilset['coil'][name]['It'] = 0

    def update_bundle(self, name):
        try:
            Nold = self.coilset['coil'][name]['Nf']
        except KeyError:
            Nold = 0
        subcoil = PF.mesh_coil(self.coilset['coil'][name], self.dCoil)
        Nf = self.coilset['coil'][name]['Nf']
        self.coil['active'][name].clear()
        for i, filament in enumerate(subcoil):
            subname = '{}_{}'.format(name, i)
            self.subcoil[subname] = filament
            self.coil['active'][name][subname] = self.subcoil[subname]
        if Nold > Nf:
            for i in range(Nf, Nold):
                del self.coilset['subcoil'][name + '_{:1.0f}'.format(i)]

    def copy_coil(self, coil_read):
        coil_copy = {}
        for strand in coil_read.keys():
            coil_copy[strand] = {}
            coil_copy[strand] = coil_read[strand].copy()
        return coil_copy

    def store_update(self, extent='full'):
        if extent == 'full':
            for var in self.log_scalar:
                self.log[var].append(getattr(self, var))
            for var in self.log_array:
                self.log[var].append(getattr(self, var).copy())
            for label in ['current', 'plasma']:
                self.log[label + '_iter'].append(self.iter[label])
        else:
            self.log['plasma'].append(
                self.copy_coil(self.eq.plasma).copy())
            self.log['position_iter'].append(self.iter['position'])

    def poloidal_angle(self):  # calculate poloidal field line angle
        if 'Bdir' in self.fix['BC']:
            index = self.fix['BC'] == 'Bdir'  # angle for last Bdir co-location
            B = self.sf.Bpoint(
                [self.fix['x'][index][-1], self.fix['z'][index][-1]])
            Tpol = 180 * np.arctan2(B[1], B[0]) / np.pi  # angle in degrees
        else:
            Tpol = 0
        return Tpol

    def PFspace(self, Lo, *args):
        Lpf = Lo[:self.nPF]
        dL = np.zeros(self.nPF)
        dLmin = 2e-2
        for i, l in enumerate(Lpf):
            L = np.append(Lpf[:i], Lpf[i + 1:])
            dL[i] = np.min(abs(l - L))
        fun = np.min(dL) - dLmin
        return fun

    def get_rss(self, vector, gamma, error=False):
        err = np.dot(self.wG, vector.reshape(-1, 1)) - self.wT
        rss = np.sum(err**2)  # residual sum of squares
        rss += gamma * np.sum(vector**2)  # Tikhonov regularization
        if error:  # return error
            return rss, err
        else:
            return rss

    def solve(self):
        vector = np.linalg.lstsq(self.wG, self.wT, rcond=None)[0].reshape(-1)
        self.rss = self.get_rss(vector)
        if self.svd:
            self.If = np.dot(self.V, self.alpha)
        else:
            self.If = vector
        self.update_current()

    def frss(self, vector, grad, gamma=1):
        self.iter['current'] += 1
        rss, err = self.get_rss(vector, gamma, error=True)
        if grad.size > 0:
            jac = 2 * self.wG.T @ self.wG @ vector
            jac -= 2 * self.wG.T @ self.wT[:, 0]
            jac += gamma * 2 * vector  # Tikhonov regularization
            grad[:] = jac
        return rss

    def Ilimit(self, constraint, alpha, grad):  # only for svd==True
        if grad.size > 0:
            grad[:self.nC] = -self.V  # lower bound
            grad[self.nC:] = self.V  # upper bound
        If = np.dot(self.V, alpha)
        constraint[:self.nC] = self.Io['lb'] - If  # lower bound
        constraint[self.nC:] = If - self.Io['ub']  # upper bound

    def set_Io(self):  # set lower/upper bounds on coil currents (Jmax)
        self.Io = {'name': self.adjust_coils, 'value': np.zeros(self.nC),
                   'lb': np.zeros(self.nC), 'ub': np.zeros(self.nC)}
        for i, name in enumerate(self.adjust_coils):  # limits in MA
            coil = self.coilset['coil'][name]
            Nf = coil['Nf']  # fillament number
            Nt = coil['Nt']  # turn number
            self.Io['value'][i] = coil['It'] / (Nf * self.Iscale)
            if name in self.limit['I']:  # per-coil ('PF1', 'CS)
                key = name
            elif name[:2] in self.limit['I']:  # per-set ('PF', 'CS', ...)
                key = name[:2]
            else:  # coil name or coil group not found in limit['Ic'], limit J
                key = 'Jmax'
            if key == 'Jmax':  # apply current density limit (default)
                Ilim = coil['dx'] * coil['dz'] * self.Jmax
                self.Io['lb'][i] = -Ilim / Nf
                self.Io['ub'][i] = Ilim / Nf
            else:
                # limit set as kA
                self.Io['lb'][i] = 1e-3 * self.limit['I'][key][0] * Nt / Nf
                self.Io['ub'][i] = 1e-3 * self.limit['I'][key][1] * Nt / Nf
                # set limit as MA.T
                # self.Io['lb'][i] = self.limit['I'][key][0] / Nf
                # self.Io['ub'][i] = self.limit['I'][key][1] / Nf
        lindex = self.Io['value'] < self.Io['lb']
        self.Io['value'][lindex] = self.Io['lb'][lindex]
        uindex = self.Io['value'] > self.Io['ub']
        self.Io['value'][uindex] = self.Io['ub'][uindex]

    def Flimit(self, constraint, vector, grad):
        if self.svd:  # convert eigenvalues to current vector
            If = np.dot(self.V, vector)
        else:
            If = vector
        self.ff.If = If
        F, dF = self.ff.set_force()  # set coil force and jacobian
        if grad.size > 0:  # calculate constraint jacobian
            # PFz lower bound
            grad[:self.nPF] = -dF[:self.nPF, :, 1]
            # PFz upper bound
            grad[self.nPF:2 * self.nPF] = dF[:self.nPF, :, 1]
            # CSsum lower
            grad[2 * self.nPF] = -np.sum(dF[self.nPF:, :, 1], axis=0)
            # CSsum upper
            grad[2 * self.nPF + 1] = np.sum(dF[self.nPF:, :, 1], axis=0)
            # evaluate each seperating gap in CS stack
            for j in range(self.nCS - 1):
                index = 2 * self.nPF + 2 + j
                # lower limit
                grad[index] = -np.sum(dF[self.nPF+j+1:, :, 1], axis=0) +\
                    np.sum(dF[self.nPF:self.nPF+j+1, :, 1], axis=0)
                # upper limit
                grad[index + self.nCS - 1] = \
                    np.sum(dF[self.nPF+j+1:, :, 1], axis=0) -\
                    np.sum(dF[self.nPF:self.nPF+j+1, :, 1], axis=0)
            # evaluate each axial gap in CS stack
            dFtp = self.tie_plate['alpha'] * \
                    np.sum(dF[self.nPF:, :, 0], axis=0)
            dFtp += np.sum(self.tie_plate['beta'].reshape((-1, 1)) * \
                np.ones((1, self.nC)) * dF[self.nPF:, :, 1], axis=0)
            dFtp += self.tie_plate['gamma'] * \
                    np.sum(dF[self.nPF:, :, 3], axis=0) 
            dFaxial = np.zeros((self.nCS+1, self.nC))
            dFaxial[-1] = dFtp
            for i in np.arange(1, self.nCS + 1):
                dFaxial[-(i+1)] = dFaxial[-i] + dF[-i, :, 1]
            for j in range(self.nCS + 1):
                #dFaxial[j] = op.approx_fprime(If, self.get_Faxial, 1e-7, j)
                #print('aprox', j, dFaxial)
                index = 2 * self.nPF + 2 + 2 * (self.nCS - 1) + j
                # lower limit
                grad[index] = -dFaxial[j]
                # upper limit
                grad[index + self.nCS + 1] = dFaxial[j]
        PFz = F[:self.nPF, 1]  # vertical force on PF coils
        PFz_limit = self.get_PFz_limit()  # PF vertical force limits
        constraint[:self.nPF] = PFz_limit[:, 0] - PFz  # PFz lower
        constraint[self.nPF:2 * self.nPF] = PFz - PFz_limit[:, 1]  # PFz upper
        FxCS = F[self.nPF:, 0]  # radial force on CS coils (vector)
        FzCS = F[self.nPF:, 1]  # vertical force on CS coils (vector)
        FcCS = F[self.nPF:, 3]  # vertical crusing force on CS coils (vector)
        CSsum = np.sum(FzCS)  # vertical force on CS stack (sum)
        # lower and upper limits applied to CSz_sum
        CSsum_limit = self.get_CSsum_limit()
        constraint[2 * self.nPF] = CSsum_limit[0] - CSsum
        constraint[2 * self.nPF + 1] = CSsum - CSsum_limit[1]
        CSsep_limit = self.get_CSsep_limit()
        for j in range(self.nCS - 1):  # evaluate CSsep for each gap
            Fsep = np.sum(FzCS[j + 1:]) - np.sum(FzCS[:j + 1])
            index = 2 * self.nPF + 2 + j
            # lower limit
            constraint[index] = CSsep_limit[j, 0] - Fsep
            # upper limit
            constraint[index + self.nCS - 1] = Fsep - CSsep_limit[j, 1]
        # CS Faxial limit
        CSaxial_limit = self.get_CSaxial_limit()
        Ftp = -self.tie_plate['preload'] 
        Ftp += self.tie_plate['alpha'] * np.sum(FxCS)
        Ftp += np.sum(self.tie_plate['beta'] * FzCS)
        Ftp += self.tie_plate['gamma'] * np.sum(FcCS)
        Faxial = np.ones(self.nCS+1)
        Faxial[-1] = Ftp
        for i in np.arange(1, self.nCS + 1):  # Faxial for each gap top-bottom
            Faxial[-(i+1)] = Faxial[-i] + FzCS[-i] - self.tie_plate['mg']
        for j in range(self.nCS + 1):  # Faxial for each gap top-bottom
            index = 2*self.nPF + 2 + 2*(self.nCS - 1) + j
            # lower limit
            constraint[index] = CSaxial_limit[j, 0] - Faxial[j]
            # upper limit
            constraint[index + self.nCS + 1] = Faxial[j] - CSaxial_limit[j, 1]
        #print('c', np.array_str(constraint[-(self.nCS+1):], precision=2))
        
    def CSlimit(self, constraint, vector, grad):
        If = vector
        self.ff.If = If
        F, dF = self.ff.set_force()  # set coil force and jacobian
        if grad.size > 0:  # calculate constraint jacobian
            # evaluate each axial gap in CS stack
            dFtp = self.tie_plate['alpha'] * \
                    np.sum(dF[self.nPF:, :, 0], axis=0)
            dFtp += np.sum(self.tie_plate['beta'].reshape((-1, 1)) * \
                np.ones((1, self.nC)) * dF[self.nPF:, :, 1], axis=0)
            dFtp += self.tie_plate['gamma'] * \
                    np.sum(dF[self.nPF:, :, 3], axis=0) 
            dFaxial = np.zeros((self.nCS+1, self.nC))
            dFaxial[-1] = dFtp
            for i in np.arange(1, self.nCS + 1):
                dFaxial[-(i+1)] = dFaxial[-i] + dF[i, :, 1]
            for j in range(self.nCS + 1):
                #dFaxial_ = op.approx_fprime(If, self.get_Faxial, 1e-7, -j)
                # upper limit
                grad[j] = dFaxial[j]
        # CS Faxial limit
        FxCS = F[self.nPF:, 0]  # radial force on CS coils (vector)
        FzCS = F[self.nPF:, 1]  # vertical force on CS coils (vector)
        FcCS = F[self.nPF:, 3]  # vertical crusing force on CS coils (vector)
        CSaxial_limit = self.get_CSaxial_limit()
        Ftp = -self.tie_plate['preload'] 
        Ftp += self.tie_plate['alpha'] * np.sum(FxCS)
        Ftp += np.sum(self.tie_plate['beta'] * FzCS)
        Ftp += self.tie_plate['gamma'] * np.sum(FcCS)
        Faxial = np.ones(self.nCS+1)
        Faxial[-1] = Ftp
        for i in np.arange(1, self.nCS + 1):  # Faxial for each gap top-bottom
            Faxial[-(i+1)] = Faxial[-i] + FzCS[-i] - self.tie_plate['mg']
        for j in range(self.nCS + 1):
            # upper limit
            constraint[j] = Faxial[j] - CSaxial_limit[j, 1]
            
    def get_Faxial(self, If, j=None):
        self.ff.If = If
        F, dF = self.ff.set_force() 
        FxCS = F[self.nPF:, 0]  # radial force on CS coils (vector)
        FzCS = F[self.nPF:, 1]  # vertical force on CS coils (vector)
        FcCS = F[self.nPF:, 3]  # vertical crusing force on CS coils (vector)
        
        Ftp = -self.tie_plate['preload'] 
        Ftp += self.tie_plate['alpha'] * np.sum(FxCS)
        Ftp += np.sum(self.tie_plate['beta'] * FzCS)
        Ftp += self.tie_plate['gamma'] * np.sum(FcCS)
        Faxial = np.zeros(self.nCS+1)
        Faxial[-1] = Ftp
        for i in np.arange(1, self.nCS + 1):  # Faxial for each gap
            Faxial[-(i+1)] = Faxial[-i] + FzCS[-i] - self.tie_plate['mg']
        if j is None:
            return Faxial
        else:
            return Faxial[j]

    def get_PFz_limit(self):
        PFz_limit = np.zeros((self.nPF, 2))
        for i, coil in enumerate(self.PFcoils):
            if coil in self.limit['F']:  # per-coil
                PFz_limit[i] = self.limit['F'][coil]
            elif coil[:2] in self.limit['F']:  # per-set
                PFz_limit[i] = self.limit['F'][coil[:2]]
            else:  # no limit
                PFz_limit[i] = [-1e16, 1e16]
        return PFz_limit

    def get_CSsep_limit(self):
        CSsep_limit = np.zeros((self.nCS - 1, 2))
        for i in range(self.nCS - 1):  # gaps, bottom-top
            gap = 'CS{}sep'.format(i)
            if gap in self.limit['F']:  # per-gap
                CSsep_limit[i] = self.limit['F'][gap]
            elif 'CSsep' in self.limit['F']:  # per-set
                CSsep_limit[i] = self.limit['F']['CSsep']
            else:  # no limit
                CSsep_limit[i] = [-1e16, 1e16]
        return CSsep_limit

    def get_CSsum_limit(self):
        CSsum_limit = np.zeros((1, 2))
        if 'CSsum' in self.limit['F']:  # per-set
            CSsum_limit = self.limit['F']['CSsum']
        else:  # no limit
            CSsum_limit = [-1e16, 1e16]
        return CSsum_limit
    
    def get_CSaxial_limit(self):
        CSaxial_limit = np.zeros((self.nCS + 1, 2))
        for i in range(self.nCS + 1):  # gaps, top-bottom
            gap = 'CS{}axial'.format(i)
            if gap in self.limit['F']:  # per-gap
                CSaxial_limit[i] = self.limit['F'][gap]
            elif 'CSaxial' in self.limit['F']:  # per-set
                CSaxial_limit[i] = self.limit['F']['CSaxial']
            else:  # no limit
                CSaxial_limit[i] = [-1e16, 1e16]
        return CSaxial_limit

    def solve_slsqp(self, flux):  # solve for constrained current vector
        self.check_state()
        self.fix_flux(flux)  # swing
        self.set_Io()  # set coil current and bounds
        #opt = nlopt.opt(nlopt.LD_SLSQP, self.nC)
        opt = nlopt.opt(nlopt.LD_MMA, self.nC)
        opt.set_min_objective(self.frss)
        opt.set_ftol_rel(1e-3)
        opt.set_xtol_abs(1e-3)
        tol = 1e-2 * np.ones(2 * self.nPF + 2 + 2 * (self.nCS - 1) + 
                             2 * (self.nCS + 1))  # 1e-3
        opt.add_inequality_mconstraint(self.Flimit, tol)
        #tol = 1e-1 * np.ones(self.nCS + 1)
        #opt.add_inequality_mconstraint(self.CSlimit, tol)
        if self.svd:  # coil current eigen-decomposition
            opt.add_inequality_mconstraint(
                self.Ilimit, 1e-3 * np.ones(2 * self.nC))
            self.alpha = opt.optimize(self.alpha)
            self.If = np.dot(self.V, self.alpha)
        else:
            opt.set_lower_bounds(self.Io['lb'])
            opt.set_upper_bounds(self.Io['ub'])
            self.If = opt.optimize(self.If)  # self.Io['value']
        '''
        print('')
        c = np.zeros(len(tol))
        self.Flimit(c, self.If, np.array([]))
        print('c', c[-(self.nCS+1):])
        print(self.get_Faxial(self.If) - self.tie_plate['limit_load'])
        print(self.get_Faxial(self.If))
        '''
        self.Io['value'] = self.If.reshape(-1)
        self.opt_result = opt.last_optimize_result()
        self.rss = opt.last_optimum_value()
        self.update_current()
        return self.rss

    def update_area(self, relax=1, margin=0):
        It = self.swing['It'][np.argmax(abs(self.swing['It']), axis=0),
                              range(self.nC)]
        for name in self.PFcoils:
            if name in self.adjust_coils:
                It = It[list(self.adjust_coils).index(name)]
            else:
                It = self.coilset['coil'][name]['It']
            if It != 0:
                It = abs(It)
                if It > self.limit['I']['PF']:
                    It = self.limit['I']['PF']  # coil area upper bound
                dA_target = It / self.Jmax  # apply current density limit
                dA = self.coilset['coil'][name]['dx'] *\
                    self.coilset['coil'][name]['dz']
                ratio = dA_target / dA
                scale = (ratio * (1 + margin))**0.5
                self.coilset['coil'][name]['dx'] *= relax * (scale - 1) + 1
                self.coilset['coil'][name]['dz'] *= relax * (scale - 1) + 1
        dl = self.fit_PF()  # fit re-sized coils to TF
        return dl

    def update_position_vector(self, Lnorm, grad):
        rss = self.update_position(Lnorm, update_area=True, store_update=True)
        if grad.size > 0:
            grad[:] = op.approx_fprime(Lnorm, self.update_position, 1e-7)
        rss_str = '\r{:d}) rss {:1.2f}mm '.format(
            self.iter['position'], 1e3 * rss)
        rss_str += 'time {:1.0f}s'.format(time.time() - self.tstart)
        rss_str += '\t\t\t'  # white space
        sys.stdout.write(rss_str)
        sys.stdout.flush()
        return rss

    def update_position_scipy(self, Lnorm):
        jac = op.approx_fprime(Lnorm, self.update_position, 5e-4)
        rss = self.update_position(Lnorm, store_update=True)
        return rss, jac

    def check_state(self):
        if not self.rhs:
            errtxt = 'inverse colocations unset, run \'self.colocate\''
            raise ValueError(errtxt)

    def update_position(self, Lnorm, update_area=False, store_update=False):
        self.check_state()
        self.iter['current'] == 0
        self.Lo['norm'] = np.copy(Lnorm)
        L = loops.denormalize_variables(Lnorm, self.Lo)
        Lpf = L[:self.nPF]
        for name, lpf in zip(self.PFcoils, Lpf):
            self.move_PF(name, Lout=lpf, norm=self.norm)
        if len(L) > self.nPF:
            Lcs = L[self.nPF:]
            Z, dZ = self.arange_CS(Lcs)
            for name, z, dz in zip(self.CScoils, Z, dZ):
                self.move_CS(name, z, dz)
        if update_area:  # converge PF coil areas
            dl_o = 0
            imax, err, relax = 15, 1e-3, 1
            dL = []
            for i in range(imax):
                dl = self.update_area(relax=relax)
                dL.append(dl)
                self.ff.set_force_field(state='active')
                self.set_foreground()
                rss = self.swing_flux()
                if abs(dl - dl_o) < err:  # coil areas converged
                    break
                else:
                    dl_o = dl
                if i > 1:
                    relax *= 0.5
                if i == imax - 1:
                    print('warning: coil areas not converged')
                    print(dL)
        else:
            self.fit_PF()  # fit coils to TF
            self.ff.set_force_field(state='active')
            self.set_foreground()
            rss = self.swing_flux()
        if store_update:
            self.iter['position'] += 1
            self.store_update()
        return rss

    def swing_flux(self, bndry=False):
        for i, flux in enumerate(self.swing['flux']):
            self.swing['rss'][i] = self.solve_slsqp(flux)
            self.ff.get_force()
            Fcoil = self.ff.Fcoil
            self.swing['FxPF'][i] = Fcoil['PF']['x']
            self.swing['FzPF'][i] = Fcoil['PF']['z']
            self.swing['FsepCS'][i] = Fcoil['CS']['sep']
            self.swing['FzCS'][i] = Fcoil['CS']['zsum']
            self.swing['Fcoil'][i] = {'F': Fcoil['F'], 'dF': Fcoil['dF']}
            self.swing['Tpol'][i] = self.poloidal_angle()
            self.swing['It'][i] = self.It
            self.swing['Isum'][i] = self.Isum
            self.swing['IsumCS'][i] = self.IsumCS
            self.swing['IsumPF'][i] = self.IsumPF
            self.swing['Imax'][i] = self.Imax
            self.swing['IcCSmax'][i] = self.IcCSmax
        self.rssID = np.argmax(self.swing['rss'])
        self.rss = self.swing['rss'][self.rssID]
        return self.rss

    def plot_swing(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        for i, name in enumerate(self.coil['active']):
            Nt = self.coilset['coil'][name]['Nt']  # turn number
            ax.plot(2*np.pi*self.swing['flux'],
                    1e-3*self.Iscale*self.swing['It'][:, i] / Nt,
                    label=name)
        ax.legend()
        plt.despine()
        ax.set_xlabel('$t$ s')
        ax.set_ylabel('$I$ kA')

    def set_sail(self):  # move pf coil out of port
        if not hasattr(self, 'Lex'):
            errtxt = 'can\'t leave if you don\'t know were you are\n'
            errtxt += 'set port locations in inv.Lex by running:\n'
            errtxt += 'excl = R.define_port_exclusions\n'
            errtxt += 'inv.Lex = R.TF.xzL(excl)'
            raise ValueError(errtxt)

        for i, (name, L) in enumerate(zip(self.PFcoils, self.Lo['value'])):
            ip = np.argmin(abs(L-self.Lex))
            dl = self.coilset['coil'][name]['rc']/self.tf.fun['out']['L']
            if (ip % 2 == 0 and L + dl > self.Lex[ip]) or \
                    (ip % 2 == 1 and L - dl < self.Lex[ip]):
                # coil in port move down if even, up if odd
                value = self.Lex[ip] - dl*(-1) ** (ip % 2)
            else:
                value = L
            if ip % 2 == 0:  # even
                ub = self.Lex[ip] - dl  # upper bound
                if ip > 0:
                    lb = self.Lex[ip-1] + dl  # lower bound
                else:
                    lb = 0.15
            else:  # odd
                if ip < self.nPF-1:
                    ub = self.Lex[ip+1] - dl  # upper bound
                else:
                    ub = 0.95
                lb = self.Lex[ip] + dl  # lower bound
            # update bounds
            self.Lo['value'][i] = value
            self.Lo['lb'][i] = lb
            self.Lo['ub'][i] = ub
            self.limit['L'][name] = [lb, ub]

    def set_Lo(self, L):  # set lower/upper bounds on coil positions
        self.nL = len(L)  # length of coil position vector
        self.Lo = {'name': [[] for _ in range(self.nL)],
                   'value': np.zeros(self.nL),
                   'lb': np.zeros(self.nL), 'ub': np.zeros(self.nL)}
        for i, (l, name) in enumerate(zip(L[:self.nPF], self.PFcoils)):
            Lo_name = 'Lpf{:1.0f}'.format(i)
            lb, ub = self.limit['L'][name]
            loops.add_value(self.Lo, i, Lo_name, l, lb, ub)
        for i, l in enumerate(L[self.nPF:]):  # CS
            Lo_name = 'Zcs{:1.0f}'.format(i)
            loops.add_value(self.Lo, i+self.nPF, Lo_name, l,
                            self.limit['L']['CS'][0], self.limit['L']['CS'][1])

    def Llimit(self, constraint, L, grad):
        PFdL = 1e-4  # minimum PF inter-coil spacing
        CSdL = 1e-4  # minimum CS coil height
        if grad.size > 0:
            grad[:] = np.zeros((self.nPF - 1 + self.nCS, len(L)))  # initalise
            for i in range(self.nPF - 1):  # PF
                grad[i, i] = 1
                grad[i, i + 1] = -1
            for i in range(self.nCS):  # CS
                grad[i + self.nPF - 1, i + self.nPF] = 1
                grad[i + self.nPF - 1, i + self.nPF + 1] = -1
        constraint[:self.nPF - 1] = L[:self.nPF - 1] - \
            L[1:self.nPF] + PFdL  # PF
        constraint[self.nPF - 1:] = L[self.nPF:-1] - \
            L[self.nPF + 1:] + CSdL  # CS

    def minimize(self, method='ls', verbose=False):
        self.iter['position'] == 0
        self.set_Lo(self.Lo['value'])  # update limits
        Lnorm = loops.normalize_variables(self.Lo)

        if method == 'bh':  # basinhopping
            # minimizer = {'method':'SLSQP','jac':True,#'args':True,
            #              'options':{'eps':1e-3}, #'jac':True,
            #             'bounds':[[0,1] for _ in range(self.nL)]}
            minimizer = {'method': 'L-BFGS-B', 'jac': True,
                         'bounds': [[0, 1] for _ in range(self.nL)]}

            res = op.basinhopping(self.update_position_scipy, Lnorm, niter=1e4,
                                  T=1e-3, stepsize=0.05, disp=True,
                                  minimizer_kwargs=minimizer,
                                  interval=50, niter_success=100)
            Lnorm, self.rss = res.x, res.fun
        elif method == 'de':  # differential_evolution
            bounds = [[0, 1] for _ in range(self.nL)]
            res = op.differential_evolution(self.update_position, bounds,
                                            args=(False,), strategy='best1bin',
                                            maxiter=100, popsize=15, tol=0.01,
                                            mutation=(0.5, 1),
                                            recombination=0.7, polish=True,
                                            disp=True)
            Lnorm, self.rss = res.x, res.fun
        elif method == 'ls':  # sequential least squares
            print('\nOptimising configuration:')
            opt = nlopt.opt(nlopt.LD_SLSQP, self.nL)
            # opt = nlopt.opt(nlopt.LD_MMA, self.nL)
            opt.set_ftol_abs(5e-3)
            opt.set_ftol_rel(5e-3)
            opt.set_stopval(50e-3)  # <x [m]
            opt.set_min_objective(self.update_position_vector)
            opt.set_lower_bounds([0 for _ in range(self.nL)])
            opt.set_upper_bounds([1 for _ in range(self.nL)])
            tol = 1e-3 * np.ones(self.nPF - 1 + self.nCS)
            opt.add_inequality_mconstraint(self.Llimit, tol)
            self.rss = opt.last_optimum_value()
            Lnorm = opt.optimize(Lnorm)
        loops.denormalize_variables(Lnorm, self.Lo)
        print('\nrss {:1.2f}mm'.format(1e3 * self.rss))
        result = opt.last_optimize_result()
        if result < 0:
            warntxt = 'optimiser exited with error code {}'.format(result)
            warn(warntxt)

    def tick(self):
        self.tloop = time.time()
        self.tloop_cpu = time.process_time()

    def tock(self):
        tw, tcpu = time.time(), time.process_time()
        dt = tw - self.tloop
        self.ttotal = tw - self.tstart
        dt_cpu = tcpu - self.tloop_cpu
        self.ttotal_cpu = tcpu - self.tstart_cpu
        dfcpu, self.fcpu = dt_cpu / dt, self.ttotal_cpu / self.ttotal
        print('dt {:1.0f}s speedup {:1.0f}%'.format(
            dt, 100 * dfcpu / self.ncpu))
        print('total {:1.0f}s speedup {:1.0f}%'.format(self.ttotal,
              100 * self.fcpu / self.ncpu))
        print('')

    def initialize_log(self):
        self.log = {}
        for var in self.log_scalar + self.log_array\
                + self.log_iter + self.log_plasma:
            self.log[var] = []
        self.iter = {'plasma': 0, 'position': 0, 'current': 0}

    def optimize(self, verbose=False):
        self.initialize_log()
        self.ztarget = self.eq.sf.Mpoint[1]
        self.ttotal, self.ttotal_cpu = 0, 0
        self.tstart, self.tstart_cpu = time.time(), time.process_time()
        self.iter['plasma'] += 1
        self.tick()
        self.add_plasma()
        self.minimize(verbose=verbose)
        self.store_update(extent='position')
        self.tock()
        self.eq.run(update=True)  # update psi map
        self.update_coils()
        return self.Lo

    def add_plasma(self):
        self.set_background()
        self.get_weight()

    def update_current(self):
        self.Isum, self.IsumCS, self.IsumPF = 0, 0, 0
        self.ff.If = self.If  # pass current to force field
        self.It = np.zeros(len(self.coil['active'].keys()))
        self.Ic = np.zeros(len(self.coil['active'].keys()))
        for j, name in enumerate(self.coil['active']):
            Nfilament = self.coilset['coil'][name]['Nf']
            Nturn = self.coilset['coil'][name]['Nt']
            self.It[j] = self.If[j] * Nfilament  # store current MAt
            self.Ic[j] = self.It[j] / Nturn  # conductor current
            self.coilset['coil'][name]['It'] = self.It[j] * self.Iscale
            self.coilset['coil'][name]['Ic'] = self.Ic[j] * self.Iscale
            self.coil['active'][name]['Isum'] = self.It[j] * self.Iscale
            for i in range(Nfilament):
                subname = name + '_{:1.0f}'.format(i)
                self.coilset['subcoil'][subname]['If'] = self.If[j] *\
                    self.Iscale
                self.coil['active'][name]['If'][i] = self.If[j] * self.Iscale
            self.Isum += abs(self.It[j])  # sum absolute current
            if name in self.CScoils:
                self.IsumCS += abs(self.It[j])
            elif name in self.PFcoils:
                self.IsumPF += abs(self.It[j])
        IcCS = np.array([self.coilset['coil'][name]['Ic']
                         for name in self.coilset['index']['CS']['name']])
        self.IcCSmax = IcCS[np.argmax(abs(IcCS))]
        imax = np.argmax(abs(self.It))
        self.Imax = self.It[imax]

    def write_swing(self):
        nC, nS = len(self.adjust_coils), len(self.Swing)
        It = np.zeros((nS, nC))
        for i, flux in enumerate(self.Swing[::-1]):
            self.solve_slsqp(flux)
            It[i] = self.It
        dataname = self.tf.dataname.replace('TF.pkl', '_currents.txt')
        with open('../Data/'+dataname, 'w') as f:
            f.write('Coil\tr [m]\tz [m]\tdr [m]\tdz [m]')
            f.write('\tI SOF [A]\tI EOF [A]\n')
            for j, name in enumerate(self.coil['active'].keys()):
                x = float(self.coilset['coil'][name]['x'])
                z = float(self.coilset['coil'][name]['z'])
                dx = self.coilset['coil'][name]['dx']
                dz = self.coilset['coil'][name]['dz']
                position = '\t{:1.3f}\t{:1.3f}'.format(x, z)
                size = '\t{:1.3f}\t{:1.3f}'.format(dx, dz)
                current = '\t{:1.3f}\t{:1.3f}\n'.format(It[0, j], It[1, j])
                f.write(name + position + size + current)


class SWING(object):

    def __init__(self, inv, sf, rss_limit=0.10, wref=0, nswing=2, plot=False,
                 output=True):
        self.nswing = nswing
        self.inv = inv
        self.rss_limit = rss_limit
        self.wref = wref

    def get_rss(self, centre):
        self.inv.set_swing(centre=centre, width=self.wref,
                           array=np.linspace(-0.5, 0.5, 2))
        rss = self.inv.update_position(self.Lnorm, update_area=True)
        return float(self.rss_limit - rss)

    def find_root(self, slim):
        swing = brentq(self.get_rss, slim[0], slim[1], xtol=0.1, maxiter=500,
                       disp=True)
        return swing

    def flat_top(self):
        print('\nCalculating swing:')
        self.Lnorm = loops.normalize_variables(self.inv.Lo)
        SOF = self.find_root([-60, 0]) - self.wref / 2
        EOF = self.find_root([0, 60]) + self.wref / 2

        self.width = 0.95 * (EOF - SOF)
        self.centre = np.mean([SOF, EOF])
        self.inv.set_swing(centre=self.centre, width=self.width,
                           array=np.linspace(-0.5, 0.5, self.nswing))
        self.inv.update_position(self.Lnorm, update_area=True)

    def energy(self):
        self.inductance(dCoil=self.inv.dCoil, Iscale=1)
        E = np.zeros(len(self.inv.swing['flux']))
        for i, Isw in enumerate(self.inv.swing['It']):
            Isw *= self.inv.Iscale
            E[i] = 0.5 * np.dot(np.dot(Isw, self.inv.M), Isw)
        return np.max(E)

    def output(self):
        self.flat_top()
        E = self.energy()
        PFvol = 0
        for name in self.inv.pf.coil:
            coil = self.inv.pf.coil[name]
            x, dx, dz = coil['x'], coil['dx'], coil['dz']
            PFvol += 2 * np.pi * x * dx * dz
        Isum = np.max(self.inv.swing['Isum'])
        Ipf = np.max(self.inv.swing['IsumPF'])
        Ics = np.max(self.inv.swing['IsumCS'])
        FzPF = np.max(self.inv.swing['FzPF'])
        FsepCS = np.max(self.inv.swing['FsepCS'])
        FzCS = np.max(self.inv.swing['FzCS'])

        print('swing divisions: {:1.0f}'.format(self.nswing))
        print('swing width {:1.0f}Vs'.format(2 * np.pi * self.width))
        print(r'max absolute PF/CS current sum {:1.1f}MA'.format(Isum))
        print(r'PF stored energy {:1.1f}GJ'.format(1e-9 * E))
        print(r'max absolute PF current sum {:1.1f}MA'.format(Ipf))
        print(r'max vertical PF force {:1.1f}MN'.format(FzPF))
        print(r'max sum CS vertical force {:1.1f}MN'.format(FzCS))
        print(r'max CS seperation force {:1.1f}MN'.format(FsepCS))
        print(r'max absolute CS current sum {:1.1f}MA'.format(Ics))
        print(r'PF material volume {:1.1f}m3'.format(PFvol))
