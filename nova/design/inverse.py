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
from pandas import DataFrame, Series, concat, isnull
from pandas.api.types import is_list_like

from amigo import geom
from amigo.time import clock
from amigo.pyplot import plt
#from nova import loops
#import nova.cross_coil as cc
#from nova.coils import PF
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.biotmethods import Target
#from nova.electromagnetic.streamfunction import SF
#from nova.force import force_field
from nova.limits.tieplate import get_tie_plate


class Inverse(CoilSet):
    
    def __init__(self):
        self._biot_instances.update({'colocate': 'colocate'})
        CoilSet.__init__(self)
        
        
    def set_foreground(self):
        self.flux = self.coil.reduce_mpc(self.colocate.flux)
        self.wG = self.flux  # full flux constraint
        
        '''
        self.G = np.zeros((self.fix['n'], self.coil._nC))  # [G][If] = [T]
        self.add_value('active')
        self.wG = self.wsqrt * self.G
        if self.svd:  # singular value dec.
            self.U, self.S, self.V = np.linalg.svd(self.wG)
            self.wG = np.dot(self.wG, self.V)  # solve in terms of alpha
        '''
        
    def set_background(self):
        self.BG = np.zeros(self.colocate.n)  # background
        #self.add_value('passive')
        
    def set_target(self):
        self.T = (self.fix['value'] - self.BG).reshape(-1, 1)
        self.wT = self.wsqrt * self.T

    #def set_target(self)
    
    def get_rss(self, vector, gamma, error=False):
        err = np.dot(self.wG, vector.reshape(-1, 1)) - self.wT
        rss = np.sum(err**2)  # residual sum of squares
        rss += gamma * np.sum(vector**2)  # Tikhonov regularization
        if error:  # return error
            return rss, err
        else:
            return rss

    def solve(self):
        Ic = np.linalg.lstsq(self.wG, 
                             self.colocate.Psi, rcond=None)[0]
        self.Ic = Ic
        
        '''
        self.rss = self.get_rss(vector)
        if self.svd:
            self.If = np.dot(self.V, self.alpha)
        else:
            self.If = vector
        self.update_current()
        '''
        
    def check_state(self):
        if self.colocate.n == 0:
            errtxt = 'colocations unset, self.colocate.add_targets'
            raise ValueError(errtxt)
        
    def solve_slsqp(self):  # solve for constrained current vector
        #self.check_state()
        #self.set_Io()  # set coil current and bounds
        #opt = nlopt.opt(nlopt.LD_SLSQP, self.nC)
        opt = nlopt.opt(nlopt.LD_MMA, self.coil._nC)
        opt.set_min_objective(self.frss)
        opt.set_ftol_rel(1e-3)
        opt.set_xtol_abs(1e-3)
        #tol = 1e-2 * np.ones(2 * self.nPF + 2 + 2 * (self.nCS - 1) + 
        #                     2 * (self.nCS + 1))  # 1e-3
        #opt.add_inequality_mconstraint(self.Flimit, tol)
        
        #opt.set_lower_bounds(self.Io['lb'])
        #opt.set_upper_bounds(self.Io['ub'])
        Ic = opt.optimize(self.coil._Ic)
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
        
    def set_weight(self, index, gradient):
        index &= (gradient > 0)  # ensure gradient > 0
        self.colocate.targets.loc[index, 'weight'] = 1 / gradient[index]
            
    def update_weight(self):        
        'update colocation weight based on inverse of absolute gradient'
        gradient = self.colocate.targets.loc[:, ['d_dx', 'd_dz']].to_numpy()
        normal = self.colocate.targets.loc[:, ['nx', 'nz']].to_numpy()
        d_dx, d_dz = gradient.T
        # compute gradient magnitudes
        gradient_L2 = np.linalg.norm(gradient, axis=1)  # L2norm
        field_index = np.array(['B' in l for l in self.colocate.targets.label])
        self.set_weight(field_index, gradient_L2)
        gradient_dot = abs(np.array([g @ n for g, n in zip(gradient, normal)]))
        bndry_index = ['bndry' in l for l in self.colocate.targets.label]
        self.set_weight(bndry_index, gradient_dot)
        # calculate mean weight
        if sum(bndry_index) > 0:
            mean_index = bndry_index
        elif sum(field_index) > 0:
            mean_index = field_index
        else:
            mean_index = slice(None)
        mean_weight = self.colocate.targets.weight[mean_index].mean()
        # not field or Psi_bndry (separatrix)
        mean_index = [not field and not bndry for field, bndry in zip(
                    field_index, bndry_index)]
        
        self.colocate.targets.loc[mean_index, 'weight'] = mean_weight
        self.wsqrt = np.sqrt(self.colocate.targets.factor * 
                             self.colocate.targets.weight)
        self.wsqrt /= np.mean(self.wsqrt)  # normalize weight
        
        
    def plot_colocate(self, tails=True):
        self.update_weight()
        
        style = DataFrame(index=['color', 'marker', 'markersize',
                                 'markeredgewidth'])
        
        plt.plot(self.colocate)
        '''
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
        '''
               

    def fix_flux(self, flux):
        if not hasattr(self, 'fix_o'):  # set once
            self.fix_o = self.fix.copy()
        index = ['psi' in name for name in self.fix.name]
        self.fix.loc[index, 'value'] = self.fix_o.loc[index, 'value'] + flux
        #self.set_target()  # adjust target flux
        
    '''
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
    '''
        
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




    '''
    def set_foreground(self):
        self.G = np.zeros((self.fix['n'], self.nC))  # [G][If] = [T]
        self.add_value('active')
        self.wG = self.wsqrt * self.G
        if self.svd:  # singular value dec.
            self.U, self.S, self.V = np.linalg.svd(self.wG)
            self.wG = np.dot(self.wG, self.V)  # solve in terms of alpha
    '''

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

    def frss(self, vector, grad, gamma=1):
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

