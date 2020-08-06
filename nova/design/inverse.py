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
from nova.electromagnetic.coilclass import CoilClass
from nova.electromagnetic.biotmethods import Target
from nova.limits.tieplate import get_tie_plate
from nova.limits.poloidal import PoloidalLimit


class Inverse(CoilClass, PoloidalLimit):
    
    def __init__(self, gamma=1e-9):
        self._biot_instances.update({'colocate': 'colocate'})
        CoilClass.__init__(self)
        PoloidalLimit.__init__(self)
        self.load_ITER_limits()
        
        self.gamma = gamma  # Tikhonov regularization
        
    
    def set_foreground(self):
        '[G][Ic] = [T]'
        self.flux = self.coil.reduce_mpc(self.colocate.flux)
        self.G = self.flux  # full flux constraint
        self.wG = self.G  # self.wsqrt * self.G
        
    def set_background(self):
        'contribution from passive coils'
        self.BG = np.zeros(self.colocate.n)  # background
        
    def set_target(self):
        self.T = (self.colocate.targets.value.values - self.BG)
        self.wT = self.T  # self.wsqrt * self.T
    
    @property
    def err(self):
        'error vector'
        return self.wG @ self.Ic - self.wT
    
    @property
    def rss(self):
        'residual sum of squares with Tikhonov regularization'
        return np.sum(self.err**2) + self.gamma * np.sum(self.Ic**2)
    
    def frss(self, Ic, grad):
        self.Ic = Ic  # update current vector
        if grad.size > 0:
            jac = 2 * self.wG.T @ self.wG @ self.Ic
            jac -= 2 * self.wG.T @ self.wT#[:, 0]
            jac += self.gamma * 2 * self.Ic  # Tikhonov regularization
            grad[:] = jac  # set gradient in-place
        return self.rss
    
    def solve_lstsq(self):
        'linear least squares solution'
        self.Ic = np.linalg.lstsq(self.wG, self.wT, rcond=None)[0]
        
    def solve(self):  # solve for constrained current vector
        #self.solve_lstsq()  # sead with least-squares solution
        
        #self.set_Io()  # set coil current and bounds
        opt = nlopt.opt(nlopt.LD_MMA, self.coil._nC)
        opt.set_min_objective(self.frss)
        opt.set_ftol_rel(1e-3)
        opt.set_xtol_abs(1e-3)
        #tol = 1e-2 * np.ones(2 * self.nPF + 2 + 2 * (self.nCS - 1) + 
        #                     2 * (self.nCS + 1))  # 1e-3
        #opt.add_inequality_mconstraint(self.Flimit, tol)
        
        lb = self.get_limit(self.coil._mpc_index, 'current', 'lower')
        ub = self.get_limit(self.coil._mpc_index, 'current', 'upper')
        
        print(lb, ub)
        opt.set_lower_bounds(1e3*lb)
        opt.set_upper_bounds(1e3*ub)
        #opt.set_lower_bounds(self.Io['lb'])
        #opt.set_upper_bounds(self.Io['ub'])
        self.Ic = opt.optimize(self.Ic)
        '''
        print('')
        c = np.zeros(len(tol))
        self.Flimit(c, self.If, np.array([]))
        print('c', c[-(self.nCS+1):])
        print(self.get_Faxial(self.If) - self.tie_plate['limit_load'])
        print(self.get_Faxial(self.If))
        '''
        self.opt_result = opt.last_optimize_result()
        
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


