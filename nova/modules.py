import pylab as pl
import numpy as np
from amigo import geom
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import scipy.optimize as op
import nlopt
from amigo.geom import String
from warnings import warn


class wrapper(object):  # first wall wrapper

    # smooth (many point) first wall profile
    def __init__(self, x, z, angle=20, dx_min=0.75, dx_max=3):
        self.x, self.z = x, z
        tx, tz = geom.tangent(self.x, self.z)
        l = geom.length(self.x, self.z)
        self.loop = {'x': IUS(l, x), 'z': IUS(l, z)}
        self.tangent = {'x': IUS(l, tx), 'z': IUS(l, tz)}
        P = np.array([x, z]).T
        st = String(P, angle=angle, dx_min=dx_min, dx_max=dx_max,
                    verbose=False)
        self.nP = st.n-2
        self.Lo = l[st.index][1:-1]
        self.dLlimit = {'min': dx_min, 'max': dx_max}
        self.nC = self.nP-1+2*(self.nP+2)  # constraint number

    def fw_corners(self, Lo):
        Lo = np.sort(Lo)
        Lo = np.append(np.append(0, Lo), 1)
        Pc = np.zeros((len(Lo)-1, 2))  # corner points
        Po = np.array([self.loop['x'](Lo), self.loop['z'](Lo)]).T
        To = np.array([self.tangent['x'](Lo), self.tangent['z'](Lo)]).T
        for i in range(self.nP+1):
            Pc[i] = geom.cross_vectors(Po[i], To[i], Po[i+1], To[i+1])
        Pc = np.append(Po[0].reshape(1, 2), Pc, axis=0)
        Pc = np.append(Pc, Po[-1].reshape(1, 2), axis=0)
        return Pc, Po

    def fw_length(self, Lo, index):
        Pc = self.fw_corners(Lo)[0]
        dL = np.sqrt(np.diff(Pc[:, 0])**2 + np.diff(Pc[:, 1])**2)
        L = np.sum(dL)
        data = [L, dL]
        return data[index]

    def fw_vector(self, Lo, grad):
        L = self.fw_length(Lo, 0)
        if grad.size > 0:
            grad[:] = op.approx_fprime(Lo, self.fw_length, 1e-6, 0)
        return L

    def fw_ms(self, Lo):
        Pc = self.fw_corners(Lo)[0]
        ms = 0  # mean square
        for pc in Pc:
            ms += np.min((pc[0]-self.x)**2 + (pc[1]-self.z)**2)
        return ms

    def fw_ms_vector(self, Lo, grad):
        ms = self.fw_ms(Lo)
        if grad.size > 0:
            grad[:] = op.approx_fprime(Lo, self.fw_ms, 1e-6)
        return ms

    def set_cmm(self, Lo, cmm, index):  # min max constraints
        ''' Lo adjusted so that all limit vaules are negitive '''
        dL = self.fw_length(Lo, 1)
        cmm[:self.nP+2] = self.dLlimit['min'] - dL
        cmm[self.nP+2:] = dL - self.dLlimit['max']
        return cmm[index]

    def Llimit(self, constraint, Lo, grad):
        dL_space = 1e-3  # minimum inter-point spacing
        if grad.size > 0:
            grad[:] = np.zeros((self.nC, self.nP))  # initalise
            for i in range(self.nP-1):  # order points
                grad[i, i] = -1
                grad[i, i + 1] = 1
            for i in range(2*self.nP+4):
                grad[self.nP-1+i, :] = op.approx_fprime(
                        Lo, self.set_cmm, 1e-6, np.zeros(2*self.nP+4), i)

        constraint[:self.nP-1] = Lo[:self.nP-1] - Lo[1:self.nP] + dL_space
        self.set_cmm(Lo, constraint[self.nP-1:], 0)

    def optimise(self):
        opt = nlopt.opt(nlopt.LD_SLSQP, self.nP)
        # opt = nlopt.opt(nlopt.LD_MMA, self.nP)
        opt.set_ftol_rel(1e-4)
        opt.set_ftol_abs(1e-4)
        opt.set_min_objective(self.fw_vector)
        opt.set_lower_bounds([0 for _ in range(self.nP)])
        opt.set_upper_bounds([1 for _ in range(self.nP)])
        tol = 1e-2 * np.ones(self.nC)
        opt.add_inequality_mconstraint(self.Llimit, tol)
        self.Lo = opt.optimize(self.Lo)
        if opt.last_optimize_result() < 0:
            warn('optimiser unconverged')
        Pc = self.fw_corners(self.Lo)[0]
        return Pc[:, 0], Pc[:, 1]

    def plot(self):
        Pc, Po = self.fw_corners(self.Lo)
        pl.plot(self.x, self.z)
        # pl.plot(Po[:, 0], Po[:, 1], 'o', ms=2)
        pl.plot(Pc[:, 0], Pc[:, 1], '-o', ms=8)
        pl.axis('equal')

if __name__ == '__main__':

    cutri, cutzi = 10.2472993945, 4.29229220147
    fw = R.BB.rb.segment['blanket_inner'].copy()

    ci = np.argmin((fw['x']-cutri)**2+(fw['z']-cutzi)**2)

    wrap = wrapper(fw['x'][:ci], fw['z'][:ci], angle=20, dx_min=0.75, dx_max=3)
    wrap.optimise()
    wrap.plot()

    dL = wrap.fw_length(wrap.Lo, 1)
    print(np.min(dL), np.max(dL))

    wrap = wrapper(fw['x'][ci:], fw['z'][ci:], angle=20, dx_min=0.75, dx_max=3)
    opt = wrap.optimise()
    wrap.plot()

    dL = wrap.fw_length(wrap.Lo, 1)
    print(np.min(dL), np.max(dL))
