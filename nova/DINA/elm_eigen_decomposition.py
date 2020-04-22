from scipy.optimize import fsolve
import numpy as np
from amigo.pyplot import plt
from scipy.integrate import odeint


class eigen:

    def __init__(self, Lp=0.23e-3, Rp=2.6e-3):
        self.set_primary_coil(Lp, Rp)

    def set_primary_coil(self, Lp, Rp):
        self.Lp = Lp
        self.Rp = Rp

    def solve_eigen(self, tau, Io):  # eigen.value (input)
        self.Io = Io
        self.tau = tau
        self.Lambda = -1/self.tau  # eigenvalues
        self.solve_eigenvector()  # solve system eigenvectors
        self.solve_dummy_inductance()  # solve dummy coil parameters

    @staticmethod
    def time_zero(x, *args):  # inital conditions
        alpha, beta, theta = x
        Ialpha, Ibeta = args[0]
        err = np.ones(3)
        err[0] = alpha*np.sin(theta) + beta * np.cos(theta)
        err[1] = alpha*np.cos(theta) - Ialpha
        err[2] = -beta*np.sin(theta) - Ibeta
        return err

    @staticmethod
    def dummy_coil(x, *args):
        Ld, Ldp = x  # dummy coil
        Lp, Rp, M = args  # primary coil
        Rd = Rp  # enforce M symetric
        err = np.ones(2)
        err[0] = -Ld*Rp/(Ld*Lp-Ldp**2) - M[0, 0]
        err[1] = -Lp*Rd/(Ld*Lp-Ldp**2) - M[1, 1]
        return err

    def solve_eigenvector(self):
        theta_o = np.pi/2  # sead eigenvector direction
        xo = [self.Io[0]/np.cos(theta_o),
              -self.Io[1]/np.sin(theta_o), theta_o]
        x = fsolve(self.time_zero, xo, args=self.Io, factor=0.01)
        self.alpha, self.beta, self.theta = x
        self.v = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                           [np.sin(self.theta), np.cos(self.theta)]])
        D = np.array([[self.Lambda[0], 0], [0, self.Lambda[1]]])
        self.M = self.v @ D @ np.linalg.inv(self.v)

    def solve_dummy_inductance(self):
        xo = self.Lp * np.array([0.5, 0.25])
        x = fsolve(self.dummy_coil, xo, args=(self.Lp, self.Rp, self.M),
                   factor=0.01)
        self.Ld, self.Lpd = x  # dummy coil self and mutual inductance
        self.L = np.array([[self.Lp, self.Lpd], [self.Lpd, self.Ld]])
        self.Linv = np.linalg.inv(self.L)

    def get_profile(self, t):
        Iexp = np.zeros((2, len(t)))
        for i in range(2):
            Iexp[i] = self.alpha * self.v[i, 0] * np.exp(self.Lambda[0]*t)
            Iexp[i] += self.beta * self.v[i, 1] * np.exp(self.Lambda[1]*t)
        return Iexp

    def dIdt(self, I, t):
        dI = self.M @ I
        return dI

    def solve_profile(self, t, Io):
        Iode = odeint(self.dIdt, Io, t)
        return Iode

    def plot_eigen(self, ax=None, color='C0', label=''):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        t = np.linspace(0, 0.25, 100)
        Iexp = self.get_profile(t)
        Iode = self.solve_profile(t, [Iexp[0, 0], 0]).T
        ax.plot(1e3*t, 1e-3*Iexp[0], '-', color=color, label=label)
        ax.plot(1e3*t, 1e-3*Iode[0], '--', color='gray')
        ax.plot(1e3*t, 1e-3*Iexp[1], '-.', color=color)
        plt.despine()
        ax.legend()
        ax.set_xlabel('$t$ ms')
        ax.set_ylabel('$I$ kA')


if __name__ is '__main__':

    ax = plt.subplots(1, 1)[1]
    LTC = eigen()
    LTC.solve_eigen(1e-3*np.array([5.1, 61.7]), [2147.5, 12617.1])
    LTC.plot_eigen(ax=ax, color='C0', label='LTC')
    ENP = eigen()
    ENP.solve_eigen(1e-3*np.array([15.9, 82.7]), [2134.2, 12846.9])
    ENP.plot_eigen(ax=ax, color='C1', label='ENP')
