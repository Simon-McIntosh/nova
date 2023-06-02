# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:49:27 2020

@author: mcintos
"""


class LumpedCapacitance:
    def __init__(self, t, P, Te):
        self.t = t
        self.P = P
        self.Te = scipy.interpolate.interp1d(t, Te)

    def Qconv(self, t, T, hA):
        return hA * (T - self.Te(t))

    def dTdt(self, t, T, hA, C, S):
        return -self.Qconv(t, T, hA) / C + S / C

    def solve(self, hA, C, S):
        sol = scipy.integrate.solve_ivp(
            self.dTdt,
            (self.t[0], self.t[-1]),
            [self.Te(self.t[0])],
            args=(hA, C, S),
            t_eval=self.t,
        )

        Qconv = self.Qconv(sol.t, sol.y, hA)
        return Qconv

    def rms(self, x):
        hA, C, S = x
        Qconv = self.solve(hA, C, S)
        return np.sqrt(np.mean((Qconv - self.P) ** 2))

    def fit(self):
        xo = 0.4, 0.1, 0.2
        sol = scipy.optimize.minimize(self.step_rms, xo)
        print(sol)
        return sol.x

    def step(self, x):
        lti = scipy.signal.lti(x[0], x[1:-1], x[-1:])
        Q = scipy.signal.step(lti, T=self.t, X0=0)[1]
        return Q

    def step_rms(self, x):
        Q = self.step(x)
        return np.sum((Q - self.P) ** 2)

    # lc = LumpedCapacitance(*spp.heat_curve())

    # hA, C, S = 0.1, 2000, 2000
    # print(lc.rms(hA, C, S))

    # x = lc.fit()

    # print(x)
    """
    Qconv = lc.solve(hA, C, S)
    plt.plot(lc.t, np.squeeze(Qconv))
    plt.plot(lc.t, lc.P)
    """
    # Q = lc.step(x)
    # plt.plot(lc.t, Q)
    # plt.plot(lc.t, lc.P)
