"""Fit lumped-capacitance thermal models to heat-output timeseries."""

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize


class LumpedCapacitance:
    """Fit an RC lumped thermal model to a cooling / heat-output curve."""

    def __init__(self, t, Te, Qdot):
        self.t = t
        self.Te = Te
        self.Qdot = Qdot
        self.Qnorm = np.linalg.norm(Qdot)
        self.Te_interp = scipy.interpolate.interp1d(t, Te)

    def dTdt(self, t, T, hA, C):
        """Return lumped-capacitance temperature derivative."""
        Te = self.Te_interp(t)
        tau = C / hA
        return -1 / tau * (T - Te)

    def solve(self, hA, C):
        """Return heat output for the given transfer and capacitance values."""
        dTo = -1 / hA * self.Qdot[0]
        sol = scipy.integrate.solve_ivp(
            self.dTdt,
            (self.t[0], self.t[-1]),
            [dTo],
            args=(hA, C),
            t_eval=self.t,
            method="RK45",
        )
        dT = sol.y
        Qdot = -hA * dT
        return Qdot

    def Qdot_err(self, x):
        """Return normalized rms error between modeled and measured heat output."""
        Qdot = self.solve(*x)
        return np.sqrt(np.mean((Qdot - self.Qdot) ** 2)) / self.Qnorm

    def fit_hA(self):
        """Fit heat-transfer coefficient and capacitance to the measured curve."""
        xo = [0.1, 1]
        sol = scipy.optimize.minimize(self.Qdot_err, xo, method="COBYLA")
        hA, C = abs(sol.x)
        err = self.Qdot_err((hA, C))
        return hA, C, err
