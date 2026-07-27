"""Fit lumped-capacitance thermal models to heat-output timeseries."""

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize


class LumpedCapacitance:
    """Fit an RC lumped thermal model to a cooling / heat-output curve."""

    #: Integrator tolerances for the model response. solve_ivp defaults to
    #: rtol=1e-3, which leaves several tenths of a percent of integration error
    #: after a few time constants -- the same order as the fit residual the
    #: optimiser is minimising, so the fitted time constant would be chasing
    #: integrator noise. A single first-order ODE is cheap enough to solve far
    #: tighter than any measurement it is fitted against.
    rtol = 1e-8
    atol = 1e-10

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
            rtol=self.rtol,
            atol=self.atol,
        )
        # solve_ivp returns one row per state variable; this model carries a
        # single temperature difference, so the heat output is one row and must
        # come back on the input timebase rather than as a (1, n) frame
        return -hA * sol.y[0]

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
