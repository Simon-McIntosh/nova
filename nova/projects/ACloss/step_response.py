"""Fit state-space model to sultan thermal-hydralic step resonse."""
import scipy
import numpy as np
import nlopt

from nova.utilities.pyplot import plt
from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.shotprofile import ShotProfile


class StepResponse:
    """Extract minimal realization state-space model from step response."""

    def __init__(self, step, minimum_gain=1e-12):
        self.t = step.t
        self.Qdot = step.Qdot
        self.minimum_gain = minimum_gain

    def generate_model(self, x, npole):
        """
        Generate linear time-invariant model.

        Parameters
        ----------
        x : array-like
            zeros, poles, gain.
        npole : int
            Number of poles.

        Returns
        -------
        lti : lti
            linear time-invariant model.

        """
        self.x = x
        self.lti = scipy.signal.ZerosPolesGain(
            [], self.pole*np.ones(npole), self.gain)

    @property
    def delay(self):
        """Return time delay."""
        return self.x[-1]

    @property
    def gain(self):
        """Return gain."""
        gain = self.x[-2]
        if gain < self.minimum_gain:
            gain = self.minimum_gain
        return gain

    @property
    def pole(self):
        """Return location of repeated pole."""
        return -self.x[0]

    @property
    def steady_state(self):
        """Return steady state step response."""
        return self.gain / (-self.pole)**self.npole

    @property
    def model_step(self):
        """
        Return model step response.

        Returns
        -------
        model_step : array-like
            model step response.

        """
        response = scipy.signal.step(self.lti, T=self.t)[1]
        return self.shift(self.t, response)

    def shift(self, t, response):
        """
        Return response shifted in time by delay.

        Parameters
        ----------
        t : array-like
            time.
        response : array-like
            model step response.

        Returns
        -------
        response : array-like
            model step response shifted by self.delay seconds.

        """
        bounds = (response[0], response[-1])
        return scipy.interpolate.interp1d(
            t+self.delay, response, fill_value=bounds, bounds_error=False)(t)

    def model_error(self, x):
        """
        Return L2norm model error.

        Parameters
        ----------
        x : array-like
            zeros, poles, gain.
        npole : int
            Number of poles.

        Returns
        -------
        error : float
            L2-norm error.

        """
        self.generate_model(x, self.npole)
        err = np.linalg.norm(self.Qdot-self.model_step, axis=0)
        return err

    def model_update(self, x, grad):
        """Return L2norm error and evaluate gradient in-place."""
        err = self.model_error(x)
        print(err)
        if len(grad) > 0:
            grad[:] = scipy.optimize.approx_fprime(x, self.model_error, 1e-4)
        return err

    def fit(self, npole, optimizer='nlopt'):
        """Fit model to shot step response."""
        self.npole = npole
        pole = 0.3
        xo = np.array([pole, self.Qdot[-1]*pole**npole, 0])
        if optimizer == 'scipy':
            res = scipy.optimize.minimize(self.model_error, xo, args=(npole))
            self.generate_model(res.x, self.npole)
        elif optimizer == 'nlopt':
            opt = nlopt.opt(nlopt.LD_MMA, len(xo))
            opt.set_min_objective(self.model_update)
            opt.set_lower_bounds(np.zeros(len(xo)))
            opt.set_ftol_rel(1e-5)
            x = opt.optimize(xo)
            self.generate_model(x, self.npole)
        else:
            raise IndexError(f'optimizer {optimizer} not in [scipy, nlopt]')

    def plot(self):
        """Plot step response."""
        plt.plot(self.t, self.Qdot)
        plt.plot(self.t, self.model_step)
        plt.plot([self.t[0], self.t[-1]], self.steady_state * np.ones(2),
                 'C7--')
        plt.xlabel('$t$ s')
        plt.ylabel(r'$\dot{Q}$ W')
        plt.despine()


if __name__ == '__main__':

    plan = TestPlan('CSJA_3', 'ac0')
    instance = ShotInstance(plan, 0)
    profile = ShotProfile(instance)
    step = profile.shotresponse.step

    stepresponse = StepResponse(step)

    stepresponse.fit(4)

    profile.plot(shif=True)
    stepresponse.plot()


    print(stepresponse.lti, stepresponse.delay, stepresponse.steady_state)
