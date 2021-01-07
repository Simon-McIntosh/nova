"""Fit state-space model to sultan thermal-hydralic step resonse."""
from dataclasses import dataclass, field
from types import SimpleNamespace

import scipy
import numpy as np
import nlopt

from nova.utilities.pyplot import plt
from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.shotprofile import ShotProfile
from nova.thermalhydralic.sultan.shotresponse import ShotResponse


@dataclass
class ResponseModel:
    """Manage optimisation data."""

    npole: int = 3
    _vector: list[float] = field(init=False, default_factory=list)
    lti: scipy.signal.lti = field(init=False, repr=False)

    @property
    def vector(self):
        """
        Manage optimization vector.

        Parameters
        ----------
        vector : list
            Optimization vector [pole, gain, delay].

        """
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = vector
        self._generate()  # regenerate lti

    @property
    def pole(self):
        """Return location of repeated pole."""
        return self._vector[0]

    @property
    def gain(self):
        """Return gain."""
        return self._vector[1]

    @property
    def delay(self):
        """Return time delay."""
        return self._vector[2]

    @property
    def step(self):
        """Return steady state step response."""
        return self.gain / self.pole**self.npole

    @property
    def label(self):
        """Return transfer function text descriptor."""
        return (fr'$\frac{{{self.gain:1.4f}}}'
                fr'{{(s+{self.pole:1.3f})^{self.npole}}}'
                fr'{{\rm e}}^{{-{self.delay:1.2f}s}}$')

    def _generate(self):
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
        self.lti = scipy.signal.ZerosPolesGain(
            [], -self.pole*np.ones(self.npole), self.gain)


@dataclass
class ModelResponse:
    """Extract minimal realization state-space model from step response."""

    time: list[float]
    heat: list[float]
    npole: int = 5
    _pole: float = 0.5
    _delay: float = 10

    def __post_init__(self):
        self.model = ResponseModel(self.npole)
        self.model.vector = self._sead

    @property
    def _sead(self):
        """
        Return sead optimization vector.

        Parameters
        ----------
        pole : float
            Location of repeated pole (positive).
        delay : float
            Time delay.

        Returns
        -------
        sead_vector : array-like
            Sead optimization vector [pole, gain, delay].

        """
        return np.array([self._pole,
                         self.heat[-1]*self._pole**self.model.npole,
                         self._delay])

    @property
    def response(self):
        """
        Return model step response.

        Returns
        -------
        model_step : array-like
            model step response.

        """
        response = scipy.signal.step(self.model.lti, T=self.time)[1]
        return self._timeshift(self.time, response)

    def _timeshift(self, time, response):
        """
        Return response shifted in time by delay.

        Parameters
        ----------
        time : array-like
            time array.
        response : array-like
            model step response.

        Returns
        -------
        response : array-like
            model step response shifted by self.delay seconds.

        """
        bounds = (response[0], response[-1])
        return scipy.interpolate.interp1d(
            time+self.model.delay, response, fill_value=bounds,
            bounds_error=False)(time)

    def model_error(self, vector):
        """
        Return L2norm model error.

        Parameters
        ----------
        vector : array-like
            Optimization vector [pole, gain, delay].
        npole : int
            Number of poles.

        Returns
        -------
        error : float
            L2-norm error.

        """
        self.model.vector = vector  # update lti model
        err = np.linalg.norm(self.heat-self.response, axis=0)
        return err

    def model_update(self, vector, grad):
        """Return L2norm error and evaluate gradient in-place."""
        err = self.model_error(vector)
        if len(grad) > 0:
            grad[:] = scipy.optimize.approx_fprime(vector,
                                                   self.model_error, 1e-6)
        return err

    def fit(self, optimizer='nlopt'):
        """Fit model to shot step response."""
        if optimizer == 'scipy':
            res = scipy.optimize.minimize(self.model_error, self._sead)
            self.model.vector = res.x
        elif optimizer == 'nlopt':
            opt = nlopt.opt(nlopt.LN_PRAXIS, 3)
            opt.set_initial_step([0.1, 0.1, 0.5])
            opt.set_min_objective(self.model_update)
            opt.set_lower_bounds(np.zeros(3))
            opt.set_ftol_rel(1e-5)
            vector = opt.optimize(self._sead)
            self.model.vector = vector
        else:
            raise IndexError(f'optimizer {optimizer} not in [scipy, nlopt]')
        return self.model.vector

    def plot(self, ax=None):
        """Plot step response."""
        if ax is None:
            ax = plt.gca()
        plt.plot(self.time, self.heat, 'C3', label='data')
        plt.plot(self.time, self.response, 'k--',
                 label=f'model {self.model.label}')
        plt.plot([self.time[0], self.time[-1]],
                 self.model.step * np.ones(2), 'C7-')
        plt.text(self.time[0], self.model.step, 'steady-state',
                 ha='left', va='bottom', color='C7')
        plt.text(self.time[-1], self.model.step,
                 f' {self.model.step:1.2f}',
                 ha='left', va='center', color='C7')
        plt.xlabel('$t$ s')
        plt.ylabel(r'$\dot{Q}$ W')
        plt.legend(loc='lower right')
        plt.despine()


if __name__ == '__main__':

    plan = TestPlan('CSJA_3', 'ac0')
    instance = ShotInstance(plan, 21)
    profile = ShotProfile(instance)

    response = ModelResponse(*profile.shotresponse.stepdata, 6)

    response.fit()
    response.plot()

    '''
        self.profile.plot(offset=True)

        heat_index = self.profile.shotresponse.heat_index
        time_offset = self.profile.lowpassdata.loc[heat_index.start,
                                                   ('t', 's')]

        print(time_offset)
        time_limit = self.profile.lowpassdata.t.iloc[-1].values[0]

        t = np.linspace(0, time_limit-time_offset)
        U = np.zeros(len(t))
        U[t <= self.stepdata.time[-1]] = 1
        y = scipy.signal.lsim(self.model.lti, U, t)[1]
        y = self._timeshift(t, y)
        plt.plot(t, y, 'C1')

    '''
