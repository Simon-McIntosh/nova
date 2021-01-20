"""Fit state-space model to sultan thermal-hydralic step resonse."""
from dataclasses import dataclass

from typing import Union

import scipy
import numpy as np
import nlopt

from nova.thermalhydralic.sultan.waveform import WaveForm
from nova.thermalhydralic.sultan.profile import Profile
from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.model import Model
from nova.utilities.pyplot import plt


@dataclass
class Response:
    """Extract minimal realization state-space model from step response."""

    waveform: Union[WaveForm, Profile, Sample, Trial, Campaign, str]
    model: Model

    def __post_init__(self):
        """Init model gain."""
        self.time, self.heat = self.response.waveform
        self.model.step = self.heat[-1]

    @property
    def mod(self):
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
        print(err)
        if len(grad) > 0:
            grad[:] = scipy.optimize.approx_fprime(vector,
                                                   self.model_error, 1e-6)
        return err

    def fit(self, optimizer='nlopt'):
        """Fit model to shot step response."""
        if optimizer == 'scipy':
            res = scipy.optimize.minimize(self.model_error, self.model.vector)
            self.model.vector = res.x
        elif optimizer == 'nlopt':
            opt = nlopt.opt(nlopt.LN_PRAXIS, self.model.parameter_number)
            opt.set_initial_step(0.05*np.array(self.model.vector))
            opt.set_min_objective(self.model_update)
            opt.set_lower_bounds(np.zeros(self.model.parameter_number))
            opt.set_ftol_rel(1e-5)
            vector = opt.optimize(self.model.vector)
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

'''
    npole: int = 6

    def __post_init__(self):


    def stepresponse(self):
        """
        Return thermo-hydralic model parameters.

        Parameters
        ----------
        npole : int, optional
            Number of repeated poles. The default is 6.

        Returns
        -------
        vector : array-like
            Optimization vector [pole, gain, delay].
        steady_state : float
            Step response steady state.

        """
        response = ModelResponse(*self.stepdata, self.npole)
        vector = response.fit()
        step = response.model.step
        return vector, step
'''

if __name__ == '__main__':

    model = LTIModel()

    response = ShotResponse('CSJA13')
    response.profile.instance.index = 11
    response.plot()

    step = StepResponse(*response.stepdata, model)
    step.fit()
    step.plot()

    '''
    plan = TestPlan('CSJA13', -1)
    instance = ShotInstance(plan, -5)
    profile = ShotProfile(instance)

    response = ModelResponse(*profile.shotresponse.stepdata, 6)

    response.fit()
    response.plot()
    '''

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
