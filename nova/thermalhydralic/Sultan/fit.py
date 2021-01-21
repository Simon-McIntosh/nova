"""Fit state-space model to sultan thermal-hydralic step resonse."""
from dataclasses import dataclass, field, InitVar
import sys

import scipy
import numpy as np
import nlopt
import pandas

from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.waveform import WaveForm
from nova.thermalhydralic.sultan.model import Model
from nova.utilities.pyplot import plt


def FitSeries():
    """Return empty fit data series."""
    return pandas.Series(index=['order', 'cutoff', 'dcgain',
                                'time_delay',
                                'waveform_amplitude', 'frequency',
                                'massflow', 'L2norm', 'steadystate'],
                         dtype=float)


@dataclass
class Fit:
    """Extract minimal realization state-space model from step response."""

    waveform_data: InitVar[pandas.DataFrame]
    model: Model
    data: dict = field(init=False, repr=False, default_factory=dict)
    _model_output: list[float] = field(init=False, repr=False)
    _coefficents: pandas.Series = field(repr=False, default_factory=FitSeries)
    reload: bool = True

    def __post_init__(self, waveform_data):
        """Init LTI model and extract waveform."""
        self._extract_data(waveform_data)
        self.model = Model(self.model.order, self.massflow*0.056,
                           2*np.max(self.heat_output), self.model.time_delay)

    def _extract_data(self, waveform_data):
        for vector in ['time', 'waveform_input', 'heat_output']:
            self.data[vector] = waveform_data[vector].to_numpy()
        for attributes in ['waveform_amplitude', 'frequency',
                           'massflow', 'samplenumber']:
            self.data[attributes] = waveform_data.attrs[attributes]

    @property
    def coefficents(self):
        """Return fitting coefficents."""
        if self.reload:
            self._coefficents['order'] = self.model.order[0]
            self._coefficents['cutoff'] = self.model.repeated_pole[0]
            for attr in ['dcgain', 'time_delay']:
                self._coefficents[attr] = getattr(self.model, attr)
            for attr in ['waveform_amplitude', 'massflow', 'frequency',
                         'steadystate']:
                self._coefficents[attr] = getattr(self, attr)
            self._coefficents['L2norm'] = self.model_error(self.model.vector)
            self.reload = False
        return self._coefficents

    @property
    def time(self):
        """Return time waveform."""
        return self.data['time']

    @property
    def waveform_input(self):
        """Return time waveform."""
        return self.data['waveform_input']

    @property
    def heat_output(self):
        """Return heat output waveform."""
        return self.data['heat_output']

    @property
    def waveform_amplitude(self):
        """Return input waveform amplitude."""
        return self.data['waveform_amplitude']

    @property
    def frequency(self):
        """Return imput waveform frequency."""
        return self.data['frequency']

    @property
    def massflow(self):
        """Return sample massflow."""
        return self.data['massflow']

    @property
    def samplenumber(self):
        """Return sample number."""
        return self.data['samplenumber']

    @property
    def model_output(self):
        """
        Return model step response.

        Returns
        -------
        model_step : array-like
            model step response.

        """
        if self.model.reload:
            output = scipy.signal.lsim2(self.model.lti, self.waveform_input,
                                        T=self.time, atol=1e-4)[1]
            self._model_output = output
            self._model_output = self._timeshift(output)
            self.model.reload = False
        return self._model_output

    def _timeshift(self, output):
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
            model step response shifted by self.time_delay seconds.

        """
        bounds = (output[0], output[-1])
        return scipy.interpolate.interp1d(
            self.time+self.model.time_delay, output, fill_value=bounds,
            bounds_error=False)(self.time)

    def model_error(self, vector):
        """
        Return L2norm model error.

        Parameters
        ----------
        vector : array-like
            Optimization vector [pole, gain, time_delay].
        npole : int
            Number of poles.

        Returns
        -------
        error : float
            L2-norm error.

        """
        self.model.vector = vector  # update lti model
        L2norm = np.linalg.norm(self.heat_output-self.model_output, axis=0)
        return 1e6*L2norm/self.samplenumber

    def model_update(self, vector, grad):
        """Return L2norm error and evaluate gradient in-place."""
        err = self.model_error(vector)
        sys.stdout.write(f'\r{err}')
        sys.stdout.flush()
        if len(grad) > 0:
            grad[:] = scipy.optimize.approx_fprime(
                vector, self.model_error, 1e-6)
        return err

    def optimize(self, optimizer='nlopt'):
        """Fit model parameters to waveform heat output."""
        if optimizer == 'scipy':
            res = scipy.optimize.minimize(self.model_error, self.model.vector)
            self.model.vector = res.x
        elif optimizer == 'nlopt':
            opt = nlopt.opt(nlopt.LN_COBYLA, self.model.parameter_number)
            opt.set_initial_step(0.25*np.array(self.model.vector))
            opt.set_min_objective(self.model_update)
            opt.set_lower_bounds(np.zeros(self.model.parameter_number))
            opt.set_ftol_rel(1e-5)
            vector = opt.optimize(self.model.vector)
            self.model.vector = vector
        else:
            raise IndexError(f'optimizer {optimizer} not in [scipy, nlopt]')
        self.reload = True
        return self.model.vector

    @property
    def steadystate(self):
        """Return steady state heat output."""
        return 0.5*self.waveform_amplitude*self.model.dcgain

    def plot(self, axes=None):
        """Plot model fit."""
        if axes is None:
            axes = plt.gca()
        axes.plot(self.time, self.heat_output, label='data')
        axes.plot(self.time, self.model_output,
                  label=f'model {self.model.label}')
        plt.plot([self.time[0], self.time[-1]],
                 self.steadystate * np.ones(2), 'C7-')
        plt.text(self.time[0], self.steadystate, 'steady-state',
                 ha='left', va='bottom', color='C7')
        plt.text(self.time[-1], self.steadystate,
                 f' {self.steadystate:1.2f}',
                 ha='left', va='center', color='C7')
        plt.xlabel('$t$ s')
        plt.ylabel(r'$\dot{Q}$ W')
        plt.legend(loc='lower right')
        plt.despine()


if __name__ == '__main__':

    sample = Sample('CSJA12', 13)
    waveform = WaveForm(sample, 0.9, pulse=True)
    model = Model(5)

    fit = Fit(waveform.data, model)

    #fit.model_output(fit.model.vector)
    fit.optimize()
    fit.plot()

    '''
    data = pandas.DataFrame(index=range(sample.samplenumber),
                            columns=FitSeries().index)
    for __, shot in sample.sequence():
        print(shot)
        fit = Fit(waveform.data, model)
        fit.optimize()
        data.iloc[shot] = fit.coefficents
    '''




    '''
    response.profile.instance.index = 11
    response.plot()

    step = StepResponse(*response.stepdata, model)
    step.fit()
    step.plot()
    '''

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
