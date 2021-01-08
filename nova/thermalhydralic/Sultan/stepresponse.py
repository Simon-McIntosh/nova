"""Fit state-space model to sultan thermal-hydralic step resonse."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import scipy
import numpy as np
import nlopt
import pandas

from nova.utilities.pyplot import plt
from nova.thermalhydralic.sultan.shotresponse import ShotResponse


@dataclass
class LTIModel:
    """Manage linear time-invariant model."""

    order: tuple[int] = field(default=(6,))
    _vector: list[float] = field(init=False, default_factory=list)
    _pole: InitVar[Union[float, list[float]]] = 0.5
    _delay: InitVar[float] = 5
    lti: scipy.signal.lti = field(init=False, repr=False)

    def __post_init__(self, _pole, _delay):
        """Init linear time-invariant model."""
        if not self._vector:  # vector unset
            self.vector = self._sead(_pole, _delay)
        else:
            self.vector = self._vector

    def _sead(self, _pole, _delay):
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
            Sead optimization vector [*pole, gain, delay].

        """
        if not pandas.api.types.is_list_like(_pole):
            _pole = [_pole for __ in range(self.system_number)]
        else:
            _pole = list(_pole)
        vector = _pole + [1, _delay]
        if len(vector) != self.parameter_number:
            raise IndexError(f'sead length {len(vector)} != '
                             f'parameter number {self.parameter_number}\n'
                             f'check _pole kwarg {_pole}')
        return vector

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
    def system_number(self):
        """Return system number, read-only."""
        return len(self.order)

    @property
    def parameter_number(self):
        """Return number of optimization parameters, read-only."""
        return len(self.order) + 2

    @property
    def repeated_pole(self):
        """Return repeated system poles."""
        return self._vector[:self.system_number]

    @property
    def pole(self) -> list[float]:
        """Return negated poles."""
        pole_list = []
        for pole, order in zip(self.repeated_pole, self.order):
            pole_list += [-pole for __ in range(order)]
        return pole_list

    @property
    def pole_gain(self):
        """Return steady state pole gain, read-only."""
        return np.prod(np.array(self.repeated_pole)**self.order)

    @property
    def gain(self):
        """Return gain."""
        return self._vector[-2]

    @property
    def delay(self):
        """Return time delay."""
        return self._vector[-1]

    @property
    def step(self):
        """Manage steady state step response."""
        return self.gain / self.pole_gain

    @step.setter
    def step(self, step):
        self._vector[-2] = self.pole_gain*step

    @property
    def label(self):
        """Return transfer function text descriptor."""
        numerator = f'{self.gain:1.4f}'
        denominator = ''.join([fr'(s+{pole:1.3f})^{order}'
                               for pole, order in
                               zip(self.repeated_pole, self.order)])
        plt.title(fr'${denominator}$')
        #return (fr'$\frac{{{self.gain:1.4f}}}'
        #        fr'{{(s+{self.pole:1.3f})^{self.npole}}}'
        #        fr'{{\rm e}}^{{-{self.delay:1.2f}s}}$')

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
        self.lti = scipy.signal.ZerosPolesGain([], self.pole, self.gain)


@dataclass
class StepResponse:
    """Extract minimal realization state-space model from step response."""

    time: list[float] = field(repr=False)
    heat: list[float] = field(repr=False)
    model: LTIModel

    def __post_init__(self):
        """Init model gain."""
        self.model.step = self.heat[-1]

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
