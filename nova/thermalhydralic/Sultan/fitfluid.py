"""Optimize fit of LTI model to fluid timeseries."""
from dataclasses import dataclass, field
from typing import Union
import sys
from types import SimpleNamespace

import scipy
import numpy as np
import nlopt
import pandas

from nova.thermalhydralic.sultan.model import Model
from nova.thermalhydralic.sultan.fluidmodel import FluidModel
from nova.utilities.pyplot import plt


@dataclass
class FitFluid:
    """Extract minimal realization state-space model from step response."""

    fluid: Union[FluidModel, Model, list[int], int]
    data: SimpleNamespace = field(init=False, repr=False,
                                  default_factory=SimpleNamespace)
    verbose: bool = True

    def __post_init__(self):
        """Init fluid model."""
        if not isinstance(self.fluid, FluidModel):
            self.fluid = FluidModel(self.fluid)
        self.data.__init__()

    def extract_data(self, waveform_data: pandas.DataFrame):
        """Extract waveform.data."""
        for vector in ['time', 'fieldratesq', 'output']:
            setattr(self.data, vector, waveform_data[vector].to_numpy())
        for attribute in ['filename', 'fieldratesq_amplitude', 'frequency',
                          'massflow', 'samplenumber']:
            setattr(self.data, attribute, waveform_data.attrs[attribute])
        self.fluid.timeseries = (self.data.time, self.data.fieldratesq)

    def initialize_model(self):
        """Init LTI model."""
        self.fluid.model.update_pole(0.056*self.data.massflow)
        dcgain = 2*self.data.fieldratesq_amplitude * np.max(self.data.output)
        if dcgain < 1e-6:
            dcgain = 1e-6
        self.fluid.model.update_dcgain(dcgain)

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
        self.fluid.model.vector = vector  # update lti model
        L2norm = np.linalg.norm(self.data.output-self.fluid.output, axis=0)
        return 1e3*L2norm/self.data.samplenumber

    def model_update(self, vector, grad):
        """Return L2norm error and evaluate gradient in-place."""
        err = self.model_error(vector)
        if self.verbose:
            sys.stdout.write(f'\r{err}')
            sys.stdout.flush()
        if len(grad) > 0:
            grad[:] = scipy.optimize.approx_fprime(
                vector, self.model_error, 1e-6)
        return err

    def _set_nlopt(self, algorithm):
        opt = nlopt.opt(getattr(nlopt, algorithm),
                        self.fluid.model.parameter_number)
        initial_step = 0.05*np.array(self.fluid.model.vector)
        initial_step[initial_step < 1e-2] = 1e-2
        opt.set_initial_step(initial_step)
        opt.set_min_objective(self.model_update)
        opt.set_lower_bounds(1e-6*np.ones(self.fluid.model.parameter_number))
        opt.set_ftol_abs(5e-3)
        return opt

    def optimize(self, waveform_data: pandas.DataFrame, optimizer='nlopt'):
        """Fit model parameters to waveform heat output."""
        self.extract_data(waveform_data)
        self.initialize_model()
        if optimizer == 'scipy':
            res = scipy.optimize.minimize(self.model_error,
                                          self.fluid.model.vector)
            self.fluid.model.vector = res.x
        elif optimizer == 'nlopt':
            try:
                opt = self._set_nlopt('LN_BOBYQA')
                vector = opt.optimize(self.fluid.model.vector)
            except nlopt.RoundoffLimited:
                opt = self._set_nlopt('LN_COBYLA')
                opt.set_ftol_rel(1e-2)
                vector = opt.optimize(self.fluid.model.vector)

            self.fluid.model.vector = vector
        else:
            raise IndexError(f'optimizer {optimizer} not in [scipy, nlopt]')

    @property
    def coefficents(self):
        """Return fitting coefficients."""
        coefficents = {}
        for attr in ['fieldratesq_amplitude', 'massflow', 'frequency']:
            coefficents[attr] = getattr(self.data, attr)
        coefficents['steadystate'] = self.steadystate
        coefficents['steadystate_error'] = self.steadystate_error
        coefficents['L2norm'] = self.model_error(self.fluid.model.vector)
        return pandas.Series(coefficents)

    @property
    def steadystate(self):
        """Return steady state heat output."""
        return 0.5*self.data.fieldratesq_amplitude*self.fluid.model.dcgain

    @property
    def steadystate_error(self):
        """Return percentage steady state error."""
        maxdata = np.max(self.data.output)
        return 100*(self.steadystate-maxdata) / maxdata

    def plot(self, axes=None):
        """Plot model fit."""
        if axes is None:
            axes = plt.gca()
        axes.plot(self.data.time, self.data.output, label='data')
        axes.plot(self.data.time, self.fluid.output,
                  label=f'model {self.fluid.model.label}')
        plt.xlabel('$t$ s')
        plt.ylabel(r'heat output W')
        plt.legend(loc='center right')
        plt.despine()
