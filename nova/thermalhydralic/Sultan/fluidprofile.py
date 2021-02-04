"""Fit state-space model to sultan thermal-hydralic step resonse."""
from dataclasses import dataclass, field, InitVar
from typing import Union

import pandas
import numpy as np

from nova.thermalhydralic.sultan.campaign import Campaign
from nova.thermalhydralic.sultan.trial import Trial
from nova.thermalhydralic.sultan.sample import Sample
from nova.thermalhydralic.sultan.profile import Profile
from nova.thermalhydralic.sultan.waveform import WaveForm
from nova.thermalhydralic.sultan.model import Model
from nova.thermalhydralic.sultan.fluidmodel import FluidModel
from nova.thermalhydralic.sultan.fitfluid import FitFluid
from nova.thermalhydralic.sultan.sultanio import SultanIO
from nova.utilities.pyplot import plt


@dataclass
class FluidProfile(SultanIO):
    """Manage fluid model and non-linear fits to experimental data."""

    profile: Union[Sample, Trial, Campaign, str]
    fluid: Union[FluidModel, Model, list[int], int] = field(default=4)
    _threshold: InitVar[float] = 0
    _delay: InitVar[bool] = False
    reload: bool = False
    _coefficents: pandas.Series = field(init=False, repr=False)
    verbose: InitVar[bool] = True

    def __post_init__(self, _threshold, _delay, verbose):
        """Init profile and waveform."""
        if not isinstance(self.profile, Profile):
            self.profile = Profile(self.profile)
        if not isinstance(self.fluid, FluidModel):
            self.fluid = FluidModel(self.fluid)
        self.profile.normalize = False
        self.fluid.model.delay = _delay
        self.waveform = WaveForm(self.profile, _threshold)
        self.fit = FitFluid(self.fluid, verbose=verbose)
        self.load_model()

    def _reload(self):
        """Reload on-demand (model, coefficents)."""
        if self.profile.sample.sourcedata.reload.fluidmodel:
            self.load_model()

    @property
    def model(self):
        """Return fluid.model."""
        self._reload()
        return self.fluid.model

    @property
    def timeseries(self):
        """Return waveform timeseries."""
        self._reload()
        return self.waveform.timeseries

    def load_model(self):
        """Load LTI fluid model."""
        if self.reload:
            self._coefficents = self.read_data()
        else:
            self._coefficents = self.load_data()
        self.model_coefficents = self._coefficents.copy()
        self.profile.sample.sourcedata.reload.fluidmodel = False

    @property
    def database(self):
        """Return database instance."""
        return self.profile.sample.trial.database

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.database.binary_filepath('fluidprofile.h5')

    @property
    def filepath(self):
        """Return full path of source datafile, read-only."""
        return self.database.datafile(self.filename)

    @property
    def filename(self):
        """Return datafile filename."""
        order = ''.join([str(order) for order in self.fit.fluid.model.order])
        threshold = str(int(1000*self.waveform.threshold))
        side = self.profile.sample.side
        delay = 1 if self.fit.fluid.model.delay else 0
        filename = f'{self.profile.sample.filename}_{side}'
        filename += f'_{order}_{threshold}_{delay}'
        return filename

    def _read_data(self):
        """Refit fluid model to waveform data."""
        self.fit.optimize(self.waveform.data)
        model_coefficents = self._flatten_coefficents(
            self.fluid.model.coefficents)
        fit_coefficents = self.fit.coefficents
        postprocess_coefficents = pandas.Series({
            'energy_model': self.pulse_energy,
            'cutoff_factor':
                model_coefficents['pole0'] / fit_coefficents['massflow']})
        return pandas.concat([model_coefficents, fit_coefficents,
                              postprocess_coefficents])

    def _flatten_coefficents(self, coefficents) -> pandas.Series:
        """Return flat coefficents float dtype."""
        distinct_poles = len(coefficents.order)
        model = pandas.Series({'distinct_poles': distinct_poles})
        for i in range(distinct_poles):
            model[f'order{i}'] = coefficents['order'][i]
            model[f'pole{i}'] = coefficents['repeated_pole'][i]
        coefficents = coefficents.drop(index=['order', 'repeated_pole'])
        coefficents = coefficents.astype(float)
        coefficents = pandas.concat([model, coefficents])
        return coefficents

    def _assemble_coefficents(self, coefficents):
        """Return coefficents in fit.model format."""
        distinct_poles = int(coefficents.distinct_poles)
        drop_list = ['distinct_poles']
        order = [None for __ in range(distinct_poles)]
        repeated_pole = [None for __ in range(distinct_poles)]
        for i in range(distinct_poles):
            order[i] = int(coefficents.loc[f'order{i}'])
            repeated_pole[i] = coefficents.loc[f'pole{i}']
            drop_list.extend([f'order{i}', f'pole{i}'])
        coefficents.drop(drop_list, inplace=True)
        coefficents = pandas.concat(
            [pandas.Series({'order': order, 'repeated_pole': repeated_pole}),
             coefficents])
        return coefficents

    @property
    def model_coefficents(self):
        """Manage fluid model fitting coefficents."""
        self._reload()
        return self.model.coefficents

    @model_coefficents.setter
    def model_coefficents(self, coefficents):
        coefficents = self._assemble_coefficents(coefficents)
        self.fluid.model.order = coefficents.order
        self.fluid.model.vector = [*coefficents.repeated_pole,
                                   coefficents.dcgain,
                                   coefficents.time_delay]
        self.fluid.model.delay = np.isclose(coefficents.delay, 1)

    @property
    def fit_coefficents(self):
        """Return fit coefficents."""
        distinct_poles = int(self.coefficents.distinct_poles)
        return self.coefficents.iloc[3 + 2*distinct_poles:]

    @property
    def coefficents(self):
        """Return response coefficents."""
        self._reload()
        return self._coefficents

    @property
    def shot_coefficents(self):
        """Return profile and response coefficents."""
        return pandas.concat([self.profile.coefficents, self.coefficents])

    @property
    def pulse_energy(self):
        """Return intergral power."""
        self.fluid.timeseries = self.waveform.timeseries(1, pulse=True)
        coldindex = np.argmax(self.fluid.time >= self.profile.cold[0])
        return np.trapz(self.fluid.output[:coldindex],
                        self.fluid.time[:coldindex])

    @property
    def steadystate(self):
        """Return steadystate."""
        return self.coefficents.steadystate

    @property
    def steadystate_error(self):
        """Return percentage steady state error."""
        return self.coefficents.steadystate_error

    def plot(self, threshold=0.85, predict=True, correct=True, TF=True,
             heat=True, axes=None):
        """
        Plot fluid response.

        Parameters
        ----------
        threshold : float, optional
            Plot cooldown threshold. The default is 0.95.
        predict : bool, optional
            Plot prediction. The default is True.
        correct : bool, optional
            Plot correction. The default is True.
        TF : bool, optional
            Plot transfer function label. The default is True.
        axes : plt.axes, optional
            Target axes. The default is None.

        Returns
        -------
        None.

        """
        if threshold < self.waveform.threshold:
            threshold = self.waveform.threshold
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        self._reload()
        self.fluid.timeseries = self.waveform.timeseries(pulse=True)
        self.fluid.plot(axes=axes, color='C3', label='fit')
        self.plot_data(threshold, axes)
        if predict and threshold > self.waveform.threshold:
            self.plot_predict(threshold, axes)
        if correct:
            self.plot_correct(threshold, axes)
        if heat:
            self.plot_heat(axes)
        if TF:
            self.plot_transfer_function(axes)
        axes.legend(ncol=4, loc='upper center', frameon=False,
                    bbox_to_anchor=(0.5, 1.13))
        plt.despine()
        axes.set_xlabel('$t$ s')
        axes.set_ylabel(r'$\dot{Q}$ W')
        plt.title(self.profile.sample.label, color='k', y=1.1)

    def plot_data(self, threshold, axes):
        """Plot sultan data."""
        timeseries = self.waveform.timeseries(threshold, pulse=True)
        axes.plot(timeseries[0], timeseries[-1], 'C0', label='data', zorder=-1)

    def plot_predict(self, threshold, axes):
        """Plot steady-state correction."""
        self.fluid.timeseries = self.waveform.timeseries(threshold, pulse=True)

        self.fluid.plot(axes=axes, color='C2', label='predict', zorder=-2)

    def plot_correct(self, threshold, axes, arrow=False):
        """Plot steady-state correction."""
        self.fluid.timeseries = self.waveform.timeseries(
            threshold, pulse=False)
        self.fluid.plot(axes=axes, color='C4', zorder=-1, label='correct')
        duration = self.fluid.timeseries[0].iloc[-1]
        maximum = self.profile.maximum
        tlim = [maximum[0], duration]
        tdelta = np.diff(tlim)[0]
        axes.plot([tlim[0] + 0.1*tdelta, tlim[1]],
                  maximum[1] * np.ones(2), ':C0')
        if arrow:
            axes.arrow(duration, maximum[1],
                       0, 0.95*0.01*maximum[1]*self.steadystate_error,
                       length_includes_head=True,
                       head_width=1, head_length=0.1, ec='C4', fc='C4')
        else:
            axes.plot(duration * np.ones(2),
                      maximum[1] * np.array(
                          [1, 1 + 0.95*0.01*self.steadystate_error]), 'C4')
        va = 'bottom' if np.sign(self.steadystate_error) < 0 else 'top'
        axes.text(duration, maximum[1] * (1 + 0.005*self.steadystate_error),
                  f' {self.steadystate_error:1.1f}%',
                  ha='left', va=va, color='C4')
        va = 'bottom' if np.sign(self.steadystate_error) >= 0 else 'top'
        axes.text(duration, self.steadystate, f' {self.steadystate:1.1f}W',
                  ha='left', va=va, color='C4')

    def plot_heat(self, axes):
        """Plot heated zone."""
        self.waveform.plot_heat(axes, label=None, zorder=-5, alpha=0.5)

    def plot_transfer_function(self, axes):
        """Plot transfer function label."""
        massflow = self.profile.sample.massflow
        transfer_function = fr'{self.model.get_label(massflow)}'
        axes.text(0.5, 0.5, transfer_function, transform=axes.transAxes,
                  fontsize='x-large', fontweight='bold',
                  ha='center', va='center', color='k',
                  bbox={'boxstyle': 'round', 'facecolor': 'none',
                        'edgecolor': 'none'})


if __name__ == '__main__':

    trial = Trial('CSJA_3', 0)
    sample = Sample(trial, 0, 'Left')

    fluidprofile = FluidProfile(sample, [4], 0, reload=False)

    #print(fluidprofile.coefficents)
    fluidprofile.plot()




