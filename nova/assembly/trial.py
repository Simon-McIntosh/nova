"""Run Monte Carlo simulations for candidate vault assemblies."""
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import cached_property
from time import time
from typing import ClassVar, Union

import numpy as np
import scipy.stats
import xarray
import xxhash

from nova.assembly import structural, electromagnetic
from nova.assembly.model import Dataset
from nova.utilities.pyplot import plt


@dataclass
class TrialAttrs:
    """Manage trial attributes."""

    samples: int = 100_000
    theta: list[float] = field(default_factory=lambda: [1.5, 1.5, 2, 2, 5, 0])
    pdf: list[str] = field(
        default_factory=lambda: ['uniform', 'uniform', 'normal', 'normal',
                                 'uniform', 'uniform'])
    modes: int = 3
    energize: Union[int, bool] = True
    wall: bool = False
    sead: int = 2025

    component: ClassVar[list[str]] = ['case', 'ccl', 'wall']
    signal: ClassVar[list[str]] = ['radial', 'tangential']
    ncoil: ClassVar[int] = 18

    @cached_property
    def _field_names(self):
        """Return list of field names."""
        return [attr.name for attr in fields(TrialAttrs)]

    @property
    def attrs(self):
        """Return trial attrs."""
        attrs = {}
        for attr in self._field_names:
            value = getattr(self, attr)
            if not isinstance(value, list):
                attrs[attr] = value
        return attrs


@dataclass
class Trial(Dataset, TrialAttrs):
    """Run stastistical analysis on trial vault assemblies."""

    filename: str = 'vault_trial'
    xxh32: xxhash.xxh32 = field(repr=False, init=False,
                                default_factory=xxhash.xxh32)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        self.energize = int(self.energize)
        self.wall = int(self.wall)
        self.group = self.group_name
        self.rng = np.random.default_rng(self.sead)
        self.structural_model = structural.Model()
        self.electromagnetic_model = electromagnetic.Model()
        super().__post_init__()

    @property
    def group_name(self):
        """Return group name as xxh32 hex hash."""
        self.xxh32.reset()
        self.xxh32.update(np.array(list(self.attrs.values()) +
                                   self.theta + self.pdf))
        return self.xxh32.hexdigest()

    def normal(self, variance: float):
        """Return sample with normal distribution."""
        scale = np.sqrt(variance)
        return self.rng.normal(scale=scale, size=(self.samples, self.ncoil))

    def uniform(self, bound: float):
        """Return sample with uniform distribution."""
        return self.rng.uniform(-bound, bound, size=(self.samples, self.ncoil))

    @contextmanager
    def timer(self):
        """Time build."""
        start_time = time()
        yield
        print(f'build time {time() - start_time:1.0f}s')

    def build(self):
        """Build Monte Carlo dataset."""
        with self.timer():
            self.build_signal()
            self.build_gap()
            self.predict_structure()
            self.predict_electromagnetic()
            if self.wall:
                self.predict_wall()
        return self.store()

    def build_signal(self):
        """Build input distributions."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data['sample'] = range(self.samples)
        self.data['index'] = range(self.ncoil)
        self.data['signal'] = self.signal
        self.data['case'] = xarray.DataArray(0., self.data.coords)
        self.data['ccl'] = xarray.DataArray(0., self.data.coords)
        self.data['wall'] = xarray.DataArray(0., self.data.coords)
        self.data['coordinate'] = ['x', 'y']
        self.data['component'] = self.component
        self.data['theta'] = ('component', 'signal'), \
            np.array(self.theta).reshape(self.data.dims['component'],
                                         self.data.dims['signal'])
        self.data['pdf'] = ('component', 'signal'), \
            np.array(self.pdf).reshape(self.data.dims['component'],
                                       self.data.dims['signal'])
        for i, component in enumerate(self.component):
            for j, signal in enumerate(self.signal):
                index = 2*i + j
                theta = self.theta[index]
                pdf = self.pdf[index]
                self.data[component].loc[..., signal] = \
                    getattr(self, pdf)(theta)

    def build_gap(self):
        """Build vault gap from radial and toroidal waveforms."""
        self.data['gap'] = xarray.zeros_like(self.data.case)
        self.data.gap[..., 0] = np.pi / self.ncoil * self.data['case'][..., 0]
        self.data.gap[:, :-1, 0] += \
            np.pi / self.ncoil * self.data['case'][:, 1:, 0].data
        self.data.gap[:, -1, 0] += \
            np.pi / self.ncoil * self.data['case'][:, 0, 0].data

        self.data.gap[..., 1] = -self.data['case'][..., 1]
        self.data.gap[:, :-1, 1] += self.data['case'][:, 1:, 1].data
        self.data.gap[:, -1, 1] += self.data['case'][:, 0, 1].data

    def predict_structure(self):
        """Run structural simulation."""
        self.data['structural'] = ('sample', 'index', 'signal'), \
            np.zeros((self.samples, self.ncoil, self.data.dims['signal']))
        if self.energize:
            gap = self.data.gap.sum(axis=-1)
            for i, signal in enumerate(self.data.signal.values):
                self.data['structural'][..., i] = \
                    self.structural_model.predict(gap, signal)

    def predict_electromagnetic(self):
        """Run electromagnetic simulation."""
        self.data['electromagnetic'] = self.data.structural.copy(deep=True)
        self.data.electromagnetic[:] += self.data.case
        self.data.electromagnetic[:] += self.data.ccl
        self.electromagnetic_model.predict(self.data.electromagnetic[..., 0],
                                           self.data.electromagnetic[..., 1])
        self.data['peaktopeak'] = 'sample', \
            self.electromagnetic_model.peaktopeak(modes=self.modes)
        self.data['offset'] = ('sample', 'coordinate'), \
            np.zeros((self.data.dims['sample'], 2))
        offset = self.electromagnetic_model.axis_offset
        self.data['offset'][..., 0] = offset.real
        self.data['offset'][..., 1] = -offset.imag
        self.data['peaktopeak_offset'] = 'sample', \
            self.electromagnetic_model.peaktopeak(modes=self.modes,
                                                  axis_offset=True)

    def predict_wall(self):
        """Predict combined wall-fieldline deviations."""
        ndiv = self.electromagnetic_model.fieldline.shape[1]
        wall_hat = np.fft.rfft(self.data.wall[..., 0])
        firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        wall_hat[..., 1] += \
            self.electromagnetic_model.axis_offset * (self.ncoil // 2)
        offset_firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        deviation = self.electromagnetic_model.fieldline.data - firstwall.data
        self.data['peaktopeak'] = 'sample', \
            self.electromagnetic_model.peaktopeak(deviation, modes=self.modes)
        offset_deviation = self.electromagnetic_model.fieldline.data - \
            offset_firstwall.data
        self.data['peaktopeak_offset'] = 'sample', \
            self.electromagnetic_model.peaktopeak(offset_deviation,
                                                  modes=self.modes)

    @property
    def pdf_text(self):
        """Return pdf text label."""
        text = ''
        for i, component in enumerate(self.component):
            if component == 'wall' and not self.wall:
                continue
            for j, signal in enumerate(self.signal):
                attr = signal[0]
                text += f'{component} '
                if attr == 't':
                    text += r'$r$'
                    attr = r'\phi'
                text += rf'$\Delta {attr}$'
                theta = self.data.theta[i, j].data
                if self.data.pdf[i, j] == 'normal':
                    pdf = rf'$\mathcal{{N}}\,(0, {theta})$'
                elif self.data.pdf[i, j] == 'uniform':
                    pdf = rf'$\mathcal{{U}}\,(\pm{theta})$'
                text += ': ' + pdf
                text += '\n'
        text += '\n'
        text += f'samples: {self.samples:,}'
        return text

    def plot(self, offset=True):
        """Plot peak to peak distribution."""
        plt.figure()
        plt.hist(self.data.peaktopeak, bins=51, density=True,
                 rwidth=0.8, label='machine axis', color='C1')
        if offset:
            plt.hist(self.data.peaktopeak_offset, bins=51,
                     density=True, rwidth=0.8, alpha=0.85, color='C2',
                     label='magnetic axis')
            plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05),
                       ncol=2, fontsize='small')
            self.label_quartile(self.data.peaktopeak_offset, 'H', color='C2',
                                height=0.15)

        self.label_quartile(self.data.peaktopeak, 'H', color='C1',
                            height=0.04)
        plt.despine()
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r'peak to peak deviation $H$, mm')
        plt.ylabel(r'$P(H)$')
        plt.text(0.95, 0.95, self.pdf_text, fontsize='small',
                 transform=axes.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='w', boxstyle='round, pad=0.5',
                           linewidth=0.5))

    def label_quartile(self, data, label: str, quartile=0.99, height=0.1,
                       color='gray'):
        """Label quartile."""
        ylim = plt.gca().get_ylim()
        yline = ylim[0] + np.array([0, height*(ylim[1] - ylim[0])])
        quartile = np.quantile(data, quartile)
        plt.plot(quartile*np.ones(2), yline, '-', color='k', alpha=0.75)
        text = rf'q(0.99): ${label}={quartile:1.1f}$'
        plt.text(quartile, yline[1], text,
                 ha='left', va='bottom', fontsize='small', color=color,
                 bbox=dict(facecolor='w', edgecolor=color))

    def plot_pdf(self, bins=51):
        """Plot pdf."""
        pdf, edges = np.histogram(self.data.peaktopeak, bins,
                                  density=True)
        plt.plot((edges[:-1] + edges[1:]) / 2, pdf)

    def plot_offset(self):
        """Plot pdf of field line axis offset."""
        offset = np.linalg.norm(self.data.offset, axis=-1)
        plt.figure()
        plt.hist(offset, bins=51, density=True, rwidth=0.8)
        plt.despine()
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r'magnetic axis offset $\zeta$, mm')
        plt.ylabel(r'$P(\zeta)$')

        self.label_quartile(offset, r'\zeta')
        plt.text(0.95, 0.95, self.pdf_text, fontsize='small',
                 transform=axes.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='w', boxstyle='round, pad=0.5',
                           linewidth=0.5))

    def plot_sample(self, sample=0):
        """Plot waveforms from single sample."""
        axes = plt.subplots(2, 1, sharex=False, sharey=False,
                            gridspec_kw=dict(height_ratios=[1, 1]))[1]
        width = 0.8
        axes[1].bar(self.data.index, self.data.gap[sample].sum(axis=-1),
                    width=width, color='gray', alpha=0.5)

        signal_width = width / self.data.dims['signal']
        offset = 0.5*signal_width - width / 2
        for i, signal in enumerate(self.data.signal):

            axes[i].bar(self.data.index + offset,
                        self.data.case[sample][:, i], width=signal_width)
            axes[i].bar(self.data.index + offset + signal_width,
                        self.data.structural[sample][:, i], width=signal_width)

        '''
        self.electromagnetic_model.predict(
            self.data.electromagnetic[sample][:, 0],
            self.data.electromagnetic[sample][:, 1])

        fieldline = self.electromagnetic_model.fieldline[0]
        print(fieldline)
        axes[-1].plot(fieldline.phi, fieldline)
        '''

        '''
                for i, attr in enumerate(['radial', 'tangential']):
                    waveform = getattr(self, attr)
                    print(waveform.shape)
                    if waveform is not None:
                        axes[0].bar(range(len(waveform)), waveform, width=0.8 - i*0.3,
                                    color=f'C{1+i}', label=attr)
                axes[0].set_xticks([])
                axes[1].plot(self.fieldline.phi, self.fieldline[sample])
                if modes is not None:
                    _fieldline = np.fft.irfft(
                        np.fft.rfft(self.fieldline[sample])[:modes+1],
                        self.fieldline.shape[1])
                    axes[1].plot(self.fieldline.phi, _fieldline, '-.', color='gray',
                                 label=f'<=n{modes}')
                    axes[1].legend(fontsize='x-small')

                plt.despine()
                axes[0].legend(fontsize='x-small', ncol=2)
                axes[0].set_ylabel(r'vault')
                axes[1].set_ylabel('field line deviation')
                axes[1].set_xlabel(r'$\phi$')
        '''



if __name__ == '__main__':

    trial = Trial(theta=[1.5, 1.5, 2, 2, 3, 0],
                  samples=1_00_000, wall=True, energize=True)
    trial.plot()

    '''
    quartile = np.quantile(trial.data.peaktopeak_offset, 0.99)
    sample = np.argmin((trial.data.peaktopeak_offset.data-quartile)**2)
    print(sample)
    print(trial.data.peaktopeak_offset[sample])
    print(trial.data.peaktopeak[sample])
    trial.plot_sample(sample)
    '''
    trial.plot_offset()
