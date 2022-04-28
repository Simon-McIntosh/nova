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

    samples: int = 200_000
    theta: list[float] = field(default_factory=lambda: [1.5, 1.5, 2, 2, 5, 0])
    pdf: list[str] = field(
        default_factory=lambda: ['uniform', 'uniform', 'normal', 'normal',
                                 'uniform', 'uniform'])
    energize: Union[int, bool] = True
    sead: int = 2025

    component: ClassVar[list[str]] = ['case', 'ccl', 'wall']
    signal: ClassVar[list[str]] = ['radial', 'tangential']
    ncoil: ClassVar[int] = 18
    ripple: ClassVar[float] = 0.11
    peaktopeak_max: ClassVar[float] = 4

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

    @property
    def group_name(self):
        """Return group name as xxh32 hex hash."""
        self.xxh32.reset()
        self.xxh32.update(np.array(list(self.attrs.values()) +
                                   self.theta + self.pdf))
        return self.xxh32.hexdigest()


@dataclass
class Trial(Dataset, TrialAttrs):
    """Run stastistical analysis on trial vault assemblies."""

    filename: str = 'vault_trial'
    xxh32: xxhash.xxh32 = field(repr=False, init=False,
                                default_factory=xxhash.xxh32)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        self.energize = int(self.energize)
        self.group = self.group_name
        self.rng = np.random.default_rng(self.sead)
        super().__post_init__()

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
            self.predict_blanket()
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
        self.data['gap'] = np.pi / self.ncoil * self.data['case'][..., 0]
        self.data.gap[:, :-1] += \
            np.pi / self.ncoil * self.data['case'][:, 1:, 0].data
        self.data.gap[:, -1] += \
            np.pi / self.ncoil * self.data['case'][:, 0, 0].data
        self.data['gap'] -= self.data['case'][..., 1]
        self.data.gap[:, :-1] += self.data['case'][:, 1:, 1].data
        self.data.gap[:, -1] += self.data['case'][:, 0, 1].data

    def predict_structure(self):
        """Run structural simulation."""
        self.data['structural'] = ('sample', 'index', 'signal'), \
            np.zeros((self.samples, self.ncoil, self.data.dims['signal']))
        if self.energize:
            model = structural.Model()
            for i, signal in enumerate(self.data.signal.values):
                self.data['structural'][..., i] = \
                    model.predict(self.data.gap, signal)

    def predict_electromagnetic(self):
        """Run electromagnetic simulation."""
        self.data['electromagnetic'] = self.data.structural.copy(deep=True)
        self.data.electromagnetic[..., 0] += self.data.case[..., 0]
        self.data.electromagnetic[..., 0] += self.data.ccl[..., 0]
        self.data.electromagnetic[..., 1] += self.data.ccl[..., 1]
        model = electromagnetic.Model()
        self.data['peaktopeak'] = 'sample', model.peaktopeak(
            self.data.electromagnetic[..., 0],
            self.data.electromagnetic[..., 1])
        self.data['offset'] = ('sample', 'coordinate'), \
            np.zeros((self.data.dims['sample'], 2))
        offset = model.offset(self.data.electromagnetic[..., 0],
                              self.data.electromagnetic[..., 1])
        self.data['offset'][..., 0] = offset.real
        self.data['offset'][..., 1] = -offset.imag
        self.data['peaktopeak_offset'] = 'sample', model.peaktopeak(
            self.data.electromagnetic[..., 0],
            self.data.electromagnetic[..., 1], offset=True)

    def predict_blanket(self, ndiv=360):
        """Predict combined wall-fieldline deviations."""
        model = electromagnetic.Model()
        fieldline = model.predict(
            self.data.electromagnetic[..., 0],
            self.data.electromagnetic[..., 1], ndiv)
        self.data['peaktopeak_blanket_nominal'] = 'sample', \
            self._predict_blanket(fieldline, False, ndiv)
        self.data['peaktopeak_blanket_offset'] = 'sample', \
            self._predict_blanket(fieldline, True, ndiv)

    def _predict_blanket(self, fieldline, offset: bool, ndiv: int):
        """Run blanket deviation simulation."""
        delta_hat = np.fft.rfft(self.data.wall, axis=1)[..., :2, 0]
        if offset:
            delta_hat[..., 0] = 0
            delta_hat[..., 1] += self.data['offset'][..., 0].data + \
                self.data['offset'][..., 0].data * 1j
        firstwall = np.fft.irfft(delta_hat, n=ndiv) * ndiv / self.ncoil
        deviation = fieldline - firstwall
        return np.max(deviation, axis=-1) - np.min(deviation, axis=-1)

    @property
    def pdf_text(self):
        """Return pdf text label."""
        text = ''
        for i, component in enumerate(self.component):
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

    def plot(self, label='quartile'):
        """Plot peak to peak distribution."""
        plt.figure()
        plt.hist(self.data.peaktopeak + self.ripple, bins=51, density=True,
                 rwidth=0.8, label='as-built')
        plt.hist(self.data.peaktopeak_offset + self.ripple, bins=51,
                 density=True, rwidth=0.8, alpha=0.5, color='gray',
                 label='n1 offset')
        plt.despine()
        plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05),
                   ncol=2, fontsize='small')
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r'peak to peak deviation $H$, mm')
        plt.ylabel(r'$P(H)$')

        if label is not None:
            ylim = axes.get_ylim()
            yline = ylim[0] + np.array([0, 0.15*(ylim[1] - ylim[0])])
        if label == 'interval':
            interval = scipy.stats.percentileofscore(
                self.data.peaktopeak + self.ripple, self.peaktopeak_max)
            plt.plot(self.peaktopeak_max*np.ones(2), yline, '-', color='gray')
            interval_text = rf'PI$(H<{self.peaktopeak_max})={interval:1.0f}\%$'
            plt.text(self.peaktopeak_max, yline[1], interval_text,
                     ha='left', va='bottom', fontsize='small', color='gray')
        elif label == 'quartile':
            self.label_quartile(self.data.peaktopeak + self.ripple, 'H')

        plt.text(0.95, 0.95, self.pdf_text, fontsize='small',
                 transform=axes.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='w', boxstyle='round, pad=0.5',
                           linewidth=0.5))

    def plot_blanket(self, label='quartile'):
        """Plot peak to peak distribution."""
        plt.figure()
        plt.hist(self.data.peaktopeak_blanket_nominal + self.ripple, bins=51,
                 density=True, rwidth=0.8, label='nominal')
        plt.hist(self.data.peaktopeak_blanket_offset + self.ripple, bins=51,
                 density=True, rwidth=0.8, alpha=0.5, color='gray',
                 label='n1 offset')
        plt.despine()
        plt.legend(loc='center', bbox_to_anchor=(0.5, 1.05),
                   ncol=2, fontsize='small')
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r'peak to peak deviation $H$, mm')
        plt.ylabel(r'$P(H)$')

        self.label_quartile(self.data.peaktopeak_blanket_offset
                            + self.ripple, 'H')

        plt.text(0.95, 0.95, self.pdf_text, fontsize='small',
                 transform=axes.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='w', boxstyle='round, pad=0.5',
                           linewidth=0.5))

    def label_quartile(self, data, label: str, quartile=0.99):
        """Label quartile."""
        ylim = plt.gca().get_ylim()
        yline = ylim[0] + np.array([0, 0.15*(ylim[1] - ylim[0])])
        quartile = np.quantile(data, quartile)
        plt.plot(quartile*np.ones(2), yline, '-', color='gray')
        interval_text = rf'q(0.99): ${label}={quartile:1.1f}$'
        plt.text(quartile, yline[1], interval_text,
                 ha='left', va='bottom', fontsize='small', color='gray')

    def plot_pdf(self, bins=51):
        """Plot pdf."""
        pdf, edges = np.histogram(self.data.peaktopeak, bins,
                                  density=True)
        plt.plot((edges[:-1] + edges[1:]) / 2, pdf)

    def plot_offset(self):
        """Plot pdf of field line axis offset."""
        plt.figure()
        plt.hist(self.data.offset, bins=51, density=True, rwidth=0.8)
        plt.despine()
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r'magnetic axis offset $\zeta$, mm')
        plt.ylabel(r'$P(\zeta)$')

        self.label_quartile(self.data.offset, r'\zeta')
        plt.text(0.95, 0.95, self.pdf_text, fontsize='small',
                 transform=axes.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='w', boxstyle='round, pad=0.5',
                           linewidth=0.5))


if __name__ == '__main__':

    trial = Trial(theta=[1.5, 1.5, 2, 2, 4, 0]).build()


    trial.plot_blanket()
    #trial.plot_offset()
