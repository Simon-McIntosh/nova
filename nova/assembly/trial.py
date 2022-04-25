"""Run Monte Carlo simulations for candidate vault assemblies."""
from dataclasses import dataclass, field, fields
from functools import cached_property
from time import time
from typing import ClassVar

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

    samples: int = 500_000
    sead: int = 2025
    case_radial: tuple[str, float] = ('uniform', 2)
    case_tangential: tuple[str, float] = ('uniform', 0)
    ccl_radial: tuple[str, float] = ('normal', 0)
    ccl_tangential: tuple[str, float] = ('normal', 0)

    ncoil: ClassVar[int] = 18

    @cached_property
    def trial_attrs(self):
        """Return list of trial attributes."""
        return [attr.name for attr in fields(TrialAttrs)]

    @cached_property
    def tuple_attrs(self):
        """Return list of tuple attributes (distributions)."""
        return [attr for attr in self.trial_attrs if
                isinstance(getattr(self, attr), tuple)]

    @property
    def attrs(self):
        """Return trial attrs."""
        attrs = {}
        for attr in self.trial_attrs:
            value = getattr(self, attr)
            if attr in self.tuple_attrs:
                distribution = value[0]
                for parameter in value[1:]:
                    distribution += f'_{parameter}'
                value = distribution
            attrs[attr] = value
        return attrs

    @property
    def group_name(self):
        """Return group name as xxh32 hex hash."""
        self.xxh32.reset()
        self.xxh32.update(np.array(list(self.attrs.values())))
        return self.xxh32.hexdigest()


@dataclass
class Trial(Dataset, TrialAttrs):
    """Run stastistical analysis on trial vault assemblies."""

    filename: str = 'vault_trial'
    xxh32: xxhash.xxh32 = field(repr=False, init=False,
                                default_factory=xxhash.xxh32)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        self.group = self.group_name
        self.rng = np.random.default_rng(self.sead)
        super().__post_init__()

    def normal(self, variance: float):
        """Return sample with normal distribution.

        Parameters
        ----------
        width: float
            Distribution width, (2 sigma)
        """
        scale = np.sqrt(variance)
        return self.rng.normal(scale=scale, size=(self.samples, self.ncoil))

    def uniform(self, bound: float):
        """Return sample with uniform distribution."""
        return self.rng.uniform(-bound, bound, size=(self.samples, self.ncoil))

    def build(self):
        """Build Monte Carlo dataset."""
        start = time()
        self.build_signal()
        self.build_gap()
        self.predict_structure()
        self.predict_electromagnetic()
        print(f'build time {time() - start:1.0f}s')
        return self.store()

    def build_signal(self):
        """Build input distributions."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data['sample'] = range(self.samples)
        self.data['index'] = range(self.ncoil)
        self.data['signal'] = ['radial', 'tangential']
        self.data['case'] = xarray.DataArray(0., self.data.coords)
        self.data['ccl'] = xarray.DataArray(0., self.data.coords)
        for attr in self.tuple_attrs:
            name = attr.split('_')[0]
            signal = attr.split('_')[1]
            value = getattr(self, attr)
            distribution = value[0]
            parameters = value[1:]
            self.data[name].loc[..., signal] = \
                getattr(self, distribution)(*parameters)

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

    def plot(self):
        """Plot peak to peak distribution."""
        plt.figure()
        plt.hist(self.data.peaktopeak, bins=51, density=True,
                 rwidth=0.8)
        plt.despine()
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r'peak to peak deviation $H$, mm')
        plt.ylabel(r'$P(H)$')

        self.plot_pdf()

        plt.text(0.65, 0.85,
                 r'$\Delta r_{ccl} = \mathcal{N}\,(0, 2)$'
                 '\n'
                 r'$\Delta r_{case} = \mathcal{U}\,(\pm 2)$',
                 transform=axes.transAxes)

    def plot_pdf(self, bins=51):
        """Plot pdf."""
        pdf, edges = np.histogram(self.data.peaktopeak, bins,
                                  density=True)
        plt.plot((edges[:-1] + edges[1:]) / 2, pdf)


if __name__ == '__main__':

    trial = Trial()
    trial.plot()
