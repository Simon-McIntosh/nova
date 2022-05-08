"""Perform post-processing analysis on Fourier perterbed TFC dataset."""
from dataclasses import dataclass, field
from typing import ClassVar

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import xarray

from nova.assembly.gap import Gap
from nova.assembly.model import ModelData, ModelBase
from nova.assembly.ccl import CCL
from nova.utilities.pyplot import plt


@dataclass
class Points(ModelData):
    """Extract midplane mesh and initalize dataset."""

    folder: str = 'TFCgapsG10'
    origin: tuple[int] = (0, 0, 0)
    ncoil: ClassVar[int] = 18
    cluster: ClassVar[int] = 1

    def slice_mesh(self):
        """Return midplane mesh."""
        mesh = CCL(self.folder, self.name, cluster=self.cluster).mesh
        return mesh.slice('z', self.origin)

    @property
    def attrs(self):
        """Return instance attributes."""
        return {attr: getattr(self, attr)
                for attr in ['filename', 'folder',
                             'origin', 'ncoil', 'cluster']}

    def initialize_dataset(self):
        """Initialize empty dataset from referance scenario."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data['scenario'] = self.mesh['scenario']
        self.data['index'] = range(1, self.ncoil+1)
        self.data['mesh'] = ['x', 'y', 'z']
        self.data['points'] = xarray.DataArray(0., self.data.coords)

    def build(self):
        """Build single mesh dataset."""
        self.mesh = self.slice_mesh()
        self.initialize_dataset()
        for scenario in self.data.scenario.data:  # store ansys data [mm]
            self.data['points'].loc[scenario][:, :3] = 1e3*self.mesh[scenario]


@dataclass
class Fourier(Points):
    """Perform Fourier deomposition on single vault simulation."""

    def initialize_dataset(self):
        """Extend points initialize dataset."""
        super().initialize_dataset()
        self.data['response'] = ['radial', 'tangential']
        self.data['delta'] = ('scenario', 'index', 'response'), \
            np.ones(tuple(self.data.dims[dim]
                          for dim in ['scenario', 'index', 'response']))

    def build(self):
        """Extend points build."""
        super().build()
        reference = Points('k0').data.sel(scenario=self.data.scenario)
        point_delta = self.data['points'] - reference['points']
        r_norm = reference['points'][..., :2].data.copy()
        r_norm /= np.linalg.norm(r_norm, axis=-1)[..., np.newaxis]
        t_norm = r_norm @ np.array([[0, -1], [1, 0]])

        self.data['delta'][..., -2] = np.einsum(
            'ijk,ijk->ij', point_delta[..., :2].data, r_norm)
        self.data['delta'][..., -1] = np.einsum(
            'ijk,ijk->ij', point_delta[..., :2].data, t_norm)
        self.fft(self.data)
        return self.store()

    def plot(self, coefficient='amplitude', scenario='TFonly'):
        """Plot fourier components."""
        plt.figure()
        axes = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        for i, response in enumerate(['radial', 'tangential']):
            axes[i].bar(self.data.mode,
                        self.data.fft.sel(scenario=scenario,
                                          response=response,
                                          coefficient=coefficient))
            axes[i].set_ylabel(f'{coefficient}')
        plt.despine()
        axes[-1].set_xlabel('Fourier mode number')


@dataclass
class BaseTransform:
    """Extract factor and phase transform from single point simulations."""

    signal_fft: xarray.DataArray
    response_fft: xarray.DataArray
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    eps: ClassVar[float] = 1e-5

    def __post_init__(self):
        """Build filter."""
        self.data['signal_fft'] = self.signal_fft
        self.data['response_fft'] = self.response_fft
        self.build()

    def build(self):
        """Build signal filter."""
        _filter = self.response_coef / self.signal_coef
        _filter_dims = _filter.dims + ('coefficient',)
        self.data['filter'] = _filter_dims, \
            np.zeros(tuple(self.data.dims[dim] for dim in _filter_dims))
        self.data.filter[..., 0] = _filter.real
        self.data.filter[..., 1] = _filter.imag
        self.data.filter[..., 2] = self.data.response_fft[..., 2] /\
            self.data.signal_fft[..., 2]
        self.data.filter[..., 3] = self.data.response_fft[..., 3] -\
            self.data.signal_fft[..., 3]
        # remove filter coefficients with low signal amplitudes
        signal_amplitude = self.data.signal_fft.sel(coefficient='amplitude')
        max_amplitude = signal_amplitude.data.max(axis=0)
        index = (signal_amplitude < self.eps*max_amplitude) | \
            (np.isclose(signal_amplitude, 0))
        index = index.data
        if index.ndim == 1:
            self.data.filter.data[index] = 0
            return
        for i in range(index.shape[1]):
            self.data.filter.data[index[:, i], :, i] = 0

    @property
    def signal_coef(self):
        """Return complex signal coefficents."""
        return self.data.signal_fft[..., 0] + 1j * self.data.signal_fft[..., 1]

    @property
    def response_coef(self):
        """Return complex response coefficents."""
        return self.data.response_fft[..., 0] + \
            1j * self.data.response_fft[..., 1]


@dataclass
class Transform:
    """Build transform dataset."""

    simulation: str
    signal: str = None
    scenario: str = 'TFonly'
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    attrs: ClassVar[list[str]] = ['delta', 'fft']

    def __post_init__(self):
        """Load datasets and compute transform."""
        self.build_signal()
        self.build_response()
        self.data['filter'] = BaseTransform(
            self.data.signal_fft, self.data.response_fft).data.filter

    def build_signal(self):
        """Build transform signal."""
        signal_data = Gap(self.simulation).data
        if self.signal is not None:
            signal_data = signal_data.sel(signal=self.signal)
        for signal in np.atleast_1d(signal_data.signal.values):
            self.data[signal] = signal_data[signal]
        for attr in self.attrs:
            self.data[f'signal_{attr}'] = signal_data[attr]
        self.data.attrs |= signal_data.attrs

    def build_response(self):
        """Build transform response."""
        response = Fourier(self.simulation).data.sel(scenario=self.scenario)
        for attr in self.attrs:
            self.data[f'response_{attr}'] = response[attr]


@dataclass
class Model(ModelBase):
    """Construct structural model."""

    name: str = 'structural'

    def _filter(self, label: str):
        """Return complex filter."""
        return self.data.filter[..., 0].sel(response=label) + \
            1j * self.data.filter[..., 1].sel(response=label)

    def build(self):
        """Build fourier component model."""
        self.data = xarray.Dataset()
        self.data['filter'] = xarray.zeros_like(Transform('k0').data.filter)
        self.data = self.data.drop('simulation')
        self.build_gap_filter()
        self.build_yaw_filter()
        self.build_roll_filter()
        return self.store()

    def build_gap_filter(self):
        """Build gap filter."""
        for mode in self.data.mode.values[1:]:
            self.data.filter[mode, :, 0] = \
                Transform(f'k{mode}', 'gap').data.filter[mode]
        self.load_filter()

    def build_yaw_filter(self, simulation='w3'):
        """Build yaw filter."""
        transform = Transform(simulation).data
        data = transform.response_delta.to_dataset()
        data = data.rename(dict(response_delta='delta'))
        data.delta[:, 0] -= self.predict_radial(transform.gap)
        data.delta[:, 1] -= self.predict_tangential(transform.gap)
        self.fft(data)
        self.data.filter[..., 2, :] = BaseTransform(
            transform.signal_fft.sel(signal='yaw'), data.fft).data.filter
        self.load_filter()

    def build_roll_filter(self, simulation='w4'):
        """Build roll filter."""
        transform = Transform(simulation).data
        data = transform.response_delta.to_dataset()
        data = data.rename(dict(response_delta='delta'))
        data.delta[:, 0] -= self.predict_radial(
            transform.gap, yaw=transform.yaw)
        data.delta[:, 1] -= self.predict_tangential(
            transform.gap, yaw=transform.yaw)
        self.fft(data)
        self.data.filter[..., 1, :] = BaseTransform(
            transform.signal_fft.sel(signal='roll'), data.fft).data.filter
        self.load_filter()

    def predict_radial(self, gap=None, roll=None, yaw=None):
        """Return radial response."""
        return self.predict('radial', gap, roll, yaw)

    def predict_tangential(self, gap=None, roll=None, yaw=None):
        """Return tangential response."""
        return self.predict('tangential', gap, roll, yaw)

    def predict(self, response, *signals):
        """Return prediction of zero-gap (operational) response waveform."""
        waveform = 0
        for i, signal in enumerate(signals):  # gap, roll, yaw
            if signal is None:
                continue
            waveform += np.fft.irfft(
                np.fft.rfft(signal) * self.filter[response][:, i])
        return waveform

    def plot_benchmark(self, simulation: str):
        """Plot structural model results."""
        transform = Transform(simulation)
        transform.data.response_delta[:] -= \
            transform.data.response_delta.mean(axis=0)
        axes = plt.subplots(3, 1, sharex=True, sharey=False,
                            gridspec_kw=dict(height_ratios=[2, 2, 2]))[1]
        width = 0.9/transform.data.dims['signal']
        color = ['C0', 'C6', 'C7']
        for i, signal in enumerate(transform.data.signal.values):
            offset = width * (i - transform.data.dims['signal']/2)
            axes[0].bar(transform.data.index + offset,
                        transform.data[signal], width=width, color=color[i])

        axes[0].set_ylabel('signal')
        axes[0].legend(transform.data.signal.values,
                       fontsize='xx-small', ncol=3)
        axes[1].bar(transform.data.index, transform.data.response_delta[:, 0],
                    width=0.85, color='C1', label='ANSYS')
        model_radial = self.predict_radial(*transform.data.signal_delta.data.T)
        axes[1].bar(transform.data.index, model_radial,
                    width=0.45, color='C2', label='Vault proxy')
        axes[1].set_ylabel(r'$\Delta r$')
        axes[0].set_title(f'Structural benchmark: {simulation}')

        axes[2].bar(transform.data.index, transform.data.response_delta[:, 1],
                    width=0.85, color='C1', label='ANSYS')
        model_tangential = self.predict_tangential(
            *transform.data.signal_delta.data.T)
        axes[2].bar(transform.data.index, model_tangential,
                    width=0.45, color='C2', label='Vault proxy')
        axes[2].set_ylabel(r'$r\Delta \phi$')

        axes[-1].set_xlabel('coil index')
        axes[-1].xaxis.set_major_locator(MultipleLocator(2))
        axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes[-1].xaxis.set_minor_locator(MultipleLocator(1))
        plt.despine()
        axes[-1].legend(fontsize='xx-small', ncol=2)

    def plot_response(self):
        """Plot radial and tangential."""
        axes = plt.subplots(1, 1, sharex=True)[1]
        axes.bar(self.data.mode,
                 self.data.filter[..., 2].sel(response='radial'),
                 label='radial')
        axes.bar(self.data.mode,
                 self.data.filter[..., 2].sel(response='tangential'),
                 width=0.5, label='tangential')
        axes.xaxis.set_major_locator(MultipleLocator(1))
        axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        plt.despine()
        plt.title('Structural response')
        plt.xlabel(r'wavenumber $k$')
        plt.ylabel('gap amplification factor')
        plt.legend()


if __name__ == '__main__':

    structural = Model()
    structural.plot_benchmark('w5')

    #structural.plot_response()
    #Gap().plot('k3')
