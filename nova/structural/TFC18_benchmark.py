"""Manage TFC18 benchmarks."""
from collections import Iterable
from dataclasses import dataclass, field
from typing import Union
import os

import matplotlib.patches as mpatches
import numpy as np
import pandas
import pyvista as pv
import scipy
import scipy.fft
import seaborn as sns

from nova.definitions import root_dir
from nova.structural.F4E_TFC18 import F4E_Data
from nova.structural.TFC18 import TFC18
from nova.utilities.pyplot import plt
from nova.utilities.addtext import linelabel


@dataclass
class BenchBase:
    """Benchmark baseclass."""

    mesh_a: pv.PolyData
    mesh_b: pv.PolyData
    case_names: list[str, str]
    scenario: Union[str, list[str, str]]
    mesh: pv.PolyData = field(init=False)

    def __post_init__(self):
        """Load dataset."""
        if isinstance(self.scenario, str):
            self.scenario = 2*[self.scenario]
        self.build()

    def build(self):
        """Build database."""
        self.mesh = self.mesh_a.copy()
        self.mesh.clear_data()
        self.mesh.field_data['case_names'] = self.case_names
        self.mesh['coil_names'] = [f'{index+1:02}' for index in range(18)]
        self.mesh['arc_length'] = self.mesh_a['arc_length']
        self.mesh['displace_a'] = self.mesh_a[self.scenario[0]]
        self.mesh['displace_b'] = self.mesh_b[self.scenario[1]]
        self.mesh['delta'] = self.mesh['displace_b'] - self.mesh['displace_a']


@dataclass
class ErrorProfile:
    """TFC18 structural error shapes."""

    name: str
    scenario: str
    key: dict = field(default_factory=lambda:
                      dict(scod='pcr100', SCOD='pcr100', F4E='f4e',
                           MAG='mag'))
    mesh: pv.PolyData = field(init=False)
    keypoints: dict = field(init=False)

    def __post_init__(self):
        """Load dataset."""
        self.load()
        self.keypoints = self.extract_cetnerline_keypoints()

    def load(self):
        """Load dataset."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            getattr(self, f'_{self.dataset}')()
            self.mesh.save(self.vtk_file)

    @property
    def dataset(self):
        """Return dataset name."""
        return self.key.get(self.name, self.name)

    def extract_cetnerline_keypoints(self):
        """Construct TFC centerline keypoint lookup."""
        coil = self.extract_coil(0, clip_x=True)
        interp_z = scipy.interpolate.interp1d(coil.points[:, 2],
                                              coil['arc_length'])
        return dict(ILIS_top=interp_z(4),
                    coil_top=coil['arc_length'][0],
                    HFS_midplane=interp_z(0),
                    LFS_midplane=0,
                    ILIS_base=interp_z(-4),
                    coil_base=coil['arc_length'][-1])

    @property
    def vtk_file(self):
        """Return vtk filepath."""
        return os.path.join(root_dir, 'data/Assembly/TFC18',
                            self.dataset + f'_{self.scenario}.vtk')

    def _f4e(self):
        """Build F4E dataset."""
        f4e = F4E_Data().mesh
        if self.scenario != 'EOB':
            raise IndexError(f'{self.scenario} not present '
                             'in F4E dataset (only EOB)')
        self.mesh = BenchBase(f4e, f4e, ['v0', 'v3'], ['v0', 'v3']).mesh

    def _pcr65(self):
        """Build PCR 100% reference case."""
        v0 = TFC18('TFCgapsG10', 'v0_65_f4e', cluster=1).mesh
        v3 = TFC18('TFCgapsG10', 'v3_65_f4e', cluster=1).mesh
        self.mesh = BenchBase(v0, v3, ['v0', 'v3'], self.scenario).mesh

    def _pcr100(self):
        """Build PCR 100% reference case."""
        v0 = TFC18('TFCgapsG10', 'v0_100_f4e', cluster=1).mesh
        v3 = TFC18('TFCgapsG10', 'v3_100_f4e', cluster=1).mesh
        self.mesh = BenchBase(v0, v3, ['v0', 'v3'], self.scenario).mesh

    def _mag(self):
        """Build MAG reference case."""
        v0 = TFC18('TFCgapsG10', 'v0', cluster=1).mesh
        v3 = TFC18('TFCgapsG10', 'v3', cluster=1).mesh
        self.mesh = BenchBase(v0, v3, ['v0', 'v3'], self.scenario).mesh

    def warp(self, factor=100, coil=None):
        """Plot benchmark cases."""
        plotter = pv.Plotter()
        for i, case in enumerate('ab'):
            if coil is None:
                mesh = self.mesh
            else:
                mesh = pv.PolyData(self.extract_coil(coil, False))
            warp = mesh.warp_by_vector(f'displace_{case}', factor=factor)
            plotter.add_mesh(warp)
        plotter.show()

    def extract_coil(self, index: int, clip_x=True):
        """Return single coil rotated to x-z reference plane."""
        cell = self.mesh.extract_cells(index)
        cell.rotate_z(-360*index / 18, transform_all_input_vectors=True)
        if clip_x:
            index = cell.points[:, 2].argmax()
            cell = cell.clip('x', (cell.points[index, 0], 0, 0))
        return cell

    def plot(self, coordinate=1, clip_x=True, axes=None):
        """Plot ensemble."""
        if axes is None:
            axes = plt.gca()
        colors = sns.color_palette('muted', n_colors=18)
        label = linelabel(ax=axes, value='', Ndiv=20, loc='start')
        for i in range(18):
            coil = self.extract_coil(i, clip_x)
            self.plot_coil(coil, coordinate=coordinate,
                           axes=axes, label=label, color=colors[i])
        label.plot()

    def plot_coil(self, coil, coordinate=0, axes=None,
                  delta=True, label=None, color=None):
        """Generate 2d plots."""
        if axes is None:
            axes = plt.gca()

        if delta:
            axes.plot(coil['abscissa'], 1e3*coil['delta'][:, coordinate],
                      color=color)
            if label:
                label.add(coil['coil_names'][0])
        else:
            axes.plot(coil['arc_length'],
                      1e3 * coil['displace_a'][:, coordinate],
                      color=color)
            if label:
                label.add(coil['coil_names'][0])
            axes.plot(coil['arc_length'],
                      1e3 * coil['displace_b'][:, coordinate],
                      color=color, ls='--')
        plt.despine()

    def clip_index(self, clip, factor=1.001):
        """Return cliping index."""
        if clip:
            coil = self.extract_coil(0)
            return coil.points[:, 0] <= factor * np.min(coil.points[:, 0])
        return slice(None)

    def shade(self, axes=None):
        """Shade inboard."""
        if axes is None:
            axes = plt.gca()
        coil = self.extract_coil(0)
        index = self.clip_index()
        n_shade = sum(index)
        ylim = axes.get_ylim()
        axes.fill_between(coil['arc_length'][index],
                          ylim[1]*np.ones(n_shade), ylim[0]*np.ones(n_shade),
                          color='gray', alpha=0.2, lw=0)

    @property
    def rms(self):
        """Calculate rms error."""
        rms = np.zeros((18, 1))
        for i in range(18):
            coil = self.extract_coil(i)
            rms[i] = np.sqrt(np.mean(
                np.linalg.norm(coil['delta'][:, :2], axis=1)**2))
        return 1e3*np.sqrt(np.mean(rms**2))

    def bar(self, keypoint: str, coordinate: int,
            width=0.8, axes=None, label=None, annotate=True, text='keypoint'):
        """Generate bar plot."""
        if axes is None:
            axes = plt.gca()
        length = self.keypoints[keypoint]
        delta = self.extract_coilset_delta(length)
        coord_label = ['r', r'\phi', 'z']
        if label == 'mode':
            err = self.error_modes(1e3*delta[:, coordinate])
            label = f'{err[1]:1.1f}, {err[2]:1.1f}'
        axes.bar(range(18), 1e3*delta[:, coordinate], width, label=label)
        if annotate:
            axes.set_xlabel('coil index')
            axes.set_ylabel(rf'$\Delta\,{coord_label[coordinate]}$ mm')
            axes.set_xticks(range(1, 18, 2))
            axes.set_xticklabels(range(2, 19, 2))
        if text:
            if text == 'keypoint':
                text = keypoint.replace('_', ' ')
            axes.text(0.95, 0.95, text,
                      transform=axes.transAxes,
                      ha='right', va='top')
        plt.despine()

    def extract_coilset_delta(self, arc_length: float):
        """Return set of coilset deltas from single arc_length."""
        delta = np.zeros((18, 3))
        for i in range(18):
            coil = self.extract_coil(i, clip_x=False)
            delta[i] = scipy.interpolate.interp1d(
                coil['arc_length'], coil['delta'], axis=0)(arc_length)
        return delta

    def error_modes(self, values, ncoil=18):
        """Return amplitudes of Fourier error modes."""
        fft_coefficents = scipy.fft.rfft(values)
        error_modes = np.zeros(1 + ncoil//2)
        error_modes[0] = fft_coefficents[0].real / ncoil
        error_modes[1:] = abs(fft_coefficents[1:]) / (ncoil//2)
        return error_modes

    def plot_wave(self, axes, values, wavenumber=1, ncoil=18):
        """Plot fft modes."""
        coef = np.zeros(ncoil//2, dtype=complex)
        fft_coefficents = scipy.fft.rfft(values)
        error_modes = self.error_modes(values)

        coef[0] = fft_coefficents[0]
        if not isinstance(wavenumber, Iterable):
            wavenumber = [wavenumber]
        label = ''
        for wn in wavenumber:
            coef[wn] = fft_coefficents[wn]
            if label:
                label += '\n'
            label += f'$k_{wn}$='
            label += f'{error_modes[wn]:1.2f}'

        nifft = ncoil
        ifft = scipy.fft.irfft(coef, n=nifft, norm='backward').real
        axes.bar(range(ncoil), values, color='lightgray')
        axes.plot(range(ncoil), values, '.-', color='C7')
        axes.plot(np.linspace(0, ncoil-1, nifft), ifft, '-', color='C6')
        axes.text(ncoil-0.5, ifft[-1], label, va='center',
                  color='C6', fontsize='small')

    def plot_keypoints(self, labels=None):
        """Plot location of refernace points."""
        if labels is None:
            labels = [key.replace('_', ' ') for key in self.keypoints]
        coil = self.extract_coil(0, False)
        points = scipy.interpolate.interp1d(coil['arc_length'],
                                            coil.points, axis=0)
        plt.plot(coil.points[:, 0], coil.points[:, 2], color='gray', lw=5)
        for i, keypoint in enumerate(self.keypoints):
            point = points(self.keypoints[keypoint])
            plt.plot(point[0], point[2], 'C0o', ms=12)
            plt.text(point[0], point[2], f' {labels[i]}',
                     va='bottom', ha='left')
        plt.axis('equal')
        plt.axis('off')

    def error_frame(self):
        """Return structural error modes dataframe."""
        index = pandas.MultiIndex.from_product(
            [['r', 'phi', 'z'], list('abcdef')], names=['coord', 'point'])
        error = pandas.DataFrame(index=index,
                                 columns=[f'k{i}' for i in range(10)])

        for i, (keypoint, label) in enumerate(zip(self.keypoints, 'cbdaef')):
            length = self.keypoints[keypoint]
            delta = self.extract_coilset_delta(length)
            for j, coord in enumerate(['r', 'phi', 'z']):
                error.loc[(coord, label), :] = \
                    self.error_modes(1e3*delta[:, j])
        return error


@dataclass
class BenchMark:
    """Perform case to case benchmarking."""

    case_a: str
    case_b: str
    scenario: Union[str, list[str, str]]
    profile: list[ErrorProfile, ErrorProfile] = field(init=False, repr=False)

    def __post_init__(self):
        """Load dataset."""
        if isinstance(self.scenario, str):
            self.scenario = 2*[self.scenario]
        self.profile = [ErrorProfile(self.case_a, self.scenario[0]),
                        ErrorProfile(self.case_b, self.scenario[1])]

    def bar(self, keypoint: str, coordinate: int, axes=None, legend=True,
            annotate=True, text=None):
        """Plot single bar."""
        if axes is None:
            axes = plt.gca()
        self.profile[0].bar(keypoint, coordinate, axes=axes, label='mode',
                            annotate=False, text=False)
        self.profile[1].bar(keypoint, coordinate, width=0.4, axes=axes,
                            label='mode', annotate=annotate, text=text)
        if legend:
            loc = 2 if coordinate == 0 else 4
            axes.legend(loc=loc, fontsize='xx-small')

    def bar_array(self, coordinate: int):
        """Generate bar plot array."""
        fig, axes_array = plt.subplots(3, 2, sharex=True, sharey=True)
        text = 'cbdaef'
        for i, (keypoint, axes) in enumerate(zip(self.profile[0].keypoints,
                                                 axes_array.reshape(-1))):
            self.bar(keypoint, coordinate, axes,
                     legend=True, annotate=False, text=text[i])
        axes_array[0][0].set_xticks(range(0, 18, 4))
        axes_array[0][0].set_xticklabels(range(0, 19, 4))
        coordinate_label = ['r', r'\phi', 'z']
        for i in range(2):
            axes_array[2][i].set_xlabel('coil index')
        axes_array[1][0].set_ylabel(
            rf'$\Delta\,{coordinate_label[coordinate]}$ mm')
        labels = self.case_labels
        case_a = mpatches.Patch(color='C0', label=labels[0])
        case_b = mpatches.Patch(color='C1', label=labels[1])
        fig.legend(handles=[case_a, case_b], ncol=2, loc='upper center')

    @property
    def case_labels(self):
        """Return case labels."""
        labels = [self.case_a, self.case_b]
        if self.scenario[0] != self.scenario[1]:  # include scenario
            labels = [f'{label} {self.profile[i].scenario}' for i, label
                      in enumerate(labels)]
        return labels

    def error_frame(self, mode: int):
        """Return comparitive error mode dataframe."""
        frames = [profile.error_frame() for profile in self.profile]
        delta = pandas.DataFrame(columns=frames[0].index)
        for i, label in enumerate(self.case_labels):
            delta.loc[label, :] = frames[i].loc[:, f'k{mode}']
        delta.loc['delta', :] = delta.iloc[1, :] - delta.iloc[0, :]
        delta.index.name = f'k{mode}'
        return delta

    def maximum_error(self):
        """Return dataframe highlighting maxiumum error in eack mode."""
        max_error = pandas.DataFrame(index=['coord', 'point', 'delta',
                                            'percent'],
                                     columns=[f'k{i}' for i in range(1, 10)])
        for mode in range(1, 10):
            error = self.error_frame(mode)
            col = np.argmax(error.loc['delta', :].abs())
            max_error.loc[['coord', 'point'], f'k{mode}'] = error.columns[col]
            max_error.loc['delta', f'k{mode}'] = error.iloc[-1, col]
            max_error.loc['percent', f'k{mode}'] = 1e2*(error.iloc[-1, col] /
                                                        error.iloc[0, col])
        return max_error


if __name__ == '__main__':

    #bench = BenchMark('F4E', 'SCOD', 'EOB')
    #bench = BenchMark('MAG', 'SCOD', 'TFonly')
    bench = BenchMark('pcr100', 'pcr65', 'EOB')
    #bench = BenchMark('SCOD', 'SCOD', ['TFonly', 'EOB'])

    #bench.bar_array(2)
    print(bench.maximum_error())

    #bench.profile[0].plot_keypoints('cbdaef')

    #bm.warp(200)
    #bm.build_pcr100()
    #bm.plot(1)
    #print(bm.rms)
    #bm.bar()
