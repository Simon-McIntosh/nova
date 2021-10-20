"""Manage TFC18 benchmarks."""
from dataclasses import dataclass, field
import os

import numpy as np
import pyvista as pv
import scipy
import seaborn as sns

from nova.definitions import root_dir
from nova.structural.F4E_TFC18 import F4E_Data
from nova.structural.TFC18gap import TFCgap
from nova.structural.TFC18 import TFC18
from nova.utilities.pyplot import plt
from nova.utilities.addtext import linelabel


@dataclass
class BenchBase:
    """Benchmark baseclass."""

    mesh_a: pv.PolyData
    mesh_b: pv.PolyData
    case_names: list[str, str]
    scenario: list[str, str]
    mesh: pv.PolyData = field(init=False)

    def __post_init__(self):
        """Load dataset."""
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
class BenchMark:
    """TFC18 structural benchmark."""

    dataset: str = 'f4e'
    mesh: pv.PolyData = field(init=False)

    def __post_init__(self):
        """Load dataset."""
        self.load()

    def load(self):
        """Load dataset."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            getattr(self, f'build_{self.dataset}')()
            self.mesh.save(self.vtk_file)

    @property
    def vtk_file(self):
        """Return vtk filepath."""
        return os.path.join(root_dir, 'data/Assembly/TFC18',
                            self.dataset + '_benchmark.vtk')

    def build_f4e(self):
        """Build F4E dataset."""
        f4e = F4E_Data().mesh
        self.mesh = BenchBase(f4e, f4e, ['v0', 'v3'], ['v0', 'v3']).mesh

    def build_pcr100(self):
        """Build PCR 100% reference case."""
        v0 = TFC18('TFCgapsG10', 'v0_100_f4e', cluster=1).mesh
        v3 = TFC18('TFCgapsG10', 'v3_100_f4e', cluster=1).mesh
        self.mesh = BenchBase(v0, v3, ['v0', 'v3'], ['EOB', 'EOB']).mesh
        self.mesh.save(self.vtk_file)

    def build_benchmark(self):
        """Build benchmark dataset."""
        io = TFCgap(file='v3_100_f4e', baseline='v0_100_f4e', cluster=1).mesh
        f4e = F4E_Data().mesh
        self.mesh = BenchBase(io, f4e, ['io', 'f4e'], ['EOB', 'v3-v0']).mesh

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
        
    def extract_coil(self, index: int, clip=1.001):
        """Return single coil rotated to x-z reference plane."""
        cell = self.mesh.extract_cells(index)
        cell.rotate_z(-360*index / 18, transform_all_input_vectors=True)
        if clip:
            cell = cell.clip('x', (clip*cell.points[:, 0].min(), 0, 0))
            cell['abscissa'] = cell.points[:, 2]
        return cell

    def plot(self, coordinate=1, clip=1.001, axes=None):
        """Plot ensemble."""
        if axes is None:
            axes = plt.gca()
        colors = sns.color_palette('muted', n_colors=18)
        label = linelabel(ax=axes, value='', Ndiv=20, loc='start')
        for i in range(18):
            coil = self.extract_coil(i, clip)
            self.plot_coil(coil, coordinate=coordinate,
                           axes=axes, label=label, color=colors[i])
        label.plot()

    def plot_coil(self, coil, coordinate=0, axes=None,
                  diffrence=True, label=None, color=None):
        """Generate 2d plots."""
        if axes is None:
            axes = plt.gca()
        if diffrence:
            axes.plot(coil['abscissa'], 1e3*coil['delta'][:, coordinate],
                      color=color)
            if label:
                label.add(coil['coil_names'][0])
        else:
            axes.plot(coil['abscissa'],
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
    
    def bar(self, width=0.8):
        """Generate bar plot."""
        delta = np.zeros((18, 3))
        for i in range(18):
            coil = self.extract_coil(i, clip=1.001)
            delta[i] = scipy.interpolate.interp1d(
                coil.points[:, 2], coil['delta'], axis=0,
                fill_value='extrapolate')(-3)
        plt.bar(range(18), 1e3*delta[:, 0], width)
        plt.despine()
            
        

if __name__ == '__main__':

    BenchMark('pcr100').bar()
    BenchMark('f4e').bar(width=0.4)
    #bm.warp(200)
    #bm.build_pcr100()
    #bm.plot(1)
    #print(bm.rms)
    #bm.bar()
