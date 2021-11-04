"""Load ccl fiducial data for ITER TF coilset."""
from dataclasses import dataclass, field
import string

import numpy as np
import pandas
import pyvista as pv
import xarray

from nova.structural.centerline import CenterLine
from nova.structural.fiducialccl import FiducialIDM
from nova.structural.gaussianprocessregressor import GaussianProcessRegressor
from nova.structural.plotter import Plotter
from nova.utilities.pyplot import plt


@dataclass
class FiducialData(Plotter, FiducialIDM):
    """Manage ccl fiducial data."""

    fill: bool = True
    sead: int = 2025
    rawdata: dict[str, pandas.DataFrame] = \
        field(init=False, repr=False, default_factory=dict)
    data: xarray.Dataset = field(init=False, repr=False)
    gpr: GaussianProcessRegressor = field(init=False, repr=False)
    mesh: pv.PolyData = field(init=False)

    def __post_init__(self):
        """Load data."""
        self.build_dataset()
        if self.fill:
            self.backfill()
        self.locate_coils()
        self.build_mesh()

    def build_dataset(self):
        """Build xarray dataset."""
        self.initialize_dataset()
        self.load_centerline()
        self.load_fiducials()
        self.load_fiducial_deltas()

    def label_coils(self, plotter, location='OD'):
        """Add coil labels."""
        plotter.add_point_labels(self.mesh[location][:18],
                                 self.mesh['label'][:18], font_size=20)

    def backfill(self):
        """Insert samples drawn from EU/JA datasets as proxy for missing."""
        metadata = xarray.Dataset(
            coords=dict(DA=['EU', 'JA'], coil=range(1, 20)))
        metadata['origin'] = ('coil',
                              ['EU', 'JA', 'EU', 'EU', 'EU', 'EU', 'JA',
                               'JA', 'EU', 'JA', 'EU', 'JA', 'JA', 'JA',
                               'JA', 'JA', 'EU', 'EU', 'JA'])
        rng = np.random.default_rng(self.sead)  # sead random number generator

        # self.data['clone'] = ('coil', np.full(self.data.dims['coil'], -1))
        self.data = self.data.assign_coords(
            clone=('coil', np.full(self.data.dims['coil'], -1)))
        fill = []
        for DA in metadata.DA:
            source = self.data.coil[self.data.origin == DA].values
            index = metadata.coil[metadata.origin == DA].values
            target = index[~np.isin(index, source)]
            sample = rng.integers(len(source), size=len(target))
            copy = self.data.sel(coil=source[sample])
            copy = copy.assign_coords(coil=target)
            copy = copy.assign_coords(clone=('coil', source[sample]))
            fill.append(copy)
        self.data = xarray.concat([self.data, *fill],
                                  dim='coil', data_vars='minimal')
        self.data = self.data.sortby('coil')

    def locate_coils(self):
        """Update data with coil's position index."""
        loc = [14, 15, 4, 17, 6, 7, 2, 3, 16, 5, 12, 13, 8, 9, 10, 11, 18, 1,
               19]
        self.data = self.data.assign_coords(
            location=('coil', [loc.index(coil) for coil in self.data.coil]))
        self.data = self.data.sortby('location')

    def build_mesh(self):
        """Build vtk mesh."""
        self.mesh = pv.PolyData()
        centerline = pv.Spline(1e-3*self.data.centerline)
        centerline['arc_length'] /= centerline['arc_length'][-1]
        for loc in self.data.location:
            if loc.coil == 19:
                continue
            coil = centerline.copy()
            coil['delta'] = 1e-3*self.data.centerline_delta.sel(coil=loc.coil)
            coil.rotate_z(20*loc.values, point=(0, 0, 0),
                          transform_all_input_vectors=True)
            midplane = coil.slice(normal='z', origin=(0, 0, 0))
            midplane.points += midplane['delta']
            coil['coil'] = [loc.coil.values]
            coil['ID'] = [midplane.points[0]]
            coil['OD'] = [midplane.points[1]]
            label = f'{loc.coil.values:02d}'
            try:
                if (clone := self.data.clone.sel(coil=loc.coil)) != -1:
                    label += f'<{clone.values}'
            except AttributeError:
                pass
            coil['label'] = [label]
            self.mesh = self.mesh.merge(coil, merge_points=False)

    def initialize_dataset(self):
        """Init xarray dataset."""
        self.data = xarray.Dataset(
            coords=dict(space=['x', 'y', 'z'],
                        target=list(string.ascii_uppercase[:8])))

    def load_fiducials(self):
        """Load ccl fiducials."""
        self.data['fiducial'] = (('target', 'space'), self.fiducials())
        target_index = [
            np.argmin(np.linalg.norm(self.data.centerline[:-1] -
                                     fiducial, axis=1))
            for fiducial in self.data.fiducial]
        self.data = self.data.assign_coords(
            target_index=('target', target_index))
        target_length = self.data.arc_length[target_index].values
        self.data = self.data.assign_coords(
            target_length=('target', target_length))
        self.data = self.data.sortby('target_length')
        self.gpr = GaussianProcessRegressor(self.data.target_length)

    def load_centerline(self):
        """Load geodesic centerline."""
        centerline = CenterLine()
        self.data['arc_length'] = centerline.mesh['arc_length']
        self.data['centerline'] = (('arc_length', 'space'),
                                   1e3*centerline.mesh.points)

    def load_fiducial_deltas(self):
        """Load fiducial deltas."""
        #'''
        delta, origin = {}, []
        for i in range(1, 20):
            index = f'{i:02d}'
            try:
                data = getattr(self, f'_tfc{index}')
                delta[i] = data[0].reindex(self.data.target)
                origin.append(data[1])
            except NotImplementedError:
                continue
        #'''
        #print(self.data.target)
        #delta, origin = FiducialIDM().data
        self.data['coil'] = list(delta)
        self.data = self.data.assign_coords(origin=('coil', origin))
        self.data['fiducial_delta'] = (('coil', 'target', 'space'),
                                       np.stack([delta[index]
                                                 for index in delta], axis=0))

        self.data['centerline_delta'] = xarray.DataArray(
            0., coords=[('coil', self.data.coil.values),
                        ('arc_length', self.data.arc_length.values),
                        ('space', self.data.space.values)])
        for coil_index in range(self.data.dims['coil']):
            for space_index in range(self.data.dims['space']):
                self.data['centerline_delta'][coil_index, :, space_index] = \
                    self.load_gpr(coil_index, space_index)

    def load_gpr(self, coil_index, space_index):
        """Return Gaussian Process regression."""
        return self.gpr.evaluate(
                        self.data.arc_length,
                        self.data.fiducial_delta[coil_index, :, space_index])

    def plot_gpr(self, coil_index, space_index):
        """Plot Gaussian Process regression."""
        self.load_gpr(coil_index, space_index)
        self.gpr.plot()

    def plot_gpr_array(self, coil_index):
        """Plot gpr array."""
        axes = plt.subplots(3, 1, sharex=True, sharey=True,
                            figsize=(4, 8))[1]
        for space_index, coord in enumerate('xyz'):
            self.load_gpr(coil_index, space_index)
            self.gpr.plot(axes=axes[space_index], text=False)
            axes[space_index].set_ylabel(fr'$\Delta{{{coord}}}$ mm')
        plt.despine()
        axes[-1].set_xlabel('arc length')
        axes[0].legend(loc='center', bbox_to_anchor=(0, 1.1, 1, 0.1))

    @staticmethod
    def fiducials():
        """Return fiducial coordinates."""
        return pandas.DataFrame(
            index=list(string.ascii_uppercase[:8]),
            columns=['x', 'y', 'z'],
            data=[[2713.7, 0., -3700.],
                  [2713.7, 0., 3700.],
                  [5334.4, 0., 6296.4],
                  [8980.4, 0., 4437.0],
                  [9587.6, 0., -3695.0],
                  [3399.7, 0., -5598.0],
                  [10733., 0., 0.],
                  [2713.7, 0., 0.]])

    def plot(self, factor=400):
        """Plot fiudicial points on coil cenerline."""
        axes = plt.subplots(1, 2, sharey=True)[1]
        for j in range(2):
            axes[j].plot(self.data.centerline[:, 0],
                         self.data.centerline[:, 2], 'gray', ls='--')
            axes[j].axis('equal')
            axes[j].axis('off')
        color = [0, 0]
        for i in range(self.data.dims['coil']):
            j = 0 if self.data.origin[i] == 'EU' else 1
            axes[j].plot(self.data.centerline[:, 0] +
                         factor*self.data.centerline_delta[i, :, 0],
                         self.data.centerline[:, 2] +
                         factor*self.data.centerline_delta[i, :, 2],
                         color=f'C{color[j]}',
                         label=f'{self.data.coil[i].values:02d}')
            axes[j].plot(self.data.fiducial[:, 0] +
                         factor*self.data.fiducial_delta[i, :, 0],
                         self.data.fiducial[:, 2] +
                         factor*self.data.fiducial_delta[i, :, 2], '.',
                         color=f'C{color[j]}')
            color[j] += 1
        for j, origin in enumerate(['EU', 'JA']):
            axes[j].legend(fontsize='x-small', loc='center',
                           bbox_to_anchor=[0.4, 0.5])
            axes[j].set_title(origin)

    def plot_single(self, coil=2, factor=500, axes=None):
        """Plot single fiducial curve."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]

        axes.plot(self.data.centerline[:, 0],
                  self.data.centerline[:, 2], 'gray', ls='--')

        for fiducial in self.data.fiducial:
            axes.plot(*fiducial[::2], 'ko')
            axes.text(*fiducial[::2], f' {fiducial.target.values}')

        axes.plot(self.data.fiducial[:, 0] +
                  factor*self.data.fiducial_delta[coil, :, 0],
                  self.data.fiducial[:, 2] +
                  factor*self.data.fiducial_delta[coil, :, 2], 'C3o')

        axes.plot(self.data.centerline[:, 0] +
                  factor*self.data.centerline_delta[coil, :, 0],
                  self.data.centerline[:, 2] +
                  factor*self.data.centerline_delta[coil, :, 2],
                  color='C0')
        axes.axis('equal')
        axes.axis('off')


if __name__ == '__main__':

    fiducial = FiducialData(fill=False)

    plotter = pv.Plotter()
    fiducial.warp(500, plotter=plotter)
    fiducial.label_coils(plotter)
    plotter.show_axes()
    plotter.show()

    #fiducial.plot()
    #fiducial.plot_gpr_array(1)
