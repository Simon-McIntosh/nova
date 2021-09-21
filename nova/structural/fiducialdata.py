"""Load ccl fiducial data for ITER TF coilset."""
from dataclasses import dataclass, field
import string

import numpy as np
import pandas
import pyvista as pv
import xarray

from nova.structural.centerline import CenterLine
from nova.structural.gaussianprocessregressor import GaussianProcessRegressor
from nova.structural.plotter import Plotter
from nova.utilities.pyplot import plt


@dataclass
class FiducialData(Plotter):
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
            if (clone := self.data.clone.sel(coil=loc.coil)) != -1:
                label += f'<{clone.values}'
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
        delta, origin = {}, []
        for i in range(1, 20):
            index = f'{i:02d}'
            try:
                data = getattr(self, f'_tfc{index}')
                delta[i] = data[0].reindex(self.data.target)
                origin.append(data[1])
            except NotImplementedError:
                continue
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

    @property
    def _tfc01(self):
        """Return TFC01 fiducial data."""
        raise NotImplementedError('TFC01 - EU - pending')

    @property
    def _tfc02(self):
        """Return TFC02 fiducial data - JA."""
        return pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.63, 0.41, 0.09],
                  [-0.68, 0.43, -0.12],
                  [-0.52, 0.92, 1.67],
                  [-2.67, 0.92, -2.35],
                  [-4.41, 0.53, -0.32],
                  [-1.9, 1.69, -0.05],
                  [-4.76, -0.68, -1.73],
                  [0.25, -0.49, 0.04]]), 'JA'

    @property
    def _tfc03(self):
        """Return TFC03 fiducidal data - 52Z4PV - F4E_D_2REWA9 v2.0."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.8,  0.14, -0.25],
                  [-0.5,  0.15, -0.21],
                  [0.2, -0.19, -0.01],
                  [0.15, -0.45, -0.01],
                  [-0.03, -0.39, -0.01],
                  [0.05, -0.23,  0.04],
                  [-0.18,  0.23,  0.08],
                  [-0.65,  0.06,  0.08],
                  [-1.16,  0.54,  0.01],
                  [-1.66,  1.19,  0.82],
                  [-1.51,  1.57,  1.57],
                  [-1.26,  0.54,  1.7],
                  [-1.4,  1.09,  1.76],
                  [-1.72,  0.61,  0.76],
                  [-2.26, -0.16, -0.01],
                  [-2.8, -0.66, -0.49],
                  [-3.42, -0.69, -0.88],
                  [-4.08, -0.59, -1.04],
                  [-4.75, -0.9, -0.85],
                  [-5.15, -0.74, -0.48],
                  [-5.31, -0.54, -0.15],
                  [-5.35, -0.7,  0.09],
                  [-4.88, -0.54,  0.32],
                  [-4.24, -0.32,  0.48],
                  [-3.6, -0.16,  0.64],
                  [-2.63,  0.35,  0.36],
                  [0.19,  1.21, -2.7],
                  [-0.22,  0.87, -0.82],
                  [-0.28,  0.37, -0.62],
                  [-0.74,  0.23, -0.6]]), 'EU'

    @property
    def _tfc04(self):
        """Return TFC04 fiducial data."""
        raise NotImplementedError('TFC04 - EU - pending')

    @property
    def _tfc05(self):
        """Return TFC05 fiducidal data - 4HMUWH - F4E_D_2PYAKN v2.0."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.66,  0.82, -0.04],
                  [-0.24, -0.11,  0.03],
                  [0.21, -0.61,  0.16],
                  [0.21, -0.67,  0.13],
                  [0.13, -0.85,  0.1],
                  [0., -0.78,  0.05],
                  [-0.12, -0.33,  0.],
                  [-0.07,  0.23, -0.03],
                  [-0.6,  0.98,  0.],
                  [-1.28,  2.03,  0.85],
                  [-1.23,  2.24,  1.64],
                  [-1.05,  1.69,  1.93],
                  [-1.17,  0.98,  2.18],
                  [-1.39,  0.89,  1.43],
                  [-1.8, -0.33,  0.91],
                  [-2.22, -0.86,  0.54],
                  [-2.7, -1.01,  0.2],
                  [-3.31, -0.96, -0.12],
                  [-3.78, -0.76, -0.26],
                  [-4.35, -0.95, -0.32],
                  [-4.27, -0.99, -0.22],
                  [-4.26, -0.97, -0.08],
                  [-4.08, -0.43,  0.11],
                  [-3.9, -0.02,  0.37],
                  [-3.51,  0.11,  0.59],
                  [-2.92,  0.72,  0.81],
                  [-0.29,  1.77, -1.96],
                  [0.06,  2.74,  0.17],
                  [0.23,  2.71,  0.04],
                  [-0.28,  2.03, -0.29]]), 'EU'

    @property
    def _tfc06(self):
        """Return TFC06 fiducidal data - 5PPCAF - F4E_D_2RTP8J v1.2 EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.5,  0.64, -0.12],
                  [0.03,  0.07, -0.08],
                  [0.55,  0.02,  0.09],
                  [0.54, -0.21,  0.02],
                  [0.47, -0.74, -0.05],
                  [0.38, -0.88,  0.08],
                  [0.22, -0.38,  0.21],
                  [-0.33,  0.27,  0.04],
                  [-0.9,  1.42, -0.04],
                  [-1.44,  1.85,  0.79],
                  [-1.21,  2.17,  1.49],
                  [-0.94,  2.25,  1.65],
                  [-1.04,  1.55,  1.66],
                  [-1.37,  1.34,  0.73],
                  [-1.89,  0.45,  0.16],
                  [-2.37, -0.22, -0.19],
                  [-2.95, -0.38, -0.48],
                  [-3.65, -0.62, -0.69],
                  [-4.41, -0.79, -0.73],
                  [-4.94, -1.01, -0.6],
                  [-4.69, -0.9, -0.1],
                  [-4.78, -0.91,  0.23],
                  [-4.41, -0.76,  0.33],
                  [-3.7, -0.49,  0.28],
                  [-3.37,  0.08,  0.41],
                  [-2.22,  1.08,  0.48],
                  [0.38,  2.25, -2.42],
                  [0.12,  2.67, -0.58],
                  [-0.01,  2.65, -0.54],
                  [-0.44,  2.32, -0.5]]), 'EU'

    @property
    def _tfc07(self):
        """Return TFC07 fiducial data."""
        raise NotImplementedError('TFC07 - JA - pending')

    @property
    def _tfc08(self):
        """Return TFC08 fiducidal data - JA."""
        return pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.69, -0.62, -0.52],
                  [-0.72, -0.72, -1.08],
                  [-1.19, 1.11, -0.08],
                  [-2.56, 0.25, -0.3],
                  [-3.17, -1.97, -0.07],
                  [-1.48, -0.9, 0.21],
                  [-3.17, 0.04, -0.22],
                  [0.34, 0.48, -2.31]]), 'JA'

    @property
    def _tfc09(self):
        """Return TFC09 fiducidal data - 2SU8F4 - F4E_D_2KYN3R v2.0 EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.3,  0.32, -0.16],
                  [-0.02,  0.32, -0.04],
                  [0.2,  0.23,  0.06],
                  [0.25, -0.05,  0.14],
                  [0.28, -0.31,  0.21],
                  [0.06, -0.25,  0.18],
                  [-0.16, -0.18,  0.15],
                  [-0.38, -0.16,  0.12],
                  [-0.64, -0.26,  0.07],
                  [-0.99, -0.11,  0.22],
                  [-1.29,  0.25,  0.5],
                  [-1.59,  0.6,  0.77],
                  [-1.88,  0.92,  1.02],
                  [-2.09,  0.79,  0.79],
                  [-2.3,  0.65,  0.56],
                  [-2.5,  0.52,  0.33],
                  [-2.72,  0.39,  0.1],
                  [-2.95,  0.23, -0.17],
                  [-3.2,  0.07, -0.44],
                  [-3.44, -0.08, -0.71],
                  [-3.34, -0.15, -0.92],
                  [-3.09, -0.16, -1.07],
                  [-2.84, -0.18, -1.22],
                  [-2.6, -0.19, -1.36],
                  [-2.35, -0.22, -1.5],
                  [-2.03, -0.23, -1.68],
                  [-1.71, -0.23, -1.83],
                  [-1.08,  0.17, -0.77],
                  [-0.9,  0.29, -0.47],
                  [-0.61,  0.32, -0.29]]), 'EU'

    @property
    def _tfc10(self):
        """Return TFC10 fiducial data - JA."""
        return pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[0.14, 0.3, -0.72],
                  [0.08, 0.17, -1.07],
                  [-1.81, -0.61, -1.25],
                  [-1.99, 0.11, -2.13],
                  [-2.13, 0.06, -0.51],
                  [0.3, 0.98, 0.85],
                  [-2.85, 0.5, -1.09],
                  [0.19, -0.36, -1.]]), 'JA'

    @property
    def _tfc11(self):
        """Return TFC11 fiducial data - 3T6WVX - F4E_D_2NNX2Y v3.0 EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.84, -0.09, -0.16],
                  [-0.31, -0.39,  0.],
                  [0.04, -0.07,  0.12],
                  [0.09,  0.04,  0.1],
                  [-0.03, -0.07,  0.07],
                  [-0.04, -0.11,  0.05],
                  [-0.12, -0.24,  0.03],
                  [-0.45, -0.07,  0.01],
                  [-1.1,  0.1, -0.01],
                  [-2.12,  0.14,  0.88],
                  [-1.83,  0.22,  1.5],
                  [-1.58,  0.23,  1.6],
                  [-1.73, -0.31,  1.65],
                  [-1.94, -0.49,  0.81],
                  [-2.32, -0.72,  0.33],
                  [-2.65, -0.73,  0.11],
                  [-3.24, -0.65, -0.27],
                  [-3.86, -0.64, -0.44],
                  [-4.47, -0.73, -0.46],
                  [-4.92, -0.22, -0.33],
                  [-4.96, -0.01, -0.17],
                  [-4.78, -0.19, -0.02],
                  [-4.45,  0.06,  0.15],
                  [-4.03,  0.22,  0.29],
                  [-3.56, -0.02,  0.44],
                  [-2.76,  0.21,  0.48],
                  [-0.29,  0.56, -2.07],
                  [-0.17,  1.57, -0.79],
                  [-0.26,  1.09, -0.7],
                  [-0.82,  0.83, -0.67]]), 'EU'

    @property
    def _tfc12(self):
        """Return TFC12 fiducial data - JA 2UD358."""
        return self.coordinate_transform(pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.21,  0.17, -0.7],
                  [-1.34, -0.82, -0.32],
                  [0.27, 0.63, -1.78],
                  [1.29, -2.06, -3.4],
                  [1.74, -0.97, -5.1],
                  [0.57, -0.89, -2.51],
                  [0.96, -2.12, -6.22],
                  [-0.77, -0.08,  0.62]])), 'JA'

    @property
    def _tfc13(self):
        """Return TFC13 fiducial data - 3B5YEM - JA."""
        return pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.3,  0.26, -0.8],
                  [-0.29, -0.32, -1.21],
                  [-2.26,  2.19,  1.43],
                  [-0.44, -0.05, -1.87],
                  [-4.78, -0.54, -1.69],
                  [0.44,  0.53, -0.44],
                  [-2.86, -1.28, -2.27],
                  [-0.08, -0.71, -1.36]]), 'JA'

    @property
    def _tfc14(self):
        """Return TFC14 fiducial data - EU."""
        raise NotImplementedError('TFC14 - JA - pending')

    @property
    def _tfc15(self):
        """Return TFC15 fiducial data - JA."""
        raise NotImplementedError('TFC15 - JA - pending')

    @property
    def _tfc16(self):
        """Return TFC16 fiducial data - JA."""
        raise NotImplementedError('TFC16 - JA - pending')

    @property
    def _tfc17(self):
        """Return TFC17 fiducial data - EU."""
        raise NotImplementedError('TFC17 - EU - pending')

    @property
    def _tfc18(self):
        """Return TFC18 fiducial data - EU."""
        raise NotImplementedError('TFC18 - EU - pending')

    @property
    def _tfc19(self):
        """Return TFC19 fiducial data - JA."""
        raise NotImplementedError('TFC19 - JA - pending')

    @staticmethod
    def coordinate_transform(mcs):
        """
        Convert ccl delta coordinates from AU (JA) to space.

        space to MCS:
            𝑋𝑀𝐶𝑆 = 5334.4 − 𝑋𝑇𝐺𝐶𝑆
            𝑌𝑀𝐶𝑆 = 29 − 𝑍𝑇𝐺𝐶𝑆
            𝑍𝑀𝐶𝑆 = 𝑌𝑇𝐺𝐶𝑆

        From MCS to space:
            𝑋𝑇𝐺𝐶𝑆 = 5334.4 − 𝑋𝑀𝐶𝑆
            𝑌𝑇𝐺𝐶𝑆 = 𝑍𝑀𝐶𝑆
            𝑍𝑇𝐺𝐶𝑆 = 29 - 𝑌𝑀𝐶𝑆

        """
        space = pandas.DataFrame(index=mcs.index, columns=mcs.columns)
        space.loc[:, 'dx'] = mcs.dz
        space.loc[:, 'dy'] = mcs.dx
        space.loc[:, 'dz'] = mcs.dy
        return space

    @staticmethod
    def read_clipboard(column_index=slice(3, 6)):
        """Read displacment data from clipboard."""
        # pylint: disable=no-member
        ccl = pandas.read_clipboard(header=None)
        ccl.set_index(0, inplace=True)
        ccl.index.name = None
        ccl = ccl.iloc[:, column_index]
        ccl.columns = ['dx', 'dy', 'dz']
        ccl = ccl.iloc[~(ccl == '-').any(axis=1).values, :]
        return ccl.dropna(0).astype(float)

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

    fiducial = FiducialData(fill=True)
    plotter = pv.Plotter()
    fiducial.warp(500, plotter=plotter)
    fiducial.label_coils(plotter)
    plotter.show_axes()
    plotter.show()

    #fiducial.plot()
    #fiducial.plot_gpr_array(1)
