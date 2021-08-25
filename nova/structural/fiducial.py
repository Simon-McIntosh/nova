"""Load ccl fiducial data for ITER TF coilset."""
from dataclasses import dataclass, field
import string

import numpy as np
import pandas
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
import xarray

from nova.structural.centerline import CenterLine
from nova.utilities.pyplot import plt


@dataclass
class FiducialData:
    """Manage ccl fiducial data."""

    rawdata: dict[str, pandas.DataFrame] = \
        field(init=False, repr=False, default_factory=dict)
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Load data."""
        self.build_dataset()

    def build_dataset(self):
        """Build xarray dataset."""
        self.initialize_dataset()
        self.load_centerline()
        self.load_fiducials()
        self.load_deltas()

    def initialize_dataset(self):
        """Init xarray dataset."""
        self.data = xarray.Dataset(
            coords=dict(space=['x', 'y', 'z'],
                        tnp=['t', 'n', 'p'],  # tangent, normal, plane
                        target=list(string.ascii_uppercase[:8])))

    def load_fiducials(self):
        """Load ccl fiducials."""
        self.data['fiducial'] = (('target', 'space'), self.targets())
        target_index = [np.argmin(
            np.linalg.norm(self.data.centerline[:-1] - fiducial, axis=1))
            for fiducial in self.data.fiducial]
        self.data = self.data.assign_coords(
            target_index=('target', target_index))
        target_length = self.data.arc_length[target_index].values
        self.data = self.data.assign_coords(
            target_length=('target', target_length))
        self.data = self.data.sortby('target_length')

    def load_centerline(self):
        """Load geodesic centerline."""
        centerline = CenterLine()
        self.data['arc_length'] = centerline.mesh['arc_length']
        self.data['centerline'] = (('arc_length', 'space'),
                                   1e3*centerline.mesh.points)
        for component in ['tangent', 'normal', 'plane']:
            self.data[component] = \
                (('arc_length', 'space'), centerline.mesh[component])

    def load_deltas(self):
        """Load fiducial deltas."""
        delta = {}
        for i in range(1, 20):
            index = f'{i:02d}'
            try:
                delta[index] = \
                    getattr(self, f'_tfc{index}').reindex(self.data.target)
            except NotImplementedError:
                continue
        self.data['coil'] = list(delta)
        self.data['delta'] = (('coil', 'target', 'space'),
                              np.stack([delta[index]
                                        for index in delta], axis=0))
        for component in ['tangent', 'normal', 'plane']:
            self.data[f'd{component[0]}'] = \
                (('coil', 'target'),
                 np.einsum('...ij,ij->...i',  self.data.delta,
                           self.data[component][self.data.target_index]))

    @staticmethod
    def targets():
        """Return fiducial targets."""
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

    @staticmethod
    def tfc06_fiducials(self):
        """Return tfc06 fiducial coordinates."""
        # CCL as-built fudicials in space for TFC06
        tfc06_fiducials = pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2',
                   'B', '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10',
                   '11', 'G', '12', '13', '14', 'E', '15', '16', '18', '19',
                   '20'],
            columns=['x', 'y', 'z', 'dx', 'dy', 'dz'],
            data=[[2713.20, 0.64, -3671.12, -0.50, 0.64, -0.12],
                  [2713.73, 0.07, -2746.08, 0.03, 0.07, -0.08],
                  [2714.25, 0.02, -1820.91, 0.55, 0.02, 0.09],
                  [2714.24, -0.21, -895.98, 0.54, -0.21, 0.02],
                  [2714.17, -0.74, 28.95, 0.47, -0.74, -0.05],
                  [2714.08, -0.88, 954.08, 0.38, -0.88, 0.08],
                  [2713.92, -0.38, 1879.21, 0.22, -0.38, 0.21],
                  [2713.37, 0.27, 2804.04, -0.33, 0.27, 0.04],
                  [2712.80, 1.42, 3728.96, -0.90, 1.42, -0.04],
                  [2919.86, 1.85, 4952.89, -1.44, 1.85, 0.79],
                  [3502.79, 2.17, 5723.39, -1.21, 2.17, 1.49],
                  [4354.16, 2.25, 6180.35, -0.94, 2.25, 1.65],
                  [5333.36, 1.55, 6327.06, -1.04, 1.55, 1.66],
                  [6365.03, 1.34, 6163.03, -1.37, 1.34, 0.73],
                  [7324.21, 0.45, 5741.86, -1.89, 0.45, 0.16],
                  [8202.53, -0.22, 5170.61, -2.37, -0.22, -0.19],
                  [8977.45, -0.38, 4465.52, -2.95, -0.38, -0.48],
                  [9722.65, -0.62, 3500.91, -3.65, -0.62, -0.69],
                  [10274.49, -0.79, 2413.77, -4.41, -0.79, -0.73],
                  [10613.46, -1.01, 1242.60, -4.94, -1.01, -0.60],
                  [10728.31, -0.90, 28.90, -4.69, -0.90, -0.10],
                  [10654.52, -0.91, -949.17, -4.78, -0.91, 0.23],
                  [10435.59, -0.76, -1905.07, -4.41, -0.76, 0.33],
                  [10076.40, -0.49, -2817.52, -3.70, -0.49, 0.28],
                  [9584.23, 0.08, -3665.59, -3.37, 0.08, 0.41],
                  [8753.28, 1.08, -4653.12, -2.22, 1.08, 0.48],
                  [7744.78, 2.25, -5460.72, 0.38, 2.25, -2.42],
                  [4284.62, 2.67, -6104.88, 0.12, 2.67, -0.58],
                  [3399.69, 2.65, -5569.54, -0.01, 2.65, -0.54],
                  [2841.96, 2.32, -4700.20, -0.44, 2.32, -0.50]])
        fiducials = pandas.DataFrame(
            index=tfc06_fiducials.index,
            columns=['x', 'y', 'z'])
        fiducials[:] = tfc06_fiducials.loc[:, 'x':'z'].values - \
            tfc06_fiducials.loc[:, 'dx':'dz'].values
        return fiducials

    @property
    def _tfc01(self):
        """Return TFC01 fiducial data."""
        raise NotImplementedError('TFC01 - EU - pending')

    @property
    def _tfc02(self):
        """Return TFC02 fiducial data - JA."""
        return self.coordinate_transform(pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.01, -0.3, -0.71],
                  [0.0, -0.41, -0.51],
                  [0.31, 1.84, -0.93],
                  [0.08, -1.95, -2.91],
                  [-0.44, -0.79, -5.18],
                  [1.06, -0.62, -2.29],
                  [-1.39, -1.93, -5.96],
                  [-0.96, -0.3, 0.73]]))

    @property
    def _tfc03(self):
        """Return TFC03 fiducidal data - EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.64, -0.07, -0.25],
                  [-0.46, -0.04, -0.21],
                  [0.04, -0.03, -0.01],
                  [0., -0.23, -0.01],
                  [-0.05, -0.42, -0.01],
                  [-0.14, -0.02,  0.04],
                  [-0.23,  0.38,  0.08],
                  [-0.55,  0.2,  0.08],
                  [-0.89,  0.08,  0.01],
                  [-1.33, -0.04,  0.23],
                  [-1.67, -0.1,  0.65],
                  [-2.01, -0.16,  1.06],
                  [-2.34, -0.23,  1.43],
                  [-2.53, -0.3,  1.15],
                  [-2.73, -0.38,  0.86],
                  [-2.92, -0.45,  0.57],
                  [-3.12, -0.53,  0.29],
                  [-3.33, -0.59,  0.03],
                  [-3.51, -0.61,  0.01],
                  [-3.68, -0.62, -0.02],
                  [-3.69, -0.52, -0.15],
                  [-3.62, -0.38, -0.31],
                  [-3.25, -0.2, -0.48],
                  [-2.82, -0.01, -0.66],
                  [-2.4,  0.18, -0.83],
                  [-1.98,  0.33, -1.28],
                  [-1.41,  0.6, -1.68],
                  [-1.11,  0.08, -0.69],
                  [-1.02, -0.06, -0.41],
                  [-0.85, -0.09, -0.3]])

    @property
    def _tfc04(self):
        """Return TFC04 fiducial data."""
        raise NotImplementedError('TFC04 - EU - pending')

    @property
    def _tfc05(self):
        """Return TFC05 fiducidal data - EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.47, -0.05, -0.04],
                  [-0.19, -0.13,  0.03],
                  [0.22, -0.25,  0.16],
                  [0.13, -0.33,  0.13],
                  [0.04, -0.41,  0.1],
                  [-0.05, -0.35,  0.05],
                  [-0.14, -0.11,  0.],
                  [-0.17,  0.04, -0.03],
                  [-0.53,  0.27,  0.],
                  [-1.01,  0.38,  0.28],
                  [-1.39,  0.31,  0.69],
                  [-1.76,  0.24,  1.1],
                  [-2.14,  0.16,  1.47],
                  [-2.42, -0.01,  1.24],
                  [-2.71, -0.19,  1.02],
                  [-2.99, -0.37,  0.8],
                  [-3.28, -0.55,  0.57],
                  [-3.57, -0.68,  0.33],
                  [-3.75, -0.57,  0.14],
                  [-3.93, -0.45, -0.06],
                  [-3.59, -0.36, -0.22],
                  [-3.28, -0.24, -0.38],
                  [-2.98, -0.12, -0.54],
                  [-2.67,  0., -0.7],
                  [-2.37,  0.11, -0.85],
                  [-2.04,  0.13, -1.08],
                  [-1.59,  0.23, -1.45],
                  [-1.17,  0.14, -0.53],
                  [-1.04,  0.12, -0.27],
                  [-0.79,  0.04, -0.13]])

    @property
    def _tfc06(self):
        """Return TFC06 fiducidal data - EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.24,  0.05, -0.12],
                  [0.03,  0.05, -0.08],
                  [0.47, -0.01,  0.09],
                  [0.38, -0.17,  0.02],
                  [0.29, -0.31, -0.05],
                  [0.23, -0.14, 0.08],
                  [0.16,  0.04, 0.21],
                  [-0.23,  0.34,  0.04],
                  [-0.63,  0.36, -0.04],
                  [-1.11,  0.39,  0.15],
                  [-1.44,  0.42,  0.53],
                  [-1.77,  0.45,  0.92],
                  [-2.1,  0.47,  1.27],
                  [-2.32,  0.33,  1.07],
                  [-2.56,  0.19,  0.86],
                  [-2.78,  0.05,  0.65],
                  [-3.01, -0.1,  0.45],
                  [-3.29, -0.24,  0.21],
                  [-3.61, -0.32,  0.],
                  [-3.92, -0.41, -0.21],
                  [-3.59, -0.36, -0.1],
                  [-3.55, -0.4, -0.12],
                  [-3.21, -0.26, -0.37],
                  [-2.82, -0.08, -0.66],
                  [-2.43,  0.09, -0.96],
                  [-1.73,  0.46, -1.11],
                  [-1.3,  0.72, -1.46],
                  [-0.92,  0.24, -0.54],
                  [-0.81,  0.11, -0.28],
                  [-0.55,  0.06, -0.17]])

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
                  [0.34, 0.48, -2.31]])

    @property
    def _tfc09(self):
        """Return TFC09 fiducidal data - EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12',  '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.33,  0.22, -0.12],
                  [-0.15,  0.33, -0.05],
                  [-0.06,  0.36,  0.01],
                  [-0.03,  0.01,  0.07],
                  [0.04, -0.27,  0.17],
                  [-0.15, -0.16,  0.1],
                  [-0.39, -0.11,  0.08],
                  [-0.61, -0.15,  0.07],
                  [-0.84, -0.28,  0.02],
                  [-0.89, -0.23,  0.05],
                  [-0.75,  0.07,  0.19],
                  [-1.27,  0.44,  0.53],
                  [-2.11,  0.87,  0.99],
                  [-1.01,  0.38,  0.42],
                  [-0.5,  0.14,  0.14],
                  [-0.37,  0.07,  0.05],
                  [-0.37,  0.04, -0.01],
                  [-0.51,  0.03, -0.08],
                  [-1.08,  0.02, -0.22],
                  [-3.17,  0.02, -0.7],
                  [-2.4,  0.01, -0.55],
                  [-1.01,  0.01, -0.25],
                  [-0.57,  0.01, -0.18],
                  [-0.44,  0.01, -0.19],
                  [-0.47,  0.01, -0.3],
                  [-0.92,  0.01, -0.81],
                  [-1.95,  0.01, -1.87],
                  [-0.52,  0.18, -0.25],
                  [-1.,  0.45, -0.4],
                  [-0.73,  0.35, -0.29]])

    @property
    def _tfc10(self):
        """Return TFC10 fiducial data - JA."""
        raise NotImplementedError('TFC10 - JA - pending')

    @property
    def _tfc11(self):
        """Return TFC11 fiducial data - EU."""
        raise NotImplementedError('TFC11 - EU - pending')

    @property
    def _tfc12(self):
        """Return TFC12 fiducial data - JA."""
        raise NotImplementedError('TFC12 - JA - pending')

    @property
    def _tfc13(self):
        """Return TFC13 fiducial data - JA."""
        raise NotImplementedError('TFC13 - JA - pending')

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
    def read_clipboard():
        """Read displacment data from clipboard."""
        # pylint: disable=no-member
        ccl = pandas.read_clipboard(header=None)
        ccl.set_index(0, inplace=True)
        ccl.index.name = None
        columns = ccl.columns
        ccl.drop(columns=columns[3:], inplace=True)
        ccl.columns = ['dx', 'dy', 'dz']
        return ccl.dropna(0)

    def plot(self):
        """Plot fiudicial points on coil cenerline."""
        axes = plt.subplots(1, 1)[1]
        axes.plot(self.data.centerline[:, 0], self.data.centerline[:, 2])
        axes.plot(self.data.fiducial[:, 0], self.data.fiducial[:, 2], 'X')
        for i in range(self.data.dims['coil']):
            axes.plot(self.data.delta[i, :, 0], self.data.delta[i, :, 0])

        plt.axis('equal')
        plt.axis('off')

    def plot_components(self):
        """Plot delta components."""
        axes = plt.subplots(3, 1)[1]
        for i in range(self.data.dims['coil']):
            index = ~np.isnan(self.data.dt[i])
            for j, coord in enumerate(['dt', 'dn', 'dp']):
                axes[j].plot(self.data.target_length[index],
                             self.data[coord][i][index], '-o')

    def infer(self, coil_index):

        length = np.append(self.data.target_length.values, 1)
        data = self.data.dn[coil_index].values
        data = np.append(data, data[0])
        index = ~np.isnan(data)
        length, data = length[index], data[index]

        data = np.append(data[-2], np.append(data, data[1]))
        length = np.append(length[-2]-1, np.append(length, 1+length[1]))
        length = length.reshape(-1, 1)

        kernel = 1.0 * RBF(length_scale=0.3, length_scale_bounds=(0.05, 1)) +\
            WhiteKernel(noise_level=0.25, noise_level_bounds=(1e-8, 1))
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       alpha=0.05).fit(length, data)
        X_ = np.linspace(0, 1, 100)
        y_mean, y_cov = gpr.predict(X_[:, np.newaxis], return_cov=True)

        plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
        plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                         y_mean + np.sqrt(np.diag(y_cov)),
                         alpha=0.5, color='k')
        #plt.plot(X_, 0.5*np.sin(3*X_), 'r', lw=3, zorder=9)
        plt.scatter(length[:, 0], data, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))

        #plt.plot(length, data)

        print(gpr.kernel_)



if __name__ == '__main__':

    fiducial = FiducialData()
    fiducial.infer(1)
