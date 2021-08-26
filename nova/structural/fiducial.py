"""Load ccl fiducial data for ITER TF coilset."""
from collections.abc import Iterable
from dataclasses import dataclass, field, InitVar
from typing import Union
import string

import numpy as np
import numpy.typing as npt
import pandas
import sklearn.gaussian_process
import xarray

from nova.structural.centerline import CenterLine
from nova.utilities.pyplot import plt


@dataclass
class GaussianProcessRegressor:
    """Fit cyclic 1D waveforms using a Gaussian Process Regressor."""

    x: InitVar[npt.ArrayLike]
    period: list[float, float] = field(default_factory=lambda: [0., 1.])
    wrap: int = 0
    regressor: sklearn.gaussian_process.GaussianProcessRegressor = None
    data: xarray.Dataset = field(init=False)

    def __post_init__(self, x):
        """Init dataset."""
        x = self.to_numpy(x)
        self.data = xarray.Dataset(coords=dict(x=x))
        self.build_regressor()

    def build_regressor(self, noise_std=0.25):
        """Build Gaussian Process Regressor."""
        if self.regressor is None:
            ExpSineSquared = sklearn.gaussian_process.kernels.ExpSineSquared(
                length_scale=0.5, length_scale_bounds=(0.1, 1e2),
                periodicity=1.0, periodicity_bounds='fixed')
            kernel = ExpSineSquared
            self.regressor = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=kernel, alpha=noise_std**2)

    @staticmethod
    def to_numpy(array):
        """Return numpy array."""
        if isinstance(array, xarray.DataArray):
            return array.values
        return array

    def fit(self, y):
        """Fit Gaussian Process Regressor."""
        self.data['y'] = ('x', self.to_numpy(y))
        _x, _y = self.data.x.to_numpy(), self.data.y.to_numpy()
        if np.isnan(_y).any():  # drop nans
            index = ~np.isnan(_y)
            _x, _y = _x[index], _y[index]
        if self.wrap > 0:
            _x = np.pad(_x, pad_width=self.wrap, mode='wrap')
            _x[:self.wrap] -= self.period[-1]
            _x[-self.wrap:] += self.period[-1]
            _y = np.pad(_y, pad_width=self.wrap, mode='wrap')
        self.data['_x'] = _x
        self.data['_y'] = _y
        self.regressor = self.regressor.fit(_x.reshape(-1, 1), _y)

    def predict(self, x_mean):
        """Sample Gaussian Process Regressor."""
        if isinstance(x_mean, int):
            x_mean = np.linspace(*self.period, x_mean)
        else:
            x_mean = self.to_numpy(x_mean)
        y_mean, y_cov = self.regressor.predict(x_mean.reshape(-1, 1),
                                               return_cov=True)
        self.data['x_mean'] = x_mean
        self.data['y_mean'] = ('x_mean', y_mean)
        self.data['y_std'] = ('x_mean', np.sqrt(np.diag(y_cov)))
        return y_mean

    def evaluate(self, x, y):
        """Return GPR prediction at x for data points y."""
        self.fit(y)
        return self.predict(x)

    def plot(self):
        """Plot current GP regression."""
        axes = plt.subplots(1, 1)[1]
        plt.scatter(self.data.x, self.data.y, c='C3', s=30, zorder=10,
                    label='fiducial data')
        axes.plot(self.data.x_mean, self.data.y_mean, 'gray', lw=2, zorder=9)
        axes.fill_between(self.data.x_mean,
                          self.data.y_mean - self.data.y_std,
                          self.data.y_mean + self.data.y_std,
                          alpha=0.15, color='k',
                          label='95% confidence')

        plt.despine()
        plt.xlabel('arc length')
        plt.ylabel('displacement, mm')
        plt.title(self.regressor.kernel_)
        plt.legend()


@dataclass
class FiducialData:
    """Manage ccl fiducial data."""

    rawdata: dict[str, pandas.DataFrame] = \
        field(init=False, repr=False, default_factory=dict)
    data: xarray.Dataset = field(init=False, repr=False)
    gpr: GaussianProcessRegressor = field(init=False, repr=False)

    def __post_init__(self):
        """Load data."""
        self.build_dataset()

    def build_dataset(self):
        """Build xarray dataset."""
        self.initialize_dataset()
        self.load_centerline()
        self.load_fiducials()
        self.load_fiducial_deltas()

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
                delta[index] = data[0].reindex(self.data.target)
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
                  [-0.96, -0.3, 0.73]])), 'JA'

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
                  [-0.85, -0.09, -0.3]]), 'EU'

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
                  [-0.79,  0.04, -0.13]]), 'EU'

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
                  [-0.55,  0.06, -0.17]]), 'EU'

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
                  [-0.73,  0.35, -0.29]]), 'EU'

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
        """Return TFC11 fiducial data - EU."""
        return pandas.DataFrame(
            index=['A', '1-A', '1', '1-21', '1-2', '1-22', '2', 'B-2', 'B',
                   '3', '4', '5', 'C', '6', '7', '8', 'D', '9', '10', '11',
                   'G', '12', '13', '14', 'E', '15', '16', '18', '19', '20'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.56, -0.19, -0.16],
                  [-0.24, -0.16,  0.],
                  [-0.01, -0.14,  0.12],
                  [-0.07, -0.12,  0.1],
                  [-0.12, -0.1,  0.07],
                  [-0.17, -0.07,  0.05],
                  [-0.22, -0.05,  0.03],
                  [-0.42, -0.03,  0.01],
                  [-0.96, -0.03, -0.01],
                  [-1.59, -0.15,  0.15],
                  [-1.98, -0.36,  0.46],
                  [-2.37, -0.57,  0.77],
                  [-2.77, -0.79,  1.08],
                  [-2.93, -0.67,  0.94],
                  [-3.1, -0.56,  0.8],
                  [-3.27, -0.44,  0.66],
                  [-3.43, -0.33,  0.52],
                  [-3.63, -0.19,  0.36],
                  [-3.83, -0.06,  0.2],
                  [-4.02,  0.08,  0.04],
                  [-3.88,  0.1, -0.17],
                  [-3.56,  0.04, -0.36],
                  [-3.23, -0.01, -0.55],
                  [-2.91, -0.07, -0.75],
                  [-2.58, -0.12, -0.94],
                  [-2.16, -0.2, -1.2],
                  [-1.73, -0.27, -1.45],
                  [-1.24, -0.26, -0.86],
                  [-1.17, -0.24, -0.57],
                  [-0.92, -0.21, -0.34]]), 'EU'

    @property
    def _tfc12(self):
        """Return TFC12 fiducial data - JA."""
        return pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[-0.21,  0.17, -0.7],
                  [-1.34, -0.82, -0.32],
                  [0.27, 0.63, -1.78],
                  [1.29, -2.06, -3.4],
                  [1.74, -0.97, -5.1],
                  [0.57, -0.89, -2.51],
                  [0.96, -2.12, -6.22],
                  [-0.77, -0.08,  0.62]]), 'JA'

    @property
    def _tfc13(self):
        """Return TFC13 fiducial data - JA."""
        return pandas.DataFrame(
            index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            columns=['dx', 'dy', 'dz'],
            data=[[0.58, -1.4, -0.56],
                  [-1.2, -1.44, -0.48],
                  [0.45, 2.15, -3.75],
                  [-1.44, -1.44, -1.96],
                  [-0.39, -2.12, -6.48],
                  [0.71, -1.34, -0.94],
                  [-1.3, -2.33, -4.92],
                  [-0.7, -1.94, 0.55]]), 'JA'

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
                         label=f'{self.data.coil[i].values}')
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





if __name__ == '__main__':

    fiducial = FiducialData()

    fiducial.plot()

    fiducial.plot_gpr(-1, 0)
