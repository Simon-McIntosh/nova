"""Perform fit for SM6 to fiducial mesurments."""

from dataclasses import dataclass, field, InitVar
from functools import cached_property
from typing import ClassVar
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import xarray

from nova.assembly.centerline import CenterLine
from nova.assembly.fiducialdata import FiducialData
from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor
from nova.utilities.pyplot import plt


@dataclass
class Transform:
    """Provide clocking transform."""

    clock: Rotation = Rotation.from_euler('z', 10, degrees=True)
    anticlock: Rotation = Rotation.from_euler('z', -10, degrees=True)


@dataclass
class Plotter:
    """Plot fidicual to target fit in cylindrical coordinates."""

    data: InitVar[xarray.Dataset]
    factor: float = 500
    fiducial_labels: bool = True

    color: ClassVar[dict[str, str]] = dict(
        fit='C1', fit_target='C0', reference='C4', reference_target='C6')
    marker: ClassVar[dict[str, str]] = dict(
        fit='X', fit_target='d', reference='X', reference_target='d')

    def __post_init__(self, data: xarray.Dataset):
        """Transform data to cylindrical coordinates."""
        self.extract(data)

    def __call__(self, label: str = 'target', stage: int = 2):
        """Plot fiducial and centerline fits."""
        if label == 'target':
            return self.target()
        if stage > 0:
            self.fiducial(label)
        if stage > 1:
            self.fiducial(f'{label}_target')
            self.centerline(label)

    def extract(self, data: xarray.Dataset):
        """Extract cartisean data and map to cylindrical coordinates."""
        self.data = xarray.Dataset()

        self.data['target'] = self.to_cylindrical(data.target)
        self.data['centerline'] = self.to_cylindrical(data.centerline)
        for attr in ['reference', 'fit']:
            self.data[attr] = self.to_cylindrical(data[attr]) - \
                    self.data.target
            for norm in ['target', 'centerline']:
                self.data[f'{attr}_{norm}'] = \
                    self.to_cylindrical(data[f'{attr}_{norm}']) - \
                    self.data[norm]

    @cached_property
    def axes(self):
        """Return axes instance."""
        axes = plt.subplots(1, 2, sharey=True,
                            gridspec_kw=dict(width_ratios=[3, 1]))[1]
        axes[0].set_xlabel('radius')
        axes[0].set_ylabel('height')
        axes[1].set_xlabel('toroidal')
        for i in range(2):
            axes[i].axis('equal')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.despine()
        return axes

    def to_cylindrical(self, data_array: xarray.DataArray) -> xarray.DataArray:
        """Retun dataarray in cylindrical coordinates."""
        cylindrical_data = data_array.copy().assign_coords(
            dict(space=['r', 'phi', 'z']))
        cylindrical_data[0] = Transform.clock.apply(data_array[0])
        cylindrical_data[1] = Transform.anticlock.apply(data_array[1])
        return cylindrical_data

    def plot_box(self, data_array: xarray.DataArray):
        """Plot bounding box around target."""

    def target(self):
        """Plot fiducial targets."""
        for i in range(2):
            self.axes[0].plot(self.data.centerline[i, :, 0],
                              self.data.centerline[i, :, 2],
                              '--', color='gray')
            self.axes[1].plot(self.data.centerline[i, :, 1],
                              self.data.centerline[i, :, 2],
                              '--', color='gray')
            self.axes[0].plot(self.data.target[i, :, 0],
                              self.data.target[i, :, 2], 'o', color='gray')
            self.axes[1].plot(self.data.target[i, :, 1],
                              self.data.target[i, :, 2], 'o', color='gray')
        if self.fiducial_labels:
            for radius, height, label in zip(
                    self.data.target[0, :, 0],
                    self.data.target[0, :, 2],
                    self.data.fiducial.values):
                self.axes[0].text(radius, height, f'{label} ',
                                  ha='right', va='center', color='gray',
                                  fontsize='x-large', zorder=-10)

    def delta(self, label: str):
        """Return displacment delta multiplied by scale factor."""
        return self.factor * self.data[f'{label}']

    def fiducial(self, label: str):
        """Plot fiducial deltas."""
        color = self.color.get(label, self.color[label.split('_')[0]])
        marker = self.marker[label]
        for i in range(2):
            delta = self.delta(label)
            self.axes[0].plot(self.data.target[i, :, 0] + delta[i, :, 0],
                              self.data.target[i, :, 2] + delta[i, :, 2],
                              color+marker)
            self.axes[1].plot(self.data.target[i, :, 1] + delta[i, :, 1],
                              self.data.target[i, :, 2] + delta[i, :, 2],
                              color+marker)

    def centerline(self, label: str):
        """Plot gpr centerline."""
        color = self.color[f'{label}_target']
        for i in range(2):
            delta = self.delta(f'{label}_centerline')
            self.axes[0].plot(self.data.centerline[i, :, 0] + delta[i, :, 0],
                              self.data.centerline[i, :, 2] + delta[i, :, 2],
                              color=color)
            self.axes[1].plot(self.data.centerline[i, :, 1] + delta[i, :, 1],
                              self.data.centerline[i, :, 2] + delta[i, :, 2],
                              color=color)


@dataclass
class SectorTransform:
    """Perform optimal sector transforms fiting fiducials to targets."""

    sector: int = 6
    infer: bool = True
    method: str = 'rms'
    variance: float = 1
    weights: list[float] = field(default_factory=lambda: [1, 1, 0.5])
    gpr: GaussianProcessRegressor = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    def __post_init__(self):
        """Load data."""
        self.load_reference()
        self.load_centerline()
        self.load_gpr()
        self.fit_reference()
        self.fit()

    def load_reference(self):
        """Load reference sector data."""
        self.data['reference'] = self.load_sector(self.sector)
        fiducial = FiducialData(fill=False).data.fiducial.drop(
            labels=['target_index']).rename(
                dict(target='fiducial')).sel(
                    fiducial=self.data.reference.fiducial)
        self.data.coords['target_length'] = \
            'fiducial', fiducial.target_length.values
        self.data['target'] = xarray.zeros_like(self.data.reference)
        self.data['target'][0] = Transform.anticlock.apply(fiducial)
        self.data['target'][1] = Transform.clock.apply(fiducial)
        '''
        self.data['reference'][0] = \
            Transform.anticlock.apply(self.data['reference'][0])
        self.data['reference'][1] = \
            Transform.clock.apply(self.data['reference'][1])
        '''
        self.data['reference'] += self.data['target']
        self.data = self.data.sortby('target_length')

    def load_centerline(self):
        """Load geodesic centerline."""
        centerline = CenterLine()
        self.data['arc_length'] = centerline.mesh['arc_length']
        self.data['nominal_centerline'] = \
            ('arc_length', 'space'), 1e3*centerline.mesh.points
        self.data['centerline'] = xarray.concat([
            self.data.nominal_centerline, self.data.nominal_centerline],
            dim='coil')
        self.data['centerline'][0] = \
            Transform.anticlock.apply(self.data['centerline'][0])
        self.data['centerline'][1] = \
            Transform.clock.apply(self.data['centerline'][1])

    def fit_reference(self):
        """Evaluate gpr for reference fiducials."""
        self.evaluate_gpr('reference', self.data.reference)

    def evaluate_gpr(self, label: str, data_array: xarray.DataArray):
        """Evaluate gpr."""
        delta = data_array - self.data.target
        target = f'{label}_target'
        centerline = f'{label}_centerline'
        self.data[target] = xarray.zeros_like(self.data.target)
        self.data[centerline] = xarray.zeros_like(self.data.centerline)
        for coil_index in range(self.data.dims['coil']):
            for space_index in range(self.data.dims['space']):
                self.gpr.fit(delta[coil_index, :, space_index])
                self.data[target][coil_index, :, space_index] = \
                    self.gpr.predict(self.data.target_length)
                self.data[centerline][coil_index, :, space_index] = \
                    self.gpr.predict(self.data.arc_length)
        self.data[f'{label}_target'] += self.data.target
        self.data[f'{label}_centerline'] += self.data.centerline

    def load_gpr(self):
        """Load gaussian process regressor."""
        self.gpr = GaussianProcessRegressor(self.data.target_length,
                                            self.variance)

    @property
    def points(self):
        """Return reference points."""
        if self.infer:  # use gpr inference
            return self.data.reference_target.copy()
        return self.data.reference.copy()

    def transform(self, x, points=None) -> xarray.DataArray:
        """Return transformed sector."""
        if points is None:
            points = self.points
        points[:] += x[:3]
        if len(x) == 6:
            rotate = Rotation.from_euler('xyz', x[-3:], degrees=True)
            for i in range(2):
                points[i] = rotate.apply(points[i])
        return points

    def delta(self, points):
        """Return coil-frame deltas."""
        delta = points - self.data['target']
        delta[0] = Transform.clock.apply(delta[0])
        delta[1] = Transform.anticlock.apply(delta[1])
        return delta

    @staticmethod
    def error_vector(delta, method='rms'):
        """Return error vector."""
        error = np.zeros(3)
        if method == 'rms':
            error[0] = np.mean(delta[:, [5, 3, 4], 0]**2)
            error[1] = np.mean(delta[..., 1]**2)
            error[2] = np.mean(delta[:, [2, 1, -1, -2], 2]**2)
            return error
        error[0] = np.max(abs(delta[:, [5, 3, 4], 0]))  # radial (A, B, H)
        error[1] = np.max(abs(delta[..., 1]))  # toroidal (all)
        error[2] = 0.5*np.max(abs(delta[:, [2, 1, -1, -2], 2]))  # (C, D, E, F)
        return error

    def transform_error(self, x, points=None, method=None):
        """Return transform error vector."""
        if method is None:
            method = self.method
        points = self.transform(x, points)
        return self.point_error(points, method)

    def weighted_transform_error(self, x, points=None, method=None):
        """Return weighted transform error vector."""
        return self.transform_error(x, points, method='max') * self.weights

    def point_error(self, points, method=None):
        """Return error vector."""
        if method is None:
            method = self.method
        delta = self.delta(points)
        return self.error_vector(delta, method)

    def max_transform_error(self, x, points=None):
        """Return maximum error."""
        return np.max(self.weighted_transform_error(x, points, method='max'))

    def rms_transform_error(self, x, points=None):
        """Return mean error."""
        return np.sqrt(np.mean(
            self.weighted_transform_error(x, points, method='rms')))

    def scalar_error(self, x):
        """Return scalar mesure for fit error."""
        return getattr(self, f'{self.method}_transform_error')(x)

    def fit(self):
        """Perform sector fit."""
        xo = np.zeros(6)
        opp = minimize(self.scalar_error, xo, method='SLSQP')
        if not opp.success:
            warnings.warn('optimization failed')
        self.data['fit'] = self.transform(opp.x)
        self.data['opp_x'] = 'transform', opp.x
        self.data['error'] = 'space', self.transform_error(opp.x)
        self.evaluate_gpr('fit', self.data.fit)

    def load_sector(self, sector: int) -> xarray.DataArray:
        """Return sector data."""
        match sector:
            case 6:
                coils = [12, 13]
                data = [[[0.27, -0.7, 2.04],
                         [-1.43, -0.2, 0.34],
                         [-2.02, 2.02, 1.46],
                         [-3.33, 0.71, -1.15],
                         [-4.64, 0.73, 0.97],
                         [-0.99, 0.87, 0.5],
                         [-5.02, -0.87, -0.01],
                         [-0.22, 0.01, 0.43]],
                        [[0.7, 1.98, -0.35],
                         [-1.33, 1., -1.29],
                         [-4.5, 2.96, 1.57],
                         [-2.28, 2.69, -1.85],
                         [-5.54, 0.87, -1.63],
                         [1.44, 1.54, -0.81],
                         [-4.04, 2.5, -2.2],
                         [0.33, -0.15, -1.32]]]
            case 26:  # inital allignment data
                coils = [12, 13]
                data = [[[0.4, -0.6, 2.1],
                         [-1.4, -0.5, 0.3],
                         [-2.3, 1.6, 1.5],
                         [-3.4, 0.1, -1.2],
                         [-4.7, -0.1, 1.0],
                         [-4.8, -1.7, 0.0],
                         [-1.1, 0.7, 0.5],
                         [-0.2, 0.0, 0.4]],
                        [[1.0, 1.8, -0.3],
                         [-1.1, 1.2, -1.3],
                         [-3.9, 3.7, 1.6],
                         [-1.8, 3.0, -1.9],
                         [-5.3, 1.8, -1.6],
                         [-3.5, 3.2, -2.2],
                         [1.7, 1.3, -0.8],
                         [0.3, -0.2, -1.3]]]
            case _:
                raise NotImplementedError(f'sector {sector} not specified')

        return xarray.DataArray(data, dims=('coil', 'fiducial', 'space'),
                                coords=dict(coil=coils,
                                fiducial=list('ABCDEFGH'),
                                space=list('xyz')))

    def plot(self, label: str):
        """Plot fits."""
        plotter = Plotter(self.data)
        plotter('target')
        if label != 'target':
            stage = 1 + int(self.infer)
            plotter(label, stage)
            plotter.axes[0].set_title(label)
            self.text_fit(plotter.axes[0], label)
        # plt.savefig('fit.png')

    def text_transform(self, axes):
        """Display text transform."""
        opp_x = self.data.opp_x.values
        deg_to_mm = 10570*np.pi/180
        axes.text(0.3, 0.5,
                  f'dx: {opp_x[0]:1.2f}mm\n' +
                  f'dy: {opp_x[1]:1.2f}mm\n' +
                  f'dz: {opp_x[2]:1.2f}mm\n' +
                  f'rx: {opp_x[3]*deg_to_mm:1.2f}mm\n' +
                  f'ry: {opp_x[4]*deg_to_mm:1.2f}mm\n' +
                  f'rz: {opp_x[5]*deg_to_mm:1.2f}',
                  va='center', ha='left',
                  transform=axes.transAxes)

    def fit_error(self, method: str):
        """Return fit error vector."""
        return self.transform_error(self.data.opp_x.values,
                                    self.data.reference.copy(), method)

    def reference_error(self, method: str):
        """Return reference error vector."""
        return self.point_error(self.data.reference, method)

    def text_fit(self, axes, label: str):
        """Display text transform."""
        error_vector = getattr(self, f'{label}_error')
        error = dict(rms=np.sqrt(error_vector('rms')),
                     max=error_vector('max'))
        text = ''
        for i, coordinate in enumerate(['radial: A,B,H', 'toroidal: all',
                                        'vertical: C,D,E,F']):
            text += '\n' + coordinate + '\n'
            for method in ['rms', 'max']:
                text += f'    {method}: {error[method][i]:1.2f}\n'
        axes.text(0.3, 0.5, text,
                  va='center', ha='left',
                  transform=axes.transAxes, fontsize='xx-small')


if __name__ == '__main__':

    transform = SectorTransform(6, True)
    #transform.plot('target')
    transform.plot('reference')
    transform.plot('fit')
