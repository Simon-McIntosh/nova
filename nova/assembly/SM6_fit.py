"""Perform fit for SM6 to fiducial mesurments."""

from dataclasses import dataclass, field, InitVar
from functools import cached_property
from typing import ClassVar

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
    variance: float = 1
    gpr: GaussianProcessRegressor = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)
    plotter: Plotter = field(init=False, repr=False)
    clock: Rotation = Transform.clock
    anticlock: Rotation = Transform.anticlock

    def __post_init__(self):
        """Load data."""
        self.load_reference()
        self.load_centerline()
        self.load_gpr()
        self.fit_reference()
        self.fit()
        self.plotter = Plotter(self.data)

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
        self.data['target'][0] = self.anticlock.apply(fiducial)
        self.data['target'][1] = self.clock.apply(fiducial)
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
            self.anticlock.apply(self.data['centerline'][0])
        self.data['centerline'][1] = \
            self.clock.apply(self.data['centerline'][1])

    def fit_reference(self):
        """Evaluate gpr for reference fiducials."""
        self.fit_gpr('reference', self.data.reference)

    def fit_gpr(self, label: str, data_array: xarray.DataArray):
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

    def transform(self, x) -> xarray.DataArray:
        """Return transformed sector."""
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
        delta[0] = self.clock.apply(delta[0])
        delta[1] = self.anticlock.apply(delta[1])
        return delta

    @staticmethod
    def error_vector(delta):
        """Return error vector."""
        '''
        error = np.zeros(3)
        error[0] = np.max(abs(delta[:, [5, 3, 4], 0]))  # radial (A, B, H)
        error[1] = np.max(abs(delta[..., 1]))  # toroidal (all)
        error[2] = 0.5*np.max(abs(delta[:, [2, 1, -1, -2], 2]))  # (C, D, E, F)
        '''

        error = np.zeros(3)
        error[0] = np.mean(delta[:, [5, 3, 4], 0]**2)
        error[1] = np.mean(delta[..., 1]**2)
        error[2] = np.mean(delta[:, [2, 1, -1, -2], 2]**2)
        return error

    def error(self, x):
        """Return fit error."""
        points = self.transform(x)
        delta = self.delta(points)
        return self.error_vector(delta)

    def max_error(self, x):
        """Return maximum error."""
        return np.max(self.error(x))

    def mean_error(self, x):
        """Return mean error."""
        return np.mean(self.error(x))

    def fit(self):
        """Perform sector fit."""
        xo = np.zeros(6)
        opp = minimize(self.mean_error, xo, method='SLSQP')
        self.data['fit'] = self.transform(opp.x)
        self.data['opp_x'] = 'tansform', opp.x
        self.data['error'] = 'space', self.error(opp.x)
        self.fit_gpr('fit', self.data.fit)

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

    def reference_error(self, stage: int):
        """Return estimate for maximum reference error."""
        if stage == 2:
            points = self.data.reference_target
        else:
            points = self.data.reference
        return np.max(self.error_vector(self.delta(points)))

    def plot(self, label: str):
        """Plot fits."""
        self.plotter('target')
        if label != 'target':
            stage = 1 + int(self.infer)
            self.plotter(label, stage)
            match label:
                case 'reference':
                    error = self.reference_error(stage)
                case 'fit':
                    error = np.max(self.data.error.values)
            self.plotter.axes[0].set_title(
                f'{label} {error:1.2f}mm (infer: {self.infer})')
            #if label == 'fit' and stage == 2:
            self.text_transform(self.plotter.axes[0])
        plt.savefig('fit.png')

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


if __name__ == '__main__':

    transform = SectorTransform(6, True)
    #transform.plot('target')
    #transform.plot('reference')
    transform.plot('fit')
