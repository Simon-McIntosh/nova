"""Perform fit for SM6 to fiducial mesurments."""

from dataclasses import dataclass, field, InitVar
from functools import cached_property
from typing import ClassVar
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import xarray

from nova.assembly.centerline import CenterLine
from nova.assembly.fiducialdata import FiducialData
from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor
from nova.assembly.spacialanalyzer import SpacialAnalyzer
from nova.assembly.transform import Rotate


@dataclass
class Plotter:
    """Plot fidicual to target fit in cylindrical coordinates."""

    cartisean_data: InitVar[xarray.Dataset]
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)
    factor: float = 750
    fiducial_labels: bool = True
    rotate: Rotate = field(init=False, default_factory=Rotate)

    color: ClassVar[dict[str, str]] = dict(
        fit='C1', fit_target='C0', reference='C4', reference_target='C6')
    marker: ClassVar[dict[str, str]] = dict(
        fit='X', fit_target='d', reference='X', reference_target='d')

    def __post_init__(self, data: xarray.Dataset):
        """Transform data to cylindrical coordinates."""
        self.extract(data)

    def __call__(self, label: str = 'target', stage: int = 2,
                 coil_index=range(2)):
        """Plot fiducial and centerline fits."""
        if label == 'target':
            return self.target()
        if stage > 0:
            self.fiducial(label, coil_index=coil_index)
        if stage > 1:
            self.fiducial(f'{label}_target', coil_index=coil_index)
            self.centerline(label, coil_index=coil_index)

    def extract(self, cartisean_data: xarray.Dataset):
        """Extract cartisean data and map to cylindrical coordinates."""
        self.data = xarray.Dataset()
        self.data['target'] = \
            self.rotate.to_cylindrical(cartisean_data.target)
        self.data['centerline'] = \
            self.rotate.to_cylindrical(cartisean_data.centerline)
        for attr in ['reference', 'fit']:
            if attr not in cartisean_data:
                continue
            self.data[attr] = \
                self.rotate.to_cylindrical(cartisean_data[attr]) - \
                self.data.target
            for norm in ['target', 'centerline']:
                self.data[f'{attr}_{norm}'] = \
                    self.rotate.to_cylindrical(
                        cartisean_data[f'{attr}_{norm}']) - \
                    self.data[norm]
            self.data[f'{attr}_centerline_sample'] = \
                cartisean_data[f'{attr}_centerline_sample'] - \
                self.data['centerline']
        self.data.target[..., 1] = 0
        self.data.centerline[..., 1] = 0

    @cached_property
    def axes(self):
        """Return axes instance."""
        axes = plt.subplots(1, 2, sharey=True,
                            gridspec_kw=dict(width_ratios=[2.5, 1]))[1]
        axes[0].set_xlabel('radius')
        axes[0].set_ylabel('height')
        axes[1].set_xlabel('toroidal')
        for i in range(2):
            axes[i].axis('equal')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            sns.despine()
        return axes

    def plot_box(self, data_array: xarray.DataArray):
        """Plot bounding box around target."""

    def target(self, coil_index=range(2)):
        """Plot fiducial targets."""
        for i in coil_index:
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
        self.axes[0].plot([800, 12000], [-8000, 8000], 'w.')
        self.axes[1].plot([-2500, 2500], [-8000, 8000], 'w.')

    def delta(self, label: str):
        """Return displacment delta multiplied by scale factor."""
        return self.factor * self.data[f'{label}']

    def fiducial(self, label: str, coil_index=range(2)):
        """Plot fiducial deltas."""
        color = self.color.get(label, self.color[label.split('_')[0]])
        marker = self.marker[label]
        for i in coil_index:
            delta = self.delta(label)
            self.axes[0].plot(self.data.target[i, :, 0] + delta[i, :, 0],
                              self.data.target[i, :, 2] + delta[i, :, 2],
                              color+marker)
            self.axes[1].plot(self.data.target[i, :, 1] + delta[i, :, 1],
                              self.data.target[i, :, 2] + delta[i, :, 2],
                              color+marker)

    def centerline(self, label: str, coil_index=range(2), samples=True):
        """Plot gpr centerline."""
        color = self.color[f'{label}_target']
        attr = f'{label}_centerline'
        if samples:
            attr += '_sample'
        for i in coil_index:
            delta = self.delta(attr)
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
    n_samples: int = 10
    files: dict[str, str] = field(default_factory=dict)
    gpr: GaussianProcessRegressor = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)

    variance: ClassVar[float] = 1
    weights: ClassVar[list[float]] = [1, 1, 0.5]

    def __post_init__(self):
        """Load data."""
        self.rotate = Rotate()
        self.spacial_analyzer = SpacialAnalyzer(
            sector=self.sector, files=self.files)
        self.load_reference()
        self.load_centerline()
        self.load_gpr()
        self.fit_reference()
        self.fit()

    def load_reference(self):
        """Load reference sector data."""
        self.data['reference'] = self.spacial_analyzer.reference_ccl
        fiducial = FiducialData(fill=False).data.fiducial.drop(
            labels=['target_index']).rename(
                dict(target='fiducial')).sel(
                    fiducial=self.data.reference.fiducial)
        self.data['target_length'] = fiducial.target_length
        self.data['target'] = self.spacial_analyzer.nominal_ccl
        self.data['target_cylindrical'] = \
            self.rotate.to_cylindrical(self.data.target)
        self.data = self.data.sortby('target_length')

    def load_centerline(self):
        """Load geodesic centerline."""
        centerline = CenterLine()
        self.data['arc_length'] = centerline.mesh['arc_length']
        self.data['nominal_centerline'] = \
            ('arc_length', 'cartesian'), 1e3*centerline.mesh.points
        self.data['centerline'] = xarray.concat([
            self.data.nominal_centerline, self.data.nominal_centerline],
            dim='coil')
        self.data['centerline'][0] = \
            self.rotate.anticlock(self.data['centerline'][0])
        self.data['centerline'][1] = \
            self.rotate.clock(self.data['centerline'][1])
        self.data['centerline_cylindrical'] = self.rotate.to_cylindrical(
                self.data.centerline)

    def fit_reference(self):
        """Evaluate gpr for reference fiducials."""
        self.evaluate_gpr('reference', self.data.reference)

    def evaluate_gpr(self, label: str, data_array: xarray.DataArray):
        """Evaluate gpr in cylindrical coordinate system."""
        delta = self.rotate.to_cylindrical(data_array) - \
            self.data.target_cylindrical
        target = f'{label}_target'
        centerline = f'{label}_centerline'
        sample = f'{label}_centerline_sample'
        self.data[target] = xarray.zeros_like(self.data.target_cylindrical)
        self.data[centerline] = \
            xarray.zeros_like(self.data.centerline_cylindrical)
        self.data[sample] = \
            xarray.zeros_like(self.data[centerline]).expand_dims(
                dict(samples=self.n_samples), axis=-1).copy()
        for coil_index in range(self.data.dims['coil']):
            for space_index in range(self.data.dims['cylindrical']):
                self.gpr.fit(delta[coil_index, :, space_index])
                self.data[target][coil_index, :, space_index] = \
                    self.gpr.predict(self.data.target_length)
                self.data[centerline][coil_index, :, space_index] = \
                    self.gpr.predict(self.data.arc_length)
                self.data[sample][coil_index, :, space_index, :] = \
                    self.gpr.sample(self.data.arc_length, self.n_samples)
        self.data[target] += self.data.target_cylindrical
        self.data[centerline] += self.data.centerline_cylindrical
        self.data[sample] += self.data.centerline_cylindrical
        for attr in [target, centerline]:
            self.data[attr] = self.rotate.to_cartesian(self.data[attr])

    def plot_gpr(self, label: str, coil_index: int, n_samples=10):
        """Load gaussian process regressor."""
        delta = self.rotate.to_cylindrical(self.data[label]) - \
            self.data.target_cylindrical
        axes = plt.subplots(3, 1, sharex=True)[1]
        for space_index in range(self.data.dims['cylindrical']):
            self.gpr.fit(delta[coil_index, :, space_index])
            self.gpr.predict(self.data.arc_length)
            self.gpr.plot(axes[space_index], text=False,
                          marker='X', color='C1', line_color='C0')
            if n_samples > 0:
                self.gpr.sample(self.data.arc_length, n_samples)
                self.gpr.plot_samples(axes[space_index])

            self.gpr.predict(self.data.target_length)
            axes[space_index].plot(self.data.target_length,
                                   self.gpr.data.y_mean, 'dC0')
            coord = str(self.data.cylindrical[space_index].values)
            coord = coord.replace('phi', r'\phi')
            axes[space_index].set_ylabel(rf'${coord}$')
        sns.despine()
        axes[-1].set_xlabel('arc length')
        axes[0].set_title(f'TF{self.data.coil[coil_index].values:1d}')
        plt.tight_layout()
        plt.savefig('gpr.png')

    def plot_coil(self, label: str, coil_index: int, n_samples=10):
        """Plot single coil."""
        plotter = Plotter(self.data)
        plotter(label, 2, coil_index=[coil_index])
        if n_samples > 0:
            self.gpr.sample(self.data.arc_length, n_samples)
            plotter.axis[0].plot()
        plotter.axes[0].set_title(f'TF{self.data.coil[coil_index].values:d}')
        plt.savefig('coil.png')

    def load_gpr(self):
        """Load gaussian process regressor."""
        self.gpr = GaussianProcessRegressor(self.data.target_length,
                                            self.variance)

    @property
    def points(self):
        """Return reference points."""
        if self.infer:  # use gpr inference
            return self.data['reference_target'].copy()
        return self.data['reference'].copy()

    def transform(self, x, points) -> xarray.DataArray:
        """Return transformed sector."""
        if points is None:
            points = self.points
        points = points[:] + x[:3]
        if len(x) == 6:
            rotate = Rotation.from_euler('XYZ', x[-3:], degrees=True)
            for i in range(2):
                points[i] = rotate.apply(points[i])
        return points

    def delta(self, points):
        """Return coil-frame deltas."""
        return self.rotate.to_cylindrical(points) - \
            self.data.target_cylindrical

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
        error[2] = np.max(abs(delta[:, [2, 1, -1, -2], 2]))  # (C, D, E, F)
        return error

    def transform_error(self, x, points, method=None):
        """Return transform error vector."""
        if method is None:
            method = self.method
        points = self.transform(x, points)
        return self.point_error(points, method)

    def weighted_transform_error(self, x, points, method=None):
        """Return weighted transform error vector."""
        return self.transform_error(x, points, method='max') * self.weights

    def point_error(self, points, method=None):
        """Return error vector."""
        if method is None:
            method = self.method
        delta = self.delta(points)
        return self.error_vector(delta, method)

    def max_transform_error(self, x, points):
        """Return maximum error."""
        return np.max(self.weighted_transform_error(x, points, method='max'))

    def rms_transform_error(self, x, points):
        """Return mean error."""
        return np.sqrt(np.mean(
            self.weighted_transform_error(x, points, method='rms')))

    def scalar_error(self, x, points):
        """Return scalar mesure for fit error."""
        return getattr(self, f'{self.method}_transform_error')(x, points)

    def fit(self):
        """Perform sector fit."""
        xo = np.ones(6)
        opt = minimize(self.scalar_error, xo, method='SLSQP', args=(None,))
        if not opt.success:
            warnings.warn(f'optimization failed {opt}')
        self.data['opt_x'] = 'transform', opt.x
        self.data['fit'] = self.transform(opt.x, self.data.reference.copy())
        self.evaluate_gpr('fit', self.data.fit)

    def plot(self, label: str, postfix=''):
        """Plot fits."""
        plotter = Plotter(self.data)
        plotter('target')
        if label != 'target':
            stage = 1 + int(self.infer)
            stage = 2
            plotter(label, stage)
            plotter.axes[0].set_title(label + postfix)
            self.text_fit(plotter.axes[0], label)
        plt.tight_layout()
        plt.savefig('fit.png')

    def plot_transform(self):
        """Plot transform text."""
        plotter = Plotter(self.data)
        plotter('target')
        self.text_transform(plotter.axes[0])
        plotter.axes[0].set_title('transform: reference -> fit')
        plt.tight_layout()
        # plt.savefig('fit.png')

    def text_transform(self, axes):
        """Display text transform."""
        opt_x = self.data.opt_x.values
        deg_to_mm = 1  # 10570*np.pi/180
        axes.text(0.8, 0.5,
                  f'dx: {opt_x[0]:1.2f}mm\n' +
                  f'dy: {opt_x[1]:1.2f}mm\n' +
                  f'dz: {opt_x[2]:1.2f}mm\n' +
                  f'rx: {opt_x[3]*deg_to_mm:1.2e}' + r'$^o$' + '\n' +
                  f'ry: {opt_x[4]*deg_to_mm:1.2e}' + r'$^o$' + '\n' +
                  f'rz: {opt_x[5]*deg_to_mm:1.2e}' + r'$^o$',
                  va='center', ha='left',
                  transform=axes.transAxes, fontsize='x-small')

    def fit_error(self, method: str):
        """Return fit error vector."""
        return self.transform_error(self.data.opt_x.values,
                                    self.data.reference.copy(), method)

    def reference_error(self, method: str):
        """Return reference error vector."""
        return self.point_error(self.data.reference.copy(), method)

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
        axes.text(0.8, 0.5, text,
                  va='center', ha='left',
                  transform=axes.transAxes, fontsize='x-small')

    def write(self):
        """Write fit to file."""
        fit_ccl = self.data.fit
        fit_ccl.attrs['group'] = 'fit_ccl'
        data = [fit_ccl]
        try:
            fit = self.transform(self.data.opt_x.values,
                                 self.spacial_analyzer.nominal)
            fit.attrs['group'] = 'fit'
            data.append(fit)
        except FileNotFoundError:
            pass
        self.spacial_analyzer.write(*data)


if __name__ == '__main__':

    transform = SectorTransform(6, True, method='rms', n_samples=5,
                                files=dict(reference_ccl='reference_ccl'))
    #transform.plot('target')
    #transform.plot('reference')
    #transform.plot('fit')
    #transform.plot_transform()
    #transform.write()

    transform.plot_gpr('reference', 0, n_samples=5)
    #transform.plot_gpr('reference', 1)
