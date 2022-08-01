"""Perform fit for SM6 to fiducial mesurments."""

from dataclasses import dataclass, field
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import xarray

from nova.assembly.centerline import CenterLine
from nova.assembly.fiducialdata import FiducialData
from nova.assembly.gaussianprocessregressor import GaussianProcessRegressor
from nova.utilities.pyplot import plt


@dataclass
class SectorTransform:
    """Perform optimal sector transforms fiting fiducials to targets."""

    infer: bool = True
    variance: float = 1
    gpr: GaussianProcessRegressor = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)
    clock: Rotation = Rotation.from_euler('z', 10, degrees=True)
    anticlock: Rotation = Rotation.from_euler('z', -10, degrees=True)

    def __post_init__(self):
        """Load data."""
        self.load_reference()
        self.load_centerline()
        self.load_gpr()
        self.fit_reference()

    def load_reference(self):
        """Load reference sector data."""
        self.data['reference'] = self.load_sector(6)
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
        error = np.zeros(3)
        error[0] = np.max(abs(delta[:, [5, 3, 4], 0]))  # radial (A, B, H)
        error[1] = np.max(abs(delta[..., 1]))  # toroidal (all)
        error[2] = 0.5*np.max(abs(delta[:, [2, 1, -1, -2]]))  # (C, D, E, F)
        return error

    def error(self, x):
        """Return fit error."""
        points = self.transform(x)
        delta = self.delta(points)
        return self.error_vector(delta)

    def max_error(self, x):
        """Return maximum absolute error."""
        return np.max(self.error(x))

    def fit(self):
        """Perform sector fit."""
        xo = np.zeros(6)
        opp = minimize(self.max_error, xo, method='SLSQP')
        self.data['fit'] = self.transform(opp.x)
        self.data['opp_x'] = 'tansform', opp.x
        self.data['error'] = 'space', self.error(opp.x)
        self.fit_gpr('fit', self.data.fit)

    def load_sector(self, sector: int) -> xarray.DataArray:
        """Return sector data."""
        match sector:
            case 6:
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

    def to_cylindrical(self, data_array: xarray.DataArray) -> xarray.DataArray:
        """Retun dataarray in cylindrical coordinates."""
        cylindrical_data = data_array.copy().assign_coords(
            dict(space=['r', 'phi', 'z']))
        cylindrical_data[0] = self.clock.apply(data_array[0])
        cylindrical_data[1] = self.anticlock.apply(data_array[1])
        return cylindrical_data

    def plot_box(self, data_array: xarray.DataArray):
        """Plot bounding box around target."""

    def plot_target(self, axes, target, centerline):
        """Plot fiducial targets."""
        for i in range(2):
            axes[0].plot(centerline[i, :, 0], centerline[i, :, 2],
                         '--', color='gray')
            axes[1].plot(centerline[i, :, 1], centerline[i, :, 2],
                         '--', color='gray')
            axes[0].plot(target[i, :, 0], target[i, :, 2], 'o', color='gray')
            axes[1].plot(target[i, :, 1], target[i, :, 2], 'o', color='gray')

    def plot_fiducial(self, axes, target, factor, delta, marker, color):
        """Plot fiducial deltas."""
        for i in range(2):
            axes[0].scatter(target[i, :, 0] + factor*delta[i, :, 0],
                            target[i, :, 2] + factor*delta[i, :, 2],
                            marker=marker, color=color)
            axes[1].scatter(target[i, :, 1] + factor*delta[i, :, 1],
                            target[i, :, 2] + factor*delta[i, :, 2],
                            marker=marker, color=color)

    def plot_centerline(self, axes, target, factor, delta, color):
        """Plot gpr centerline."""
        for i in range(2):
            axes[0].plot(target[i, :, 0] + factor*delta[i, :, 0],
                         target[i, :, 2] + factor*delta[i, :, 2],
                         color=color)
            axes[1].plot(target[i, :, 1] + factor*delta[i, :, 1],
                         target[i, :, 2] + factor*delta[i, :, 2],
                         color=color)

    def plot(self, factor: float = 500):
        """Plot fidicual to target fit in cylindrical coordinates."""
        target = self.to_cylindrical(self.data.target)
        centerline = self.to_cylindrical(self.data.centerline)
        reference = self.to_cylindrical(self.data.reference) - target
        reference_centerline = \
            self.to_cylindrical(self.data.reference_centerline) - centerline
        fit = self.to_cylindrical(self.data.fit) - target
        fit_centerline = \
            self.to_cylindrical(self.data.fit_centerline) - centerline
        axes = plt.subplots(1, 2, sharey=True,
                            gridspec_kw=dict(width_ratios=[3, 1]))[1]
        self.plot_target(axes, target, centerline)
        self.plot_fiducial(axes, target, factor, reference, 'd', 'C0')
        self.plot_centerline(axes, centerline, factor,
                             reference_centerline, 'C0')

        self.plot_fiducial(axes, target, factor, fit, 's', 'C1')
        self.plot_centerline(axes, centerline, factor, fit_centerline, 'C1')

        for _radius, _height, label in zip(target[0, :, 0], target[0, :, 2],
                                           self.data.fiducial.values):
            axes[0].text(_radius, _height, f'{label} ', ha='right',
                         color='gray')
        axes[0].set_xlabel('radius')
        axes[0].set_ylabel('height')
        axes[1].set_xlabel('toroidal')

        reference_error = np.max(
            self.error_vector(self.delta(self.data.reference)))
        minmax_error = np.max(abs(self.data.error.values))
        legend = [Line2D([0], [0], markerfacecolor='C0', marker='d',
                         color='w',
                         label=f'reference {reference_error:1.1f}mm'),
                  Line2D([0], [0], markerfacecolor='C1', marker='s',
                         color='w',
                         label=f'max error {minmax_error:1.1f}mm')]
        axes[0].legend(handles=legend, bbox_to_anchor=[1, 1.1], ncol=2)
        opp_x = self.data.opp_x.values
        deg_to_mm = 10570*np.pi/180
        axes[0].text(0.35, 0.5,
                     f'dx: {opp_x[0]:1.2f}mm\n' +
                     f'dy: {opp_x[1]:1.2f}mm\n' +
                     f'dz: {opp_x[2]:1.2f}mm\n' +
                     f'rx: {opp_x[3]*deg_to_mm:1.2f}mm\n' +
                     f'ry: {opp_x[4]*deg_to_mm:1.2f}mm\n' +
                     f'rz: {opp_x[5]*deg_to_mm:1.2f}',
                     va='center', ha='left',
                     transform=axes[0].transAxes)
        for i in range(2):
            axes[i].axis('equal')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.despine()


if __name__ == '__main__':

    transform = SectorTransform(True)
    transform.fit()
    transform.plot()
