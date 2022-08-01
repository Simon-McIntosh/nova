"""Perform fit for SM6 to fiducial mesurments."""

from dataclasses import dataclass, field
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import xarray

from nova.assembly.fiducialdata import FiducialData
from nova.utilities.pyplot import plt


@dataclass
class SectorTransform:
    """Perform optimal sector transforms fiting fiducials to targets."""

    data: xarray.Dataset = field(init=False, repr=False,
                                 default_factory=xarray.Dataset)
    clock: Rotation = Rotation.from_euler('z', 10, degrees=True)
    anticlock: Rotation = Rotation.from_euler('z', -10, degrees=True)

    def __post_init__(self):
        """Load sector data."""
        self.data['reference'] = self.load_sector(6)
        fiducial = FiducialData(fill=False).data.fiducial.drop(
            labels=['target_index', 'target_length']).rename(
                dict(target='fiducial')).sel(
                    fiducial=self.data.reference.fiducial)
        self.data['target'] = xarray.zeros_like(self.data.reference)
        self.data['target'][0] = self.anticlock.apply(fiducial)
        self.data['target'][1] = self.clock.apply(fiducial)

        self.data['reference'][0] = \
            self.anticlock.apply(self.data['reference'][0])
        self.data['reference'][1] = \
            self.clock.apply(self.data['reference'][1])
        self.data['reference'] += self.data['target']

    def transform(self, x) -> xarray.DataArray:
        """Return transformed sector."""
        points = self.data.reference.copy()
        points[:] += x[:3]
        if len(x) == 6:
            rotate = Rotation.from_euler('xyz', x[-3:], degrees=True)
            for i in range(2):
                points[i] = rotate.apply(points[i])
        return points

    def delta(self, x):
        """Return coil-frame deltas."""
        delta = self.transform(x) - self.data['target']
        delta[0] = self.clock.apply(delta[0])
        delta[1] = self.anticlock.apply(delta[1])
        return delta

    def error(self, x):
        """Return fit error."""
        delta = self.delta(x)
        error = np.zeros(3)
        error[0] = np.max(abs(delta[:, [0, 1, -1], 0]))  # radial fit (A, B, H)
        error[1] = np.max(abs(delta[..., 1]))  # toroidal fit
        error[2] = 0.5*np.max(abs(delta[:, 3:-1, 2]))  # vertical fit
        return error

    def max_error(self, x):
        """Return maximum absolute error."""
        error = self.error(x)
        return np.max(abs(error))

    def fit(self):
        """Perform sector fit."""
        xo = np.zeros(6)
        opp = minimize(self.max_error, xo, method='SLSQP')
        delta = self.transform(opp.x) - self.data['target']
        self.data['fit'] = xarray.zeros_like(self.data.reference)
        self.data['fit'][0] = self.anticlock.apply(delta[0])
        self.data['fit'][1] = self.clock.apply(delta[1])
        self.data['fit'] += self.data['target']
        self.data['opp_x'] = 'tansform', opp.x
        self.data['error'] = 'space', self.error(opp.x)

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

    def plot(self, factor: float = 250):
        """Plot fidicual to target fit."""
        reference = self.data.reference - self.data.target
        fit = self.data.fit - self.data.target
        radius = np.linalg.norm(self.data.target[..., :2], axis=-1)

        toroidal = xarray.zeros_like(self.data.target[..., 1])
        toroidal[0] = self.clock.apply(self.data.target[0])[:, 1]
        toroidal[1] = self.anticlock.apply(self.data.target[1])[:, 1]
        height = self.data.target[..., 2]

        axes = plt.subplots(1, 2, sharey=True,
                            gridspec_kw=dict(width_ratios=[3, 1]))[1]
        for i in range(2):

            axes[0].plot(radius[i], height[i], 'o', color='gray')
            axes[0].plot(radius[i] +
                         factor*np.linalg.norm(reference[i, :, :2], axis=-1),
                         height[i] + factor*reference[i, :, 2], 'C0d')
            axes[0].plot(radius[i] +
                         factor*np.linalg.norm(fit[i, :, :2], axis=-1),
                         height[i] + factor*fit[i, :, 2], 'C1s')

            axes[1].plot(toroidal[i], height[i], 'o', color='gray')
            axes[1].plot(toroidal[i] +
                         factor*Rotation.from_euler(
                             'x', 10 - i*20, degrees=True).apply(
                             reference[i])[:, 1],
                         height[i] + factor*reference[i, :, 2], 'C0d')
            axes[1].plot(toroidal[i] +
                         factor*Rotation.from_euler(
                             'x', 10 - i*20, degrees=True).apply(
                                 fit[i])[:, 1],
                         height[i] + factor*fit[i, :, 2], 'C1s')
        for _radius, _height, label in zip(radius[0], height[0],
                                           self.data.fiducial.values):
            axes[0].text(_radius, _height, f'{label} ', ha='right',
                         color='gray')
        axes[0].set_xlabel('radius')
        axes[0].set_ylabel('height')
        axes[1].set_xlabel('toroidal')
        minmax_error = np.max(abs(self.data.error.values))

        legend = [Line2D([0], [0], markerfacecolor='C0', marker='d',
                         color='w', label='lstsq 3.7mm'),
                  Line2D([0], [0], markerfacecolor='C1', marker='s',
                         color='w',
                         label=f'min max error {minmax_error:1.1f}mm')]
        axes[0].legend(handles=legend, bbox_to_anchor=[1, 1.1], ncol=2)

        for i in range(2):
            axes[i].axis('equal')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.despine()


if __name__ == '__main__':

    transform = SectorTransform()
    transform.fit()
    transform.plot()
