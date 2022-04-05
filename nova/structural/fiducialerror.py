"""Calculate placement error from fiducial data."""
from dataclasses import dataclass, field

from collections.abc import Iterable
import numpy as np
import scipy.fft
import scipy.optimize
import scipy.spatial.transform
import sklearn.neighbors
import xarray

from nova.structural.fiducialdata import FiducialData
from nova.utilities.pyplot import plt


@dataclass
class FiducialError:
    """Manage fiducial error estimates."""

    data: xarray.Dataset
    radial_weight: float = -1.
    kdtree: sklearn.neighbors.KDTree = field(init=False)

    def __post_init__(self):
        """Init fit dataarray - perform coilset fit."""
        self.kdtree = sklearn.neighbors.KDTree(self.data.centerline[:-1, :])
        self.initialize_fit()

    def initialize_fit(self):
        """Initialise fit delta."""
        self.data['fit_delta'] = (('coil', 'arc_length', 'space'),
                                  np.zeros(self.data.centerline_delta.shape))
        self.data['translate'] = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        self.data['fit_translate'] = (
            ('coil', 'translate'),
            np.zeros((self.data.dims['coil'], self.data.dims['translate'])))

    def l2norm(self, candidate):
        """Return L2 norm of baseline-centerline delta."""
        index = self.kdtree.query(candidate, 2, False).reshape(-1, 2)

        delta = candidate[index[:, 0]] - self.data.centerline
        segment = candidate[index[:, 1]] - candidate[index[:, 0]]
        segment /= np.linalg.norm(segment, axis=1).reshape(-1, 1)

        delta -= segment * np.einsum('ij,ij->i', delta, segment).reshape(-1, 1)
        if self.radial_weight != 0:
            radius = np.linalg.norm(self.data.centerline[:, :2], axis=1)
            radius.shape = (-1, 1)
            weight = radius**self.radial_weight
            return np.linalg.norm(weight * delta) / np.sum(weight)
        return np.linalg.norm(delta)

    def transform(self, translate, centerline, origin):
        """Return delta transformed centerline."""
        centerline = np.copy(centerline)
        centerline -= origin
        rotate = scipy.spatial.transform.Rotation.from_euler(
            'xyz', translate[3:], degrees=True)
        centerline = rotate.apply(centerline)
        centerline += origin
        centerline += translate[:3]
        return centerline

    def fit_error(self, translate, centerline, origin):
        """Return l2norm of transformed centerline."""
        candidate = self.transform(translate, centerline, origin)
        return self.l2norm(candidate)

    def fit_coil(self, index):
        """Fit coil to baseline."""
        origin = (np.min(self.data.centerline[:, 0].values), 0, 0)
        centerline = self.data.centerline + self.data.centerline_delta[index]
        optimize_result = scipy.optimize.minimize(
            self.fit_error, np.zeros(6), args=(centerline, origin))
        fit = self.transform(optimize_result.x, centerline, origin)
        self.data['fit_delta'][index] = fit - self.data.centerline
        self.data['fit_translate'][index] = optimize_result.x

    def fit_coilset(self, radial_weight=None):
        """Fit coilset to baseline."""
        if radial_weight is not None:
            self.radial_weight = radial_weight
        for index in range(self.data.dims['coil']):
            self.fit_coil(index)

    def plot(self, index, factor=500):
        """Plot centerlines."""
        axes = plt.subplots(2, 2, sharey='row', sharex='col',
                            gridspec_kw=dict(height_ratios=[8, 1],
                                             width_ratios=[8, 1]))[1]
        centerline = self.data.centerline
        centerline_delta = self.data.centerline_delta[index]
        fit_delta = self.data.fit_delta[index]

        for index, coord in zip(((0, 0), (0, 1), (1, 0)),
                                ((0, 2), (1, 2), (0, 1))):
            axes[index].plot(centerline[:, coord[0]], centerline[:, coord[1]],
                             '--', color='gray', label='nominal')
            axes[index].plot(centerline[:, coord[0]] +
                             factor*centerline_delta[:, coord[0]],
                             centerline[:, coord[1]] +
                             factor*centerline_delta[:, coord[1]], '-',
                             label='CCL')
            axes[index].plot(centerline[:, coord[0]] +
                             factor*fit_delta[:, coord[0]],
                             centerline[:, coord[1]] +
                             factor*fit_delta[:, coord[1]], '-',
                             label='fit')
            axes[index].set_aspect('equal')
            axes[index].axis('off')
        axes[1, 1].axis('off')
        axes[0, 0].legend(loc='center', bbox_to_anchor=(1.5, 0.5))

    def plot_coilset(self, factor=500):
        """Plot coilset centerlines."""
        axes = plt.subplots(1, 2)[1]
        centerline = self.data.centerline
        origin = self.data.origin.values
        coil = self.data.coil.values
        clone = self.data.clone.values
        for axid in range(2):
            axes[axid].plot(centerline[:, 0], centerline[:, 2],
                            '--', color='gray')
        for index in range(18):
            fit_delta = self.data.fit_delta[index]
            axid = 0 if origin[index] == 'EU' else 1
            label = f'{coil[index]:02d}'
            if clone[index] != -1:
                label += f'<{clone[index]:02d}'
            axes[axid].plot(centerline[:, 0] + factor*fit_delta[:, 0],
                            centerline[:, 2] + factor*fit_delta[:, 2], '-',
                            label=label)
        for axid in range(2):
            axes[axid].set_aspect('equal')
            axes[axid].axis('off')
            axes[axid].legend(loc='center', fontsize='xx-small')
        axes[0].set_title('EU')
        axes[1].set_title('JA')
        if self.radial_weight == -1:
            plt.suptitle(r'Inboard fit $\propto r^{-1}$')
        if self.radial_weight == 0:
            plt.suptitle('Equal weight')
        if self.radial_weight == 1:
            plt.suptitle(r'Outboard fit $\propto r$')

    def plot_wave(self, axes, values, wavenumber=1, ncoil=18):
        """Plot fft modes."""
        if wavenumber is None:
            axes.bar(range(ncoil), values, color='C0')
            return
        coef = np.zeros(ncoil//2, dtype=complex)
        error_modes = np.zeros(1 + ncoil//2)
        fft_coefficents = scipy.fft.rfft(values)
        error_modes[0] = fft_coefficents[0].real / ncoil
        error_modes[1:] = abs(fft_coefficents[1:]) / (ncoil//2)

        coef[0] = fft_coefficents[0]
        if not isinstance(wavenumber, Iterable):
            wavenumber = [wavenumber]
        label = ''
        for wn in wavenumber:
            coef[wn] = fft_coefficents[wn]
            if label:
                label += '\n'
            label += f' $k_{wn}$='
            label += f'{error_modes[wn]:1.2f}'

        nifft = ncoil
        ifft = scipy.fft.irfft(coef, n=nifft, norm='backward').real
        axes.bar(range(ncoil), values, color='lightgray')
        axes.plot(range(ncoil), values, '.-', color='C7')
        axes.plot(np.linspace(0, ncoil-1, nifft), ifft, '-', color='C6')
        axes.text(ncoil-0.5, ifft[-1], label, va='center',
                  color='C6', fontsize='small')

    def plot_error(self):
        """Plot placement error."""
        axes = plt.subplots(3, 2, sharex=True, sharey=True)[1]
        plt.subplots_adjust(wspace=0.45)
        loc = range(18)
        wavenumber = [1, 2, 3]
        for i, coord in enumerate(['x', 'y', 'z']):
            delta = self.data.fit_translate.sel(translate=coord)[loc]
            self.plot_wave(axes[i, 0], delta.values, wavenumber)
            axes[i, 0].set_ylabel(rf'$\Delta${coord} mm')

        for i, (coord, radius, title) in enumerate(
                zip(['ry', 'rx', 'rz'], [4.5, 4.5, 0.25],
                    ['pitch', 'roll', 'yaw'])):
            delta = 1e3*radius*self.data.fit_translate.sel(
                translate=coord)[loc] * np.pi/180
            self.plot_wave(axes[i, 1], delta.values, wavenumber)

        axes[2, 0].set_xticks([0, 9, 17])

        axes[0, 0].set_title('translate')
        axes[0, 1].set_title('rotate')

        axes[2, 0].set_xlabel('coil index')
        axes[2, 1].set_xlabel('coil index')
        plt.despine()


if __name__ == '__main__':

    data = FiducialData(fill=True, sead=2025).data
    error = FiducialError(data)
    error.fit_coilset(radial_weight=-1)

    error.plot_coilset()

    #error.plot(0)
    error.plot_error()
