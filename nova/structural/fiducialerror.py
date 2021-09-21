"""Calculate placement error from fiducial data."""
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
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
    radial_weight: bool = True

    def __post_init__(self):
        """Initalize fit dataarray - perform coilset fit."""
        self.calculate_error()

    #def load_dataset(self):

    def calculate_error(self):
        """Calculate ccl placment error."""
        self.initialize_fit()
        self.fit_coilset()

    def initialize_fit(self):
        """Initialise fit delta."""
        self.data['fit_delta'] = (('coil', 'arc_length', 'space'),
                                  np.zeros(self.data.centerline_delta.shape))
        self.data['trans'] = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        self.data['fit_trans'] = (('coil', 'trans'),
                                  np.zeros((self.data.dims['coil'],
                                            self.data.dims['trans'])))


    def l2norm(self, centerline, tree):
        """Return L2 norm of baseline-centerline delta."""
        centerline_index = tree.query(centerline)[1].reshape(-1)
        delta = centerline[centerline_index] - self.data.centerline
        if self.radial_weight:
            radius = np.linalg.norm(self.data.centerline[:, :2], axis=1)
            radius.shape = (-1, 1)
            return np.linalg.norm(radius**-1 * delta) / np.sum(radius**-1)
        return np.linalg.norm(delta)

    def transform(self, trans, centerline, origin):
        """Return delta transformed centerline."""
        centerline = np.copy(centerline)
        centerline -= origin
        rotate = scipy.spatial.transform.Rotation.from_euler(
            'xyz', trans[3:], degrees=True)
        centerline = rotate.apply(centerline)
        centerline += origin
        centerline += trans[:3]
        return centerline

    def fit_error(self, trans, centerline, origin, tree):
        """Return l2norm of transformed centerline."""
        candidate = self.transform(trans, centerline, origin)
        return self.l2norm(candidate, tree)

    def fit_coil(self, index):
        """Fit coil to baseline."""
        origin = (np.min(self.data.centerline[:, 0].values), 0, 0)
        centerline = self.data.centerline + self.data.centerline_delta[index]
        tree = sklearn.neighbors.KDTree(self.data.centerline)
        optimize_result = scipy.optimize.minimize(
            self.fit_error, np.zeros(6), args=(centerline, origin, tree))
        fit = self.transform(optimize_result.x, centerline, origin)
        self.data['fit_delta'][index] = fit - self.data.centerline
        self.data['fit_trans'][index] = optimize_result.x

    def fit_coilset(self):
        """Fit coilset to baseline."""
        for index in range(self.data.dims['coil']):
            self.fit_coil(index)

    def plot(self, index, factor=500):
        """Plot centerlines."""
        axes = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 2]))[1]
        centerline = self.data.centerline
        centerline_delta = self.data.centerline_delta[index]
        fit_delta = self.data.fit_delta[index]

        for i, ycoord in enumerate([2, 1]):
            axes[i].plot(centerline[:, 0], centerline[:, ycoord],
                         '--', color='gray', label='nominal')
            axes[i].plot(centerline[:, 0] + factor*centerline_delta[:, 0],
                         centerline[:, ycoord] +
                         factor*centerline_delta[:, ycoord], '-')
            axes[i].plot(centerline[:, 0] + factor*fit_delta[:, 0],
                         centerline[:, ycoord] +
                         factor*fit_delta[:, ycoord], '-')
            axes[i].axis('equal')
            axes[i].axis('off')



if __name__ == '__main__':

    data = FiducialData(fill=True, sead=2025).data
    error = FiducialError(data)

    #error.plot(8)
