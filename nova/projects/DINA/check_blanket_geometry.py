
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas
import scipy

from nova.database.filepath import FilePath
from nova.utilities.pyplot import plt


@dataclass
class DINA_Blanket(FilePath):
    """Compare DINA blanket geometries."""

    datapath: str = 'data/DINA/geometry'

    def __post_init__(self):
        """Set filepath."""
        self.set_path(self.datapath)

    @cached_property
    def mat_data(self):
        """Return vessel mat data."""
        file = self.file('env_VS_coils2', extension='.mat')
        data = pandas.DataFrame(
            scipy.io.loadmat(file, simplify_cells=True)['device']['vessel'])
        data['coordinates'] = [self.oblique_patch(*coords) for coords in
                               data.loc[:, ['rc', 'zc', 'dl', 'dh',
                                            'alpha', 'beta']].values]
        return data

    @staticmethod
    def oblique_patch(radius, height, length_alpha, length_beta,
                      alpha, beta):
        """Return skewed polygon."""
        alpha -= np.pi/2
        radius = radius + np.array(
            [0, length_alpha * np.cos(alpha),
             length_alpha * np.cos(alpha)
             - length_beta * np.sin(beta),
             -length_beta * np.sin(beta)])
        height = height + np.array(
            [0, length_alpha * np.sin(alpha),
             length_alpha * np.sin(alpha)
             + length_beta * np.cos(beta),
             length_beta * np.cos(beta)])
        return np.append(radius.reshape(-1, 1),
                         height.reshape(-1, 1), axis=1)

    @cached_property
    def json_data(self):
        """Return json vessel data."""
        file = self.file('blanket', extension='.json')
        return pandas.read_json(file)

    def plot_patch(self, data: pandas.DataFrame, **kwargs):
        """Plot patch."""
        for patch in data.coordinates:
            loop = np.append(patch, patch[:1], axis=0)
            plt.plot(*loop.T, **kwargs)

    def plot(self):
        """Plot comparison of mat and json blanket geometries."""
        self.plot_patch(self.json_data, color='r', lw=0.75)
        self.plot_patch(self.mat_data, color='b', lw=0.75)

        plt.plot(self.mat_data['rc'], self.mat_data['zc'], 'X')

        plt.axis('equal')
        plt.axis('off')

    def save(self):
        """Save mat data to file."""
        file = self.file('DINA_blanket', extension='.json')
        self.mat_data.to_json(file)


if __name__ == '__main__':

    blanket = DINA_Blanket()
    blanket.plot()
    blanket.save()
