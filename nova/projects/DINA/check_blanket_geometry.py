from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas
import scipy

from nova.database.filepath import FilePath
import matplotlib.pyplot as plt


@dataclass
class DINA_Blanket(FilePath):
    """Compare DINA blanket geometries."""

    datapath: str = "data/DINA/geometry"

    def __post_init__(self):
        """Set filepath."""
        self.set_path(self.datapath)

    @cached_property
    def mat_data(self, imas=False):
        """Return vessel mat data."""
        file = self.file("env_VS_coils2", extension=".mat")
        data = pandas.DataFrame(
            scipy.io.loadmat(file, simplify_cells=True)["device"]["vessel"]
        )

        radius, height, length_alpha, length_beta, alpha, beta = data.loc[
            :, ["rc", "zc", "dl", "dh", "alpha", "beta"]
        ].values.T

        if imas:
            # apply corrections to DINA .mat data
            alpha -= np.pi / 2
            radius += 0.5 * (length_alpha * np.cos(beta) - length_beta * np.sin(alpha))
            height += 0.5 * (length_alpha * np.sin(beta) + length_beta * np.cos(alpha))

            data["alpha"] = alpha
            data["rc"] = radius
            data["zc"] = height

            data["coordinates"] = [
                self.oblique_patch(*coords)
                for coords in zip(
                    radius, height, length_alpha, length_beta, alpha, beta
                )
            ]
            return data
        coordinates = self.DINA_patch(
            radius, height, length_alpha, length_beta, alpha, beta
        )
        data["coordinates"] = [patch for patch in coordinates]
        return data

    @staticmethod
    def oblique_patch(radius, height, length_alpha, length_beta, alpha, beta):
        """Return skewed polygon."""
        radius = radius + np.array(
            [
                0,
                length_alpha * np.cos(alpha),
                length_alpha * np.cos(alpha) - length_beta * np.sin(beta),
                -length_beta * np.sin(beta),
            ]
        )
        height = height + np.array(
            [
                0,
                length_alpha * np.sin(alpha),
                length_alpha * np.sin(alpha) + length_beta * np.cos(beta),
                length_beta * np.cos(beta),
            ]
        )
        return np.append(radius.reshape(-1, 1), height.reshape(-1, 1), axis=1)

    @staticmethod
    def DINA_patch(radius, height, length_alpha, length_beta, alpha, beta):
        """Return skewed polygon - translation from vessel_runs_plot.m."""
        coordinates = np.zeros((len(radius), 4, 2))

        coordinates[:, 0, 0] = radius - 0.5 * (
            length_alpha * np.cos(beta) + length_beta * np.cos(alpha)
        )
        coordinates[:, 0, 1] = height - 0.5 * (
            length_alpha * np.sin(beta) + length_beta * np.sin(alpha)
        )

        coordinates[:, 1, 0] = coordinates[:, 0, 0] + length_alpha * np.cos(beta)
        coordinates[:, 1, 1] = coordinates[:, 0, 1] + length_alpha * np.sin(beta)

        coordinates[:, 2, 0] = (
            coordinates[:, 0, 0]
            + length_alpha * np.cos(beta)
            + length_beta * np.cos(alpha)
        )
        coordinates[:, 2, 1] = (
            coordinates[:, 0, 1]
            + length_alpha * np.sin(beta)
            + length_beta * np.sin(alpha)
        )

        coordinates[:, 3, 0] = coordinates[:, 0, 0] + length_beta * np.cos(alpha)
        coordinates[:, 3, 1] = coordinates[:, 0, 1] + length_beta * np.sin(alpha)

        return coordinates

    @cached_property
    def json_data(self):
        """Return json vessel data."""
        file = self.file("blanket", extension=".json")
        return pandas.read_json(file)

    def plot_patch(self, data: pandas.DataFrame, **kwargs):
        """Plot patch."""
        for patch in data.coordinates:
            loop = np.append(patch, patch[:1], axis=0)
            plt.plot(*loop.T, **kwargs)

    def plot(self):
        """Plot comparison of mat and json blanket geometries."""
        self.plot_patch(self.json_data, color="C3", lw=0.75)
        self.plot_patch(self.mat_data, color="C0", lw=0.75)

        # plt.plot(self.mat_data['rc'], self.mat_data['zc'], 'x',
        #          ms=6, mew=1, mfc=None, color='C0')
        plt.axis("equal")
        plt.axis("off")

    def save(self):
        """Save mat data to file."""
        file = self.file("DINA_blanket", extension=".json")
        self.mat_data.to_json(file)


if __name__ == "__main__":
    blanket = DINA_Blanket()

    plt.set_aspect(0.9)
    blanket.plot()
    blanket.save()
