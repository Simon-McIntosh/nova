"""Perform Gaussian Process Regression on Fiducial Data Points."""

from dataclasses import dataclass, field, InitVar

import numpy as np
import numpy.typing as npt
import sklearn.gaussian_process
import xarray

import matplotlib.pyplot as plt


@dataclass
class GaussianProcessRegressor:
    """Fit cyclic 1D waveforms using a Gaussian Process Regressor."""

    x: InitVar[npt.ArrayLike]
    variance: float = 0.5
    period: list[float] = field(default_factory=lambda: [0.0, 1.0])
    wrap: int = 0
    regressor: sklearn.gaussian_process.GaussianProcessRegressor = None
    data: xarray.Dataset = field(init=False)

    def __post_init__(self, x):
        """Init dataset."""
        x = self.to_numpy(x)
        self.data = xarray.Dataset(coords={"x": x})
        self.build_regressor()

    def build_regressor(self):
        """Build Gaussian Process Regressor."""
        if self.regressor is None:
            expsinesquared = sklearn.gaussian_process.kernels.ExpSineSquared(
                length_scale=0.85,
                length_scale_bounds="fixed",
                periodicity=1.0,
                periodicity_bounds="fixed",
            )
            constant = sklearn.gaussian_process.kernels.ConstantKernel(
                constant_value=1e-8, constant_value_bounds=(1e-22, 1e2)
            )
            kernel = expsinesquared + constant
            self.regressor = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=kernel, alpha=self.variance
            )

    @staticmethod
    def to_numpy(array):
        """Return numpy array."""
        if isinstance(array, xarray.DataArray):
            return array.values
        return array

    def fit(self, y):
        """Fit Gaussian Process Regressor."""
        self.data["y"] = "x", self.to_numpy(y)
        _x, _y = self.data.x.to_numpy(), self.data.y.to_numpy()
        if np.isnan(_y).any():  # drop nans
            index = ~np.isnan(_y)
            _x, _y = _x[index], _y[index]
        if self.wrap > 0:
            _x = np.pad(_x, pad_width=self.wrap, mode="wrap")
            _x[: self.wrap] -= self.period[-1]
            _x[-self.wrap :] += self.period[-1]
            _y = np.pad(_y, pad_width=self.wrap, mode="wrap")
        self.data["_x"] = _x
        self.data["_y"] = _y
        self.regressor = self.regressor.fit(_x.reshape(-1, 1), _y)

    def x_space(self, x: int | npt.ArrayLike) -> np.ndarray:
        """Return input as numpy array."""
        if isinstance(x, int):
            return np.linspace(self.period[0], self.period[1], x)
        return self.to_numpy(x)

    def predict(self, x_mean, return_std=False):
        """Return mean Gaussian Process Regressor."""
        x_mean = self.x_space(x_mean)
        y_mean, y_std = self.regressor.predict(x_mean[:, np.newaxis], return_std=True)
        try:
            self.data = self.data.drop_vars(["x_mean", "y_mean", "y_std"])
        except ValueError:
            pass
        self.data["x_mean"] = x_mean
        self.data["y_mean"] = ("x_mean", y_mean)
        self.data["y_std"] = ("x_mean", y_std)
        if return_std:
            return y_mean, y_std
        return y_mean

    def sample(self, x_sample, n_samples: int):
        """Return mean Gaussian Process Regressor."""
        x_sample = self.x_space(x_sample)
        y_sample = self.regressor.sample_y(x_sample[:, np.newaxis], n_samples)
        try:
            self.data = self.data.drop_vars(["samples", "x_sample", "y_sample"])
        except ValueError:
            pass
        self.data["x_sample"] = x_sample
        self.data["samples"] = range(n_samples)
        self.data["y_sample"] = ("x_sample", "samples"), y_sample
        return y_sample

    def plot_samples(self, axes):
        """Plot gpr samples."""
        axes.plot(self.data.x_sample, self.data.y_sample, "gray", lw=0.5, zorder=-5)

    def evaluate(self, x, y):
        """Return GPR prediction at x for data points y."""
        self.fit(y)
        return self.predict(x)

    def plot(
        self,
        stage=2,
        axes=None,
        text=True,
        label="fiducial_data",
        marker="o",
        marker_color="C3",
        line_color="gray",
        fill_color="k",
    ):
        """Plot current GP regression."""
        if axes is None:
            axes = plt.subplots(1, 1)[1]
        axes.scatter(
            self.data.x,
            self.data.y,
            marker=marker,
            color=marker_color,
            alpha=0.5,
            s=30,
            zorder=10,
            label=label,
        )
        if stage > 0:
            axes.plot(self.data.x_mean, self.data.y_mean, line_color, lw=2, zorder=9)
        if stage > 1:
            axes.fill_between(
                self.data.x_mean,
                self.data.y_mean - 2 * self.data.y_std,
                self.data.y_mean + 2 * self.data.y_std,
                alpha=0.15,
                color=fill_color,
                label="95% confidence",
            )
        if stage > 2:
            data_y = self.predict(self.data.x)
            axes.plot(self.data.x, data_y, "d", color=marker_color, alpha=0.5)

        if text:
            plt.despine()
            plt.xlabel("arc length")
            plt.ylabel("displacement, mm")
            plt.title(self.regressor.kernel_)
            plt.legend()
