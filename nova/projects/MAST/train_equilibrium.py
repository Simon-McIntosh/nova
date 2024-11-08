"""Train plasma equilibrium reconstructor."""

from dataclasses import dataclass, field
from functools import cached_property
import pathlib

import appdirs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import sklearn.metrics
import sklearn.pipeline
import xarray as xr

# load plasma volume dataset
path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast/equilibrium"
train = xr.open_dataset(path / "train.nc")
test = xr.open_dataset(path / "test.nc")

# chose signals - tip: filter by attrs
# print(train.filter_by_attrs(units="Wb"))  # all attributes with units of Webbers
# print(train.filter_by_attrs(units="V"))  # all attributes with units of Volts
# print(train.filter_by_attrs(group="magnetics"))  # all attributes in the magnetics group


# transform signal and target xarray Datasets into columar data
def to_pandas(dataset: xr.Dataset, attrs: list[str]) -> pd.DataFrame:
    """Return set of Dataset attributes as a concatanated Pandas DataFrame."""
    return pd.concat(
        [dataset[attr].transpose("time", ...).to_pandas() for attr in attrs], axis=1
    )


signals = ["flux_loops"]  # select input signals
X = to_pandas(train, signals)
y = train.magnetic_flux.data.reshape((train.sizes["time"], -1))

# generate train and test set based on shot index
index = train.shot_index.values == 3
X_train, X_test, y_train, y_test = X[~index], X[index], y[~index], y[index]


pipeline = sklearn.pipeline.make_pipeline(
    # sklearn.preprocessing.RobustScaler(),
    sklearn.linear_model.LinearRegression(),
    # sklearn.model_selection.GridSearchCV(
    #    sklearn.kernel_ridge.KernelRidge(kernel="rbf"),
    #    param_grid={"alpha": np.logspace(-6, 1, 5), "gamma": np.logspace(-6, 1, 5)},
    # ),
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

error = sklearn.metrics.mean_absolute_error(y_test, y_pred)
print(f"mean absolute error {error:1.6f}")

# make predictions for test set
magnetic_flux = pipeline.predict(to_pandas(test, signals))

prediction = pd.DataFrame(magnetic_flux)
prediction.index.name = "index"
prediction.to_csv(path / "linear_regression.csv")

pd.read_csv(path / "solution.csv", index_col="index").drop("Usage", axis=1).to_csv(
    path / "perfect.csv"
)

mean = prediction.copy()
mean.loc[:] = y_train.mean()
mean.to_csv(path / "mean.csv")
mean.to_csv(path / "sample_submission.csv")


@dataclass
class Contour:
    """Manage multiple contour plots."""

    magnetic_flux: xr.DataArray | None = None
    levels: int = 31
    _handles: list = field(init=False, repr=False, default_factory=list)

    @cached_property
    def axes(self):
        """Manage axes instance."""
        self.fig, axes = plt.subplots(figsize=(3, 4.5))
        axes.set_aspect("equal")
        axes.set_axis_off()
        return axes

    @cached_property
    def shape(self):
        """Return flux map 2D shape."""
        return self.magnetic_flux.sizes["z"], self.magnetic_flux.sizes["major_radius"]

    def plot(self, data: xr.DataArray | np.ndarray = None, label=None, **kwargs):
        """Create contour map from magnetic flux data, store contour levels."""
        if isinstance(data, xr.DataArray):
            self.magnetic_flux = data
            data = data.values
        kwargs = {"colors": "gray", "linestyles": "-", "levels": self.levels} | kwargs
        try:
            contour = self.axes.contour(
                self.magnetic_flux.major_radius,
                self.magnetic_flux.z,
                data.reshape(self.shape),
                **kwargs,
            )
        except AttributeError:
            raise AttributeError(
                "Grid coordinates major_radius and z not found on "
                "magnetic_flux DataArray."
            )
        self.levels = contour.levels
        if label:
            color = kwargs.get("colors", "gray")
            self._handles.append(
                plt.matplotlib.lines.Line2D([0], [0], label=label, color=color)
            )
        return contour

    def legend(self):
        """Add legend to plot."""
        plt.legend(
            handles=self._handles, loc="center", bbox_to_anchor=[0.5, 1.08], ncol=1
        )
        self._handles = []

    def __call__(self, efit: np.ndarray, prediction: np.ndarray):
        """Plot a comparision between EFIT++ ground truth and prediction."""
        return [
            self.plot(efit, colors="gray", label="EFIT++"),
            self.plot(prediction, colors="C0", label="Prediction"),
        ]

    def _next_image(self, efit: np.ndarray, prediction: np.ndarray) -> PIL.Image.Image:
        """Yield poloidal flux contour images."""
        del self.axes  # clear instance axes
        self._handles = []  # clear legend handles
        self.levels = np.linspace(efit.min(), efit.max(), 51)
        contours = self(efit[-1], prediction[-1])
        self.legend()
        for _efit, _prediction in zip(efit, prediction):
            for contour in contours:
                contour.remove()
            contours = self(_efit, _prediction)
            self.fig.canvas.draw()
            yield PIL.Image.fromarray(np.array(self.fig.canvas.buffer_rgba()))

    def to_gif(self, efit: np.ndarray, prediction: np.ndarray):
        """Save gif animation of frame-wise efit-prediction mapping."""
        # self.fig.tight_layout(pad=0)
        imgs = [image for image in self._next_image(efit, prediction)]
        imgs[0].save(
            path / "equilibrium_animation.gif",
            save_all=True,
            append_images=imgs,
            duration=100,
            loop=0,
            minimize_size=True,
        )


sns.set_context("notebook")
contour = Contour(train.magnetic_flux)

time_index = 20
contour(y_test[time_index], y_pred[time_index])
# contour.legend()

contour.to_gif(y_test, y_pred)

# make_gif(train.magnetic_flux)
