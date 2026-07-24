# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:05:16 2024

@author: mcintos
"""
import warnings


import intake
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.kernel_ridge import KernelRidge

warnings.simplefilter(action="ignore", category=FutureWarning)

import s3fs
import xarray as xr
import dask
import zarr

"""
endpoint = "https://s3.echo.stfc.ac.uk"
fs = s3fs.S3FileSystem(endpoint_url=endpoint, anon=True)
shot = zarr.open_group(fs.get_mapper("s3://mast/level1/shots/30419.zarr/"))
shot["asm"]
"""


catalog = intake.open_catalog("https://mastapp.site/intake/catalog.yml")


shot_id = 30419 + 5


def get_data(shot_id: int):
    """Return psi-2d dataset."""
    psirz = (
        catalog.level1.sources(url=f"s3://mast/level1/shots/{shot_id}.zarr/efm")
        .to_dask()
        .psirz.dropna(dim="profile_r")
    )

    currents = (
        catalog.level1.sources(url=f"s3://mast/level1/shots/{shot_id}.zarr/amc")
        .to_dask()
        .filter_by_attrs(
            name=lambda name: name is not None
            and ("plasma" in name or "coil_current" in name)
        )
        .interp({"time": psirz.time})
        .to_pandas()
    )
    magnetics = (
        catalog.level1.sources(url=f"s3://mast/level1/shots/{shot_id}.zarr/amb")
        .to_dask()
        .load()
    )

    camera = (
        catalog.level1.sources(url=f"s3://mast/level1/shots/{shot_id}.zarr/rbb")
        .to_dask()
        .interp({"time": psirz.time}, kwargs={"fill_value": "extrapolate"})
    )
    camera = pd.DataFrame(
        camera.data.values.reshape(camera.sizes["time"], -1), index=camera.time
    )

    mse = catalog.level1.sources(
        url=f"s3://mast/level1/shots/{shot_id}.zarr/ams"
    ).to_dask()

    flux = (
        magnetics.filter_by_attrs(units="Wb")
        .interpolate_na(dim="time")
        .interp({"time": psirz.time})
        .to_pandas()
    )

    field = (
        magnetics.filter_by_attrs(units="T")
        .interpolate_na(dim="time")
        .interp({"time": psirz.time})
        .to_pandas()
    )
    """
    volts = (
        magnetics.filter_by_attrs(units="V")
        .interpolate_na(dim="time")
        .interp({"time": psirz.time})
        .to_pandas()
    )
    """

    saddle = (
        catalog.level1.sources(url=f"s3://mast/level1/shots/{shot_id}.zarr/asm")
        .to_dask()
        .sad_m.dropna(dim="time")
        .interp({"time": psirz.time}, kwargs={"fill_value": "extrapolate"})
        .to_pandas()
        .T
    )

    # return camera.values, psirz
    # return flux, psirz
    return (
        pd.concat(
            [
                field,
                flux,
                saddle,
                # volts,
                currents,
            ],
            axis=1,
        ),
        psirz,
    )  #


"""
itime = 0
axes = plt.subplot()
axes.set_aspect("equal")
axes.set_axis_off()

axes.contour(
    psirz.profile_r, psirz.profile_z, psirz[itime], 51, colors="gray", linestyles="-"
)
"""
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model

# reg = linear_model.Lasso(alpha=0.1)
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np

"""
model = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),
    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3, 1e-6], "gamma": np.logspace(-2, 2, 5)},
)
"""

"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
"""

from sklearn.pipeline import make_pipeline
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

model = make_pipeline(
    # sklearn.decomposition.PCA(),
    sklearn.preprocessing.RobustScaler(),
    # sklearn.ensemble.HistGradientBoostingRegressor(),
    # sklearn.linear_model.RidgeCV(alphas=np.logspace(-3, 2, 5)),
    # KernelRidge(alpha=0.1, gamma=1e-4, kernel="rbf"),
    sklearn.model_selection.GridSearchCV(
        sklearn.kernel_ridge.KernelRidge(kernel="rbf"),
        param_grid={"alpha": np.logspace(-6, 1, 5), "gamma": np.logspace(-6, 1, 5)},
    ),
    # sklearn.ensemble.GradientBoostingRegressor(random_state=0),
)


currents, psirz = get_data(shot_id)
model.fit(currents, psirz.data.reshape(psirz.sizes["time"], -1))


offset = 4
itime_pred = 15
currents_test, psirz_test = get_data(shot_id + offset)
psirz_pred = model.predict(currents_test).reshape(
    psirz_test.sizes["time"],
    psirz_test.sizes["profile_r"],
    psirz_test.sizes["profile_z"],
)


axes = plt.subplot()
axes.set_aspect("equal")
axes.set_axis_off()
levels = axes.contour(
    psirz_test.profile_r,
    psirz_test.profile_z,
    psirz_test[itime_pred],
    51,
    colors="gray",
    linestyles="-",
).levels
axes.contour(
    psirz_test.profile_r,
    psirz_test.profile_z,
    psirz_pred[itime_pred],
    levels=levels,
    colors="C0",
    linestyles="-",
)
