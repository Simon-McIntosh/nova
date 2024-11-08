"""Access MAST data from object store."""

import numpy as np

import iis

import sklearn
import sklearn.pipeline
import sklearn.model_selection
import sklearn.kernel_ridge
import xarray as xr


shot_id = 30420
shot_id = 15585

test_id = 15212

dataset = np.array([15585, 15212, 15010, 14998])

"""
test = np.array([])

for shot in train:
    print(shot)
    try:
        _ = iis.Shot(shot).to_dask("equilibrium", "magnetic_flux")
    except FileNotFoundError:
        print(f"object store not found for {shot}")
"""


channels = [
    # "center_column",
    "coil_currents",
    # "coil_voltages",
    "flux_loops",
    # "outer_discrete",
    # "saddle_coils",
]

shot = iis.Shot(shot_id)
target = shot.to_dask("equilibrium", "magnetic_flux")
# signal = shot.to_dask("magnetics", "saddle_coils").interp({"time": target.time})
signal = shot.to_pandas("magnetics", channels, time=target.time)

"""
signal_train, signal_test, target_train, target_test = (
    sklearn.model_selection.train_test_split(
        signal,
        target.data.reshape((target.sizes["time"], -1)),
        test_size=0.2,
        shuffle=True,
        random_state=3,
    )
)
"""

pipeline = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.RobustScaler(),
    sklearn.model_selection.GridSearchCV(
        sklearn.kernel_ridge.KernelRidge(kernel="rbf"),
        param_grid={"alpha": np.logspace(-6, 1, 5), "gamma": np.logspace(-6, 1, 5)},
    ),
)
pipeline.fit(signal, target.data.reshape((target.sizes["time"], -1)))

"""
test_reshape = (target_test.shape[0],) + target.shape[1:]
magnetic_flux_test = target_test.reshape(test_reshape)
magnetic_flux_predict = pipeline.predict(signal_test).reshape(test_reshape)

itime = -1
levels, axes = flux_map(magnetic_flux_test[itime])
flux_map(magnetic_flux_predict[itime], axes=axes, levels=levels, colors="C0")
"""

# independant test
test_shot = iis.Shot(test_id)
target = test_shot.to_dask("equilibrium", "magnetic_flux")
# signal = test_shot.to_dask("magnetics", "saddle_coils").interp({"time": target.time})
signal = test_shot.to_pandas("magnetics", channels, time=target.time)
itime = -5
prediction = pipeline.predict(signal).reshape(target.shape)
levels, axes = iis.flux_map(target[itime])
iis.flux_map(prediction[itime], axes=axes, levels=levels, colors="C0")

"""
    equilibrium.major_radius,
    equilibrium.z,
    magnetic_flux_test[itime],
    colors="gray",
    linestyles="-",
    levels=51,
).levels
axes.contour(
    equilibrium.major_radius,
    equilibrium.z,
    magnetic_flux_predict[itime],
    colors="C0",
    linestyles="-",
    levels=levels,
)
"""
