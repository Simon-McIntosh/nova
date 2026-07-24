import appdirs
import intake
import matplotlib.pyplot as plt
import numpy as np

catalog = intake.open_catalog("https://mastapp.site/intake/catalog.yml")
storage_options = {
    "cache_storage": appdirs.user_cache_dir("fair-mast", False),
    "s3": {"anon": True, "endpoint_url": "https://s3.echo.stfc.ac.uk"},
}

sources = catalog.index.level1.sources().read()


def to_dask(shot_id: int, name: str):
    """Return dask dataset."""
    shot_df = sources.loc[sources.shot_id == shot_id]
    url = shot_df.loc[shot_df.name == name].iloc[0].url
    return catalog.level1.sources(url=url).to_dask()


shot_id = 30419

# efit_url = shot_df.loc[shot_df.name == "efm"].iloc[0].url
# currents_url = shot_df.loc[shot_df.name == "amc"].iloc[0].url


magnetics = to_dask(shot_id, "amb")
# flux = magnetics.filter_by_attrs(units="Wb").to_dataframe()
# flux.interpolate(inplace=True)  # interpolate NaN values

field = magnetics.filter_by_attrs(units="T").to_dataframe().interpolate()


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


def get_data(shot_id: int):
    magnetics = to_dask(shot_id, "amb")
    flux = magnetics.filter_by_attrs(units="Wb").to_dataframe().interpolate()
    # flux.interpolate(inplace=True)  # interpolate NaN values
    field = magnetics.filter_by_attrs(units="T").to_dataframe().interpolate()
    plasma_current = to_dask(shot_id, "amc").plasma_current

    X = pd.concat([flux, field], axis=1).loc[0:0.6]

    # X = flux.loc[0:0.6]

    y = plasma_current.interp({"time": X.index})
    return X, y


X, y = get_data(shot_id)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)

hgbt = HistGradientBoostingRegressor()
# hgbt.fit(X_train, y_train)
hgbt.fit(X, y)

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet

X, y = get_data(shot_id)

hgbt = Ridge(alpha=0.1)
hgbt.fit(X, y)

y_pred = hgbt.predict(X)

plt.plot(y)
plt.plot(y_pred)

offset = 11
X, y = get_data(shot_id + offset)

y_pred = hgbt.predict(X)

plt.figure()
plt.plot(y)
plt.plot(y_pred)


# url = sources_df.loc[sources_df.name == "efm"].url.iloc[-1]
# dataset = catalog.level1.sources(url=efit_url, storage_options=storage_options)
# dataset = dataset.to_dask()
