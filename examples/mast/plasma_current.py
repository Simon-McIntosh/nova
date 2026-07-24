"""Prepare plasma current train and test datasets."""

import pathlib

import appdirs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import iis

# segment source dataset
source_ids = np.array([15585, 15212, 15010, 14998, 30410, 30418, 30420])

rng = np.random.default_rng(7)
rng.shuffle(source_ids)
source_ids = pd.Series(source_ids)

split_ids = {
    "train": source_ids[:5],
    "test": source_ids[5:],
}

# plot plasma current time traces
sns.set_context("paper")
plt.subplots(figsize=(8, 5.5))
for shot_index, shot_id in enumerate(split_ids["train"]):
    shot = iis.Shot(shot_id)
    shot["summary"].plasma_current.plot(label=f"shot {shot_index}")
plt.legend()
plt.xlabel("time s")
plt.ylabel("plasma current kA")
sns.despine()


def to_dataframe(shot_ids: pd.Series, channels=None, columns=None):
    """Return concatanated dataframe for the list of input ids."""
    dataframes = []
    if channels is None:
        channels = list(iis.Shot(shot_ids.iloc[0])["magnetics"])
    for shot_index, shot_id in shot_ids.items():
        shot = iis.Shot(shot_id)
        target = shot.to_pandas("summary", "plasma_current", multi_index=False)
        dataframe = shot.to_pandas(
            "magnetics", channels, time=target.index, multi_index=False
        )
        dataframe.sort_index(axis=1, inplace=True)
        dataframe["shot_index"] = shot_index
        dataframe["plasma_current"] = target.plasma_current
        if columns is not None:
            dataframe.columns = columns
        dataframes.append(dataframe.reset_index(names=["time"]))
    dataframe = pd.concat(dataframes, ignore_index=True, axis=0)
    dataframe.index.rename("index", inplace=True)
    return dataframe


channels = ["center_column", "outer_discrete"]
columns = to_dataframe(source_ids[:1], channels=channels).columns
dataset = {
    mode: to_dataframe(shot_ids, channels=channels, columns=columns[1:])
    for mode, shot_ids in split_ids.items()
}

# extract solution
solution = dataset["test"].loc[:, ["plasma_current"]]
solution["Usage"] = dataset["test"].shot_index.map({5: "Public", 6: "Private"})
# delete solution from test file
dataset["test"].drop("plasma_current", axis=1, inplace=True)

# write to file
path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast"
(path / "plasma_current").mkdir(exist_ok=True)
dataset["train"].to_csv(path / "plasma_current/train.csv")
dataset["test"].to_csv(path / "plasma_current/test.csv")
solution.to_csv(path / "plasma_current/solution.csv")

solution.loc[:, "plasma_current"].to_csv(path / "plasma_current/perfect.csv")


sns.set_context("paper")
axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)[1]
shot_id = split_ids["train"][0]
shot = iis.Shot(shot_id)
magnetics = to_dataframe(pd.Series(shot_id), channels=channels, columns=columns[1:])
axes[0].plot(magnetics.time, magnetics.iloc[:, 1:-2])
shot["summary"].plasma_current.plot(
    label=f"shot {shot_index}", ax=axes[1], color="gray"
)
axes[0].set_ylabel("diagnostic signal V")
axes[1].set_xlabel("time s")
axes[1].set_ylabel("plasma current kA")

sns.despine(plt.gcf())


"""
pd.concat(
    [
        shot.to_pandas("magnetics", "center_column"),
        shot.to_pandas("magnetics", "flux_loops"),
    ],
    axis=1,
)

dataframe = shot.to_pandas("magnetics", "center_column")
dataframe.columns = pd.MultiIndex.from_product(
    [["center_column"], dataframe.columns], names=("diagnostic", "channel")
)
"""


"""

channels = [
    # "center_column",
    "coil_currents",
    # "coil_voltages",
    "flux_loops",
    # "outer_discrete",
    # "saddle_coils",
]

shot_id =

target = shot.to_dask("equilibrium", "magnetic_flux")
# signal = shot.to_dask("magnetics", "saddle_coils").interp({"time": target.time})
signal = shot.to_pandas("magnetics", channels, time=target.time)
"""
