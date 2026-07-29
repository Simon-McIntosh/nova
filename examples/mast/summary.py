# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:08:27 2024

@author: mcintos
"""
import appdirs

import intake
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.cluster
import sklearn.pipeline

storage_options = {
    "cache_storage": appdirs.user_cache_dir("fair-mast", False),
    "s3": {"anon": True, "endpoint_url": "https://s3.echo.stfc.ac.uk"},
}


def to_dask(shot_id: int, name: str):
    """Return dask dataset."""
    shot_df = sources.loc[sources.shot_id == shot_id]
    url = shot_df.loc[shot_df.name == name].iloc[0].url
    return catalog.level1.sources(url=url, storage_options=storage_options).to_dask()


shot_id = 30419

catalog = intake.open_catalog("https://mastapp.site/intake/catalog.yml")
df = pd.DataFrame(catalog.index.level1.shots().read())
summary = df.loc[df.campaign == "M9"]

sources = catalog.index.level1.sources().read()

catalog = intake.open_catalog("https://mastapp.site/intake/catalog.yml")
shot_df = pd.DataFrame(catalog.index.level1.shots().read())

for _, (shot_id, tipmax) in summary.loc[-10:, ["shot_id", "cpf_tipmax"]].iterrows():
    url = shot_df.loc[shot_df.shot_id == shot_id].iloc[0].url

    # shot_df = sources.loc[sources.shot_id == shot_id]
    # url = shot_df.loc[shot_df.name == "rbb"].iloc[0].url
    print(url)
    # data = catalog.level1.shots(url=url, group="rbb").to_dask()  # .sel(time=tipmax)


sources = catalog.index.level1.sources().read()
shot_df = sources.loc[sources.shot_id == shot_id]

summary.loc[:, "Te"] = 1e-3 * summary.loc[:, "cpf_te0_max"]
summary.loc[:, "nTtau"] = 1e-23 * summary.loc[
    :, ["cpf_ne0_ipmax", "cpf_te0_ipmax", "cpf_tautot_ipmax"]
].prod(axis=1)


# apply clustering algorithm
X = summary.loc[:, ["cpf_ne0_ipmax", "cpf_te0_ipmax", "cpf_tautot_ipmax"]].dropna()
X = X.loc[(X > 0).all(axis=1)]
logX = np.log(X)
model = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    sklearn.cluster.DBSCAN(eps=0.2, min_samples=6),
)
model.fit(logX)

summary["labels"] = -2 * np.ones(summary.shape[0])
summary.loc[X.index, "labels"] = model["dbscan"].labels_

axes = plt.subplot()
sns.scatterplot(summary, x="Te", y="nTtau", hue="current_range", ax=axes)
axes.set_xscale("log")
axes.set_yscale("log")
axes.set_xlabel("temperature keV")
axes.set_ylabel(r"fusion triple product $10^{20} $keVm$^{-3}$s")


from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show

shot_ids = np.array(
    [
        29352,
        29368,
        29454,
        29474,
        29581,
        29582,
        29602,
        29662,
        29948,
        30095,
        30162,
        30165,
        30177,
        30179,
        30189,
        30254,
        30260,
        30285,
        30300,
        30301,
        30302,
        30303,
        30304,
        30305,
        30306,
        30310,
        30311,
        30312,
        30313,
        30314,
        30316,
        30317,
        30318,
        30319,
        30320,
        30321,
        30322,
        30323,
        30324,
        30325,
        30335,
        30336,
        30337,
        30338,
        30339,
        30342,
        30343,
        30345,
        30348,
        30349,
        30350,
        30351,
        30352,
        30353,
        30354,
        30355,
        30356,
        30357,
        30358,
        30359,
        30360,
        30361,
        30362,
        30366,
        30367,
        30368,
        30369,
        30370,
        30371,
        30372,
        30373,
        30374,
        30375,
        30376,
        30377,
        30378,
        30379,
        30380,
        30383,
        30384,
        30388,
        30389,
        30390,
        30396,
        30397,
        30398,
        30399,
        30400,
        30404,
        30405,
        30406,
        30407,
        30409,
        30410,
        30411,
        30412,
        30413,
        30416,
        30417,
        30418,
        30419,
        30420,
        30421,
        30422,
        30423,
        30424,
        30425,
        30426,
        30427,
        30428,
        30430,
        30439,
        30440,
        30441,
        30443,
        30444,
        30445,
        30447,
        30448,
        30449,
        30450,
        30451,
        30454,
        30455,
        30456,
        30457,
        30458,
        30459,
        30460,
        30461,
        30462,
        30463,
        30464,
        30465,
        30466,
        30467,
        30468,
        30469,
        30470,
        30471,
    ]
)

TITLE = "Fusion Tripple Product for MAST M9 Campaign"
TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"

p = figure(
    tools=TOOLS,
    toolbar_location="above",
    width=1200,
    title=TITLE,
    x_axis_type="log",
    y_axis_type="log",
)
p.toolbar.logo = "grey"
p.background_fill_color = "#efefef"
p.xaxis.axis_label = "temperature keV)"
p.yaxis.axis_label = "fusion triple product $10^{20} $keVm$^{-3}$s"
p.grid.grid_line_color = "white"
p.hover.tooltips = [
    ("index", "@index"),
    ("Te", "@Te"),
    ("nTtau:", "@nTtau"),
]

"""
summary.loc[[current is not None for current in summary.current_range], "current"] = [
    float(current.split()[0])
    for current in summary.current_range
    if current is not None
]
"""
source = ColumnDataSource(summary.iloc[shot_ids])

p.scatter(
    "Te",
    "nTtau",
    size=12,
    source=source,
    color="current",
    line_color="black",
    alpha=0.9,
)
show(p)
"""
shot_a = np.nanargmin(abs(np.log(summary.Te) - 2) + abs(np.log(summary.nTtau) + 6))
plt.plot(
    summary.iloc[shot_a].Te,
    summary.iloc[shot_a].nTtau,
    marker="s",
    ms=12,
    color="k",
)
"""
