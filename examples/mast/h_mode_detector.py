# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:08:33 2024

@author: mcintos
"""
import intake
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import tree, svm

catalog = intake.open_catalog("https://mastapp.site/intake/catalog.yml")
sources = catalog.index.level1.sources().read()

from sklearn.model_selection import train_test_split

shot_id = 30419


def get_data(shot_id: int):
    """Return labeled H-mode detector dataset."""
    asm = catalog.level1.sources(
        url=f"s3://mast/level1/shots/{shot_id}.zarr/asm"
    ).to_dask()
    xim = catalog.level1.sources(
        url=f"s3://mast/level1/shots/{shot_id}.zarr/xim"
    ).to_dask()

    X = xim.to_pandas()
    y = asm.hm_rating.interp({"hm_time": xim.time}).data.astype(int)
    return X, y


X, y = get_data(shot_id)

clf = svm.LinearSVC()

# clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

offset = -2

X, y = get_data(shot_id + offset)

y_pred = clf.predict(X)

axes = plt.subplot()
axes.plot(X.index, y, label="ground truth")
axes.plot(X.index, y_pred, "--", label="prediction")
axes.set_xlabel("time s")
axes.set_ylabel("H-mode detector")
sns.despine()
axes.legend()

axes = plt.subplot()
