"""Train plasma current."""

import pathlib

import appdirs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast/plasma_current"

pipe = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    # sklearn.linear_model.LinearRegression(),
    sklearn.ensemble.HistGradientBoostingRegressor(loss="absolute_error"),
)

train = pd.read_csv(path / "train.csv")
X, y = train.drop("plasma_current", axis=1), train.plasma_current
X.drop(["index", "time", "shot_index"], axis=1, inplace=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X,
    y,
    test_size=0.3,
    shuffle=True,
    random_state=3,
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

mape = sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE {mape:1.3f}")
mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
print(f"MAE {mae:1.3f}")

test = pd.read_csv(path / "test.csv")
# test.drop(["index", "time", "shot_index"], axis=1, inplace=True)

pipe.fit(X, y)

"""
submission = pd.DataFrame(pipe.predict(test), columns=["plasma_current"])
submission.index.name = "index"
submission.to_csv(path / "sample_submission.csv")
solution = pd.read_csv(path / "solution.csv")


sns.set_context("notebook")
axes = plt.subplots(figsize=(6, 4))[1]

sort_index = np.argsort(X_test.time)
_X_test = X_test.iloc[sort_index]
_y_test = y_test.iloc[sort_index]
for shot_index in np.unique(X_train.shot_index):
    index = _X_test.shot_index == shot_index
    axes.plot(_X_test.loc[index, "time"], _y_test.loc[index], "--", color="gray")
    axes.plot(_X_test.loc[index, "time"], pipe.predict(_X_test)[index])
axes.set_xlabel("time s")
axes.set_ylabel("plasma current kA")
sns.despine()


mape_private = sklearn.metrics.mean_absolute_percentage_error(
    solution.plasma_current, submission
)
print(f"MAPE private {mape_private:1.3f}")
mae_private = sklearn.metrics.mean_absolute_error(
    submission,
    solution.plasma_current,
)
print(f"MAE private {mae_private:1.3f}")
"""

"""
for shot_index in np.unique(test.shot_index):
    index = test.shot_index == shot_index
    test.loc[index].plot(x="time", y="plasma_current", ax=axes)
    test.loc[index].plot(x="time", y="predict", ax=axes)
"""
