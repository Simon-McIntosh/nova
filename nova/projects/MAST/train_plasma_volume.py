"""Train model on plasma volume proton camera dataset."""

import pathlib

import appdirs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.decomposition
import sklearn.ensemble
import sklearn.kernel_ridge
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.model_selection
import xarray as xr

# load plasma volume dataset
path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast/plasma_volume"
train = xr.open_dataset(path / "train.nc")
test = xr.open_dataset(path / "test.nc")


X = train.frame.values.reshape(train.sizes["shot_id"], -1)
y = train.plasma_volume

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.3, random_state=7
)

pipeline = sklearn.pipeline.make_pipeline(
    sklearn.decomposition.KernelPCA(n_components=25),
    sklearn.linear_model.LinearRegression(),
)

parameter_grid = {
    "kernelpca__n_components": (10, 20),
    "kernelpca__gamma": np.logspace(-6, 6, 3),
}

grid = sklearn.model_selection.GridSearchCV(
    pipeline,
    param_grid=parameter_grid,
    n_jobs=-1,
)


grid.fit(X_train, y_train)
y_predict = grid.predict(X_test)
R2 = sklearn.metrics.r2_score(y_test, y_predict)
print(f"model R2 {R2:1.3f}")

volume = grid.predict(test.frame.values.reshape(test.sizes["shot_id"], -1))
solution = pd.DataFrame(
    {"plasma_volume": volume}, index=pd.Index(test.shot_id, name="shot_id")
)
solution.to_csv(path / "linear_regression.csv")

pd.read_csv(path / "solution.csv", index_col="shot_id").drop("Usage", axis=1).to_csv(
    path / "perfect.csv"
)

mean = solution.copy()
mean.loc[:, "plasma_volume"] = y_train.mean()
mean.to_csv(path / "mean.csv")
mean.to_csv(path / "sample_submission.csv")

train["test"] = "test_samples", y_test.values
train["predict"] = "test_samples", y_predict

sns.set_context("notebook")
plt.figure(figsize=(8, 6))
sns.scatterplot(train, x="test", y="predict", label=f"model R2={R2:1.3f}")
plt.axis("equal")
plt.xlabel(r"test volume m$^3$")
plt.ylabel(r"predict volume m$^3$")
xlim = plt.xlim()
plt.plot(xlim, xlim, "-", color="gray", label="ideal")
sns.despine()
plt.legend()
