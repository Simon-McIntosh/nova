"""An noise fitting example."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.optimize
import scipy.fft

rng = np.random.default_rng(seed=4)

point_number = 5
phi = np.linspace(0, 2 * np.pi, point_number, endpoint=False)

data = rng.random(point_number)

Cn = scipy.fft.rfft(data, n=5)

test_data = rng.random(point_number)


point_number_hr = 5001
phi_hr = np.linspace(0, 2 * np.pi, point_number_hr)

train_model = scipy.fft.irfft(Cn, n=point_number, axis=0)

train_model_hr = (
    point_number_hr / point_number * scipy.fft.irfft(Cn, n=point_number_hr, axis=0)
)


def point_error(data, model):
    """Return L2 norm point error."""
    return np.linalg.norm(data - model, axis=0)


train_error = point_error(data, train_model)
test_error = point_error(test_data, train_model)


level = 2

plt.figure(figsize=(6, 4))

plt.plot(phi, data, "o", label="train data", alpha=0.75)
plt.plot(2 * np.pi, 1.1, "C0.", alpha=0)
plt.plot(0, -0.05, "C0.", alpha=0)

if level > 0:
    plt.plot(
        phi_hr, train_model_hr, "-", zorder=-10, label=f"train model {train_error:1.2f}"
    )
if level > 1:
    plt.plot(phi, test_data, "o", label=f"test data {test_error:1.2f}", alpha=0.75)

plt.xlabel(r"$\phi$")
plt.ylabel("value")
sns.despine()
plt.legend(loc="center", bbox_to_anchor=(-0.05, 1.05, 1, 0.1), ncol=3)
