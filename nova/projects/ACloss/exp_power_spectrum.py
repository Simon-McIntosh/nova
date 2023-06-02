import numpy as np
import scipy

import matplotlib.pyplot as plt

t, dt = np.linspace(0, 1000, 2**9, retstep=True)


# plt.plot(t, y)
for tau in [4, 8, 16]:
    y = -40e3 / tau * np.exp(-t / tau)
    f, Pxx = scipy.signal.periodogram(y, 1 / dt, scaling="spectrum")
    plt.plot(f[1:-1], Pxx[1:-1] * 1e-6, label=rf"$\tau$={tau:1.0f}s")
plt.plot(0.1 * np.ones(2), [0, Pxx[1] * 1e-6], "gray", ls="--")
# plt.xscale('log')
# plt.yscale('log')

plt.despine()
plt.xlabel("$f$ Hz")
plt.ylabel("$I^2$ MA$^2$")
plt.legend()
