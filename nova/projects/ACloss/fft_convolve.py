import os.path

import numpy as np
import scipy.fft
import scipy.interpolate
import scipy.signal

from nova.definitions import root_dir
from nova.utilities.png_tools import data_mine, data_load
import matplotlib.pyplot as plt


path = os.path.join(root_dir, "inputs/ACLoss/")
# data_mine(path, 'JASTEC6_2T', (0, 12), (0, 160))
H = data_load(path, "JASTEC6_2T", date="2020_11_19")[0][0]

f_data, P_data = H["x"], H["y"]
P_interp = scipy.interpolate.interp1d(f_data, P_data, fill_value="extrapolate")

# plt.plot(f_data, P_data, 'o')

dt = 0.025

n = 2**10
f = scipy.fft.fftfreq(n, dt)
print(f[1], f.max())
P = np.zeros(n)
P[1 : n // 2] = P_interp(f[1 : n // 2])
P[n // 2 + 1 :] = P[1 : n // 2][::-1]
P[n // 2] = P[n // 2 - 1]
# plt.plot(f, P)

p = scipy.fft.ifft(P, n)
t = dt * np.arange(n)

# plt.plot(t, p)

fe = 5

t = dt * np.arange(2 * n)
Be = 0.3 * np.sin(t * 2 * np.pi * fe)

pe = scipy.signal.convolve(Be, p, mode="valid").real
te = dt * np.arange(len(pe))
pe[0] = 0

print(np.sqrt(np.mean(pe**2)))

plt.plot(t, Be)
plt.plot(te, pe)
