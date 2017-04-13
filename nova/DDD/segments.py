import numpy as np
import pylab as pl

N = 93
delta = 31 / N


dL = np.array([1.019, 2.301, 1.014, 2.301, 2.123, 15.074, 1.051])

rem = []
N = np.arange(80, 140)
for n in N:
    delta = 31 / n
    rem.append(np.max(np.abs(np.round(dL / delta) - dL / delta)))

i = np.argmin(rem)
print(N[i])


j = np.argmin(abs(N - 100))
print(rem[j])

pl.plot(N, rem)
