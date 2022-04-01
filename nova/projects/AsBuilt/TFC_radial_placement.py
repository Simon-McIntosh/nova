
import numpy as np

from nova.utilities.pyplot import plt

ncoil = 18

rng = np.random.default_rng(2025)
radius = 2*rng.random((100000, ncoil)) - 1
radius *= 6.9
#radius *= 4.9
#radius *= 4.25


radius = np.array([[4.911188, 3.720625, 4.503144, 3.109321, -1.781115,
                    1.395654, -4.430507, -4.286455, -4.567845, -4.41067,
                    -3.732528, 3.300678,  3.650736, 1.563206, -4.555027,
                    4.775682, 2.731185,  4.198865]])


fft = np.fft.fft(radius, axis=-1)
amplitude = abs(fft) / (ncoil // 2)
amplitude[:, 0] /= 2
if ncoil % 2 == 0:
    amplitude[:, ncoil // 2] /= 2
phase = np.angle(fft)

phi, dphi = np.linspace(0, 2*np.pi, 150, endpoint=False, retstep=True)
phi = phi[:, np.newaxis]

coef = [0.33, 1.63]

Bnorm = 0
for wavenumber in range(1, ncoil//2 + 1):
    Hfit = coef[1]*amplitude[:, wavenumber]
    Bnorm += Hfit / 4 * np.sin(wavenumber*phi + phase[:, wavenumber])

deviation = dphi * np.cumsum(Bnorm, axis=0)
peaktopeak = coef[0] + deviation.max(axis=0) - deviation.min(axis=0)

plt.figure()
plt.plot(phi, deviation[:, 0])

print(peaktopeak[0])

if len(radius) > 1:
    print(4/np.quantile(peaktopeak, 0.99))

    plt.figure()
    plt.hist(peaktopeak, bins=51, rwidth=0.85)
