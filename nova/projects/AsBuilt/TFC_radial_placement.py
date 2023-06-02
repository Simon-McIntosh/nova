import os

import numpy as np

from nova.definitions import root_dir
from nova.utilities.png_tools import data_load
import matplotlib.pyplot as plt

ncoil = 18

rng = np.random.default_rng(2025)
radius = 2 * rng.random((100000, ncoil)) - 1
factor = 7.5
radius *= factor
# radius *= 4.9
# radius *= 4.25

"""
radius = np.array([[4.911188, 3.720625, 4.503144, 3.109321, -1.781115,
                    1.395654, -4.430507, -4.286455, -4.567845, -4.41067,
                    -3.732528, 3.300678,  3.650736, 1.563206, -4.555027,
                    4.775682, 2.731185,  4.198865]])
"""
"""
radius = np.array(
      [[ 4.911188, -4.046573, -1.700848],
       [ 3.720625,  0.684551, -0.772447],
       [ 4.503144, -4.844046, -3.677593],
       [ 3.109321, -4.185436, -4.728377],
       [-1.781115, -0.273149, -4.963377],
       [ 1.395654,  2.234939, -3.250931],
       [-4.430507,  1.728926, -3.246353],
       [-4.286455,  3.105658, -4.289813],
       [-4.567845,  0.44772 , -1.126167],
       [-4.41067 ,  1.413355, -4.54465 ],
       [-3.732528, -2.271867, -4.153696],
       [ 3.300678,  4.616676,  3.136483],
       [ 3.650736,  4.525728,  4.327046],
       [ 1.563206,  4.12104 ,  4.560215],
       [-4.555027,  4.891656,  3.943722],
       [ 4.775682, -4.252579,  2.248062],
       [ 2.731185, -4.310566,  2.761704],
       [ 4.198865, -4.017274, -0.864311]]).T
"""

fft = np.fft.fft(radius, axis=-1)
amplitude = abs(fft) / (ncoil // 2)
amplitude[:, 0] /= 2
if ncoil % 2 == 0:
    amplitude[:, ncoil // 2] /= 2
phase = np.angle(fft)

phi, dphi = np.linspace(0, 2 * np.pi, 150, endpoint=False, retstep=True)
phi = phi[:, np.newaxis]

coef = [1.1, 0.61]

Bnorm = coef[0] * np.ones_like(amplitude[:, 0]) * np.sin(ncoil * phi)
for wavenumber in range(1, ncoil // 2 + 1):
    Bnorm += (
        coef[1]
        * amplitude[:, wavenumber]
        * np.sin(wavenumber * phi + phase[:, wavenumber])
    )

deviation = dphi * np.cumsum(Bnorm, axis=0)
peaktopeak = deviation.max(axis=0) - deviation.min(axis=0)

index = 2

plt.figure()

path = os.path.join(root_dir, "input/Assembly/")
data = data_load(path, f"peaktopeak_case{index+1}", date="2022_04_01")[0][0]
plt.plot(data["x"], data["y"], "-", label="Energopul")
plt.plot(phi, deviation[:, index], "--", label="FFT")

plt.xlabel(r"$\phi$")
plt.ylabel(r"field line deviation, $h$ mm")
plt.despine()
plt.legend()
plt.title(f"case {index+1}")


print(peaktopeak[index])

Hmax = 6
if len(radius) > 3:
    CI = np.quantile(peaktopeak, 0.99)
    print(Hmax / CI)

    plt.figure()
    plt.hist(peaktopeak, bins=51, rwidth=0.85)
    plt.yticks([])
    plt.xlabel("peak to peak deviation, $H$ mm")
    plt.ylabel("$P(H)$")
    plt.despine()
    ylim = plt.ylim()
    plt.plot(
        [CI, CI],
        [ylim[0], ylim[0] + 0.8 * (ylim[1] - ylim[0])],
        "--",
        color="gray",
        label=f"99% $\Delta r < ${factor:1.1f}mm",
    )
    plt.legend()
    plt.title(f"require H<{Hmax}mm")
