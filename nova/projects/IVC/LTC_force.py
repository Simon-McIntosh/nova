import os
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir
from amigo.pyplot import plt

path = os.path.join(class_dir(nep), "../Data/LTC/")
# data_mine(path, 'VS3_force', [0, 0.08], [60e3, 260e3])

points = data_load(path, "VS3_force", date="2018_03_15")[0]

LTC = {}
LTC["lowerVS"] = {"t": points[0]["x"], "Fmag": points[0]["y"]}
LTC["upperVS"] = {"t": points[1]["x"], "Fmag": points[1]["y"]}

for coil in LTC:
    plt.plot(1e3 * LTC[coil]["t"], 1e-3 * LTC[coil]["Fmag"])

plt.despine()
plt.xlabel("$t$ ms")
plt.ylabel("$|F|$ kNm$^{-1}$")
plt.legend()
