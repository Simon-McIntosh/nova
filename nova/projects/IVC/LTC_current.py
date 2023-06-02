import os
import nep
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt

path = os.path.join(class_dir(nep), "../Data/LTC/")
# data_mine(path, 'VS3_current', [0, 0.08], [35e3, 85e3])
# data_mine(path, 'VS3_current_VDE', [0, 1.25], [0, 64e3])

# points = data_load(path, 'VS3_current', date='2018_03_15')[0]
points = data_load(path, "VS3_current_VDE", date="2018_05_24")[0]

t = points[0]["x"]
Ic = points[0]["y"]

plt.plot(1e3 * t, 1e-3 * Ic)

plt.despine()
plt.xlabel("$t$ ms")
plt.ylabel("$|I|$ kA")
plt.legend()
