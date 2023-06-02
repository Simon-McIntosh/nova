import os
import nep
from nep.DINA.capacitor_discharge import power_supply
from amigo.png_tools import data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from collections import OrderedDict
import numpy as np

discharge = OrderedDict()

path = os.path.join(class_dir(nep), "../Data/LTC/")
points = data_load(path, "VS3_discharge", date="2018_02_28")[0]
t, Io = points[1]["x"], points[1]["y"]  # jacket + vessel
discharge["LTC"] = {"t": t - 10, "I": Io}

path = os.path.join(class_dir(nep), "../Data/Energopul/")
points = data_load(path, "IVS3_discharge", date="2018_03_01")[0]
t, Io = points[1]["x"], points[1]["y"]  # jacket + vessel
discharge["ENP"] = {"t": t, "I": Io}

ps = power_supply(nturn=4, vessel=True, scenario=-1, code="IO", impulse=False)
ps.solve(Io=60e3, plot=False, tend=0.1)
discharge["Nova"] = {"t": ps.data["t"], "I": ps.data["Ivec"][:, 0]}

plt.set_context("talk")
# plt.figure()
for code, color in zip(discharge, ["C0", "C2", "C3"]):
    plt.plot(
        1e3 * discharge[code]["t"], 1e-3 * discharge[code]["I"], color=color, label=code
    )

tau_bare = 77.86 * 1e-3
t_bare = 1e-3 * np.linspace(0, 100, 100)
Ibare = 60e3 * np.exp(-t_bare / tau_bare)
plt.plot(1e3 * t_bare, 1e-3 * Ibare, "--", color="lightgray", label="bare conductor")

plt.legend()
plt.despine()
plt.xlabel("$t$ ms")
plt.ylabel("$I$ kA")
plt.xlim([0, 100])
