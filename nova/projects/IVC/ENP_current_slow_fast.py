import os
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from nep.DINA.read_plasma import read_plasma
import numpy as np


path = os.path.join(class_dir(nep), "../Data/Energopul/")
# data_mine(path, 'current_slow_fast', [0, 0.13], [52e3, 90e3])

ENP = {}
points = data_load(path, "current_slow_fast", date="2018_03_16")[0]
ENP["t"] = points[0]["x"]
ENP["I"] = points[0]["y"]

pl = read_plasma("disruptions")
trip, Ivs3_data, Ivs3, Ivs3_fun = pl.Ivs3_single(11, plot=False, dz_trip=2)

for mode, color in zip(Ivs3_fun, ["gray", "C0", "C3"]):
    plt.plot(1e3 * pl.t, 1e-3 * Ivs3_fun[mode](pl.t), label=mode, color=color)

plt.plot(1e3 * (trip["t_cq"] + ENP["t"]), 1e-3 * ENP["I"], "C4", label="Energopul")
plt.plot(1e3 * (trip["t_cq"] + ENP["t"]), -1e-3 * ENP["I"], ":C4")

ylim = plt.gca().get_ylim()
plt.plot(
    1e3 * trip["t_dz"] * np.ones(2),
    ylim,
    "--",
    alpha=0.7,
    color="gray",
    zorder=-10,
    label=r"$|\Delta z|>$" + "{:1.2f}m".format(pl.dz_trip),
)
plt.plot(
    1e3 * trip["t_cq"] * np.ones(2),
    ylim,
    "-.",
    alpha=0.7,
    color="gray",
    zorder=-10,
    label="current quench",
)

plt.despine()
plt.legend()
plt.xlabel("$t$ ms")
plt.ylabel("$I_{vs3}$ kA")
plt.title(pl.name)
