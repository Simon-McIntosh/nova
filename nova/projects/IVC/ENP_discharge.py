import os
import nep
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from amigo.addtext import linelabel
from read_dina import timeconstant

path = os.path.join(class_dir(nep), "../Data/Energopul/")

# data_mine(path, 'IVS3_discharge', [0, 0.13], [5e3, 60e3])
points = data_load(path, "IVS3_discharge", date="2018_03_01")[0]

to, Io = points[0]["x"], points[0]["y"]  # bare conductor
td, Id = points[1]["x"], points[1]["y"]  # jacket + vessel

tc = timeconstant(to, Io, trim_fraction=0)

Io_o, tau_o, tfit_o, Ifit_o = tc.nfit(1)  # bare single fit
tc.load(td, Id)  # replace with coupled discharge curve
Io_d, tau_d, tfit_d, Ifit_d = tc.nfit(3)

plt.plot(1e3 * (to - to[0]), 1e-3 * Io, "C0-", label="bare conductor")
txt_o = timeconstant.ntxt(Io_o / Io[0], tau_o)
plt.plot(1e3 * (tfit_o - to[0]), 1e-3 * Ifit_o, "C1--", label="exp fit " + txt_o)
plt.plot(1e3 * (td - td[0]), 1e-3 * Id, "C2-", label="conductor + passive structures")
txt_d = timeconstant.ntxt(Io_d / Id[0], tau_d)
plt.plot(1e3 * (tfit_d - td[0]), 1e-3 * Ifit_d, "C3--", label="exp fit " + txt_d)
plt.despine()
plt.xlabel("$t$ ms")
plt.ylabel("$I$ kA")
plt.legend()
