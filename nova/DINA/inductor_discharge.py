import numpy as np
from nep.DINA.read_plasma import read_plasma
from read_dina import locate, get_folders
import nep
from amigo.IO import class_dir
from os.path import join
from amigo.pyplot import plt

directory = join(class_dir(nep), '../Scenario_database/disruptions')
folders = get_folders(directory)

Lvs = 1.52e-3
Rvs = 12.01e-3


plt.figure()
pl = read_plasma(directory, folder=folders[3])

Iind = 1e3*pl.Iind
t = 1e-3*pl.t

didt = np.gradient(Iind, t)

vin = pl.Iind*Rvs + Lvs*didt

plt.plot(t, Iind*1e-3)