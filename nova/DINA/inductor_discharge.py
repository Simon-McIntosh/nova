import numpy as np
from nep.DINA.read_plasma import read_plasma
from amigo.pyplot import plt


Lvs = 1.52e-3
Rvs = 12.01e-3


plt.figure()
pl = read_plasma('disruptions')
pl.read_file(3)

Ivs_o = 1e3*pl.Ivs_o
t = 1e-3*pl.t

didt = np.gradient(Ivs_o, t)

vin = pl.Ivs_o*Rvs + Lvs*didt

plt.plot(t, Ivs_o*1e-3)