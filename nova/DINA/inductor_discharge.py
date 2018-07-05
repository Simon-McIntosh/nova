import numpy as np
from nep.DINA.read_plasma import read_plasma
from amigo.pyplot import plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d


Lvs3_DINA = 1.52e-3
Rvs3_DINA = 12.01e-3

Lvs3_LTC = 1.52e-3
Rvs3_LTC = 17.66e-3


plt.figure()
pl = read_plasma('disruptions')
pl.read_file(3)

Ivs3_o = pl.Ivs3_o
t = pl.t

didt = np.gradient(Ivs3_o, t)
vin = pl.Ivs3_o*Rvs3_DINA + Lvs3_DINA*didt
vin_fun = interp1d(t, vin, fill_value='extrapolate')


def dIdt_fun(I, t):
    g = (vin_fun(t) - I*Rvs3_LTC)/Lvs3_LTC
    return g


Iode = odeint(dIdt_fun, 0, t)

plt.plot(t, Ivs3_o*1e-3)
plt.plot(t, Iode*1e-3)
