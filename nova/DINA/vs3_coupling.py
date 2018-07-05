import numpy as np
from scipy.integrate import odeint
from amigo.pyplot import plt
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir
import os

path = os.path.join(class_dir(nep), '../Data/LTC/')
points = data_load(path, 'VS3_discharge_main_report', date='2018_06_25')[0]

to, Io = points[0]['x'], points[0]['y']  # bare conductor
td, Id = points[1]['x'], points[1]['y']  # jacket + vessel
to -= to[0]
td -= td[0]


tau_vs3 = 78e-3  # vs3 loop time constant
Rvs3 = 17.66e-3
Lvs3 = tau_vs3 * Rvs3
Io_vs3 = Io[0]

M = np.array([Lvs3])
Rc = np.array([Rvs3])


def dIdt(Ic, t, *args):  # current rate (function of odeint)
    Minv = args[0]
    Rc = args[1]
    Idot = np.dot(-Minv, Ic*Rc)  # -IR + vbg
    return Idot

if len(M) == 1:
    Minv = 1/M
else:
    Minv = np.linalg.inv(M)  # inverse for odeint

Iode = odeint(dIdt, Io_vs3, to, args=(Minv, Rc)).T


plt.plot(to, Io, 'C0')
plt.plot(to, Iode[0, :], 'C1--')
