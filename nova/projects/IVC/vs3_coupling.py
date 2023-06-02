import numpy as np
from scipy.integrate import odeint
from amigo.pyplot import plt
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize

path = os.path.join(class_dir(nep), "../Data/LTC/")
points = data_load(path, "VS3_discharge_main_report", date="2018_06_25")[0]

to, Io = points[0]["x"], points[0]["y"]  # bare conductor
td, Id = points[1]["x"], points[1]["y"]  # jacket + vessel
to -= to[0]
td -= td[0]


tau_vs3 = 78e-3  # vs3 loop time constant
Rvs3 = 17.66e-3
Lvs3 = tau_vs3 * Rvs3


def dIdt(Ic, t, *args):  # current rate (function of odeint)
    Minv = args[0]
    Rc = args[1]
    Idot = np.dot(-Minv, Ic * Rc)  # -IR + vbg
    return Idot


def get_waveform(x, *args):
    Lvs3 = args[0]
    Rvs3 = args[1]
    Ivs3 = args[2]
    t = args[3]
    if len(x) == 3:  # coil pair
        Lvv = x[:2]
        Rvv = [x[2]]
        n = 2
    elif len(x) == 6:  # coil triplet
        Lvv = x[:4]
        Rvv = x[4:]
        n = 3
    M = np.zeros((n, n))
    R = np.zeros(n)
    Io = np.zeros(n)
    M[0, 0] = Lvs3
    M[1, 1] = Lvv[0]
    M[0, 1] = Lvv[1]
    M[1, 0] = Lvv[1]
    R[1] = Rvv[0]
    if n == 3:
        M[2, 2] = Lvv[2]
        M[0, 2] = Lvv[3]
        M[2, 0] = Lvv[3]
        R[2] = Rvv[1]
    Io[0] = Ivs3
    R[0] = Rvs3
    R[:] = Rvs3
    R[-1] *= 0.2
    Minv = np.linalg.inv(M)  # inverse for odeint
    Iode = odeint(dIdt, Io, t, args=(Minv, R)).T
    return Iode


def fit_waveform(x, *args):
    Ivs3_ref = args[-1]
    Iode = get_waveform(x, *args[:-1])
    err = np.linalg.norm(Iode[0] - Ivs3_ref) / np.linalg.norm(Ivs3_ref)
    return err


t = np.linspace(0, td[-1], int(5e3))
Ivs3_ref = interp1d(td, Id)(t)

fsead = 0.7
# xo = [Lvs3, 0.3*Lvs3, Rvs3]
xo = [Lvs3, fsead * Lvs3, Lvs3, fsead * Lvs3, Rvs3, Rvs3]

x = minimize(
    fit_waveform,
    xo,
    method="Nelder-Mead",
    options={"xatol": 1e-5},
    args=(Lvs3, Rvs3, Id[0], t, Ivs3_ref),
).x

err = fit_waveform(x, Lvs3, Rvs3, Id[0], t, Ivs3_ref)
Iode = get_waveform(x, Lvs3, Rvs3, Id[0], t)

print(err, x, x[0] / x[-1])

plt.plot(td, Id, "C2")
plt.plot(t, Iode[0], "C3-")
plt.plot(t, Iode[1], "C4-")
plt.plot(t, Iode[2], "C3-.")
