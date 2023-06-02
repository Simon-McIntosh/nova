import numpy as np
from amigo.pyplot import plt
from nep.DINA.read_plasma import read_plasma
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from amigo.addtext import linelabel
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import SchemDraw as schem
import SchemDraw.elements as e


R = 17.66e-3
L = 1.52e-3

tau = L / R


pl = read_plasma("disruptions")
pl.read_file(3)

trip_t = pl.trip_vs3(eps=1e-3)[1]  # ms

t = np.linspace(trip_t, pl.t[-1], int(len(pl.t)))  # equal spacing, s
Iind = interp1d(pl.t, pl.Ivs_o)(t)  # A
Iind_lp = savgol_filter(Iind, 11, 3, mode="mirror")  # lowpass filter

imax = np.argmax(Iind_lp)
tmax = t[imax]
dt_exp = t[-1] - tmax

io = np.argmin(abs(t - tmax - 0.2 * dt_exp))

t_exp = t[io:]
I_exp = Iind_lp[io:]
n_exp = len(t_exp)


def fit(x, *args):
    A, B, tau = x
    t_exp, I_exp = args
    I_fit = A + B * np.exp(-t_exp / tau)
    err = np.linalg.norm(I_exp - I_fit)
    return err


A, B, tau = minimize(fit, [1e4, 1e4, 100e-3], args=(t_exp, I_exp)).x
I_fit = A + B * np.exp(-t / tau)

text = linelabel(postfix="", value="", loc="max")
plt.plot(1e3 * t, 1e-3 * Iind_lp, label="DINA")
plt.plot(1e3 * t[io:], 1e-3 * Iind_lp[io:], label="DINA subset")
plt.plot(1e3 * t, 1e-3 * I_fit, "--", label=r"fit $I=A+Be^{-t/\tau_{approx}}$")
text.add(r"$\tau_{approx}=$" + "{:1.1f}ms".format(1e3 * tau))
text.plot()
plt.despine()
plt.legend(loc=4)
plt.xlabel("$t$ ms")
plt.ylabel("$I_{VS3}$ kA")


ax_schem = inset_axes(plt.gca(), width="30%", height="30%", loc=1)
d = schem.Drawing(unit=4, inches_per_unit=1.5)
V1 = d.add(e.LINE, d="down")

d.add(e.RES, d="right", label="{:1.2f}m$\Omega$".format(1e3 * R))
d.add(e.LINE, d="up")
d.add(e.INDUCTOR, to=V1.start, toplabel="{:1.2f}mH".format(1e3 * L))
d.draw(ax=ax_schem)
