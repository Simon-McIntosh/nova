from scipy.integrate import odeint
import numpy as np
from amigo.pyplot import plt
from nep.DINA.read_plasma import read_plasma
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import SchemDraw as schem
import SchemDraw.elements as e
from amigo.addtext import linelabel

R = 17.66e-3
L = 1.52e-3

tau = L/R


pl = read_plasma('disruptions')
pl.read_file(3)

trip_t = pl.trip_vs3(eps=1e-3)[1]  # ms

t = np.linspace(trip_t, pl.t[-1], int(len(pl.t)))  # equal spacing, s
Iind = interp1d(pl.t, pl.Ivs_o)(t)  # A

Iind_lp = savgol_filter(Iind, 11, 3, mode='mirror')  # lowpass filter

dIdt = np.gradient(Iind_lp, t)

vs = Iind_lp*R + L*dIdt  # voltage source
vs_fun = interp1d(t, vs, fill_value='extrapolate')


def dIdt_fun(I, t):
    # g = -I*R/L
    g = (vs_fun(t) - I*R)/L
    return g


Iode_plus = Iind[0]+odeint(dIdt_fun, 60e3, t)
Iode_zero = Iind[0]+odeint(dIdt_fun, 0, t)
Iode_minus = Iind[0]+odeint(dIdt_fun, -60e3, t)

Iexp = 60*np.exp(-(t-trip_t*1e-3)/tau)

ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)[1]
ax[0].plot(1e3*t, 1e-3*vs_fun(t), 'C3')
ax[0].set_ylabel('$v_{source}$ kV')

ax[0].text(0.5, 0.9, r'$V_{source} = IR + L \frac{dI}{dt}$',
           transform=ax[0].transAxes,
           ha='center', bbox=dict(facecolor='grey', alpha=0.25),
           fontsize=15)

ax_schem = inset_axes(ax[0], width='30%', height='60%', loc=1)

d = schem.Drawing(unit=4, inches_per_unit=1.5)
V1 = d.add(e.SOURCE_V, label='$V_{source}$')
d.add(e.RES, d='right', label='{:1.2f}m$\Omega$'.format(1e3*R))
d.add(e.LINE, d='down')
d.add(e.INDUCTOR, to=V1.start, botlabel='{:1.2f}mH'.format(1e3*L))

d.draw(ax=ax_schem)

text = linelabel(postfix='', ax=ax[1], value='')
text_max = linelabel(loc='max', Ndiv=5, postfix='kA', ax=ax[1], value='1.1f')
ax[1].plot(1e3*t, 1e-3*Iind_lp, 'C3', label='$I_{DINA}$')
text.add('DINA')
text_max.add('')
ax[1].plot(1e3*t, np.zeros(len(t)), '-_', color='gray', alpha=0.4)
ax[1].plot(1e3*t, 1e-3*Iode_plus, 'C0-', label='$I_{ODE}$, $I_o=+60$kA')
text.add('numerical, $I_o=+60$kA')
ax[1].plot(1e3*t, 60*np.exp(-(t-trip_t)/tau)+1e-3*Iind_lp, '--', color='gray')
text.add('arithmetic sum')

text_max.add('')
ax[1].plot(1e3*t, 1e-3*Iode_zero, 'C0--', label='$I_{ODE}$, $I_o=0$kA')
text.add('numerical, $I_o=0$kA')
text_max.add('')
ax[1].plot(1e3*t, 1e-3*Iode_minus, 'C0-', label='$I_{ODE}$, $I_o=-60$kA')
text.add('numerical, $I_o=-60$kA')
ax[1].plot(1e3*t, -60*np.exp(-(t-trip_t)/tau)+1e-3*Iind_lp, '--', color='gray')
text.add('arithmetic sum')

# ax[1].set_xlim([0, 160])
ax[1].set_xlabel('$t$ ms')
ax[1].set_ylabel('$I_{vs}$ kA')
# ax[1].legend()
text.plot()
text_max.plot(Ralign=True)

plt.despine()
plt.setp(ax[0].get_xticklabels(), visible=False)
