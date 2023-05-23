import os
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
import numpy as np
from nep.DINA.capacitor_discharge import power_supply
from amigo.addtext import linelabel
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from nep.DINA.circuits import impulse_capacitor


path = os.path.join(class_dir(nep), '../Data/EM/')
# data_mine(path, 'VS3_risetime', [5, 6], [0, 3e5])
points = data_load(path, 'VS3_risetime', date='2018_07_20')[0]
t = points[0]['x'] - 5.2
Ic = points[0]['y']/4
to_index = np.argmin(abs(t - 0))
clip_index = np.argmin(abs(t - 0.7))
t = t[to_index:clip_index]
Ic = Ic[to_index:clip_index]

Imax = np.max(Ic)
t10 = t[Ic > 0.1*Imax][0]
t90 = t[Ic > 0.9*Imax][0]
dt_rise = t90 - t10

ax = plt.subplots(1, 1)[1]
text = linelabel(value='', ax=ax, Ndiv=20)
ax.plot(1e3*t, 1e-3*Ic, color='gray')
text.add('PS referance')

ps = power_supply(nturn=4, vessel=True, scenario=-1, code='IO')
ps.solve(Io=0, sign=1, nturn=4, t_pulse=0.3, impulse=True, plot=False,
         pulse_phase=0, t_end=t[-1], origin='start')
ax.plot(1e3*ps.data['t'], 1e-3*ps.data['Ivec'][:, 0], '-')
text.add('flat-top')
ps.solve(Io=0, sign=1, nturn=4, t_pulse=0.0, impulse=True, plot=False,
         pulse_phase=0, t_end=t[-1], origin='start')
ax.plot(1e3*ps.data['t'], 1e-3*ps.data['Ivec'][:, 0], '-')
text.add('spike')

ax_circuit = inset_axes(ax, width='45%', height='45%', loc=1)
impulse_capacitor(ax=ax_circuit)

text.plot()
plt.despine()
ax.set_xlabel('$t$ ms')
ax.set_ylabel('$I_{vs3}$ kA')
plt.set_context('notebook')


print('risetime {:1.1f}ms'.format(1e3*(t90 - t10)))


