import os
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir
from amigo.pyplot import plt
from nep.DINA.capacitor_discharge import power_supply
from amigo.addtext import linelabel


LTC = {}
path = os.path.join(class_dir(nep), '../Data/LTC/')
points = data_load(path, 'VS3_current', date='2018_03_15')[0]
LTC['md'] = {'t': points[0]['x'], 'I': -points[0]['y']}
points = data_load(path, 'VS3_current_VDE', date='2018_05_24')[0]
LTC['VDE'] = {'t': points[0]['x'], 'I': -points[0]['y']}


Nova = {}
ps = power_supply(nturn=4, vessel=True, code='IO', origin='peak')
ps.solve(Io=0, sign=-1, nturn=4, t_pulse=0.3, scenario=3,
         impulse=True, plot=False)
Nova['md'] = {'t': ps.data['t'], 'I': ps.data['Ivec'][:, 0]}
ps.solve(plot=False, scenario=11, impulse=False)
Nova['VDE'] = {'t': ps.data['t'], 'I': ps.data['Ivec'][:, 0]}

ax = plt.subplots(2, 1)[1]
text = [linelabel(ax=_ax, loc='min', value='1.1f', Ndiv=15,
                  postfix='kA') for _ax in ax]
ax[0].plot(1e3*LTC['md']['t'], 1e-3*LTC['md']['I'], 'C0', label='LTC')
text[0].add('')
ax[0].plot(1e3*Nova['md']['t'], 1e-3*Nova['md']['I'], 'C3', label='Nova')
text[0].add('')
ax[1].plot(1e3*LTC['VDE']['t'], 1e-3*LTC['VDE']['I'], 'C0')
text[1].add('')
ax[1].plot(1e3*Nova['VDE']['t'], 1e-3*Nova['VDE']['I'], 'C3')
text[1].add('')

for _text in text:
    _text.plot(Ralign=True)

ax[0].legend()
plt.despine()
for _ax, name in zip(ax, [ps.pl.dina.folders[3], ps.pl.dina.folders[11]]):
    _ax.set_ylabel('$I$ kA')
    _ax.text(0.5, 0.9, name, transform=_ax.transAxes,
             ha='center', va='top', bbox=dict(facecolor='w',
                                              ec='gray', lw=1,
                                              boxstyle='round', pad=0.5))
ax[-1].set_xlabel('$t$ ms')
plt.legend()
