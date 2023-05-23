from nep.DINA.capacitor_discharge import power_supply
from amigo.pyplot import plt
import os
import nep_data
from amigo.png_tools import data_load
from amigo.IO import class_dir


ps = power_supply(nturn=4, vessel=True, scenario=0, code='Nova',
                  Ip_scale=15/15, read_txt=False, vessel_model='full',
                  impulse=False)
ps.solve()

path = os.path.join(class_dir(nep_data), 'Energopul/')
points = data_load(path, 'Ivs3_plasma_movement', date='2019_02_05')[0]
ts, Is = points[0]['x'], points[0]['y']  # static plasma discharge
tm, Im = points[1]['x'], points[1]['y']  # mobile plasma discharge

ax = plt.subplots(1, 1)[1]
ax.plot(1e3*ts, 1e-3*Is, 'C3-.', label='ENP (static plasma model)')
ax.plot(1e3*tm, 1e-3*Im, 'C3-', label='ENP (dynamic plasma model)')
ax.plot(1e3*ps.data['t'], 1e-3*ps.data['Ivec'][:, 0], 'C0-', label='Nova')
ax.set_xlim([0, 400])
ax.set_xlabel('$t$ ms')
ax.set_ylabel('$I_{vs3}$ kA')
ax.legend()
plt.despine()
