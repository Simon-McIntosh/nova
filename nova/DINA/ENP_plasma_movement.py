import os
import nep_data
from amigo.png_tools import data_mine, data_load
from amigo.IO import class_dir
from amigo.pyplot import plt

path = os.path.join(class_dir(nep_data), 'Energopul/')

# data_mine(path, 'Ivs3_plasma_movement', [0, 1], [-60e3, 0])
points = data_load(path, 'Ivs3_plasma_movement', date='2019_02_05')[0]

ts, Is = points[0]['x'], points[0]['y']  # static plasma discharge
tm, Im = points[1]['x'], points[1]['y']  # mobile plasma discharge

plt.plot(1e3*ts, 1e-3*Is, 'C0-', label='static plasma')
plt.plot(1e3*tm, 1e-3*Im, 'C3-', label='mobile plasma')
plt.despine()
plt.xlabel('$t$ ms')
plt.ylabel('$I$ kA')
plt.legend()
